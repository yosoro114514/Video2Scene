"""Complete the visually annotated scene with measured support surfaces and a room shell."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d
import trimesh

from video2scene.artifacts import ViPEArtifacts, clean_depth, unproject
from video2scene.geometry import (
    level_plane_rotation,
    point_cloud,
    proxy_mesh,
    transform_points,
)
from video2scene.reconstruction import export_models
from render_preview import render
from PIL import Image, ImageDraw, ImageFont


def measure_surfaces(annotation, artifacts, base):
    wanted = {o["frame"] for s in annotation["surfaces"] for o in s["observations"]}
    frames = {
        f.index: f
        for f in artifacts.frames(scale=base["scale_factor"])
        if f.index in wanted
    }
    transform = np.asarray(base["vipe_to_scene_rigid"])
    measurements = {}
    for surface in annotation["surfaces"]:
        samples = []
        for obs in surface["observations"]:
            f = frames[obs["frame"]]
            mask = np.zeros(f.depth.shape, np.uint8)
            cv2.fillPoly(mask, [np.array(obs["polygon"], np.int32)], 1)
            depth = clean_depth(f.depth, base["parameters"]["max_depth"])
            xyz = unproject(depth, f.intrinsics, f.c2w, mask.astype(bool), step=2)
            samples.append(transform_points(xyz, transform))
        points = np.asarray(
            point_cloud(np.vstack(samples)).voxel_down_sample(0.015).points
        )
        cloud = point_cloud(points)
        o3d.utility.random.seed(29)
        plane, ids = cloud.segment_plane(
            distance_threshold=0.025, ransac_n=3, num_iterations=1000
        )
        inliers = points[ids]
        measurements[surface["id"]] = dict(
            points=points,
            inliers=inliers,
            center=np.median(inliers, axis=0),
            plane=np.asarray(plane),
            frames=sorted({o["frame"] for o in surface["observations"]}),
        )
    return measurements


def make_part(
    identifier, label, center, dimensions, rotation, color, assumptions, parent=None
):
    return dict(
        id=identifier,
        label=label,
        name_zh=label,
        parent_id=parent or identifier,
        primitive="box",
        center=np.asarray(center).tolist(),
        dimensions=np.asarray(dimensions).tolist(),
        rotation=np.asarray(rotation).tolist(),
        color=[*color, 255],
        source="Observed RGB surface regions + ViPE depth + procedural structural completion",
        assumptions=assumptions,
    )


def group_mesh(parts):
    return trimesh.util.concatenate([proxy_mesh(o) for o in parts])


def flatten_group(parts, anchor):
    """Rotate all parts together to make the selected thin support slab horizontal."""
    normal = np.asarray(anchor["rotation"])[:, 2]
    if normal[1] < 0:
        normal = -normal
    matrix = level_plane_rotation(normal)
    pivot = np.array(anchor["center"])
    for obj in parts:
        obj["center"] = (pivot + matrix @ (np.array(obj["center"]) - pivot)).tolist()
        obj["rotation"] = (matrix @ np.array(obj["rotation"])).tolist()
        obj["assumptions"].append(
            "Support slab made horizontal; all parts of this entity rotate together. This corrects estimated depth tilt using a support prior."
        )
    return matrix


def complete_room(base, measured, annotation):
    spec = copy.deepcopy(base)
    spec["objects"] = [o for o in spec["objects"] if o["id"].startswith("object_")]
    groups = {}
    for obj in spec["objects"]:
        groups.setdefault(obj["parent_id"], []).append(obj)
    front = {
        "monitor_silver_front",
        "monitor_black_front",
        "keyboard_front",
        "mouse_front",
        "telephone",
        "book_vision",
        "gamepad",
    }
    rear = {"monitor_rear_window", "keyboard_rear"}
    table_for = {
        key: "table_rear"
        if key in rear
        else "table_front"
        if key in front
        else "table_reverse"
        for key in groups
    }
    floor_y = float(measured["floor"]["center"][1])
    # Adjacent desk panels share one top plane; the two raw estimates differ by ~4 cm.
    main_y = float(
        np.mean(
            [
                measured["table_front"]["center"][1],
                measured["table_reverse"]["center"][1],
            ]
        )
    )
    heights = {
        "table_front": main_y,
        "table_reverse": main_y,
        "table_rear": float(measured["table_rear"]["center"][1]),
    }
    # Use a vertical-wall prior. Keep the scene coordinate system and raw scan unchanged.
    nx = measured["window_wall"]["plane"][:3].copy()
    nx[1] = 0
    nx /= np.linalg.norm(nx)
    object_center = np.mean([o["center"] for o in spec["objects"]], axis=0)
    if nx @ (measured["window_wall"]["center"] - object_center) < 0:
        nx *= -1
    nz = np.cross(nx, [0, 1, 0])
    rotation = np.column_stack([nx, [0, 1, 0], nz])
    adjustments = []
    for key, parts in groups.items():
        old_center = group_mesh(parts).centroid.copy()
        old_min = float(group_mesh(parts).bounds[0, 1])
        anchor = None
        if key == "laptop":
            anchor = next(o for o in parts if o["id"].endswith("_base"))
        elif key.startswith(("keyboard_", "book_")):
            anchor = parts[0]
        angle = 0.0
        if anchor:
            mat = flatten_group(parts, anchor)
            angle = float(
                np.rad2deg(np.arccos(np.clip((np.trace(mat) - 1) / 2, -1, 1)))
            )
        hinge_shift = None
        if key == "laptop":
            screen = next(o for o in parts if o["id"].endswith("_screen"))
            base_rotation = np.asarray(anchor["rotation"])
            up = base_rotation[:, 2] * np.sign(base_rotation[1, 2])
            hinge = (
                np.asarray(anchor["center"])
                + base_rotation[:, 1] * anchor["dimensions"][1] / 2
                + up * anchor["dimensions"][2] / 2
            )
            screen_edge = (
                np.asarray(screen["center"])
                - np.asarray(screen["rotation"])[:, 1] * screen["dimensions"][1] / 2
            )
            hinge_shift = hinge - screen_edge
            for part in parts:
                if "_screen" in part["id"]:
                    part["center"] = (np.asarray(part["center"]) + hinge_shift).tolist()
                    part["assumptions"].append(
                        "Screen lower edge joined to the base rear edge to correct inconsistent independent depth-plane estimates; no joint dynamics."
                    )
        top = heights[table_for[key]]
        delta = top - float(group_mesh(parts).bounds[0, 1])
        for obj in parts:
            obj["center"][1] += delta
            obj["supported_by"] = "surface_" + table_for[key] + "_top"
            obj["assumptions"].append(
                "Entity placed on its assigned desk by a shared vertical translation; support constraints change the estimated pose."
            )
        adjustments.append(
            dict(
                entity=key,
                table=table_for[key],
                original_bottom=old_min,
                support_height=top,
                vertical_translation=delta,
                levelling_degrees=angle,
                screen_to_base_shift=None
                if hinge_shift is None
                else hinge_shift.tolist(),
                centroid_shift=(group_mesh(parts).centroid - old_center).tolist(),
            )
        )

    local = {
        key: group_mesh(parts).vertices @ rotation for key, parts in groups.items()
    }
    main_points = np.vstack([p for key, p in local.items() if key not in rear])
    # Footprints enclose the supported objects; invisible tabletop edges are inferred.
    lo, hi = main_points.min(axis=0), main_points.max(axis=0)
    lo[[0, 2]] -= [0.12, 0.12]
    hi[[0, 2]] += [0.12, 0.12]
    front_mid = np.median([local[k][:, 2].mean() for k in front])
    reverse_mid = np.median(
        [local[k][:, 2].mean() for k in groups if k not in rear | front]
    )
    seam = float((front_mid + reverse_mid) / 2)
    footprints = {
        "table_front": [float(lo[0]), float(hi[0]), seam, float(hi[2])],
        "table_reverse": [float(lo[0]), float(hi[0]), float(lo[2]), seam],
    }
    rear_points = np.vstack(
        [p for key, p in local.items() if key in rear]
        + [measured["table_rear"]["inliers"] @ rotation]
    )
    rlo, rhi = rear_points.min(axis=0), rear_points.max(axis=0)
    footprints["table_rear"] = [
        float(rlo[0] - 0.12),
        float(rhi[0] + 0.12),
        float(rlo[2] - 0.12),
        float(rhi[2] + 0.12),
    ]
    assumptions = [
        annotation["method"],
        "Monocular metric scale remains estimated.",
        "Rectangular boundaries and hidden table legs are inferred, not recovered geometry.",
    ]
    structure = []
    names = {
        "table_front": "前侧工作桌",
        "table_reverse": "后侧工作桌",
        "table_rear": "靠窗小桌",
    }
    table_thickness = 0.04
    for key, (x0, x1, z0, z1) in footprints.items():
        top = heights[key]
        center = (
            np.array([(x0 + x1) / 2, top - table_thickness / 2, (z0 + z1) / 2])
            @ rotation.T
        )
        structure.append(
            make_part(
                "surface_" + key + "_top",
                names[key] + "桌面",
                center,
                [x1 - x0, table_thickness, z1 - z0],
                rotation,
                [228, 227, 216],
                assumptions
                + [
                    "Adjacent main desks share a level top estimated from both observed patches."
                ],
                key,
            )
        )
        leg_h = top - table_thickness - floor_y
        if leg_h <= 0:
            raise ValueError("Desk must be above the observed floor")
        for i, (x, z) in enumerate(
            [
                (x0 + 0.12, z0 + 0.12),
                (x1 - 0.12, z0 + 0.12),
                (x0 + 0.12, z1 - 0.12),
                (x1 - 0.12, z1 - 0.12),
            ]
        ):
            center = np.array([x, floor_y + leg_h / 2, z]) @ rotation.T
            structure.append(
                make_part(
                    f"surface_{key}_leg_{i}",
                    names[key] + "桌腿",
                    center,
                    [0.055, leg_h, 0.055],
                    rotation,
                    [75, 79, 80],
                    assumptions
                    + [
                        "Four square metal legs are a simple support model; actual hidden leg construction is unknown."
                    ],
                    key,
                )
            )
    observed_xwall = float(np.median(measured["window_wall"]["inliers"] @ nx))
    observed_zback = float(np.median(measured["side_wall"]["inliers"] @ nz))
    all_x0 = min(f[0] for f in footprints.values())
    all_x1 = max(f[1] for f in footprints.values())
    all_z0 = min(f[2] for f in footprints.values())
    all_z1 = max(f[3] for f in footprints.values())
    # Allow an explicit small completion offset if noisy walls cut through furniture.
    xwall = max(observed_xwall, all_x1 + 0.12)
    zback = min(observed_zback, all_z0 - 0.12)
    xmin, zmax = all_x0 - 0.75, all_z1 + 0.80
    wall_height, wall_thickness = 2.35, 0.10
    center = (
        np.array([(xmin + xwall) / 2, floor_y - 0.045, (zback + zmax) / 2]) @ rotation.T
    )
    structure.append(
        make_part(
            "surface_floor",
            "地板",
            center,
            [xwall - xmin, 0.09, zmax - zback],
            rotation,
            [173, 155, 127],
            assumptions
            + [
                "Floor elevation is measured from visible floor patches. The rectangular crop extends beyond the furniture; full room dimensions are unknown."
            ],
        )
    )
    center = (
        np.array(
            [xwall + wall_thickness / 2, floor_y + wall_height / 2, (zback + zmax) / 2]
        )
        @ rotation.T
    )
    structure.append(
        make_part(
            "surface_wall_window",
            "窗侧墙面",
            center,
            [wall_thickness, wall_height, zmax - zback],
            rotation,
            [224, 226, 220],
            assumptions
            + [
                "Vertical wall normal/offset use the visible wall patch, with a vertical prior. Height 2.35 m and full extent are completion assumptions."
            ],
        )
    )
    center = (
        np.array(
            [(xmin + xwall) / 2, floor_y + wall_height / 2, zback - wall_thickness / 2]
        )
        @ rotation.T
    )
    structure.append(
        make_part(
            "surface_wall_side",
            "侧墙面",
            center,
            [xwall - xmin, wall_height, wall_thickness],
            rotation,
            [214, 218, 212],
            assumptions
            + [
                "Side wall position uses the visible rear vertical patch; perpendicularity to the window wall and height are room priors. This is a cutaway corner, not the full room."
            ],
        )
    )
    spec["objects"] = structure + spec["objects"]
    spec["structure"] = dict(
        method=annotation["method"],
        floor_height=floor_y,
        room_rotation=rotation.tolist(),
        footprints=footprints,
        table_heights=heights,
        wall_bounds=dict(
            xmin=xmin, xwall=xwall, zback=zback, zmax=zmax, height=wall_height
        ),
        observed_wall_offsets=dict(window=observed_xwall, side=observed_zback),
        support_adjustments=adjustments,
        note="Raw ViPE scan unchanged; modeled poses have support corrections. Partial room, unseen boundaries completed.",
    )
    spec["parameters"]["modelling"] = "templates_with_room_support_constraints"
    spec["stats"]["model_parts"] = len(spec["objects"])
    spec["stats"]["structural_parts"] = len(structure)
    spec["stats"]["planar_patches"] = 6
    return spec


def install_viewer(folder):
    html = Path("video2scene/viewer.html").read_text(encoding="utf-8")
    html = html.replace(
        "<title>Video2Scene · 三维场景</title>",
        "<title>桌面与房间 · Video2Scene</title>",
    )
    html = html.replace(
        "<h1>Video2Scene</h1>",
        "<h1>桌面与房间</h1><p>已补充桌面、桌腿、地板和两面墙。各物体已放到对应桌面上。可关闭墙面查看内部。</p>",
    )
    html = html.replace(
        'id="scan" type="checkbox" checked', 'id="scan" type="checkbox"'
    )
    html = html.replace(
        'id="objects" type="checkbox"', 'id="objects" type="checkbox" checked'
    )
    html = html.replace(
        '<label><input id="planes" type="checkbox"> 平面几何体</label>',
        '<label><input id="furniture" type="checkbox" checked> 桌面与桌腿</label><label><input id="floor" type="checkbox" checked> 地板</label><label><input id="walls" type="checkbox" checked> 墙面</label>',
    )
    html = html.replace(
        "function kind(name){return name.startsWith('scan')?'scan':name.startsWith('surface')?'planes':'objects'}",
        "function kind(name){return name.startsWith('scan')?'scan':name.startsWith('surface_table')?'furniture':name.startsWith('surface_floor')?'floor':name.startsWith('surface_wall')?'walls':'objects'}",
    )
    html = html.replace(
        "document.querySelector('#planes').checked=true;",
        "document.querySelector('#furniture').checked=true;",
    )
    html = html.replace("`${spec.label}\\n尺寸", "`${spec.name_zh||spec.label}\\n尺寸")
    html = html.replace(
        "单位为估计米制。简化几何体的隐藏面及厚度包含建模假设。",
        "尺寸为估计值。桌腿、墙高和未见边界为简化补全；此处展示房间的一角。",
    )
    html = html.replace(
        'href="scene.json" download>场景参数</a>',
        'href="scene.json" download>场景参数</a> · <a href="model.obj" download>模型 OBJ</a>',
    )
    html = html.replace(
        "bounds=new THREE.Box3().setFromObject(root);",
        "bounds=new THREE.Box3();for(const mesh of meshes)if(kind(mesh.name)!=='scan')bounds.expandByObject(mesh);",
    )
    old_reset = "function reset(top=false){controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(top?0:1.3,top?2.5:0.8,top?0.001:1.4).multiplyScalar(radius));camera.near=Math.max(radius/1000,.001);camera.far=radius*100;camera.updateProjectionMatrix();controls.update();}"
    new_reset = "function reset(top=false){let r=meta.structure.room_rotation;let b=meta.structure.wall_bounds;let local=new THREE.Vector3((b.xmin+b.xwall)/2,meta.structure.floor_height+.55,(b.zback+b.zmax)/2);let rot=new THREE.Matrix3().set(...r.flat());let target=local.applyMatrix3(rot);controls.target.copy(target);let offset=new THREE.Vector3(top?0:-1.1,top?2.5:1.15,top?.001:1.4).applyMatrix3(rot).multiplyScalar(radius);camera.position.copy(target).add(offset);camera.near=Math.max(radius/1000,.001);camera.far=radius*100;camera.updateProjectionMatrix();controls.update();}"
    if old_reset not in html:
        raise ValueError("Viewer reset template changed")
    html = html.replace(old_reset, new_reset)
    html = html.replace(
        "renderer.setSize(innerWidth,innerHeight);",
        "renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.setSize(innerWidth,innerHeight);",
        1,
    )
    html = html.replace(
        "o.material.side=THREE.DoubleSide;",
        "o.material.side=THREE.DoubleSide;o.castShadow=kind(o.name)!=='scan';o.receiveShadow=true;",
    )
    html = html.replace(
        "visibility();reset();",
        "light.castShadow=true;light.shadow.mapSize.set(2048,2048);light.shadow.camera.left=-radius*2;light.shadow.camera.right=radius*2;light.shadow.camera.top=radius*2;light.shadow.camera.bottom=-radius*2;light.shadow.camera.far=30;light.shadow.bias=-.0005;visibility();reset();",
    )
    if (folder / "source_video.mp4").exists():
        video_panel = (
            '<details open style="position:absolute;right:20px;top:20px;width:360px;max-width:42vw;z-index:2;background:#fffffff0;border:1px solid #d9e1e8;border-radius:10px;padding:12px">'
            '<summary style="cursor:pointer;font-weight:600;margin-bottom:8px">原始输入视频 · 16 秒</summary>'
            '<video controls playsinline preload="metadata" style="display:block;width:100%;border-radius:6px" src="source_video.mp4"></video>'
            '<p style="font-size:12px;margin:8px 0 0">可暂停、拖动进度或全屏查看。点击标题可收起。<a href="source_video.mp4" target="_blank" rel="noopener">单独打开</a></p></details>'
        )
        html = html.replace(
            '<script type="module">', video_panel + '\n<script type="module">'
        )
    (folder / "index.html").write_text(html, encoding="utf-8")


def validate_supports(spec, folder):
    exported = trimesh.load(folder / "model.glb", force="scene")
    checks = []
    for adjustment in spec["structure"]["support_adjustments"]:
        entity = adjustment["entity"]
        names = [o["id"] for o in spec["objects"] if o.get("parent_id") == entity]
        verts = np.vstack([exported.geometry[name].vertices for name in names])
        bottom = verts[:, 1].min()
        error = float(bottom - adjustment["support_height"])
        anchor_name = next(
            (
                name
                for name in names
                if name.endswith("_foot") or name.endswith("_base")
            ),
            names[0],
        )
        anchor_error = float(
            exported.geometry[anchor_name].bounds[0, 1] - adjustment["support_height"]
        )
        local = verts[verts[:, 1] < bottom + 0.005] @ np.array(
            spec["structure"]["room_rotation"]
        )
        candidates = (
            ["table_rear"]
            if adjustment["table"] == "table_rear"
            else ["table_front", "table_reverse"]
        )
        supported = np.zeros(len(local), bool)
        for table in candidates:
            x0, x1, z0, z1 = spec["structure"]["footprints"][table]
            supported |= (
                (local[:, 0] >= x0 - 1e-5)
                & (local[:, 0] <= x1 + 1e-5)
                & (local[:, 2] >= z0 - 1e-5)
                & (local[:, 2] <= z1 + 1e-5)
            )
        checks.append(
            dict(
                entity=entity,
                height_error_m=error,
                support_part=anchor_name,
                support_part_height_error_m=anchor_error,
                contact_vertices_inside_table=bool(supported.all()),
            )
        )
    leg_errors = [
        abs(float(m.bounds[0, 1]) - spec["structure"]["floor_height"])
        for name, m in exported.geometry.items()
        if "_leg_" in name
    ]
    report = dict(
        meshes=len(exported.geometry),
        all_watertight=all(m.is_watertight for m in exported.geometry.values()),
        finite_vertices=all(
            bool(np.isfinite(m.vertices).all()) for m in exported.geometry.values()
        ),
        supports=checks,
        max_leg_floor_gap_m=max(leg_errors),
        note="Checks modeled contact consistency after constraints, not reconstruction accuracy.",
    )
    assert report["all_watertight"] and report["finite_vertices"]
    assert all(
        abs(c["height_error_m"]) < 1e-5
        and abs(c["support_part_height_error_m"]) < 1e-5
        and c["contact_vertices_inside_table"]
        for c in checks
    )
    assert report["max_leg_floor_gap_m"] < 1e-5
    (folder / "structure_validation.json").write_text(json.dumps(report, indent=2))
    return report


def render_room(spec, folder):
    scene = trimesh.load(folder / "model.glb", force="scene")
    mesh = scene.to_geometry()
    rotation = np.array(spec["structure"]["room_rotation"])
    b = spec["structure"]["wall_bounds"]
    target = (
        np.array(
            [
                (b["xmin"] + b["xwall"]) / 2,
                spec["structure"]["floor_height"] + 1.0,
                (b["zback"] + b["zmax"]) / 2,
            ]
        )
        @ rotation.T
    )
    span = max(b["xwall"] - b["xmin"], b["zmax"] - b["zback"])
    direction = np.array([-1.1, 1.15, 1.4])
    direction /= np.linalg.norm(direction)
    eye = target + (direction @ rotation.T) * span * 2.2
    canvas = Image.new("RGB", (1500, 990), (237, 241, 245))
    canvas.paste(render(mesh, eye, target, width=1500, height=860), (0, 70))
    draw = ImageDraw.Draw(canvas)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    draw.text(
        (28, 20),
        "Room corner | desks, floor, walls and supported objects",
        font=ImageFont.truetype(font, 29),
        fill="#203849",
    )
    draw.text(
        (28, 944),
        "Visible surface positions from ViPE; unseen extents/legs completed. Object poses adjusted to rest on desks.",
        font=ImageFont.truetype(font, 19),
        fill="#4c6477",
    )
    canvas.save(folder / "room_preview.png")
    canvas.save(folder / "preview.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("outputs/tum_desk/scene_refined/self_modelled"),
    )
    p.add_argument("--vipe-dir", type=Path, default=Path("outputs/tum_desk/vipe"))
    p.add_argument(
        "--annotations", type=Path, default=Path("examples/tum_desk_structure.json")
    )
    p.add_argument("--measure-only", action="store_true")
    args = p.parse_args()
    backup = args.scene_dir / "before_room"
    source = (
        backup / "scene.json"
        if (backup / "scene.json").exists()
        else args.scene_dir / "scene.json"
    )
    base = json.loads(source.read_text(encoding="utf-8"))
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    measured = measure_surfaces(
        annotation, ViPEArtifacts(args.vipe_dir, base["sequence"]), base
    )
    report = {
        key: {
            "center": m["center"].tolist(),
            "plane": m["plane"].tolist(),
            "inliers": len(m["inliers"]),
            "samples": len(m["points"]),
            "quantiles": np.quantile(m["points"], [0.05, 0.95], axis=0).tolist(),
        }
        for key, m in measured.items()
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.measure_only:
        return
    if not backup.exists():
        backup.mkdir()
        for name in [
            "scene.json",
            "model.glb",
            "model.obj",
            "scene.glb",
            "index.html",
            "preview.png",
            "reprojection.jpg",
            "validation.json",
        ]:
            if (args.scene_dir / name).exists():
                shutil.copyfile(args.scene_dir / name, backup / name)
    spec = complete_room(base, measured, annotation)
    scan = trimesh.load(args.scene_dir / "scan.glb", force="scene").to_geometry()
    export_models(spec, args.scene_dir, scan)
    shutil.copyfile(args.annotations, args.scene_dir / "structure_annotations.json")
    install_viewer(args.scene_dir)
    validation = validate_supports(spec, args.scene_dir)
    render_room(spec, args.scene_dir)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
