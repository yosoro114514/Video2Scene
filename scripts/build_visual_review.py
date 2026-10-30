"""Controlled comparison using polygons authored by the assistant from RGB frames.

Stage A changes labels, masks and persistent IDs, keeping the old box fitter.
Stage B additionally uses observed panel corners and explicit simple shape priors.
Neither stage runs YOLO or changes ViPE depth/poses/the scan/alignment.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import open3d as o3d
import trimesh

from video2scene.artifacts import ViPEArtifacts, clean_depth, unproject
from video2scene.geometry import fit_box, point_cloud, transform_points
from video2scene.reconstruction import export_models


def collect(annotation, artifacts, transform, scale, max_depth, review_dir):
    wanted = {o["frame"] for e in annotation["entities"] for o in e["observations"]}
    frames = {f.index: f for f in artifacts.frames(scale=scale) if f.index in wanted}
    if wanted - frames.keys():
        raise ValueError(f"Missing annotated frames: {wanted - frames.keys()}")
    results, overlays = {}, {}
    rng = np.random.default_rng(14)
    for entity in annotation["entities"]:
        observations, points, colors = [], [], []
        edge_color = tuple(int(x) for x in rng.integers(45, 240, 3))
        for obs in entity["observations"]:
            f = frames[obs["frame"]]
            if [f.depth.shape[1], f.depth.shape[0]] != annotation["image_size"]:
                raise ValueError("Annotation image resolution does not match artifacts")
            mask = np.zeros(f.depth.shape, np.uint8)
            polygon = np.asarray(obs["polygon"], np.int32)
            cv2.fillPoly(mask, [polygon], 1)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
            depth = clean_depth(f.depth, max_depth)
            xyz = unproject(depth, f.intrinsics, f.c2w, mask, step=2)
            if len(xyz) < 12:
                raise ValueError(f"Insufficient depth: {entity['id']} frame {f.index}")
            cloud = point_cloud(xyz).voxel_down_sample(0.01)
            labels = np.asarray(cloud.cluster_dbscan(eps=0.07, min_points=4))
            if np.any(labels >= 0):
                xyz = np.asarray(cloud.points)[
                    labels == np.bincount(labels[labels >= 0]).argmax()
                ]
            else:
                xyz = np.asarray(cloud.points)
            xyz = transform_points(xyz, transform)
            observations.append(dict(annotation=obs, frame=f, points=xyz))
            points.append(xyz)
            colors.append(np.median(f.rgb[mask & (depth > 0)], axis=0))
            if f.index not in overlays:
                overlays[f.index] = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            canvas = overlays[f.index]
            cv2.polylines(canvas, [polygon], True, edge_color, 2, cv2.LINE_AA)
            anchor = polygon.mean(axis=0).astype(int)
            cv2.putText(
                canvas,
                entity["id"],
                tuple(anchor),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                edge_color,
                1,
                cv2.LINE_AA,
            )
        xyz = np.asarray(point_cloud(np.vstack(points)).voxel_down_sample(0.015).points)
        results[entity["id"]] = dict(
            points=xyz,
            observations=observations,
            color=[*np.median(colors, axis=0).astype(int).tolist(), 255],
        )
    for index, canvas in overlays.items():
        cv2.imwrite(str(review_dir / f"annotated_{index:03d}.jpg"), canvas)
    return results


def record(
    entity,
    suffix,
    center,
    dims,
    rotation,
    primitive,
    color,
    source,
    frames,
    assumptions,
):
    return dict(
        id=f"object_{entity['id']}_{suffix}",
        parent_id=entity["id"],
        label=entity["label"],
        name_zh=entity["name_zh"],
        primitive=primitive,
        center=np.asarray(center).tolist(),
        dimensions=np.asarray(dims).tolist(),
        rotation=np.asarray(rotation).tolist(),
        color=color,
        source=source,
        frame_indices=sorted(set(frames)),
        assumptions=assumptions,
    )


def panel_from_observation(observation, transform, thickness):
    """Fit a plane within a visual polygon, then intersect its four observed corner rays."""
    frame, points, annotation = (
        observation["frame"],
        observation["points"],
        observation["annotation"],
    )
    cloud = point_cloud(points)
    o3d.utility.random.seed(17)
    _, ids = cloud.segment_plane(
        distance_threshold=0.015, ransac_n=3, num_iterations=500
    )
    inliers = points[ids] if len(ids) >= 12 else points
    center = np.median(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    c2w = transform @ frame.c2w
    uv = np.asarray(annotation["quad"], float)
    fx, fy, cx, cy = frame.intrinsics
    dirs = (
        np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(4)])
        @ c2w[:3, :3].T
    )
    denominator = dirs @ normal
    if np.any(abs(denominator) < 0.03):
        raise ValueError("Panel reference is too edge-on for a stable fit")
    distance = ((center - c2w[:3, 3]) @ normal) / denominator
    if np.any(distance <= 0):
        raise ValueError("Panel fitted behind the camera")
    corners = c2w[:3, 3] + dirs * distance[:, None]
    right = (corners[1] - corners[0] + corners[2] - corners[3]) / 2
    up = (corners[0] - corners[3] + corners[1] - corners[2]) / 2
    right /= np.linalg.norm(right)
    up -= (up @ right) * right
    up /= np.linalg.norm(up)
    rotation = np.column_stack([right, up, np.cross(right, up)])
    local = corners @ rotation
    low, high = local.min(axis=0), local.max(axis=0)
    dims = high - low
    dims[2] = thickness
    return (
        corners.mean(axis=0),
        dims,
        rotation,
        dict(
            reference_frame=frame.index,
            plane_inliers=len(inliers),
            depth_samples=len(points),
        ),
    )


def build(annotation, measured, base_spec, mode):
    spec = copy.deepcopy(base_spec)
    spec["objects"] = [
        copy.deepcopy(o) for o in base_spec["objects"] if o["id"].startswith("surface")
    ]
    transform = np.asarray(base_spec["vipe_to_scene_rigid"])
    table = next(o for o in spec["objects"] if o["id"] == "surface_00")
    tabletop = table["center"][1] + 0.0125
    template_assumptions = [
        "Labels, visible polygons and cross-view IDs were authored by the assistant after looking at RGB frames.",
        "ViPE scale and depth remain estimated; hidden surfaces are not measured.",
    ]
    for entity in annotation["entities"]:
        data = measured[entity["id"]]
        frames = [o["frame"] for o in entity["observations"]]
        center, dims, rotation = fit_box(data["points"])
        if mode == "boxes":
            obj = record(
                entity,
                "box",
                center,
                dims,
                rotation,
                "box",
                data["color"],
                "Assistant polygons + original multi-view box fitter",
                frames,
                template_assumptions
                + [
                    "Same fit_box function as the YOLO baseline; annotation scope for monitors is the panel."
                ],
            )
            obj["observations"] = len(data["points"])
            spec["objects"].append(obj)
            continue
        kind = entity["template"]
        references = [
            o
            for o in data["observations"]
            if o["frame"].index == entity["preferred_frame"]
            and "quad" in o["annotation"]
        ]
        if kind in {"monitor", "laptop", "slab"}:
            for obs in references:
                part = obs["annotation"]["part"]
                thickness = (
                    0.035 if kind == "monitor" else 0.018 if kind == "laptop" else 0.025
                )
                pc, pd, pr, fit_info = panel_from_observation(obs, transform, thickness)
                assumptions = template_assumptions + [
                    f"Thickness {thickness:.3f} m is a modelling prior; footprint and orientation come from the annotated plane/corners."
                ]
                shell = record(
                    entity,
                    part,
                    pc,
                    pd,
                    pr,
                    "box",
                    entity["color"],
                    "Assistant annotation + depth-plane/corner fit",
                    frames,
                    assumptions,
                )
                shell["fit"] = fit_info
                spec["objects"].append(shell)
                if kind == "monitor" or (kind == "laptop" and part == "screen"):
                    side = -1 if entity["id"] == "monitor_black_reverse" else 1
                    color = (
                        [24, 28, 35, 255] if kind == "monitor" else [109, 156, 187, 255]
                    )
                    spec["objects"].append(
                        record(
                            entity,
                            part + "_display",
                            pc + pr[:, 2] * side * (thickness / 2 + 0.001),
                            pd * [0.91, 0.86, 0] + [0, 0, 0.002],
                            pr,
                            "box",
                            color,
                            "Simple display surface",
                            frames,
                            assumptions
                            + [
                                "Bezel proportions and display colour are visual approximations."
                            ],
                        )
                    )
                if kind == "monitor":
                    bottom = pc - pr[:, 1] * pd[1] / 2
                    base_y = min(tabletop, bottom[1] - 0.045)
                    stand_h = max(bottom[1] - base_y, 0.045)
                    spec["objects"].append(
                        record(
                            entity,
                            "stand",
                            [bottom[0], base_y + stand_h / 2, bottom[2]],
                            [0.035, stand_h, 0.035],
                            np.eye(3),
                            "cylinder",
                            [45, 47, 50, 255],
                            "Procedural monitor stand",
                            frames,
                            template_assumptions
                            + [
                                "Stand dimensions and support height are inferred; this is not a measurement of the hidden mount."
                            ],
                        )
                    )
                    spec["objects"].append(
                        record(
                            entity,
                            "foot",
                            [bottom[0], base_y, bottom[2]],
                            [pd[0] * 0.5, 0.015, pd[0] * 0.45],
                            np.eye(3),
                            "cylinder",
                            [45, 47, 50, 255],
                            "Procedural monitor foot",
                            frames,
                            template_assumptions
                            + [
                                "Approximate oval base; width proportional to the measured panel."
                            ],
                        )
                    )
            if not references:
                raise ValueError(f"No corner reference for {entity['id']}")
        else:
            primitive = "ellipsoid" if kind == "ellipsoid" else "box"
            spec["objects"].append(
                record(
                    entity,
                    "body",
                    center,
                    dims,
                    rotation,
                    primitive,
                    entity["color"],
                    "Assistant polygon + measured bounding geometry",
                    frames,
                    template_assumptions,
                )
            )
    spec["parameters"]["segment"] = "assistant_visual_annotations"
    spec["parameters"]["modelling"] = mode
    spec["annotation_method"] = annotation["method"]
    spec["annotation_scope"] = annotation["scope"]
    spec["stats"]["semantic_objects"] = len(annotation["entities"])
    spec["stats"]["model_parts"] = len(spec["objects"])
    spec["stats"].pop("elapsed_seconds", None)
    spec["entities"] = [
        {k: v for k, v in e.items() if k != "observations"}
        for e in annotation["entities"]
    ]
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--annotations", type=Path, default=Path("examples/tum_desk_self_labels.json")
    )
    p.add_argument(
        "--base-scene", type=Path, default=Path("outputs/tum_desk/scene_refined")
    )
    p.add_argument("--vipe-dir", type=Path, default=Path("outputs/tum_desk/vipe"))
    args = p.parse_args()
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    base = json.loads((args.base_scene / "scene.json").read_text())
    review = Path("outputs/tum_desk/visual_review")
    review.mkdir(exist_ok=True, parents=True)
    measured = collect(
        annotation,
        ViPEArtifacts(args.vipe_dir, base["sequence"]),
        np.asarray(base["vipe_to_scene_rigid"]),
        base["scale_factor"],
        base["parameters"]["max_depth"],
        review,
    )
    scan = trimesh.load(args.base_scene / "scan.glb", force="scene").to_geometry()
    for mode, name in [("boxes", "self_boxes"), ("templates", "self_modelled")]:
        output = args.base_scene / name
        spec = build(annotation, measured, base, mode)
        export_models(spec, output, scan)
        for filename in ["scan.glb", "scan.obj", "input_frame.jpg"]:
            shutil.copyfile(args.base_scene / filename, output / filename)
        shutil.copyfile(args.annotations, output / "visual_annotations.json")
        print(name, json.dumps(spec["stats"]), flush=True)
    counts = dict(
        yolo=Counter(
            o["label"] for o in base["objects"] if o["id"].startswith("object")
        ),
        assistant=Counter(e["label"] for e in annotation["entities"]),
    )
    (review / "recognition_comparison.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
