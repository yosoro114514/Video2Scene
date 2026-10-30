"""Auditable P1 bridge from frozen scene geometry to SpatialLM and back.

This optional branch never changes baseline scene objects. Layout inputs must be
the upstream metric layout.txt, after upstream positive-shift restoration.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

C = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
SIGNS = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.sum(SIGNS[i] != SIGNS[j]) == 1
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def similarity(scale=1.0, yaw=0.0, translation=(0, 0, 0)):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ C
    matrix = np.eye(4)
    matrix[:3, :3] = scale * r
    matrix[:3, 3] = translation
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite transform")
    return matrix


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def transform_camera(camera, matrix):
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3] / scale @ camera[:3, :3]
    out[:3, 3] = transform_points(camera[None, :3, 3], matrix)[0]
    return out


def timestamps(path):
    # Decode actual presentation timestamps, including variable frame rate inputs.
    import shutil

    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    if not shutil.which("ffprobe"):
        p = Path(path).resolve().as_posix()
        args[-1] = "/mnt/" + p[0].lower() + p[2:]
        args = ["wsl", "-d", "Ubuntu-22.04", "--exec"] + args
    data = json.loads(subprocess.check_output(args, stderr=subprocess.PIPE))
    return [float(f["best_effort_timestamp_time"]) for f in data["frames"]]


def prepare(scene_file, output):
    import open3d as o3d

    from .artifacts import ViPEArtifacts

    scene_file, output = Path(scene_file).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = json.loads(scene_file.read_text(encoding="utf-8"))
    if not scene["coordinate_system"].startswith("right-handed Y-up"):
        raise ValueError("Expected frozen Y-up geometry scene")
    cloud_file = scene_file.with_name("points.ply")
    pc = o3d.io.read_point_cloud(str(cloud_file))
    xyz = np.asarray(pc.points).copy()
    if not len(xyz) or not pc.has_colors() or not np.isfinite(xyz).all():
        raise ValueError("Expected nonempty finite RGB point cloud")
    matrix = similarity()
    transformed = transform_points(xyz, matrix)
    pc.points = o3d.utility.Vector3dVector(transformed)
    if not o3d.io.write_point_cloud(str(output / "points.ply"), pc):
        raise OSError("PLY export failed")
    base = scene_file.parent.parent
    input_info = json.loads((base / "input.json").read_text(encoding="utf-8"))
    source = Path(input_info["source"])
    source_pts = timestamps(source)
    artifacts = ViPEArtifacts(base / "vipe", scene["sequence"])
    cams = {c["index"]: c for c in scene["cameras"]}
    required = set(cams)
    if not required <= set(artifacts.indices):
        raise ValueError("Frozen cameras lack matching pose/depth/intrinsics frame IDs")
    selected = {sorted(cams)[i] for i in np.linspace(0, len(cams) - 1, 7, dtype=int)}
    capture = output / "capture"
    capture.mkdir()
    records, max_pose_error, max_projection_error = [], 0.0, 0.0
    vipe_to_scene = np.asarray(scene["vipe_to_scene_rigid"])
    for f in artifacts.frames(scale=scene["scale_factor"]):
        if f.index not in cams:
            continue
        cam = cams[f.index]
        pose = np.asarray(cam["c2w"])
        pose_error = float(np.max(np.abs(vipe_to_scene @ f.c2w - pose)))
        max_pose_error = max(max_pose_error, pose_error)
        if (
            pose_error > 1e-5
            or not np.allclose(cam["intrinsics"], f.intrinsics)
            or cam["resolution"] != [f.rgb.shape[1], f.rgb.shape[0]]
        ):
            raise ValueError(f"Frame {f.index}: frozen camera mismatch")
        spatial_cam = transform_camera(pose, matrix)
        camera_points = np.array([[0.1, 0.2, 1.0], [-0.1, -0.2, 2.0]])
        world = transform_points(camera_points, pose)
        local = transform_points(
            transform_points(world, matrix), np.linalg.inv(spatial_cam)
        )
        max_projection_error = max(
            max_projection_error, float(np.max(np.abs(local - camera_points)))
        )
        source_id = input_info["sampled_source_indices"][f.index]
        record = {
            "frame_id": f.index,
            "source_frame_id": source_id,
            "source_timestamp_seconds": source_pts[source_id],
            "timestamp_source": "ffprobe decoded frame PTS",
            "resolution": cam["resolution"],
            "intrinsics": cam["intrinsics"],
            "c2w": cam["c2w"],
            "world_frame": "scene_y_up",
            "depth_kind": "camera_z",
            "scale_status": "estimated",
            "selected_for_visual_check": f.index in selected,
        }
        if f.index in selected:
            rgb_name, depth_name, mask_name = (
                f"rgb_{f.index:04d}.png",
                f"depth_{f.index:04d}.npy",
                f"valid_{f.index:04d}.npy",
            )
            cv2.imwrite(str(capture / rgb_name), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            np.save(capture / depth_name, f.depth)
            np.save(capture / mask_name, np.isfinite(f.depth) & (f.depth > 0))
            record.update(rgb=rgb_name, depth=depth_name, valid_mask=mask_name)
        else:
            record["artifact_reference"] = {
                "rgb": str(artifacts.rgb_path),
                "depth_archive": str(artifacts.depth_path),
                "depth_entry": artifacts.depth_names[f.index],
            }
        records.append(record)
    (capture / "frames.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    save_json(output / "baseline.json", scene)
    files = [
        scene_file,
        cloud_file,
        source,
        base / "input.json",
        artifacts.rgb_path,
        artifacts.depth_path,
        artifacts.root / "pose" / f"{artifacts.sequence}.npz",
        artifacts.root / "intrinsics" / f"{artifacts.sequence}.npz",
    ]
    manifest = {
        "schema": "video2scene.spatiallm_test/1",
        "scene_scope": "partial_scene",
        "scale_status": "estimated",
        "source_files": {str(p): sha256(p) for p in files},
        "scene_to_spatial": matrix.tolist(),
        "spatial_to_scene": np.linalg.inv(matrix).tolist(),
        "yaw_alignment": "unknown; no room-axis assumption",
        "floor": "unverified; observed origin only",
        "points": len(xyz),
        "spatial_bounds": [transformed.min(0).tolist(), transformed.max(0).tolist()],
        "exported_cloud_sha256": sha256(output / "points.ply"),
        "capture_sha256": sha256(capture / "frames.jsonl"),
        "coordinate_roundtrip_max_error": float(
            np.max(np.abs(transform_points(transformed, np.linalg.inv(matrix)) - xyz))
        ),
        "camera_match_max_error": max_pose_error,
        "projection_roundtrip_max_error": max_projection_error,
        "frame_count": len(records),
        "visual_frame_ids": sorted(selected),
        "source_commit": "8913c44d84a450c53e9340b13317f8cf7144a738",
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def parse_layout(text, inverse):
    """Strict data parser: never eval model output or silently drop bad lines."""
    entries, seen = [], set()
    scale = np.cbrt(np.linalg.det(inverse[:3, :3]))
    rot = inverse[:3, :3] / scale
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(
            r"\s*((wall|door|window|bbox)_\d+)\s*=\s*(Wall|Door|Window|Bbox)\(([^()]*)\)\s*",
            line,
        )
        if not match or match[2].capitalize() != match[3]:
            raise ValueError(f"Malformed layout line: {line}")
        name, kind = match[1], match[2]
        if name in seen:
            raise ValueError(f"Duplicate candidate: {name}")
        seen.add(name)
        args = [v.strip() for v in match[4].split(",")]
        if len(args) != {"wall": 8, "bbox": 8, "door": 6, "window": 6}[kind]:
            raise ValueError(f"Wrong argument count: {line}")
        nums = np.array([float(v) for v in args[(0 if kind == "wall" else 1) :]])
        if not np.isfinite(nums).all():
            raise ValueError(f"Nonfinite candidate: {name}")
        item = {
            "id": "spatiallm:" + name,
            "type": kind,
            "source_line": line,
            "state": "proposed",
            "scale_status": "estimated",
            "extent_kind": "estimated_full_extent",
            "world_frame": "scene_y_up",
        }
        if kind == "bbox":
            x, y, z, angle, sx, sy, sz = nums
            if min(sx, sy, sz) <= 0:
                raise ValueError("Nonpositive box size")
            c, s = np.cos(angle), np.sin(angle)
            item.update(
                label=args[0].strip("\"'"),
                center=transform_points([[x, y, z]], inverse)[0].tolist(),
                dimensions=(nums[4:] * scale).tolist(),
                rotation=(rot @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])).tolist(),
            )
        elif kind == "wall":
            # Upstream outputs zero-thickness wall surfaces by design.
            if (
                nums[6] <= 0
                or nums[7] < 0
                or np.linalg.norm(nums[3:6] - nums[:3]) < 1e-8
            ):
                raise ValueError("Degenerate wall")
            item.update(
                label=kind,
                endpoints=transform_points(nums[:6].reshape(2, 3), inverse).tolist(),
                height=nums[6] * scale,
                thickness=nums[7] * scale,
                up=rot[:, 2].tolist(),
            )
        else:
            if min(nums[3:]) <= 0:
                raise ValueError("Nonpositive opening size")
            item.update(
                label=kind,
                wall_id="spatiallm:" + args[0].strip("\"'"),
                center=transform_points([nums[:3]], inverse)[0].tolist(),
                width=nums[3] * scale,
                height=nums[4] * scale,
                up=rot[:, 2].tolist(),
            )
        entries.append(item)
    walls = {e["id"] for e in entries if e["type"] == "wall"}
    if any(e.get("wall_id") not in walls for e in entries if "wall_id" in e):
        raise ValueError("Opening references missing wall")
    return entries


def corners(item, entries):
    if item.get("type", "bbox") == "wall":
        a, b = np.asarray(item["endpoints"])
        up = np.asarray(item["up"])
        direction = (b - a) / np.linalg.norm(b - a)
        rotation = np.column_stack([direction, np.cross(up, direction), up])
        return (
            (SIGNS * [np.linalg.norm(b - a), item["thickness"], item["height"]] / 2)
            @ rotation.T
            + (a + b) / 2
            + up * item["height"] / 2
        )
    if item.get("type") in {"door", "window"}:
        wall = next(e for e in entries if e["id"] == item["wall_id"])
        a, b = np.asarray(wall["endpoints"])
        direction = (b - a) / np.linalg.norm(b - a)
        up = np.asarray(item["up"])
        rotation = np.column_stack([direction, np.cross(up, direction), up])
        return (
            SIGNS * [item["width"], wall["thickness"], item["height"]] / 2
        ) @ rotation.T + item["center"]
    return (SIGNS * np.asarray(item["dimensions"]) / 2) @ np.asarray(
        item["rotation"]
    ).T + item["center"]


def draw_candidates(image, entries, camera, color):
    pose = np.asarray(camera["c2w"])
    fx, fy, cx, cy = camera["intrinsics"]
    h, w = image.shape[:2]
    for item in entries:
        if (
            not all(k in item for k in ("center", "dimensions", "rotation"))
            and "type" not in item
        ):
            continue
        local = transform_points(corners(item, entries), np.linalg.inv(pose))
        anchors = []
        for i, j in EDGES:
            a, b = local[i].copy(), local[j].copy()
            if max(a[2], b[2]) < 0.02:
                continue
            if a[2] < 0.02:
                a += (b - a) * ((0.02 - a[2]) / (b[2] - a[2]))
            if b[2] < 0.02:
                b += (a - b) * ((0.02 - b[2]) / (a[2] - b[2]))
            uv = np.array([a[:2] / a[2], b[:2] / b[2]]) * [fx, fy] + [cx, cy]
            uv = np.clip(uv, -1000000, 1000000).astype(int)
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), tuple(uv[0]), tuple(uv[1]))
            if ok:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
                anchors.append(p1)
        if anchors:
            anchor = min(anchors, key=lambda v: v[1])
            cv2.putText(
                image,
                item.get("label", "?"),
                (anchor[0], max(15, anchor[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def evaluate(folder, layout_file, output=None):
    folder, layout_file = Path(folder), Path(layout_file)
    output = Path(output) if output else folder
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        sha256(folder / "points.ply") != manifest["exported_cloud_sha256"]
        or sha256(folder / "capture/frames.jsonl") != manifest["capture_sha256"]
    ):
        raise ValueError("Frozen input changed")
    entries = parse_layout(
        layout_file.read_text(), np.array(manifest["spatial_to_scene"])
    )
    scene = json.loads((folder / "baseline.json").read_text())
    save_json(
        output / "candidates.json",
        {
            "schema": "video2scene.spatiallm_candidates/1",
            "state": "needs_visual_review",
            "layout_sha256": sha256(layout_file),
            "candidates": entries,
        },
    )
    records = [
        json.loads(s)
        for s in (folder / "capture/frames.jsonl").read_text().splitlines()
    ]
    comparisons = output / "comparisons"
    comparisons.mkdir(exist_ok=True)
    tiles = []
    for r in records:
        if not r["selected_for_visual_check"]:
            continue
        original = cv2.imread(str(folder / "capture" / r["rgb"]))
        panels = [
            original,
            draw_candidates(original.copy(), scene["objects"], r, (60, 220, 80)),
            draw_candidates(original.copy(), entries, r, (0, 175, 255)),
        ]
        for panel, label in zip(panels, ["SOURCE", "BASELINE", "SpatialLM"]):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (20, 20, 20), -1)
            cv2.putText(
                panel,
                f"{label} / frame {r['frame_id']}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        combined = np.concatenate(panels, axis=1)
        cv2.imwrite(str(comparisons / f"frame_{r['frame_id']:04d}.jpg"), combined)
        if r["frame_id"] in [manifest["visual_frame_ids"][i] for i in (0, 3, 6)]:
            tiles.append(
                cv2.resize(
                    combined,
                    (1200, round(combined.shape[0] * 1200 / combined.shape[1])),
                )
            )
    cv2.imwrite(str(output / "comparison.jpg"), np.concatenate(tiles))
    summary = {
        "baseline_objects": len(scene["objects"]),
        "spatiallm_objects": sum(e["type"] == "bbox" for e in entries),
        "spatiallm_structures": sum(e["type"] != "bbox" for e in entries),
        "labels": [e["label"] for e in entries],
        "accuracy": "not measured; no independent annotated ground truth",
        "scene_scope": "partial_scene",
    }
    save_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    export = sub.add_parser("prepare")
    export.add_argument("--scene-json", required=True)
    export.add_argument("--output", required=True)
    check = sub.add_parser("evaluate")
    check.add_argument("--input", required=True)
    check.add_argument("--layout", required=True)
    check.add_argument(
        "--output", help="Separate assessment directory for each variant"
    )
    a = p.parse_args(argv)
    result = (
        prepare(a.scene_json, a.output)
        if a.action == "prepare"
        else evaluate(a.input, a.layout, a.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
