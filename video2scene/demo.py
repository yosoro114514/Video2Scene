"""Original synthetic tabletop example using the real editor and scene repository.

No video, dataset, model weights, credentials, or AI calls are used. Blender
builds/exports on CPU. The reference image is a deterministic triangle render.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import uuid
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from .api_v1.__main__ import create_host
from .api_v1.server import make_server
from .capture_evidence import register_capture
from .capture_evidence.coordinates import project, transform_to_matrix, transform_points
from .config import blender_path
from .contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, validate
from .contracts_v1.mocks import example

SCENE_ID = "demo_tabletop"


def request(tool="inspect_scene", params=None):
    return {"contract_version": CONTRACT_VERSION, "scene_id": SCENE_ID,
            "request_id": "demo_" + uuid.uuid4().hex, "tool": tool,
            "params": params or {"revision": "head"}}


def _meshes():
    def box(extents, center):
        mesh = trimesh.creation.box(extents)
        mesh.apply_translation(center)
        return mesh
    table = [box([1.2, .06, .8], [0, .72, 0])]
    table += [box([.07, .69, .07], [x, .345, z]) for x in (-.51, .51) for z in (-.31, .31)]
    table = trimesh.util.concatenate(table)
    cup = trimesh.creation.annulus(r_min=.035, r_max=.045, height=.10, sections=32)
    cup.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    cup.apply_translation([0, .05, 0])
    base = trimesh.creation.cylinder(radius=.045, height=.006, sections=32)
    base.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    base.apply_translation([0, .003, 0])
    cup = trimesh.util.concatenate([cup, base])
    table.visual.face_colors = [164, 116, 76, 255]
    cup.visual.face_colors = [49, 139, 137, 255]
    return {"table_00": table, "t001": cup}


def _camera():
    position = np.array([1.25, 1.55, 1.8])
    forward = np.array([0., .45, 0.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4); c2w[:3, :3] = np.stack([right, down, forward], axis=1); c2w[:3, 3] = position
    return np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]), c2w


def create_sample(storage_root, scene_id=SCENE_ID):
    """Write original geometry and a synthetic capture; return an unimported snapshot."""
    root = Path(storage_root).resolve()
    folder = root / scene_id / "sample"
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = example("model_SceneSnapshot")
    snapshot.update(scene_id=scene_id, revision="demo_r000", parent_revision=None,
                    capture_revision="demo_capture_v1", geometry_revision="demo_geometry_v1")
    meshes = _meshes()
    snapshot["objects"] = snapshot["objects"][:2]
    K, c2w = _camera()
    triangles = []
    for obj in snapshot["objects"]:
        mesh = meshes[obj["instance_id"]]
        asset = folder / (obj["instance_id"] + ".glb")
        mesh.export(asset)
        obj["asset"] = {"asset_version": "synthetic-" + file_hash(asset)[:16],
                        "artifact": {"uri": f"artifact://{scene_id}/sample/{asset.name}", "sha256": file_hash(asset),
                                     "media_type": "model/gltf-binary", "role": "object_geometry"},
                        "native_frame": "asset_y_up", "native_to_object": np.eye(4).tolist()}
        obj["local_bounds_m"] = {"min": mesh.bounds[0].tolist(), "max": mesh.bounds[1].tolist()}
        obj["transform"] = {"translation_m": [0, 0, 0] if obj["instance_id"] == "table_00" else [.15, .75, .05],
                            "rotation_xyzw": [0, 0, 0, 1], "scale": [1, 1, 1]}
        obj["support_id"] = None if obj["instance_id"] == "table_00" else "table_00"
        obj["category"] = "table" if obj["instance_id"] == "table_00" else "cup"
        obj["evidence_ids"] = []
        obj["geometry_source"] = "inferred"
        for state in obj["field_state"].values():
            state.update(source="derived", locked=False, decision_ref=None)
        world = transform_points(transform_to_matrix(obj["transform"]), mesh.vertices)
        pixels, depth = project(world, K, c2w)
        for i, face in enumerate(mesh.faces):
            shade = .60 + .40 * abs(float(mesh.face_normals[i] @ np.array([.3, .8, .5]) / np.linalg.norm([.3, .8, .5])))
            color = tuple(int(c * shade) for c in mesh.visual.face_colors[i][:3])
            triangles.append((float(depth[face].mean()), pixels[face], color))
    image = Image.new("RGB", (640, 480), "#eef2f4")
    draw = ImageDraw.Draw(image)
    for _, points, color in sorted(triangles, key=lambda row: row[0], reverse=True):
        draw.polygon([tuple(point) for point in points], fill=color)
    draw.text((20, 20), "Video2Scene / synthetic editing example", fill="#263c48")
    image_path = folder / "reference.png"; image.save(image_path)
    frame = {"source_frame_id": "synthetic-000", "source_time_s": None, "preview_time_s": 0.,
             "width": 640, "height": 480, "mask": None, "K": K.tolist(), "c2w": c2w.tolist(),
             "source_to_image": np.eye(3).tolist(), "selection_reason": "Original synthetic triangle render; not camera footage.",
             "rgb": {"uri": f"artifact://{scene_id}/sample/reference.png", "sha256": file_hash(image_path),
                     "media_type": "image/png", "role": "source_rgb"}}
    register_capture(root, scene_id, snapshot["capture_revision"], snapshot["geometry_revision"], [frame],
                     metadata={"origin": "original procedural sample", "model_calls": 0,
                               "reconstruction_evidence": False, "license": "Apache-2.0"})
    validate(snapshot, "SceneSnapshot")
    (folder / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def create_demo_host(output, *, blender=None):
    host = create_host(output, blender=blender_path(blender))
    try:
        asyncio.run(host.repo.inspect(request(), host.context()))
    except ContractError as exc:
        if exc.code != "NOT_FOUND":
            raise
        snapshot = create_sample(host.root)
        asyncio.run(host.repo.import_scene(snapshot, host.context()))
        host.store.register_artifacts(snapshot)
    return host


async def export_scene(host):
    view = await host.repo.inspect(request(), host.context())
    scene = copy.deepcopy(view["snapshot"])
    candidate = example("model_CandidateRef")
    candidate.update(candidate_id="demo_export_" + uuid.uuid4().hex, scene_id=SCENE_ID,
                     base_revision=view["head"]["revision"], base_head_seq=view["head"]["head_seq"],
                     candidate_hash=canonical_hash(scene))
    draft = {k: v for k, v in scene.items() if k not in ("revision", "parent_revision")}
    result = await host.backend.build({"candidate": candidate, "scene": draft}, host.context())
    host.store.register_artifacts(result)
    record = {"head": view["head"], "snapshot_hash": canonical_hash(scene), "model_calls": 0,
              "files": [{"uri": ref["uri"], "role": ref["role"], "sha256": ref["sha256"]}
                        for ref in result["artifact_refs"] if ref["role"] in {"scene_blend", "scene_glb"}]}
    (host.root / "demo_export.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def demo_server(host, port):
    server = make_server(host, port=port)
    parent = server.RequestHandlerClass
    class Handler(parent):
        def do_POST(self):
            if urlsplit(self.path).path != "/demo/export":
                return super().do_POST()
            try:
                self.body()  # Enforce the existing local-origin and JSON boundary.
                self.json(asyncio.run(export_scene(host)))
            except ContractError as exc:
                self.error(exc)
            except (ValueError, TypeError):
                self.error(ContractError("INVALID_ARGUMENT", "Invalid export request"))
    server.RequestHandlerClass = Handler
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--blender", help="Blender executable; otherwise use local configuration/PATH")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    if not blender_path(args.blender).is_file():
        parser.error("Blender is required. Set V2S_BLENDER or pass --blender /path/to/blender")
    host = create_demo_host(args.output, blender=args.blender)
    try:
        if args.export_only:
            print(json.dumps(asyncio.run(export_scene(host)), indent=2))
            return
        host.recover()
        server = demo_server(host, args.port)
        print(f"Synthetic demo: http://127.0.0.1:{server.server_port}/?scene_id={SCENE_ID}&demo=1", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        if host.jobs:
            host.jobs.shutdown(wait=True)


if __name__ == "__main__":
    main()
