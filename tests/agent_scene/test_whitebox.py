"""Synthetic CPU checks for frame identity, coordinates, and evidence binding."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from video2scene.artifacts import Frame
from video2scene.agent_scene import whitebox
from video2scene.agent_scene.evidence import MaskEvidenceService
from video2scene.capture_evidence.storage import resolve_artifact
from video2scene.contracts_v1 import ContractError
from video2scene.validation_v1.validator import _geometry_issues


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def frame(index=0):
    return Frame(index, np.full((40, 60, 3), 120, dtype=np.uint8),
                 np.full((40, 60), 3., dtype=np.float32),
                 np.array([45., 45., 30., 20.]), np.eye(4))


def test_frames_use_explicit_index_mapping_not_sequence_position(tmp_path, monkeypatch):
    save(tmp_path / "frames/manifest.json", {"frames": [
        {"frame_index": 0, "source_frame_index": 11},
        {"frame_index": 1, "source_frame_index": 42},
        {"frame_index": 2, "source_frame_index": 99}]})
    items = [frame(2), frame(0), frame(1)]
    class FakeArtifacts:
        def __init__(self, root):
            assert Path(root) == tmp_path / "vipe"
        def frames(self):
            return iter(items)
    monkeypatch.setattr(whitebox, "ViPEArtifacts", FakeArtifacts)
    result = whitebox.frames_by_source(tmp_path)
    assert result[11] is items[1]
    assert result[42] is items[2]
    assert result[99] is items[0]


def test_floor_fit_recovers_tilt_without_becoming_a_reflection():
    x, z = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    local = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    rotation = Rotation.from_euler("xyz", [17, 0, -8], degrees=True).as_matrix()
    points = local @ rotation.T + [1., 2., 3.]
    normal, center, inliers, residual = whitebox.fit_plane(points)
    actual_up = rotation[:, 1]
    if normal @ actual_up < 0:
        normal = -normal
    assert normal @ actual_up > .999999
    assert residual < 1e-10
    right = rotation[:, 0]
    scene_rotation = np.stack((right, normal, np.cross(right, normal)))
    assert np.linalg.det(scene_rotation) == pytest.approx(1.)
    transformed = (inliers - center) @ scene_rotation.T
    assert np.max(np.abs(transformed[:, 1])) < 1e-10


@pytest.fixture
def exported(tmp_path, monkeypatch):
    run = tmp_path / "run"
    sam = run / "sam"
    source = 1938
    transform = np.eye(4)
    # Proper right-handed rotation; both camera and geometry use it.
    transform[:3, :3] = Rotation.from_euler("y", 18, degrees=True).as_matrix()
    transform[:3, 3] = [.2, 0, .3]
    save(run / "geometry_proposals.json", {"scene_from_vipe": transform.tolist(),
        'objects':[{'object_id':'chair_01','accepted_frame_indices':[source],'rejected_frame_indices':[2222]}]})
    save(run / "prior.json", {"floor_observations": [{"source_frame_index": source}]})
    save(run / "frames/manifest.json", {"preview_fps": 10})
    monkeypatch.setattr(whitebox, "frames_by_source", lambda _: {source: frame()})
    masks = sam / "chair_01/target_masks"
    masks.mkdir(parents=True)
    mask = np.zeros((40, 60), dtype=bool)
    mask[8:32, 20:40] = True
    np.savez_compressed(masks / f"{source:05d}.npz", mask=mask, source_frame_index=np.int64(source))
    np.savez_compressed(masks / '02222.npz', mask=np.ones_like(mask), source_frame_index=np.int64(2222))
    common = {"yaw_deg": 0., "basis": "mixed", "reason": "synthetic test"}
    layout = {"objects": [{**common, "object_id": "chair_01", "kind": "object", "category": "furniture/chair",
                            "center_m": [0., .5, 3.], "dimensions_m": [.8, 1., .6], "yaw_deg": 90., "support_object_id": "floor"}],
              "structures": [{**common, "object_id": "floor", "kind": "floor", "category": "structure/floor",
                              "center_m": [0., -.025, 3.], "dimensions_m": [5., .05, 5.], "support_object_id": "unknown"}]}
    layout["recommended_refinement_target"] = "chair_01"
    save(run / "layout.json", layout)
    save(sam / "result.json", {"status": "complete", "objects": [{"object_id": "chair_01"}]})
    snapshot = whitebox.export_snapshot(run, layout, sam)
    return run, snapshot, transform


def test_export_preserves_y_up_local_bounds_yaw_support_and_camera(exported):
    run, snapshot, transform = exported
    chair, floor = snapshot["objects"]
    assert chair["support_id"] == "floor"
    assert floor["kind"] == "floor"
    assert floor["support_id"] is None
    rotation = Rotation.from_quat(chair["transform"]["rotation_xyzw"]).as_matrix()
    assert rotation @ [0., 1., 0.] == pytest.approx([0., 1., 0.])
    assert rotation @ [1., 0., 0.] == pytest.approx([0., 0., -1.])
    mesh_path = resolve_artifact(run / "runtime", chair["asset"]["artifact"])
    scene = trimesh.load(mesh_path, force="scene", process=False)
    assert scene.bounds == pytest.approx(np.array([[-.4, -.5, -.3], [.4, .5, .3]]), abs=1e-6)
    capture_path = run / "runtime/redwood-agent/captures" / f"{snapshot['capture_revision']}.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert len(capture["frames"]) == 2
    for item in capture["frames"]:
        assert item['source_time_s'] is None
        assert item['preview_time_s'] == 0.0
        assert np.array(item["c2w"]) == pytest.approx(transform)
        assert capture["metadata"]["original_source_indices"][item["source_frame_id"]] == 1938
    assert _geometry_issues(snapshot, {"support_tolerance_m": .04, "collision_penetration_m": .01, "collision_fraction": .2}) == {}


class FakeRepository:
    def __init__(self, snapshot):
        self.snapshot = snapshot
    async def inspect(self, request, context):
        return {"snapshot": self.snapshot, "head": {"revision": self.snapshot["revision"], "head_seq": 0}}


def select(service, snapshot, target):
    return asyncio.run(service.select({"contract_version": "video2scene/1.0", "scene_id": snapshot["scene_id"],
        "request_id": "test-selection", "tool": "select_evidence", "params": {"revision": snapshot["revision"],
        "target_ids": [target], "purpose": "review", "n": 3}}, None))


def test_object_evidence_is_its_observed_mask(exported):
    run, snapshot, _ = exported
    service = MaskEvidenceService(run / "runtime", FakeRepository(snapshot))
    result = select(service, snapshot, "chair_01")
    assert all(f["mask"] is not None and f["source_frame_id"].endswith("-chair_01") for f in result["frames"])
    assert all("Observed SAM3.1" in f["selection_reason"] for f in result["frames"])


def test_structure_evidence_does_not_borrow_another_object_mask(exported):
    run, snapshot, _ = exported
    service = MaskEvidenceService(run / "runtime", FakeRepository(snapshot))
    result = select(service, snapshot, "floor")
    assert all(f["mask"] is None for f in result["frames"])


def test_mask_evidence_rejects_capture_geometry_mismatch(exported):
    run, snapshot, _ = exported
    changed = copy.deepcopy(snapshot)
    changed["geometry_revision"] = "geom-new"
    service = MaskEvidenceService(run / "runtime", FakeRepository(changed))
    with pytest.raises(ContractError, match="geometry"):
        select(service, changed, "chair_01")


@pytest.mark.skipif(os.environ.get("RUN_AGENT_BLENDER_SMOKE") != "1", reason="Explicit CPU Blender smoke")
def test_seed_scene_build_render_and_resume_with_cpu_blender(exported):
    from video2scene.agent_scene.run import seed_scene
    run, snapshot, _ = exported
    asyncio.run(seed_scene(run, snapshot))
    build = json.loads((run / "whitebox_build.json").read_text(encoding="utf-8"))
    assert build["initial_seed_import"] is True
    assert build["human_edits_performed"] is False
    assert build["head"] == {"revision": snapshot["revision"], "head_seq": 0}
    reopened = json.loads((Path(build["path"]) / "reopen.json").read_text(encoding="utf-8"))
    assert reopened["passed"] and reopened["max_local_bounds_error_m"] < 1e-5
    assert set(reopened["objects"]) == {"chair_01", "floor"}
    rendered = json.loads((run / "whitebox_render.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(rendered["path"]) / "result.json").read_text(encoding="utf-8"))
    assert manifest["device"] == "CPU"
    assert manifest["renders"][0]["source_frame_id"] == "rgb-01938-chair_01"
    assert manifest["renders"][0]["camera_projection_max_error_px"] < 1e-3
    assert Path(rendered["preview"]).is_file()
    # An untouched seed can resume without another build or changing the head.
    build_dirs = set((run / "runtime/redwood-agent/assets").iterdir())
    asyncio.run(seed_scene(run, snapshot))
    assert set((run / "runtime/redwood-agent/assets").iterdir()) == build_dirs
