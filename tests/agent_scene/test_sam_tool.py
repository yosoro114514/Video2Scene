"""CPU-only checks for tool provenance and non-conflating identity selection."""
from __future__ import annotations

import copy
import ast
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from video2scene.agent_scene.sam_tool import sha256, validate_inputs, validate_request
from video2scene.agent_scene.sam_worker import (
    choose_anchor, run_object, install_uncached_refinement_merge, initial_propagation_quality,
    propagation_direction,
)


def request():
    return {"version": 1, "frames_dir": "rgb", "manifest": "frames.json", "requests": [
        {"object_id": "chair_01", "category": "furniture/chair", "prompt": "chair",
         "seed_source_frame_index": 20, "positive_points_normalized": [[.2, .5]],
         "negative_points_normalized": []}]}


def inputs(tmp_path):
    folder = tmp_path / "rgb"
    folder.mkdir()
    frames = []
    for i, source in enumerate((10, 20, 30)):
        name = f"{i:06d}.jpg"
        path = folder / name
        Image.new("RGB", (12, 10), (80, 90, 100)).save(path)
        frames.append({"file": name, "source_frame_index": source, "sha256": sha256(path)})
    (tmp_path / "frames.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return folder, frames


def mask_pair():
    masks = np.zeros((2, 10, 12), dtype=bool)
    masks[0, 2:9, 1:5] = True
    masks[1, 2:9, 7:11] = True
    return np.array([2, 9]), masks, np.array([.7, .99])


def test_request_bounds_and_unsupported_mode():
    value = request()
    assert validate_request(value)["timeout_s"] == 900
    for mutation in (
        lambda v: v.update(timeout_s=3600),
        lambda v: v["requests"][0].update(positive_points_normalized=[[float("nan"), .5]]),
        lambda v: v["requests"][0].update(object_id="../../other"),
        lambda v: v["requests"][0].update(mask_file="old_mask.png"),
        lambda v: v["requests"].append(copy.deepcopy(v["requests"][0])),
    ):
        other = copy.deepcopy(value)
        mutation(other)
        with pytest.raises(ValueError):
            validate_request(other)


def test_input_mapping_hash_and_seed_are_checked(tmp_path):
    folder, frames = inputs(tmp_path)
    value, frozen = validate_inputs(request(), tmp_path)
    assert [r["source_frame_index"] for r in frozen] == [10, 20, 30]
    assert frozen[0]["width"] == 12
    assert frozen[0]["sample_index"] == 0
    Image.new("RGB", (12, 10), (200, 90, 100)).save(folder / frames[0]["file"])
    with pytest.raises(ValueError, match="hash"):
        validate_inputs(request(), tmp_path)


def test_path_escape_and_unsampled_seed_rejected(tmp_path):
    inputs(tmp_path)
    value = request()
    value["requests"][0]["seed_source_frame_index"] = 15
    with pytest.raises(ValueError, match="Seed"):
        validate_inputs(value, tmp_path)
    manifest = json.loads((tmp_path / "frames.json").read_text())
    manifest["frames"][0]["file"] = "../../outside.jpg"
    (tmp_path / "frames.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="within frames_dir"):
        validate_inputs(request(), tmp_path)


def test_anchor_uses_point_not_highest_category_score():
    item = validate_request(request())["requests"][0]
    identity, proof = choose_anchor(*mask_pair(), item)
    assert identity == 2
    assert proof["method"] == "positive_points"
    item["positive_points_normalized"] = []
    identity, proof = choose_anchor(*mask_pair(), item)
    assert identity is None
    assert proof["method"] == "unresolved"
    item["hint_box_normalized"] = [7 / 12, .2, 4 / 12, .7]
    identity, proof = choose_anchor(*mask_pair(), item)
    assert identity == 9


def test_overlapping_seed_masks_do_not_arbitrarily_assign_identity():
    item = validate_request(request())["requests"][0]
    ids, masks, scores = mask_pair()
    masks[1] = masks[0]
    identity, proof = choose_anchor(ids, masks, scores, item)
    assert identity is None
    assert proof["method"] == "ambiguous"


class FakePredictor:
    def __init__(self):
        self.calls = []

    def handle_request(self, value):
        self.calls.append(value)
        if value["type"] == "start_session":
            return {"session_id": "fake"}
        if value["type"] == "add_prompt":
            ids, masks, scores = mask_pair()
            if "points" in value:
                assert "text" not in value
                assert value["obj_id"] == 2
                ids, masks, scores = ids[:1], masks[:1], scores[:1]
            return {"frame_index": value["frame_index"], "outputs": {
                "out_obj_ids": ids, "out_binary_masks": masks, "out_probs": scores}}
        return {}

    def handle_stream_request(self, value):
        self.calls.append(value)
        assert value["propagation_direction"] == "both"
        assert value["start_frame_index"] == 1
        assert value["max_frame_num_to_track"] is None
        for index in (1, 2, 1, 0):
            ids, masks, scores = mask_pair()
            # Another newly detected category instance must not leak into target.
            ids[1] = 30
            if index == 2:
                ids, masks, scores = ids[1:], masks[1:], scores[1:]
            yield {"frame_index": index, "outputs": {
                "out_obj_ids": ids, "out_binary_masks": masks, "out_probs": scores}}


def test_bidirectional_masks_keep_source_ids_and_target_identity(tmp_path):
    folder, _ = inputs(tmp_path)
    value, frames = validate_inputs(request(), tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    predictor = FakePredictor()
    result = run_object(predictor, value["requests"][0], frames, folder, output, deadline=time.monotonic() + 60)
    assert result["status"] == "complete"
    assert result["selected_sam_id"] == 2
    assert result["frames_with_target"] == 2
    assert [r["source_frame_index"] for r in result["frames"]] == [10, 20, 30]
    with np.load(output / "chair_01/target_masks/00030.npz") as data:
        assert not data["mask"].any()
        assert not data["target_present"]
        assert data["source_frame_index"] == 30
    with np.load(output / "chair_01/target_masks/00020.npz") as data:
        assert data["mask"].sum() == 28
    removals = [c for c in predictor.calls if c["type"] == "remove_object"]
    assert removals == [{"type": "remove_object", "session_id": "fake", "frame_index": 1,
                         "obj_id": 9, "is_user_action": False}]
    assert predictor.calls[-1]["type"] == "close_session"
    assert len(result["duplicates"]) == 2
    assert (output / result["previews"][0]).exists()


def test_pinned_source_uncached_refinement_merge_regression():
    # Execute only the upstream pure dictionary method, without importing SAM,
    # torch, loading weights or accessing a GPU. This reproduces the real bug.
    source = Path(__file__).resolve().parents[2] / "ccfa-workfiles/experiments/sam31-evaluation/cache/sam3/sam3/model/sam3_multiplex_tracking.py"
    if not source.exists():
        pytest.skip("Pinned local SAM source unavailable")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sam3MultiplexTracking")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_sam2_output")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    PinnedModel = type("PinnedModel", (), {"_build_sam2_output": namespace["_build_sam2_output"]})
    model = PinnedModel()
    mask = np.ones((1, 10, 12), dtype=bool)
    state = {"cached_frame_outputs": {0: {2: mask}}}
    assert model._build_sam2_output(state, 1, {2: mask}) == {}
    proof = install_uncached_refinement_merge(model)
    result = model._build_sam2_output(state, 1, {2: mask})
    assert result[2] is mask
    assert 1 not in state["cached_frame_outputs"]
    assert model._build_sam2_output(state, 1) == {}
    assert model._build_sam2_output(state, 0)[2] is mask
    replacement = np.zeros_like(mask)
    assert model._build_sam2_output(state, 0, {2: replacement})[2] is replacement
    assert state["cached_frame_outputs"][0][2] is mask
    with pytest.raises(ValueError, match="actual mask"):
        model._build_sam2_output(state, 1, {2: None})
    assert proof["original_source_sha256"] == sha256(source)
    assert len(proof["original_method_sha256"]) == 64
    assert proof["synthetic_masks"] is False


def test_first_target_seed_only_fails_before_later_objects():
    with pytest.raises(RuntimeError, match="non-seed"):
        initial_propagation_quality({"status": "complete", "frames_with_target": 1}, 158)
    initial_propagation_quality({"status": "complete", "frames_with_target": 2}, 158)
    initial_propagation_quality({"status": "complete", "frames_with_target": 1}, 1)
    initial_propagation_quality({"status": "needs_prompt"}, 158)


def pinned_action_history_parser():
    source = Path(__file__).resolve().parents[2] / "ccfa-workfiles/experiments/sam31-evaluation/cache/sam3/sam3/model/sam3_multiplex_tracking.py"
    if not source.exists():
        pytest.skip("Pinned local SAM source unavailable")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sam3MultiplexTrackingWithInteractivity")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_parse_action_history_for_propagation")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


class BoundaryPredictor(FakePredictor):
    """Actual upstream action parser, synthetic masks; no SAM/GPU imports."""
    def __init__(self, frame_count=3):
        super().__init__()
        self.frame_count = frame_count
        self.history = [{"type": "refine", "obj_ids": [2], "frame_idx": None}]
        self.cache = {}
        self.parser = pinned_action_history_parser()

    def handle_stream_request(self, value):
        self.calls.append(value)
        seed = value["start_frame_index"]
        directions = [False, True] if value["propagation_direction"] == "both" else [value["propagation_direction"] == "backward"]
        for reverse in directions:
            kind, identities = self.parser(None, self.history, self.frame_count)
            self.history.append({"type": kind, "obj_ids": identities, "frame_idx": seed})
            order = range(seed-1, -1, -1) if reverse else range(seed, self.frame_count)
            for index in order:
                if kind == "propagation_fetch":
                    outputs = self.cache[index]
                else:
                    ids, masks, scores = mask_pair()
                    outputs = {"out_obj_ids": ids[:1], "out_binary_masks": masks[:1], "out_probs": scores[:1]}
                    self.cache[index] = outputs
                if value.get("evict_cached_frame_outputs"):
                    self.cache.pop(index, None)
                yield {"frame_index": index, "outputs": outputs}


def test_original_both_direction_reproduces_last_seed_missing_cache():
    predictor = BoundaryPredictor(96)
    stream = predictor.handle_stream_request({"propagation_direction": "both", "start_frame_index": 95,
                                              "evict_cached_frame_outputs": True})
    assert next(stream)["frame_index"] == 95
    with pytest.raises(KeyError) as exc:
        next(stream)
    assert exc.value.args == (94,)


@pytest.mark.parametrize("seed_source,expected", [(10,"forward"),(20,"both"),(30,"backward")])
def test_endpoint_and_interior_prompts_cover_all_original_frames(tmp_path, seed_source, expected):
    folder, _ = inputs(tmp_path)
    req = request();req["requests"][0]["seed_source_frame_index"] = seed_source
    value, frames = validate_inputs(req, tmp_path)
    output = tmp_path / "output";output.mkdir()
    predictor = BoundaryPredictor()
    result = run_object(predictor, value["requests"][0], frames, folder, output, deadline=time.monotonic()+60)
    assert result["status"] == "complete"
    assert result["propagation_direction"] == expected
    assert result["frames_with_target"] == 3
    assert [row["source_frame_index"] for row in result["frames"]] == [10,20,30]
    assert all(item["type"] != "propagation_fetch" for item in predictor.history)
    for original in (10,20,30):
        with np.load(output / f"chair_01/target_masks/{original:05d}.npz") as data:
            assert data["source_frame_index"] == original and data["mask"].sum() == 28


def test_single_frame_prompt_uses_forward_only():
    assert propagation_direction(0, 1) == "forward"
