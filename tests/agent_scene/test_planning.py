"""CPU checks for actual-prior schema normalization and SAM request compatibility."""
import copy
from pathlib import Path

import pytest

from video2scene.agent_scene.planning import normalize_prior, validate_prior, read
from video2scene.agent_scene.sam_tool import validate_request


def prior():
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs/redwood_agent_20261029/prior.json"
    if not path.exists():
        pytest.skip("The actual Redwood model response is not available in this checkout")
    return read(path)


def test_actual_model_prior_is_valid_and_sam_compatible():
    value = validate_prior(prior())
    request = {"version": 1, "frames_dir": "frames", "manifest": "frames/manifest.json", "timeout_s": 1200,
               "requests": [{"object_id": item["object_id"], "category": item["category"], "prompt": item["sam_prompt"],
                             "seed_source_frame_index": item["seed_source_frame_index"],
                             "positive_points_normalized": item["positive_points_normalized"],
                             "negative_points_normalized": item["negative_points_normalized"],
                             "hint_box_normalized": item["hint_box_normalized"]} for item in value["objects"]]}
    assert len(validate_request(request)["requests"]) == 11


def test_only_exact_known_id_plus_prose_is_normalized(tmp_path):
    value = prior()
    identity = value["objects"][0]["object_id"]
    value["suggested_refinement_target"] = identity + "（可见证据充分）"
    original = copy.deepcopy(value)
    normalized = normalize_prior(value, tmp_path)
    assert normalized["suggested_refinement_target"] == identity
    assert value == original
    assert (tmp_path / "prior_normalization.json").is_file()
    value["suggested_refinement_target"] = identity + "_other"
    with pytest.raises(ValueError, match="Unknown suggested"):
        normalize_prior(value, tmp_path)


def test_unseen_seed_and_bad_points_are_rejected():
    value = prior()
    value["objects"][0]["seed_source_frame_index"] = 1
    with pytest.raises(ValueError, match="actually supplied"):
        validate_prior(value)
    value = prior()
    value["objects"][0]["positive_points_normalized"] = [[-0.1, .5]]
    with pytest.raises(ValueError, match="prompt point"):
        validate_prior(value)
