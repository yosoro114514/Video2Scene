import numpy as np
import pytest

from video2scene.semantic_review import (
    conservative_groups,
    duplicate_geometry,
    parse_json,
    validate_observations,
)


def test_review_schema_rejects_missing_duplicate_or_unknown_kept_labels():
    good = {
        "observations": [
            {
                "frame": 0,
                "decision": "keep",
                "label": "monitor",
                "reason": "screen and stand",
            },
            {
                "frame": 20,
                "decision": "reject",
                "label": None,
                "reason": "tabletop background",
            },
        ],
        "same_object_across_views": False,
    }
    assert set(validate_observations(good, [0, 20], ["monitor"])) == {0, 20}
    with pytest.raises(ValueError, match="Missing review frames"):
        validate_observations(good, [0, 20, 40], ["monitor"])
    good["observations"][0]["label"] = "invented_class"
    with pytest.raises(ValueError, match="outside inventory"):
        validate_observations(good, [0, 20], ["monitor"])
    good["observations"][0]["label"] = "monitor"
    good["observations"][1]["frame"] = 0
    with pytest.raises(ValueError, match="duplicated"):
        validate_observations(good, [0, 20], ["monitor"])


def test_duplicate_groups_do_not_make_unsupported_transitive_merges():
    result = conservative_groups([1, 2, 3], [(1, 2), (2, 3)])
    assert {1, 2, 3} not in result
    assert sorted(map(len, result)) == [1, 2]
    assert conservative_groups([1, 2, 3], [(1, 2), (2, 3), (1, 3)]) == [{1, 2, 3}]


def test_simultaneously_separate_objects_cannot_merge_despite_depth_noise():
    xyz = np.random.default_rng(1).uniform(0, 0.1, (50, 3))
    left = np.zeros((20, 20), bool)
    right = left.copy()
    left[:, :8] = True
    right[:, 12:] = True
    a = {"id": 1, "frames": [10], "points": xyz}
    b = {"id": 2, "frames": [10], "points": xyz + 0.01}
    assert duplicate_geometry(a, b, {10: {1: left, 2: right}}) is None
    assert duplicate_geometry(a, b, {10: {1: left, 2: left}}) is not None


def test_repeated_same_frame_containment_nominates_partial_full_duplicate():
    xyz = np.random.default_rng(1).uniform(0, 0.1, (50, 3))
    full = np.ones((20, 20), bool)
    part = np.zeros((20, 20), bool)
    part[5:15, 5:15] = True
    a = {"id": 1, "frames": [10, 11], "points": xyz}
    b = {"id": 2, "frames": [10, 11], "points": xyz + 0.4}
    masks = {i: {1: part, 2: full} for i in [10, 11]}
    proposal = duplicate_geometry(a, b, masks)
    assert proposal["supporting_frames"] == [10, 11]
    # A geometric proposal alone is never sufficient to merge.
    assert conservative_groups([1, 2], []) == [{1}, {2}]


def test_json_parser_tolerates_fences_but_rejects_no_object():
    assert parse_json('```json\n{"ok":true}\n```') == {"ok": True}
    with pytest.raises(ValueError):
        parse_json("uncertain")


def test_file_review_rejects_stale_source_and_never_loads_local_model(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace
    from video2scene.artifacts import Frame
    from video2scene.grounded_sam2 import save_masks, load_masks
    from video2scene import semantic_review as review

    cache = tmp_path / "raw"
    cache.mkdir()
    raw = dict(
        status="complete",
        frame_indices=[0, 1, 2, 3],
        tracks={
            "1": dict(label="book", seed_frames=[0, 2], detection_scores=[0.7, 0.6])
        },
        signature=dict(
            settings=dict(scale=1, max_depth=8),
            inventory=dict(categories=[dict(label="book")]),
        ),
    )
    (cache / "manifest.json").write_text(json.dumps(raw))
    for index in raw["frame_indices"]:
        save_masks(cache / f"{index:06d}.npz", {1: np.ones((12, 12), bool)})
    decisions = tmp_path / "decisions.json"
    value = dict(
        reviewer="test assistant",
        source_manifest_sha256="stale",
        decisions={
            "1": dict(
                decision="keep",
                label="book",
                reason="book cover",
                same_object=True,
                overrides={
                    "2": dict(
                        decision="reject", label=None, reason="track drifted into table"
                    )
                },
            )
        },
    )
    decisions.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different detection cache"):
        review.load_decision_file(
            decisions, raw, review.file_digest(cache / "manifest.json")
        )
    value["source_manifest_sha256"] = review.file_digest(cache / "manifest.json")
    decisions.write_text(json.dumps(value))

    def no_model(*args, **kwargs):
        raise AssertionError("File review must not construct an inference model")

    monkeypatch.setattr(review, "LocalVisualReviewer", no_model)

    def frames(scale):
        return [
            Frame(
                i,
                np.zeros((12, 12, 3), np.uint8),
                np.ones((12, 12), np.float32),
                np.array([10, 10, 6, 6]),
                np.eye(4),
            )
            for i in range(4)
        ]

    artifacts = SimpleNamespace(frames=frames)
    output = review.run_review(
        artifacts, cache, tmp_path / "review", decision_file=decisions
    )
    assert set(load_masks(output / "000000.npz")) == {1}
    assert set(load_masks(output / "000001.npz")) == {1}
    assert load_masks(output / "000002.npz") == {}
    assert load_masks(output / "000003.npz") == {}
