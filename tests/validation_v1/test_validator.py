"""Behavior tests use explicit fixture assets; they do not assert Blender quality."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video2scene.contracts_v1 import ContractError, canonical_hash, validate
from video2scene.validation_v1 import CandidateValidator, camera_fingerprint


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env(tmp_path):
    data = json.loads((Path(__file__).resolve().parents[2] / "docs/development/contracts/examples.json").read_text(encoding="utf-8"))
    examples = {x["type"]: x["value"] for x in data["positive"] if x["name"].startswith("model_")}
    candidate = copy.deepcopy(examples["CandidateSnapshot"])
    candidate["candidate"]["state"] = "built"
    base = copy.deepcopy(candidate["scene"])
    base.update(revision="r000", parent_revision=None)
    evidence = copy.deepcopy(examples["EvidenceBundle"])
    refs = [o["asset"]["artifact"] for o in candidate["scene"]["objects"]] + [evidence["frames"][0]["rgb"]]
    mapping = {}
    for index, ref in enumerate(refs):
        path = tmp_path / f"fixture_{index}.bin"
        path.write_bytes(b"explicit-test-fixture-" + str(index).encode())
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping[ref["uri"]] = path
    for before, after in zip(base["objects"], candidate["scene"]["objects"]):
        before["asset"] = copy.deepcopy(after["asset"])
    candidate["scene"]["objects"][1]["transform"]["translation_m"][0] = 0.1
    candidate["candidate"]["candidate_hash"] = canonical_hash(candidate["scene"])
    evidence["evidence_hash"] = canonical_hash({k:v for k,v in evidence.items() if k != "evidence_hash"})
    metadata = {"operation": "patch", "intent": "design_edit", "decision_ref": "decision1",
                "target_fields": {"t001": ["transform.translation_m"]}, "constraints": {},
                "artifact_refs": [], "base_snapshot": base}
    class Repository:
        async def read_candidate(self, identity, digest, context):
            return copy.deepcopy(candidate)
        async def candidate_metadata(self, identity, context):
            return copy.deepcopy(metadata)
    class Evidence:
        async def get(self, identity, context):
            return copy.deepcopy(evidence)
    validator = CandidateValidator(tmp_path, repository=Repository(), evidence_service=Evidence(),
                                   artifact_resolver=lambda ref: mapping[ref["uri"]])
    return SimpleNamespace(candidate=candidate, evidence=evidence, metadata=metadata,
                           validator=validator, mapping=mapping, root=tmp_path)


def check(env, criteria="tabletop_design_v1"):
    return run(env.validator.validate(env.candidate, env.evidence, criteria, None))


def test_B15_design_movement_does_not_require_matching_old_photo(env):
    report = check(env)
    validate(report, "ValidationReport")
    assert report["technical_pass"] and report["committable"]
    assert report["visual_pass"] is None
    assert any(c["name"] == "visual_execution" and c["passed"] is None for c in report["checks"])
    assert run(env.validator.get_report(report["validation_id"])) == report


def test_reconstruction_missing_visual_is_not_success(env):
    env.metadata["intent"] = "reconstruction_correction"
    report = check(env, "tabletop_reconstruction_v1")
    assert report["technical_pass"] and not report["committable"]
    assert report["visual_pass"] is None


def test_reconstruction_wrong_intent_criteria_rejected(env):
    with pytest.raises(ContractError, match="intent"):
        check(env, "tabletop_reconstruction_v1")


def test_B16_candidate_payload_cannot_borrow_identity(env):
    other = copy.deepcopy(env.candidate)
    other["scene"]["objects"][1]["category"] = "bottle"
    with pytest.raises(ContractError) as caught:
        run(env.validator.validate(other, env.evidence, "tabletop_design_v1", None))
    assert caught.value.code == "EVIDENCE_MISMATCH"


def test_B16_criteria_payload_cannot_change_server_thresholds(env):
    criteria = env.validator.get_criteria("tabletop_design_v1")
    criteria["allow_visual_null"] = False
    with pytest.raises(ContractError) as caught:
        run(env.validator.validate(env.candidate, env.evidence, criteria, None))
    assert caught.value.code == "EVIDENCE_MISMATCH"


def test_changed_persisted_report_is_rejected_after_restart(env):
    report = check(env)
    path = env.root / "validation_v1/reports" / (report["validation_id"] + ".json")
    changed = copy.deepcopy(report)
    changed["visual_pass"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    restarted = CandidateValidator(env.root)
    with pytest.raises(ContractError) as caught:
        run(restarted.get_report(report["validation_id"]))
    assert caught.value.code == "STALE_ARTIFACT"


def test_stale_asset_content_prevents_commit(env):
    next(iter(env.mapping.values())).write_bytes(b"different asset")
    report = check(env)
    assert not report["technical_pass"] and not report["committable"]
    assert any(c["name"] == "artifact_freshness" and c["passed"] is False for c in report["checks"])


def test_scope_lock_and_non_target_preservation(env):
    table = env.candidate["scene"]["objects"][0]
    env.metadata["base_snapshot"]["objects"][0]["field_state"]["category"]["locked"] = True
    table["category"] = "arbitrary"
    report = check(env)
    assert not report["technical_pass"] and not report["committable"]


def test_new_support_gap_is_rejected(env):
    env.candidate["scene"]["objects"][1]["transform"]["translation_m"][1] += 0.5
    report = check(env)
    assert not report["technical_pass"] and not report["committable"]


def test_evidence_revision_must_match_base(env):
    env.evidence["scene_revision"] = "r_new"
    with pytest.raises(ContractError) as caught:
        check(env)
    assert caught.value.code == "EVIDENCE_MISMATCH"


def test_visual_callback_exception_not_silently_waived_in_design(env):
    async def failed(*args):
        raise TimeoutError("fixture timeout")
    env.validator.visual_checker = failed
    report = check(env)
    assert not report["committable"] and report["visual_pass"] is None


def test_visual_bool_without_bound_render_is_not_a_pass(env):
    env.validator.visual_checker = lambda *args: True
    report = check(env)
    assert not report["committable"] and report["visual_pass"] is None


def test_visual_wrong_camera_is_not_a_pass(env):
    ref = copy.deepcopy(env.evidence["frames"][0]["rgb"])
    candidate = env.candidate["candidate"]
    render = {"candidate_id": candidate["candidate_id"], "candidate_hash": candidate["candidate_hash"],
              "evidence_id": env.evidence["evidence_id"], "evidence_hash": env.evidence["evidence_hash"],
              "frame_ids": [env.evidence["frames"][0]["source_frame_id"]], "view_mode": "normal",
              "camera_hash": "0" * 64, "artifact_refs": [ref]}
    assert camera_fingerprint(env.evidence) != render["camera_hash"]
    env.validator.visual_checker = lambda *args: {"passed": True, "render_report": render,
                                                 "details": "Injected test callback; not a real visual quality assessment"}
    report = check(env)
    assert not report["committable"] and report["visual_pass"] is None
