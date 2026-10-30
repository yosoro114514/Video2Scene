"""Behavior checks use real SQLite/files and explicit mock build/validator data."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from video2scene.contracts_v1 import ContractError, ExecutionContext, canonical_hash, validate
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.scene_state import SQLiteSceneRepository

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = json.loads((ROOT / "docs/development/contracts/examples.json").read_text(encoding="utf8"))


def sample(name):
    return copy.deepcopy(next(item["value"] for item in EXAMPLES["positive"] if item["name"] == name))


def run(awaitable):
    return asyncio.run(awaitable)


def request(tool, params, token=None, request_id=None):
    value = {"contract_version": "video2scene/1.0", "request_id": request_id or tool,
             "scene_id": "demo_scene", "tool": tool, "params": params}
    if token:
        value.update(base_revision=token["revision"], base_head_seq=token["head_seq"])
    return value


class MockValidator:
    def __init__(self):
        self.reports = {}

    async def get_report(self, validation_id, context):
        return copy.deepcopy(self.reports[validation_id])


class Harness:
    def __init__(self, path, seed=True):
        self.path = Path(path)
        self.assets = ArtifactResolver(self.path / "artifacts")
        self.validator = MockValidator()
        self.evidence = sample("model_EvidenceBundle")
        self.criteria = {"criteria_id": "tabletop_design_v1", "intent": "design_edit", "allow_visual_null": True}
        self.criteria["criteria_hash"] = canonical_hash(self.criteria)
        self.repo = SQLiteSceneRepository(self.path / "repository", validator=self.validator,
            evidence_getter=lambda key, ctx: copy.deepcopy(self.evidence),
            criteria_getter=lambda key, ctx: copy.deepcopy(self.criteria), artifact_resolver=self.assets.resolve)
        self.ctx = ExecutionContext(actor="user", decision_ref="decision_1",
            allowed_object_ids=frozenset({"t001", "table_00"}),
            allowed_fields=frozenset({"category", "support_id", "transform.translation_m", "transform.rotation_xyzw", "transform.scale", "asset.geometry", "asset.materials"}),
            storage_root=self.path)
        if seed:
            scene = sample("model_SceneSnapshot")
            for obj in scene["objects"]:
                obj["asset"]["artifact"] = self.artifact("assets/" + obj["instance_id"] + ".glb", obj["instance_id"].encode())
            run(self.repo.import_scene(scene, self.ctx, source_hash="a" * 64))

    def artifact(self, key, body=b"explicit mock GLB bytes"):
        uri = "artifact://demo_scene/" + key
        path = self.assets.path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"uri": uri, "sha256": hashlib.sha256(body).hexdigest(), "media_type": "model/gltf-binary", "role": "asset"}

    def view(self, revision="head"):
        return run(self.repo.inspect(request("inspect_scene", {"revision": revision}), self.ctx))

    def patch_request(self, edits=None, token=None, request_id="patch_1"):
        return request("apply_scene_patch", {"intent": "design_edit", "decision_ref": "decision_1",
            "edits": edits if edits is not None else [{"instance_id": "t001", "field": "transform.scale", "value": [1, 1.05, 1]}]},
            token or self.view()["head"], request_id)

    def patch(self, **kwargs):
        return run(self.repo.prepare_patch(self.patch_request(**kwargs), self.ctx))

    def build(self, candidate):
        out = {"artifact_refs": [self.artifact("builds/" + candidate["candidate_id"] + ".glb")], "asset_updates": [], "input_fingerprint": candidate["candidate_hash"]}
        return run(self.repo.attach_build(candidate["candidate_id"], candidate["candidate_hash"], out, self.ctx))

    def report(self, candidate, validation_id="validation_1"):
        report = sample("model_ValidationReport")
        report.update(validation_id=validation_id, candidate_id=candidate["candidate_id"], candidate_hash=candidate["candidate_hash"],
            evidence_id=self.evidence["evidence_id"], evidence_hash=self.evidence["evidence_hash"], criteria_hash=self.criteria["criteria_hash"])
        self.validator.reports[validation_id] = report
        return report

    def commit_request(self, candidate, request_id="commit_1", validation_id="validation_1"):
        return request("commit_candidate", {"candidate_id": candidate["candidate_id"], "candidate_hash": candidate["candidate_hash"],
            "validation_id": validation_id, "decision_ref": "decision_1"},
            {"revision": candidate["base_revision"], "head_seq": candidate["base_head_seq"]}, request_id)

    def commit(self, candidate, **kwargs):
        return run(self.repo.commit(self.commit_request(candidate, **kwargs), self.ctx))

    def history(self, action, request_id=None, token=None):
        return run(self.repo.history(request("revision_history", {"action": action, "decision_ref": "decision_1"},
            token or self.view()["head"], request_id or action), self.ctx))


def assert_code(code, fn):
    with pytest.raises(ContractError) as exc:
        fn()
    assert exc.value.code == code


def test_b01_b02_b03_b14_full_transaction_and_aba(tmp_path):
    h = Harness(tmp_path)
    original = h.view()
    patch_request = h.patch_request()
    staged = run(h.repo.prepare_patch(patch_request, h.ctx))
    assert h.view() == original
    assert run(h.repo.prepare_patch(patch_request, h.ctx)) == staged
    altered = copy.deepcopy(patch_request)
    altered["params"]["edits"][0]["value"] = [1, 2, 1]
    assert_code("IDEMPOTENCY_CONFLICT", lambda: run(h.repo.prepare_patch(altered, h.ctx)))
    built = h.build(staged)
    assert built["candidate_id"] != staged["candidate_id"]
    assert built["candidate_hash"] != staged["candidate_hash"]
    h.report(built)
    committed = h.commit(built)
    assert h.commit(built) == committed
    assert run(h.repo.get_request_result("demo_scene", "commit_1")) == committed
    h.repo = SQLiteSceneRepository(h.repo.root, validator=h.validator, evidence_getter=lambda key, ctx: h.evidence,
        criteria_getter=lambda key, ctx: h.criteria, artifact_resolver=h.assets.resolve)
    assert h.commit(built) == committed
    assert h.view("r000")["head"]["revision"] == committed["revision"]
    assert h.view()["snapshot"]["objects"][1]["transform"]["scale"] == [1, 1.05, 1]
    undone = h.history("undo")
    assert undone["revision"] == "r000" and undone["head_seq"] == 2
    assert_code("STALE_REVISION", lambda: h.patch(token=original["head"], request_id="stale_aba"))
    redone = h.history("redo")
    assert redone["revision"] == committed["revision"] and redone["head_seq"] == 3
    assert len(run(h.repo.read_outbox("demo_scene"))) == 3
    for event in run(h.repo.read_outbox("demo_scene")):
        validate(event, "SceneEvent")


@pytest.mark.parametrize("edits,code", [
    ([{"instance_id":"t001","field":"category","value":"cup"},{"instance_id":"missing","field":"category","value":"box"}], "NOT_FOUND"),
    ([{"instance_id":"t001","field":"category","value":"cup"},{"instance_id":"t001","field":"category","value":"mug"}], "INVALID_ARGUMENT"),
    ([{"instance_id":"t001","field":"transform.rotation_xyzw","value":[0,0,0,2]}], "INVALID_ARGUMENT"),
    ([{"instance_id":"t001","field":"transform.scale","value":[1,0,1]}], "INVALID_ARGUMENT"),
    ([{"instance_id":"t001","field":"category","value":"mug"}], "NO_CHANGE"),
    ([{"instance_id":"t001","field":"support_id","value":"missing"}], "CONSTRAINT_VIOLATION"),
    ([{"instance_id":"table_00","field":"support_id","value":"t001"}], "CONSTRAINT_VIOLATION"),
])
def test_b04_b06_atomic_rejection(tmp_path, edits, code):
    h = Harness(tmp_path)
    before = h.view()
    assert_code(code, lambda: h.patch(edits=edits))
    assert h.view() == before
    with h.repo._db() as db:
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_b05_host_grants_and_locks(tmp_path):
    h = Harness(tmp_path, seed=False)
    scene = sample("model_SceneSnapshot")
    scene["objects"][1]["field_state"]["transform.scale"]["locked"] = True
    run(h.repo.import_scene(scene))
    no_grant = replace(h.ctx, actor="agent", allowed_fields=frozenset({"category"}))
    assert_code("FORBIDDEN_SCOPE", lambda: run(h.repo.prepare_patch(h.patch_request(), no_grant)))
    forged = h.patch_request()
    forged["params"]["decision_ref"] = "forged"
    assert_code("FORBIDDEN_SCOPE", lambda: run(h.repo.prepare_patch(forged, h.ctx)))
    candidate = h.patch()
    snapshot = run(h.repo.read_candidate(candidate["candidate_id"], candidate["candidate_hash"], h.ctx))
    assert snapshot["scene"]["objects"][1]["field_state"]["transform.scale"]["locked"] is True


def test_stale_candidate_cannot_overwrite_new_edit(tmp_path):
    h = Harness(tmp_path)
    first, second = h.build(h.patch()), h.build(h.patch(request_id="patch_2", edits=[{"instance_id":"t001","field":"category","value":"cup"}]))
    h.report(first, "val_first")
    h.report(second, "val_second")
    h.commit(second, request_id="commit_second", validation_id="val_second")
    after = h.view()
    assert_code("STALE_REVISION", lambda: h.commit(first, validation_id="val_first"))
    assert h.view() == after


@pytest.mark.parametrize("mutation,code", [
    ("other_candidate", "EVIDENCE_MISMATCH"), ("evidence_hash", "EVIDENCE_MISMATCH"),
    ("criteria_hash", "EVIDENCE_MISMATCH"), ("intent", "CONSTRAINT_VIOLATION"),
    ("visual_required", "CONSTRAINT_VIOLATION"), ("asset_tampered", "STALE_ARTIFACT"),
])
def test_commit_rechecks_server_bindings_and_artifacts(tmp_path, mutation, code):
    h = Harness(tmp_path)
    built = h.build(h.patch())
    report = h.report(built)
    if mutation == "other_candidate": report["candidate_id"] = "another_candidate"
    elif mutation == "evidence_hash": h.evidence["evidence_hash"] = "b" * 64
    elif mutation == "criteria_hash": h.criteria["criteria_hash"] = "c" * 64
    elif mutation == "intent": report["intent"] = "reconstruction_correction"
    elif mutation == "visual_required": h.criteria["allow_visual_null"] = False
    elif mutation == "asset_tampered": h.assets.path("artifact://demo_scene/assets/t001.glb").write_bytes(b"tampered")
    assert_code(code, lambda: h.commit(built))
    assert h.view()["head"] == {"revision": "r000", "head_seq": 0}


def test_refinement_is_isolated_and_new_asset_required(tmp_path):
    h = Harness(tmp_path)
    req = sample("request_refine_asset")
    req.update(scene_id="demo_scene", base_revision="r000", base_head_seq=0)
    req["params"].update(target_id="t001", allowed_fields=["asset.geometry","asset.materials"], decision_ref="decision_1", evidence_id=h.evidence["evidence_id"])
    staged = run(h.repo.prepare_refinement(req, h.ctx))
    snapshot = run(h.repo.read_candidate(staged["candidate_id"], staged["candidate_hash"], h.ctx))
    obj = copy.deepcopy(snapshot["scene"]["objects"][1])
    obj["asset"]["asset_version"] = "new_asset"
    obj["asset"]["artifact"] = h.artifact("assets/new.glb", b"newly constructed mock mesh")
    output = {"artifact_refs":[obj["asset"]["artifact"]], "asset_updates":[{"instance_id":"t001","asset":obj["asset"],"local_bounds_m":obj["local_bounds_m"]}], "input_fingerprint":staged["candidate_hash"]}
    built = run(h.repo.attach_build(staged["candidate_id"], staged["candidate_hash"], output, h.ctx))
    assert run(h.repo.attach_build(staged["candidate_id"], staged["candidate_hash"], output, h.ctx)) == built
    changed = run(h.repo.read_candidate(built["candidate_id"], built["candidate_hash"], h.ctx))["scene"]
    assert changed["objects"][0] == snapshot["scene"]["objects"][0]
    assert changed["objects"][1]["transform"] == snapshot["scene"]["objects"][1]["transform"]
    assert h.view()["head"]["revision"] == "r000"


def test_new_commit_clears_redo(tmp_path):
    h = Harness(tmp_path)
    first = h.build(h.patch())
    h.report(first)
    h.commit(first)
    h.history("undo")
    second = h.build(h.patch(request_id="patch_2"))
    h.report(second, "validation_2")
    h.commit(second, request_id="commit_2", validation_id="validation_2")
    assert_code("NO_CHANGE", lambda: h.history("redo"))


def test_concurrent_commits_have_one_winner(tmp_path):
    h = Harness(tmp_path)
    first = h.build(h.patch())
    second = h.build(h.patch(request_id="patch_2", edits=[{"instance_id":"t001","field":"category","value":"cup"}]))
    h.report(first, "val_first")
    h.report(second, "val_second")
    def commit_one(candidate, label):
        try:
            return h.commit(candidate, request_id="commit_" + label, validation_id="val_" + label)
        except ContractError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit_one, first, "first"), pool.submit(commit_one, second, "second")]
        results = [f.result(timeout=15) for f in futures]
    assert sum(isinstance(value, dict) for value in results) == 1
    assert "STALE_REVISION" in results
    assert h.view()["head"]["head_seq"] == 1
    assert len(run(h.repo.read_outbox("demo_scene"))) == 1


def test_cached_build_still_checks_artifact_hash(tmp_path):
    h = Harness(tmp_path)
    staged = h.patch()
    artifact = h.artifact("builds/replay.glb")
    output = {"artifact_refs": [artifact], "asset_updates": [], "input_fingerprint": staged["candidate_hash"]}
    run(h.repo.attach_build(staged["candidate_id"], staged["candidate_hash"], output, h.ctx))
    h.assets.path(artifact["uri"]).write_bytes(b"changed after success")
    assert_code("STALE_ARTIFACT", lambda: run(h.repo.attach_build(staged["candidate_id"], staged["candidate_hash"], output, h.ctx)))


@pytest.mark.parametrize("point,committed", [("commit_after_artifacts",False), ("commit_before_db_commit",False), ("commit_after_db_commit",True)])
def test_real_process_crash_and_recovery(tmp_path, point, committed):
    h = Harness(tmp_path)
    built = h.build(h.patch())
    report = h.report(built)
    state = {"request":h.commit_request(built), "report":report, "evidence":h.evidence, "criteria":h.criteria, "point":point}
    config = tmp_path / "crash.json"
    config.write_text(json.dumps(state), encoding="utf8")
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name("crash_worker.py")), str(tmp_path)], cwd=ROOT, capture_output=True, timeout=30)
    assert completed.returncode == 77, completed.stderr.decode(errors="replace")
    recovered = Harness(tmp_path, seed=False)
    head = recovered.view()["head"]
    assert head["head_seq"] == int(committed)
    result = run(recovered.repo.get_request_result("demo_scene", "commit_1"))
    assert (result is not None) == committed
    recovered.validator.reports[report["validation_id"]] = report
    recovered.evidence, recovered.criteria = state["evidence"], state["criteria"]
    retry = recovered.commit(built)
    assert retry["head_seq"] == 1
    assert len(run(recovered.repo.read_outbox("demo_scene"))) == 1
