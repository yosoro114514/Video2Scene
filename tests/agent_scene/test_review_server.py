"""Confirmation/HTTP behavior with explicit mock scene edits and no model worker."""
import asyncio
import copy
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from video2scene.agent_scene.review_server import create_review_host, make_review_server
from video2scene.contracts_v1 import ContractError, canonical_hash

ROOT = Path(__file__).resolve().parents[2]


class MockJobs:
    def __init__(self):
        self.started = 0
        self.submitted = []

    def start(self):
        self.started += 1

    async def submit(self, request, context):
        self.submitted.append(copy.deepcopy(request))
        return {"job_id": "mock_only"}


@pytest.fixture
def review(tmp_path):
    examples = json.loads((ROOT / "docs/development/contracts/examples.json").read_text(encoding="utf-8"))
    snapshot = copy.deepcopy(next(row["value"] for row in examples["positive"] if row["name"] == "model_SceneSnapshot"))
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    host = create_review_host(tmp_path)
    assert host.jobs is None
    host.backend = object()
    host.jobs = MockJobs()
    host.started_commands = []
    host._start_command = lambda command: host.started_commands.append(copy.deepcopy(command))
    host.test_view = host.current_view()
    host.current_view = lambda: copy.deepcopy(host.test_view)

    async def inspect(request, context):
        return copy.deepcopy(host.test_view)
    host.repo.inspect = inspect
    return host


def target(host):
    return next(obj for obj in host.test_view["snapshot"]["objects"] if obj["kind"] == "object")


def payload(host):
    return {"scene_id": host.scene_id, **host.test_view["head"], "target_id": target(host)["instance_id"]}


def mock_saved_edit(host):
    """Explicit test fixture, never a claim of a real user operation."""
    obj = target(host)
    base = copy.deepcopy(host.test_view["head"])
    scale = copy.deepcopy(obj["transform"]["scale"])
    scale[1] *= 1.05
    command = {"contract_version": "video2scene/1.0", "command_id": "mock_human_numeric", "session_id": "mock_browser",
               "scene_id": host.scene_id, "base_revision": base["revision"], "base_head_seq": base["head_seq"],
               "selected_ids": [obj["instance_id"]], "intent": "design_edit", "kind": "patch",
               "edits": [{"instance_id": obj["instance_id"], "field": "transform.scale", "value": scale}]}
    record, _ = host.store.accept_command(command)
    record.update(state="succeeded", message="Explicit mock numeric save")
    host.store.save_command(host.scene_id, command["command_id"], record)
    obj["transform"]["scale"] = scale
    host.test_view["head"] = {"revision": "mock_edited", "head_seq": 1}
    host.test_view["snapshot"]["revision"] = "mock_edited"
    return command


def assert_error(code, call):
    with pytest.raises(ContractError) as error:
        call()
    assert error.value.code == code


def test_confirmation_requires_saved_effective_numeric_edit(review):
    assert review.review_status()["require_edit"] is True
    assert review.review_status()["human_edit_saved"] is False
    assert_error("FORBIDDEN_SCOPE", lambda: review.confirm(payload(review)))
    assert not review.confirmation_path.exists()
    review.test_view["head"]["head_seq"] = 1
    target(review)["transform"]["scale"][1] *= 1.02
    assert_error("FORBIDDEN_SCOPE", lambda: review.confirm(payload(review)))
    assert not review.confirmation_path.exists()


def test_furniture_support_is_refinable_but_room_floor_is_not():
    from video2scene.agent_scene.review_server import refinable_object
    assert refinable_object({'kind':'support','category':'furniture/table'})
    assert not refinable_object({'kind':'floor','category':'structure/floor'})
    assert not refinable_object({'kind':'support','category':'furnishing/rug'})


def test_recovery_status_requires_exact_confirmation_and_keeps_failed_command(review):
    from video2scene.contracts_v1 import file_hash
    mock_saved_edit(review)
    review.confirm(payload(review))
    confirmation=json.loads(review.confirmation_path.read_text(encoding='utf-8'))
    record=review.store.command(review.scene_id,confirmation['command_id'])
    record.update(state='failed',message='original failure')
    review.store.save_command(review.scene_id,confirmation['command_id'],record)
    recovery={'phase':'validating','message':'recovering actual authored code',
              'confirmed_base':review.test_view['head'],'target_id':confirmation['target_id'],
              'confirmation_sha256':file_hash(review.confirmation_path)}
    path=review.run_root/'recovery_status.json'
    path.write_text(json.dumps(recovery),encoding='utf-8')
    status=review.review_status()
    assert status['phase']=='validating'
    assert status['original_attempt']['state']=='failed'
    assert review.store.command(review.scene_id,confirmation['command_id'])['state']=='failed'
    recovery['confirmation_sha256']='0'*64
    path.write_text(json.dumps(recovery),encoding='utf-8')
    assert review.review_status()['phase']=='failed'


def test_recovery_completion_requires_current_head_and_refined_asset(review):
    from video2scene.contracts_v1 import file_hash
    mock_saved_edit(review);review.confirm(payload(review))
    confirmation=json.loads(review.confirmation_path.read_text(encoding='utf-8'))
    record=review.store.command(review.scene_id,confirmation['command_id'])
    record.update(state='failed',message='original failure')
    review.store.save_command(review.scene_id,confirmation['command_id'],record)
    base=copy.deepcopy(review.test_view['head'])
    review.test_view['head']={'revision':'recovered_revision','head_seq':2}
    target(review)['asset']['artifact']['sha256']='a'*64
    bound={'confirmed_base':base,'target_id':confirmation['target_id'],'confirmation_sha256':file_hash(review.confirmation_path)}
    (review.run_root/'recovery_status.json').write_text(json.dumps(dict(bound,phase='complete',message='saved')),encoding='utf-8')
    completion=dict(bound,status='complete',head=review.test_view['head'],artifact_refs=[{'role':'object_geometry','sha256':'a'*64}])
    (review.run_root/'recovery_completion.json').write_text(json.dumps(completion),encoding='utf-8')
    assert review.review_status()['phase']=='complete'
    review.test_view['head']={'revision':'unrelated_revision','head_seq':3}
    assert review.review_status()['phase']=='failed'


def test_status_edit_eligibility_tracks_effective_correction_and_recommendation(review):
    original = copy.deepcopy(target(review)["transform"]["scale"])
    (review.run_root / "prior.json").write_text(json.dumps({"suggested_refinement_target": target(review)["instance_id"]}))
    mock_saved_edit(review)
    status = review.review_status()
    assert status["human_edit_saved"] is True
    assert status["recommended_target_id"] == target(review)["instance_id"]
    (review.run_root / "status.json").write_text(json.dumps({"recommended_refinement_target": "not_present"}))
    assert "recommended_target_id" not in review.review_status()
    (review.run_root / "layout.json").write_text(json.dumps({"recommended_refinement_target": target(review)["instance_id"]}))
    assert review.review_status()["recommended_target_id"] == target(review)["instance_id"]
    target(review)["transform"]["scale"] = original
    assert review.review_status()["human_edit_saved"] is False


def test_incomplete_confirmation_is_uncertain_and_never_relaunched(review):
    review.confirmation_path.write_text(json.dumps({"confirmed": True, "command_id": "partial"}))
    status = review.review_status()
    assert status["phase"] == "failed" and status["ready_for_review"] is False
    assert_error("RESOURCE_EXHAUSTED", lambda: review.confirm(payload(review)))
    assert review.started_commands == [] and review.jobs.started == 0


def test_confirmation_persists_before_launch_and_is_idempotent(review):
    mock_saved_edit(review)
    call = payload(review)
    observed = []
    review._start_command = lambda command: observed.append((review.confirmation_path.is_file(), copy.deepcopy(command)))
    first = review.confirm(call)
    second = review.confirm(call)
    assert first["command_id"] == second["command_id"]
    assert second["duplicate"] is True
    assert len(observed) == 1 and observed[0][0] is True
    assert review.jobs.started == 1
    saved = json.loads(review.confirmation_path.read_text(encoding="utf-8"))
    assert saved["confirmed"] is True
    assert saved["human_corrections"]["effective_numeric_changes"]
    assert saved["snapshot_hash"] == canonical_hash(review.test_view["snapshot"])
    changed = dict(call, head_seq=2)
    assert_error("IDEMPOTENCY_CONFLICT", lambda: review.confirm(changed))


def test_stale_invalid_structure_and_pending_confirmation_rejected(review):
    original = mock_saved_edit(review)
    call = payload(review)
    assert_error("STALE_REVISION", lambda: review.confirm(dict(call, head_seq=0)))
    assert_error("INVALID_ARGUMENT", lambda: review.confirm(dict(call, head_seq=True)))
    assert_error("INVALID_ARGUMENT", lambda: review.confirm(dict(call, role="user")))
    obj = target(review)
    obj["kind"] = "surface"
    assert_error("INVALID_ARGUMENT", lambda: review.confirm(call))
    obj["kind"] = "object"
    original["command_id"] = "pending_numeric"
    review.store.accept_command(original)
    assert_error("RESOURCE_EXHAUSTED", lambda: review.confirm(payload(review)))


def test_public_language_and_raw_tool_cannot_bypass_gate(review):
    command = {"contract_version": "video2scene/1.0", "command_id": "fake_refine", "session_id": "client",
               "scene_id": review.scene_id, "base_revision": review.test_view["head"]["revision"], "base_head_seq": 0,
               "selected_ids": [target(review)["instance_id"]], "intent": "design_edit", "kind": "language", "text": "精修"}
    assert_error("FORBIDDEN_SCOPE", lambda: review.submit_command(command))
    request = {"scene_id": review.scene_id, "request_id": "client_refine", "params": {"target_id": target(review)["instance_id"]}}
    assert_error("FORBIDDEN_SCOPE", lambda: asyncio.run(review._refine(request, review.context())))
    mock_saved_edit(review)
    review.confirm(payload(review))
    assert_error("FORBIDDEN_SCOPE", lambda: asyncio.run(review._refine(request, review.context(actor="user"))))
    assert review.jobs.submitted == []


def test_only_confirmed_host_context_receives_bounded_budget(review):
    mock_saved_edit(review)
    review.confirm(payload(review))
    confirmation = review._confirmation()
    cid = confirmation["command_id"]
    decision = "decision_" + canonical_hash([review.scene_id, cid])[:32]
    identity = target(review)["instance_id"]
    context = review.context(actor="user", session_id=confirmation["session_id"], decision_ref=decision,
                             allowed_object_ids=frozenset([identity]), allowed_fields=frozenset(["asset.geometry", "asset.materials"]))
    request = {"contract_version": "video2scene/1.0", "request_id": "req_" + canonical_hash([review.scene_id, cid, "refine"])[:40],
               "scene_id": review.scene_id, "tool": "refine_asset", "base_revision": confirmation["revision"], "base_head_seq": confirmation["head_seq"],
               "params": {"target_id": identity, "intent": "design_edit", "allowed_fields": ["asset.geometry", "asset.materials"],
                          "evidence_id": "mock_evidence", "constraints": {"keep_root_transform": True, "keep_support": True, "keep_bounds": True},
                          "budget": {"max_model_calls": 1, "max_iterations": 1, "timeout_s": 600}, "decision_ref": decision}}
    asyncio.run(review._refine(request, context))
    assert review.jobs.submitted[0]["params"]["budget"] == {"max_model_calls": 4, "max_iterations": 2, "timeout_s": 1200}
    review.test_view["head"]["head_seq"] += 1
    assert_error("STALE_REVISION", lambda: asyncio.run(review._refine(request, context)))


def test_restart_does_not_replay_uncertain_external_command(review):
    mock_saved_edit(review)
    receipt = review.confirm(payload(review))
    assert len(review.started_commands) == 1
    review.started_commands.clear()
    review.recover()
    assert review.started_commands == []
    record = review.store.command(review.scene_id, receipt["command_id"])
    assert record["state"] == "failed"
    assert record["error"]["details"]["outcome"] == "uncertain"
    status = review.review_status()
    assert status["phase"] == "failed"
    assert status["head"] == review.test_view["head"]
    duplicate = review.confirm(payload(review))
    assert duplicate["duplicate"] is True and review.started_commands == []


def test_status_reflects_actual_head_and_only_command_success_completes(review):
    mock_saved_edit(review)
    assert review.review_status()["head"]["head_seq"] == 1
    receipt = review.confirm(payload(review))
    assert review.review_status()["phase"] == "refining"
    review.backend_update({"stage": "independent_review", "target_id": target(review)["instance_id"]})
    assert review.review_status()["phase"] == "validating"
    record = review.store.command(review.scene_id, receipt["command_id"])
    record.update(state="succeeded", message="Mock commit completed")
    review.store.save_command(review.scene_id, receipt["command_id"], record)
    assert review.review_status()["phase"] == "complete"


def test_http_routes_keep_v1_and_reject_stale_confirmation(review):
    mock_saved_edit(review)
    server = make_review_server(review, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/agent/") as response:
            assert response.status == 200 and "确认布局" in response.read().decode()
        with urllib.request.urlopen(base + "/agent/review.mjs") as response:
            assert response.headers.get_content_type() == "text/javascript"
        with urllib.request.urlopen(base + "/") as response:
            assert "场景工作台" in response.read().decode()
        with urllib.request.urlopen(base + "/agent/status") as response:
            assert json.load(response)["head"] == review.test_view["head"]
        request = urllib.request.Request(base + "/agent/confirm", data=json.dumps(dict(payload(review), head_seq=0)).encode(),
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 409
        assert not review.confirmation_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def mock_batch(host, completed_count=0, phase="running"):
    """Isolated local ledger fixtures; never claims real provider execution."""
    from video2scene.agent_scene.batch_refinement import locked_layout
    from video2scene.contracts_v1 import file_hash
    folder = host.run_root / "batch_furniture_001"
    folder.mkdir()
    ids = [f"mock_furniture_{index:02}" for index in range(10)]
    prototype = copy.deepcopy(target(host))
    base = copy.deepcopy(host.test_view["snapshot"])
    for identity in ids:
        obj = copy.deepcopy(prototype)
        obj["instance_id"] = identity
        base["objects"].append(obj)
    base["revision"] = "mock_batch_base"
    base_head = {"revision": base["revision"], "head_seq": 2}
    auth = {"authorized": True, "authorization_source": "current_user_chat", "scene_id": host.scene_id,
            "batch_id": folder.name, "target_ids": ids, "baseline_head": base_head,
            "baseline_snapshot_hash": canonical_hash(base)}

    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    auth_path = save(folder / "authorization.json", auth)
    baseline = {"scene_id": host.scene_id, "head": base_head, "snapshot_hash": canonical_hash(base),
                "layout_hash": canonical_hash(locked_layout(base, ids)), "authorization_sha256": file_hash(auth_path), "target_ids": ids}
    for key in ("capture_record", "initial_snapshot", "geometry_proposals"):
        path = save(folder / f"mock_{key}.json", {"fixture": key})
        baseline[key] = {"path": str(path), "sha256": file_hash(path)}
    save(folder / "baseline.json", baseline)
    save(folder / "baseline_snapshot.json", base)
    snapshots = {base["revision"]: copy.deepcopy(base)}
    current, head, completed = copy.deepcopy(base), base_head, []
    for index, identity in enumerate(ids[:completed_count]):
        before = head
        current["parent_revision"] = head["revision"]
        current["revision"] = f"mock_batch_commit_{index}"
        head = {"revision": current["revision"], "head_seq": before["head_seq"] + 1}
        obj = next(row for row in current["objects"] if row["instance_id"] == identity)
        artifact = host.root / host.scene_id / "assets" / f"mock_{identity}.glb"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"Isolated fixture for {identity}; not a GLB".encode())
        obj["asset"]["asset_version"] = f"mock_{index}"
        obj["asset"]["artifact"] = {"uri": f"artifact://{host.scene_id}/assets/{artifact.name}", "sha256": file_hash(artifact),
                                         "media_type": "model/gltf-binary", "role": "object_geometry"}
        proof = {"status": "complete", "target_id": identity, "before_head": before, "head": head,
                 "asset": copy.deepcopy(obj["asset"]), "locked_fields_unchanged": True, "non_targets_unchanged": True,
                 "camera_evidence_unchanged": True}
        path = save(folder / "targets" / f"{index + 1:03}_{identity}" / "completion.json", proof)
        completed.append({"target_id": identity, "before_head": before, "head": head, "asset": copy.deepcopy(obj["asset"]),
                          "completion_path": str(path), "completion_sha256": file_hash(path), "provider_call_receipts": []})
        snapshots[current["revision"]] = copy.deepcopy(current)
    host.test_view = {"head": head, "snapshot": current}

    async def inspect(request, context):
        revision = request["params"]["revision"]
        return copy.deepcopy(host.test_view if revision == "head" else {"head": host.test_view["head"], "snapshot": snapshots[revision]})
    host.repo.inspect = inspect
    record = {"scene_id": host.scene_id, "batch_id": folder.name, "authorization_sha256": baseline["authorization_sha256"],
              "baseline_head": base_head, "baseline_snapshot_hash": baseline["snapshot_hash"], "baseline_layout_hash": baseline["layout_hash"],
              "target_ids": ids, "phase": phase, "stage": "independent_review", "active_target_id": ids[completed_count] if completed_count < 10 else None,
              "progress": {"completed": completed_count, "total": 10}, "completed": completed, "current_head": head,
              "message": "Explicit isolated batch fixture", "updated_at": "mock"}
    save(folder / "status.json", record)
    return folder, record, save


def test_batch_progress_validates_baseline_and_recommends_active_target(review):
    folder, record, save = mock_batch(review, 2)
    status = review.review_status()
    assert status["phase"] == "validating"
    assert status["batch"]["completed_count"] == 2 and status["batch"]["layout_preserved"] is True
    assert status["recommended_target_id"] == record["active_target_id"]
    assert status["batch"]["targets"][0]["state"] == "accepted"
    assert status["batch"]["targets"][2]["state"] == "validating"
    assert status["confirmation"] is None and not review.confirmation_path.exists()
    assert_error("FORBIDDEN_SCOPE", lambda: review.confirm(payload(review)))
    assert review.jobs.started == 0


@pytest.mark.parametrize("corruption", ["scene", "authorization", "layout", "asset", "capture", "count", "receipt"])
def test_batch_cannot_claim_complete_with_mismatched_proof(review, corruption):
    from video2scene.contracts_v1 import file_hash
    folder, record, save = mock_batch(review, 10, "complete")
    if corruption == "scene":
        record["scene_id"] = "different_scene"
    elif corruption == "authorization":
        record["authorization_sha256"] = "0" * 64
    elif corruption == "layout":
        review.test_view["snapshot"]["objects"][0]["transform"]["translation_m"][0] += .1
    elif corruption == "asset":
        identity = record["target_ids"][0]
        next(obj for obj in review.test_view["snapshot"]["objects"] if obj["instance_id"] == identity)["asset"]["artifact"]["sha256"] = "0" * 64
    elif corruption == "capture":
        (folder / "mock_capture_record.json").write_text("changed")
    elif corruption == "count":
        record["completed"].pop()
    elif corruption == "receipt":
        path = Path(record["completed"][0]["completion_path"])
        proof = json.loads(path.read_text())
        proof["locked_fields_unchanged"] = False
        save(path, proof)
        record["completed"][0]["completion_sha256"] = file_hash(path)
    save(folder / "status.json", record)
    status = review.review_status()
    assert status["phase"] == "failed" and status["batch"]["phase"] == "invalid"
    assert status["batch"]["editing_locked"] is True
    assert not review.confirmation_path.exists()


def test_batch_blocks_web_mutations_then_allows_edits_after_verified_completion(review):
    folder, record, save = mock_batch(review, 10, "running")
    command = {"contract_version": "video2scene/1.0", "command_id": "mock_after_batch", "session_id": "mock_browser",
               "scene_id": review.scene_id, "base_revision": review.test_view["head"]["revision"],
               "base_head_seq": review.test_view["head"]["head_seq"], "selected_ids": [target(review)["instance_id"]],
               "intent": "design_edit", "kind": "patch", "edits": [{"instance_id": target(review)["instance_id"],
                                                                         "field": "transform.scale", "value": [1.1, 1, 1]}]}
    assert_error("FORBIDDEN_SCOPE", lambda: review.submit_command(command))
    assert_error("FORBIDDEN_SCOPE", lambda: asyncio.run(review.execute({"tool": "commit_candidate"})))
    record.update(phase="complete", stage="complete")
    save(folder / "status.json", record)
    status = review.review_status()
    assert status["phase"] == "complete" and status["batch"]["editing_locked"] is False
    review.submit_command(command)
    assert len(review.started_commands) == 1
    review.test_view["head"] = {"revision": "mock_later_human_edit", "head_seq": 13}
    review.test_view["snapshot"]["revision"] = "mock_later_human_edit"
    review.test_view["snapshot"]["objects"][0]["transform"]["scale"][0] += .1
    status = review.review_status()
    assert status["phase"] == "needs_user" and status["batch"]["phase"] == "scene_changed"
    assert status["batch"]["editing_locked"] is False and status["batch"]["layout_preserved"] is False


def mock_room_finish(host, complete=False):
    from video2scene.agent_scene.review_server import ROOM_REPLACE_IDS, ROOM_ADD_IDS
    from video2scene.contracts_v1 import file_hash
    folder = host.run_root / "room_finish_001"
    folder.mkdir()
    ids = list(ROOM_REPLACE_IDS + ROOM_ADD_IDS)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    base = copy.deepcopy(host.test_view["snapshot"])
    prototype = copy.deepcopy(target(host))
    base["objects"] = [obj for obj in base["objects"] if obj["instance_id"] not in ids]
    for identity in ROOM_REPLACE_IDS:
        obj = copy.deepcopy(prototype)
        obj.update(instance_id=identity, kind="floor" if identity == "floor" else "wall", category="structure/room")
        base["objects"].append(obj)
    base["revision"] = "mock_room_baseline"
    base_head = {"revision": base["revision"], "head_seq": 12}
    auth = {"run_id": folder.name, "authorized": True, "scene_id": host.scene_id, "authorization_source": "current_user_chat",
            "user_instruction": "Isolated fixture authorizes room work", "authorized_at": "mock", "baseline_head": base_head,
            "baseline_snapshot_hash": canonical_hash(base), "targets": ids,
            "replace_asset_ids": list(ROOM_REPLACE_IDS), "add_object_ids": list(ROOM_ADD_IDS)}
    auth_path = save(folder / "authorization.json", auth)
    save(folder / "baseline_snapshot.json", base)
    capture_path = save(folder / "mock_capture.json", {"fixture": "camera record"})
    save(folder / "capture_binding.json", {"path": str(capture_path), "sha256": file_hash(capture_path)})
    record = {"scene_id": host.scene_id, "run_id": folder.name, "authorization_sha256": file_hash(auth_path),
              "baseline_head": base_head, "current_head": base_head, "targets": ids, "completed": [],
              "active_target": "floor", "phase": "running", "stage": "authoring", "message": "Isolated room fixture"}
    current, head = copy.deepcopy(base), base_head
    if complete:
        for identity in ids:
            obj = next((obj for obj in current["objects"] if obj["instance_id"] == identity), None)
            if obj is None:
                obj = copy.deepcopy(prototype)
                obj["instance_id"] = identity
                current["objects"].append(obj)
            asset_path = host.root / host.scene_id / "assets" / f"mock_room_{identity}.glb"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(f"Isolated artifact fixture {identity}; no geometry execution".encode())
            obj["asset"]["asset_version"] = f"mock_room_{identity}"
            obj["asset"]["artifact"] = {"uri": f"artifact://{host.scene_id}/assets/{asset_path.name}", "sha256": file_hash(asset_path),
                                       "media_type": "model/gltf-binary", "role": "object_geometry"}
        current["revision"] = "mock_room_complete"
        current["parent_revision"] = base["revision"]
        head = {"revision": current["revision"], "head_seq": 13}
        final_path = save(folder / "final_snapshot.json", current)
        proof = {"status": "complete", "scene_id": host.scene_id, "run_id": folder.name, "authorization_sha256": file_hash(auth_path),
                 "baseline_head": base_head, "head": head, "final_snapshot_sha256": file_hash(final_path),
                 "assets": {obj["instance_id"]: obj["asset"] for obj in current["objects"] if obj["instance_id"] in ids},
                 "preservation": {"existing_layout": True, "non_targets": True}}
        proof_path = save(folder / "completion.json", proof)
        record.update(phase="complete", stage="complete", current_head=head, active_target=None,
                      completed=[{"instance_id": identity, "status": "accepted"} for identity in ids], completion_sha256=file_hash(proof_path))
    host.test_view = {"head": head, "snapshot": copy.deepcopy(current)}
    snapshots = {base["revision"]: base, current["revision"]: current}

    async def inspect(request, context):
        revision = request["params"]["revision"]
        return copy.deepcopy(host.test_view if revision == "head" else {"head": host.test_view["head"], "snapshot": snapshots[revision]})
    host.repo.inspect = inspect
    save(folder / "status.json", record)
    return folder, record, save


def test_room_progress_takes_priority_over_completed_furniture_history(review):
    folder, record, save = mock_room_finish(review)

    async def furniture_history(view):
        return {"phase": "scene_changed", "editing_locked": False, "current_target_id": None,
                "completed_count": 10, "total": 10, "message": "Older furniture batch result"}
    review._batch_status = furniture_history
    record["completed"] = [{"instance_id": "wall_tv", "status": "prepared"}]
    save(folder / "status.json", record)
    status = review.review_status()
    assert status["phase"] == "refining" and status["head"]["head_seq"] == 12
    assert status["room_finish"]["completed_count"] == 0
    assert status["room_finish"]["targets"][1]["state"] == "prepared"
    assert status["batch"]["history_only"] is True
    assert status["recommended_target_id"] == "floor"
    assert status["message"] != "Older furniture batch result"
    assert_error("FORBIDDEN_SCOPE", lambda: review.confirm(payload(review)))
    assert_error("FORBIDDEN_SCOPE", lambda: asyncio.run(review.execute({"tool": "apply_scene_patch"})))
    assert review.jobs.started == 0 and not review.confirmation_path.exists()


@pytest.mark.parametrize("fault", ["authorization", "early_accepted", "capture", "layout", "proof", "asset_bytes"])
def test_room_finish_rejects_unauthorized_or_unproven_success(review, fault):
    folder, record, save = mock_room_finish(review, complete=fault not in ("authorization", "early_accepted", "capture"))
    if fault == "authorization":
        record["authorization_sha256"] = "0" * 64
    elif fault == "early_accepted":
        record["completed"] = [{"instance_id": "floor", "status": "accepted"}]
    elif fault == "capture":
        (folder / "mock_capture.json").write_text("changed")
    elif fault == "layout":
        review.test_view["snapshot"]["objects"][0]["transform"]["scale"][0] += .2
    elif fault == "proof":
        record["completion_sha256"] = "0" * 64
    elif fault == "asset_bytes":
        (review.root / review.scene_id / "assets" / "mock_room_floor.glb").write_bytes(b"changed")
    save(folder / "status.json", record)
    status = review.review_status()
    assert status["phase"] == "failed" and status["room_finish"]["phase"] == "invalid"
    assert status["room_finish"]["editing_locked"] is True


def test_room_completion_shows_new_head_and_allows_later_edits(review):
    mock_room_finish(review, complete=True)
    status = review.review_status()
    assert status["phase"] == "complete" and status["head"]["head_seq"] == 13
    assert status["room_finish"]["completed_count"] == 7 and status["room_finish"]["layout_preserved"] is True
    assert status["room_finish"]["editing_locked"] is False
    review._batch_edit_guard()
    review.test_view["head"] = {"revision": "mock_after_room_edit", "head_seq": 14}
    review.test_view["snapshot"]["revision"] = "mock_after_room_edit"
    review.test_view["snapshot"]["objects"][0]["transform"]["scale"][0] += .2
    status = review.review_status()
    assert status["phase"] == "needs_user" and status["room_finish"]["phase"] == "scene_changed"
    assert status["room_finish"]["editing_locked"] is False
