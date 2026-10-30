"""Real queue/lease/SQLite tests with explicit CPU-only mock refinement."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_state"))
from test_repository import Harness, assert_code, run, sample
from video2scene.contracts_v1 import ContractError, validate
from video2scene.job_service import SQLiteJobService


def refine_request(h, request_id="refine_1", timeout_s=20):
    value = sample("request_refine_asset")
    value.update(scene_id="demo_scene", base_revision="r000", base_head_seq=0, request_id=request_id)
    value["params"].update(target_id="t001", allowed_fields=["asset.geometry", "asset.materials"],
        evidence_id=h.evidence["evidence_id"], decision_ref="decision_1", intent="design_edit",
        budget={"max_iterations":1,"timeout_s":timeout_s})
    return value


def control(service, receipt, action="status"):
    return run(service.control({"contract_version":"video2scene/1.0", "request_id":"control_"+uuid4().hex,
        "scene_id":"demo_scene","tool":"job_control","params":{"job_id":receipt["job_id"],"action":action}}))


class Backend:
    def __init__(self, h, gate=None, stop_gate=None):
        self.h, self.gate, self.stop_gate = h, gate, stop_gate
        self.calls = 0
        self.started = threading.Event()

    async def refine(self, spec, scene, evidence, context):
        self.calls += 1
        self.started.set()
        while self.gate is not None and not self.gate.is_set():
            if context.cancel_event.is_set():
                while self.stop_gate is not None and not self.stop_gate.is_set():
                    await asyncio.sleep(.01)
                context.check_cancelled()
            await asyncio.sleep(.01)
        context.check_cancelled()
        obj = copy.deepcopy(next(obj for obj in scene["objects"] if obj["instance_id"] == spec["target_id"]))
        obj["asset"]["asset_version"] = "asset_" + uuid4().hex
        obj["asset"]["artifact"] = self.h.artifact("generated/" + uuid4().hex + ".glb", b"new mock refined mesh")
        return {"artifact_refs":[obj["asset"]["artifact"]], "asset_updates":[{"instance_id":obj["instance_id"],"asset":obj["asset"],"local_bounds_m":obj["local_bounds_m"]}], "input_fingerprint":spec["candidate"]["candidate_hash"]}


def make_service(h, backend, autostart=False):
    return SQLiteJobService(h.path / "jobs", h.repo, lambda key,ctx:h.evidence, backend, autostart=autostart)


def await_terminal(service, receipt, timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        record=control(service,receipt)
        if record["state"] in ("succeeded","failed","cancelled","stale"):
            return record
        time.sleep(.02)
    raise AssertionError("Job did not finish")


def test_idempotent_queue_success_does_not_commit(tmp_path):
    h=Harness(tmp_path); backend=Backend(h); service=make_service(h,backend)
    req=refine_request(h)
    receipt=run(service.submit(req,h.ctx))
    assert run(service.submit(req,h.ctx))==receipt
    changed=copy.deepcopy(req);changed["params"]["constraints"]["keep_bounds"]=False
    assert_code("IDEMPOTENCY_CONFLICT",lambda:run(service.submit(changed,h.ctx)))
    assert service.run_next() is True
    record=control(service,receipt)
    validate(record,"JobRecord")
    assert record["state"]=="succeeded" and record["result"]["state"]=="built"
    assert backend.calls==1 and h.view()["head"]["head_seq"]==0
    assert control(service,receipt,"cancel")==record
    assert run(service.get_by_request("demo_scene","refine_1"))==receipt
    assert run(service.submit(req,h.ctx))==receipt
    assert service.run_next() is False


def test_cancel_queued_confirms_no_execution(tmp_path):
    h=Harness(tmp_path); backend=Backend(h); service=make_service(h,backend)
    receipt=run(service.submit(refine_request(h),h.ctx))
    assert control(service,receipt,"cancel")["state"]=="cancelled"
    assert service.run_next() is False and backend.calls==0


def test_cancel_running_waits_for_actual_stop(tmp_path):
    h=Harness(tmp_path); gate=threading.Event(); stop=threading.Event()
    backend=Backend(h,gate,stop); service=make_service(h,backend,True)
    try:
        receipt=run(service.submit(refine_request(h),h.ctx))
        assert backend.started.wait(3)
        assert control(service,receipt,"cancel")["state"]=="cancel_requested"
        time.sleep(.1)
        assert control(service,receipt)["state"]=="cancel_requested"
        stop.set()
        assert await_terminal(service,receipt)["state"]=="cancelled"
        assert h.view()["head"]["head_seq"]==0
    finally:
        gate.set();stop.set();service.shutdown()


def test_timeout_stops_then_failed(tmp_path):
    h=Harness(tmp_path); gate=threading.Event();backend=Backend(h,gate)
    service=make_service(h,backend,True)
    try:
        receipt=run(service.submit(refine_request(h,timeout_s=1),h.ctx))
        record=await_terminal(service,receipt)
        assert record["state"]=="failed" and record["error"]["code"]=="TIMEOUT"
    finally:
        gate.set();service.shutdown()


def test_b12_background_stale_preserves_new_edit(tmp_path):
    h=Harness(tmp_path);gate=threading.Event();backend=Backend(h,gate);service=make_service(h,backend,True)
    try:
        receipt=run(service.submit(refine_request(h),h.ctx))
        assert backend.started.wait(3)
        candidate=h.build(h.patch());h.report(candidate);committed=h.commit(candidate)
        gate.set()
        record=await_terminal(service,receipt)
        assert record["state"]=="stale" and record["error"]["code"]=="STALE_REVISION"
        assert h.view()["head"]["revision"]==committed["revision"]
    finally:
        gate.set();service.shutdown()


def test_resource_lease_prevents_parallel_workers(tmp_path):
    h=Harness(tmp_path);gate=threading.Event();backend=Backend(h,gate)
    first=make_service(h,backend,True); second=make_service(h,backend)
    try:
        receipt=run(first.submit(refine_request(h),h.ctx))
        receipt2=run(first.submit(refine_request(h,"refine_2"),h.ctx))
        assert backend.started.wait(3)
        assert second.run_next() is False
        assert backend.calls==1
        gate.set()
        assert await_terminal(first,receipt)["state"]=="succeeded"
        assert await_terminal(first,receipt2)["state"]=="succeeded"
        assert backend.calls==2
    finally:
        gate.set();first.shutdown();second.shutdown()


def test_real_worker_crash_is_uncertain_and_never_reexecutes(tmp_path):
    h=Harness(tmp_path)
    state={"request":refine_request(h),"context":SQLiteJobService._ctx_data(h.ctx),"evidence":h.evidence}
    (tmp_path/"job_crash.json").write_text(json.dumps(state),encoding="utf8")
    child=subprocess.run([sys.executable,str(Path(__file__).with_name("crash_worker.py")),str(tmp_path)],capture_output=True,timeout=30)
    assert child.returncode==79,child.stderr.decode(errors="replace")
    backend=Backend(h);service=make_service(h,backend)
    receipt=run(service.get_by_request("demo_scene","refine_1"))
    record=control(service,receipt)
    assert record["state"]=="failed" and record["error"]["details"]["outcome"]=="uncertain"
    assert run(service.submit(refine_request(h),h.ctx))==receipt
    assert service.run_next() is False and backend.calls==0
    assert h.view()["head"]["head_seq"]==0
