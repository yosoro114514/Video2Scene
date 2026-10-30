import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scene_state"))
from test_repository import Harness, assert_code, run, sample
from video2scene.agent_runtime import CodexPlanner, DeterministicPlanner, PiProvider
from video2scene.contracts_v1 import ContractError


def command():
    return {"contract_version":"video2scene/1.0","command_id":"command_1","session_id":"session_1",
        "scene_id":"demo_scene","base_revision":"r000","base_head_seq":0,"selected_ids":["t001"],
        "intent":"design_edit","kind":"language","text":"再高 5%"}


def test_b18_compounds_current_height_and_clarifies(tmp_path):
    h=Harness(tmp_path);view=h.view();planner=DeterministicPlanner();cmd=command()
    first=run(planner(cmd,view,h.ctx))
    view["snapshot"]["objects"][1]["transform"]["scale"]=first["edits"][0]["value"]
    second=run(planner(cmd,view,h.ctx))
    assert second["edits"][0]["value"]==pytest.approx([1,1.1025,1])
    cmd["text"]="把杯子挪过去一点"
    assert run(planner(cmd,view,h.ctx))["action"]=="clarify"
    cmd["text"]="不要增加高度 5%"
    assert run(planner(cmd,view,h.ctx))["action"]=="clarify"
    assert h.view()["head"]["head_seq"]==0


def test_editor_explicit_refinement_command(tmp_path):
    h=Harness(tmp_path);cmd=command()
    cmd["text"]="精修选中对象的几何与材质，保持根变换、支撑和局部边界。"
    planner=DeterministicPlanner()
    assert run(planner(cmd,h.view(),h.ctx))=={"action":"refine","target_id":"t001"}
    cmd["selected_ids"]=["t001","table_00"]
    assert run(planner(cmd,h.view(),h.ctx))["action"]=="clarify"


class Evidence:
    def __init__(self,bundle):self.bundle=bundle
    async def select(self,request,context):return copy.deepcopy(self.bundle)


def test_b11_actual_attachment_paths_replay_and_no_model_tools(tmp_path):
    h=Harness(tmp_path);bundle=sample("model_EvidenceBundle")
    for index,frame in enumerate(bundle["frames"]):
        frame["rgb"]=h.artifact(f"images/{index}.png",b"explicit mock PNG payload")
    calls=[]
    def fake_invoke(prompt,images,output,**kwargs):
        calls.append({"images":images,"kwargs":kwargs})
        assert all(Path(p).read_bytes()==b"explicit mock PNG payload" for p in images)
        assert "source_frame_id" in prompt
        assert kwargs["max_tool_calls"]==0 and kwargs["model"]=="gpt-6-astra"
        return {"action":"edit","edits":[{"instance_id":"t001","field":"transform.scale","value_json":"[1,1.05,1]"}],"target_id":"","question":"","observation":"mock observation"}
    planner=CodexPlanner(tmp_path/"calls",Evidence(bundle),h.assets.resolve,invoke_fn=fake_invoke)
    first=run(planner(command(),h.view(),h.ctx));second=run(planner(command(),h.view(),h.ctx))
    assert first==second and len(calls)==1
    changed=command();changed["text"]="double height"
    assert_code("IDEMPOTENCY_CONFLICT",lambda:run(planner(changed,h.view(),h.ctx)))


def test_failed_model_call_is_not_automatically_repeated(tmp_path):
    h=Harness(tmp_path);bundle=sample("model_EvidenceBundle")
    for index,frame in enumerate(bundle["frames"]):frame["rgb"]=h.artifact(f"images/{index}.png",b"mock")
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1);raise TimeoutError("simulated lost response")
    planner=CodexPlanner(tmp_path/"calls",Evidence(bundle),h.assets.resolve,invoke_fn=fail)
    assert_code("TIMEOUT",lambda:run(planner(command(),h.view(),h.ctx)))
    assert_code("RESOURCE_EXHAUSTED",lambda:run(planner(command(),h.view(),h.ctx)))
    assert len(calls)==1


def test_pi_missing_capability_is_explicit():
    async def exercise():
        async for _ in PiProvider().run({}):pass
    assert_code("TOOL_UNAVAILABLE",lambda:run(exercise()))
