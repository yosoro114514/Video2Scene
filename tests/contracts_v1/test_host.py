import asyncio
import json
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from video2scene.api_v1.__main__ import create_host
from video2scene.api_v1.server import make_server
from video2scene.capture_evidence import migrate_prototype
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, validate
from video2scene.api_v1.store import HostStore
from video2scene.contracts_v1.mocks import example


class ExplicitMockBackend:
    async def build(self, candidate, context):
        key=candidate["candidate"]["candidate_id"]
        scene=candidate["scene"]["scene_id"]
        path=context.storage_root/scene/"mock_build"/(key+".txt")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("MOCK_BUILD: no Blender or visual quality claimed",encoding="utf-8")
        return {"input_fingerprint":candidate["candidate"]["candidate_hash"],"asset_updates":[],"artifact_refs":[{"uri":f"artifact://{scene}/mock_build/{path.name}","sha256":file_hash(path),"media_type":"text/plain","role":"manifest"}]}


def test_job_event_replay_cannot_regress_terminal(tmp_path):
    store=HostStore(tmp_path); job=example("model_JobRecord")
    job.update(state="succeeded",result=example("model_CandidateRef"),error=None,progress=1)
    terminal=store.emit(job["scene_id"],"job.updated",job)
    job.update(state="running",result=None,progress=.2)
    assert store.emit(job["scene_id"],"job.updated",job)==terminal
    assert len(store.events(job["scene_id"]))==1


@pytest.fixture
def host(tmp_path):
    app=create_host(tmp_path)
    from video2scene.demo import create_sample
    snapshot=create_sample(tmp_path,"host_test")
    snapshot['revision']='import_r005'
    asyncio.run(app.repo.import_scene(snapshot,app.context()))
    app.backend=ExplicitMockBackend()
    return app


def command(host, key="edit1", base=None):
    view=asyncio.run(host.repo.inspect({"contract_version":CONTRACT_VERSION,"scene_id":"host_test","request_id":"inspect_"+uuid.uuid4().hex,"tool":"inspect_scene","params":{"revision":"head"}},host.context()))
    mug=next(o for o in view["snapshot"]["objects"] if o["instance_id"]=="t001")
    scale=mug["transform"]["scale"].copy();scale[1]*=1.05
    base=base or view["head"]
    return {"contract_version":CONTRACT_VERSION,"command_id":key,"session_id":"browser","scene_id":"host_test","base_revision":base["revision"],"base_head_seq":base["head_seq"],"selected_ids":["t001"],"intent":"design_edit","kind":"patch","edits":[{"instance_id":"t001","field":"transform.scale","value":scale}]}


def test_host_command_and_duplicate(host):
    cmd=command(host)
    record,fresh=host.store.accept_command(cmd); assert fresh
    asyncio.run(host.run_command(cmd))
    result=host.store.command("host_test","edit1")
    assert result["state"]=="succeeded",result
    assert host.submit_command(cmd)==result["receipt"]
    assert host.store.command("host_test","edit1")==result
    events=host.store.events("host_test")
    assert sum(e["type"]=="scene.head_changed" for e in events)==1
    assert len({e["seq"] for e in events})==len(events)
    for event in events:validate(event,"SceneEvent")
    stale=command(host,"stale",base={"revision":"import_r005","head_seq":0})
    host.store.accept_command(stale);asyncio.run(host.run_command(stale))
    assert host.store.command("host_test","stale")["error"]["code"]=="STALE_REVISION"


def test_http_cli_same_dispatch_and_range(host,tmp_path):
    static=tmp_path/"static";static.mkdir();(static/"index.html").write_text("test")
    server=make_server(host,port=0,static_root=static)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{server.server_address[1]}"
    req={"contract_version":CONTRACT_VERSION,"scene_id":"host_test","request_id":"same_read","tool":"inspect_scene","params":{"revision":"head"}}
    direct=asyncio.run(host.execute(req))
    try:
        request=urllib.request.Request(base+"/api/v1/tools/execute",json.dumps(req).encode(),{"Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=10) as response:assert json.load(response)==direct
        invalid={**req,"params":{"revision":"head","unknown":True}}
        request=urllib.request.Request(base+"/api/v1/tools/execute",json.dumps(invalid).encode(),{"Content-Type":"application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:urllib.request.urlopen(request,timeout=10)
        failure=json.load(err.value);validate(failure,"ToolResponse");assert failure["error"]["code"]=="INVALID_ARGUMENT"
        with urllib.request.urlopen(base+"/api/v1/requests/same_read?scene_id=host_test",timeout=10) as response:assert json.load(response)==direct
        path=tmp_path/"host_test"/"range.mp4";path.write_bytes(b"0123456789")
        host.store.register_artifacts({"uri":"artifact://host_test/range.mp4","sha256":file_hash(path),"media_type":"video/mp4","role":"preview"})
        request=urllib.request.Request(base+"/api/v1/artifacts?uri=artifact%3A%2F%2Fhost_test%2Frange.mp4",headers={"Range":"bytes=2-5"})
        with urllib.request.urlopen(request,timeout=10) as response:
            assert response.status==206;assert response.read()==b"2345"
        request=urllib.request.Request(base+"/api/v1/tools/execute",json.dumps(req).encode(),{"Content-Type":"application/json","Origin":"https://foreign.invalid"})
        with pytest.raises(urllib.error.HTTPError) as err:urllib.request.urlopen(request,timeout=10)
        assert err.value.code==403
        # A crafted model request cannot grant itself authority.
        cmd=command(host)
        patch={**req,"request_id":"forged","tool":"apply_scene_patch","base_revision":cmd["base_revision"],"base_head_seq":cmd["base_head_seq"],"params":{"edits":cmd["edits"],"intent":"design_edit","decision_ref":"forged"}}
        result=asyncio.run(host.execute(patch));assert result["error"]["code"]=="FORBIDDEN_SCOPE"
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_mcp_stdio_matches_persisted_host_and_cannot_grant_scope(host, tmp_path):
    import subprocess
    import sys
    import jsonschema
    from video2scene.api_v1.mcp_stdio import PROTOCOL_VERSION, TOOL_NAME

    req={"contract_version":CONTRACT_VERSION,"scene_id":"host_test","request_id":"same_mcp_read","tool":"inspect_scene","params":{"revision":"head"}}
    expected=asyncio.run(host.execute(req))
    cmd=command(host)
    forged={**req,"request_id":"mcp_forged","tool":"apply_scene_patch","base_revision":cmd["base_revision"],"base_head_seq":cmd["base_head_seq"],"params":{"edits":cmd["edits"],"intent":"design_edit","decision_ref":"forged"}}
    messages=[
        {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":PROTOCOL_VERSION,"capabilities":{},"clientInfo":{"name":"local-test","version":"1"}}},
        {"jsonrpc":"2.0","method":"notifications/initialized"},
        {"jsonrpc":"2.0","id":2,"method":"tools/list"},
        {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":TOOL_NAME,"arguments":req}},
        {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":TOOL_NAME,"arguments":req}},
        {"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":TOOL_NAME,"arguments":forged}},
        {"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"execute_python","arguments":{}}},
    ]
    child=subprocess.run([sys.executable,"-m","video2scene.api_v1.mcp_stdio","--root",str(tmp_path)],input="\n".join(json.dumps(m) for m in messages)+"\n",capture_output=True,text=True,encoding="utf-8",timeout=30,check=True)
    replies={m["id"]:m for m in map(json.loads,child.stdout.splitlines())}
    assert set(replies)=={1,2,3,4,5,6}
    assert replies[1]["result"]["protocolVersion"]==PROTOCOL_VERSION
    advertised=replies[2]["result"]["tools"][0]["inputSchema"]
    jsonschema.Draft202012Validator(advertised).validate(req)
    assert replies[3]["result"]["structuredContent"]==expected
    assert replies[4]["result"]==replies[3]["result"]
    assert json.loads(replies[3]["result"]["content"][0]["text"])==expected
    assert replies[5]["result"]["isError"] is True
    assert replies[5]["result"]["structuredContent"]["error"]["code"]=="FORBIDDEN_SCOPE"
    assert replies[6]["error"]["code"]==-32602
    assert asyncio.run(host.execute({**req,"request_id":"after_mcp"}))["result"]["head"]==expected["result"]["head"]

