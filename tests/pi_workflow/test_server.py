"""HTTP boundary checks on isolated projects. No real model or human approvals."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from video2scene.pi_workflow.service import WorkflowService
from video2scene.pi_workflow.server import make_server


@pytest.fixture
def running(tmp_path,monkeypatch):
    service=WorkflowService(tmp_path/'workflow',port=0)
    monkeypatch.setattr(service,'_start',lambda *args,**kwargs:None)
    server=make_server(service,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path,body=None,headers=None,raw=None):
        data=raw if raw is not None else json.dumps(body).encode() if body is not None else None
        hdr={'Content-Type':'application/json',**(headers or {})}
        try:response=urlopen(Request(base+path,data=data,headers=hdr),timeout=15)
        except HTTPError as error:response=error
        result=response.read()
        return response.status,json.loads(result) if response.headers.get_content_type()=='application/json' else result
    yield service,request,tmp_path
    server.shutdown();server.server_close();thread.join();service.close()


def test_workbench_and_import_idempotency(running):
    service,request,tmp=running
    assert request('/workflow/')[0]==200
    assert request('/workflow/app.mjs')[0]==200
    video=tmp/'source.mp4';video.write_bytes(b'explicit-test-video-not-decoded')
    payload={'request_id':'same-create','source_path':str(video),'instruction':'fixture task'}
    code,project=request('/workflow/api/projects',payload);assert code==201
    assert request('/workflow/api/projects',payload)[1]['id']==project['id']
    assert len(service.projects())==1
    assert len(service.get(project['id'])['messages'])==1
    assert request('/workflow/api/projects',{**payload,'name':'changed'})[0]>=400


def test_rolling_capabilities(running):
    service,request,_=running
    code,capabilities=request('/workflow/api/capabilities')
    assert code==200 and capabilities['rolling_refinement'] is True
    assert capabilities['default_object_concurrency']==3
    assert capabilities['max_object_concurrency']==4
    assert capabilities['resources']==service.resources.snapshot()
    assert capabilities['resources']['limits']=={'model':3,'blender':1}


def test_internal_token_and_write_origin(running):
    service,request,tmp=running
    assert request('/workflow/internal/tool',{})[0]>=400
    assert request('/workflow/api/projects',{},headers={'Origin':'https://untrusted.example'})[0]>=400
    assert request('/workflow/api/upload?name=test.mp4',raw=b'fixture',headers={'Origin':'https://untrusted.example'})[0]>=400


def test_upload_and_artifact_scope(running):
    service,request,tmp=running
    code,upload=request('/workflow/api/upload?name=clip.mp4',raw=b'fixture')
    assert code==201
    code,project=request('/workflow/api/projects',{'request_id':'upload-test','source_path':upload['source_path']})
    assert code==201
    pid=project['id']
    assert request(f'/workflow/artifacts?project_id={pid}&path=../../workflow.sqlite')[0]>=400
    assert request('/workflow/api/projects/missing')[0]==404


def test_messages_replay_once_and_pause_without_approval(running):
    service,request,tmp=running
    video=tmp/'clip.mp4';video.write_bytes(b'fixture')
    _,p=request('/workflow/api/projects',{'request_id':'message-project','source_path':str(video)})
    route=f"/workflow/api/projects/{p['id']}"
    payload={'request_id':'one-message','message':'inspect fixture','selected_ids':[]}
    first=request(route+'/messages',payload)
    second=request(route+'/messages',payload)
    assert first==second and first[0]==200
    assert len(service.get(p['id'])['messages'])==1
    assert request(route+'/control',{'action':'approve_whitebox','revision':'no-scene','head_seq':0})[0]>=400
    assert request(route+'/control',{'action':'pause'})[1]['phase']=='paused'
    assert service.get(p['id'])['acceptance'] is None
