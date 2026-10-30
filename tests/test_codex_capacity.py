import json
from types import SimpleNamespace
import pytest

from video2scene.codex_capacity import capacity_rejection
from video2scene import codex_review


def rejected(output,extra=()):
    events=[{'type':'thread.started','thread_id':'fixture'},{'type':'turn.started'},*extra,
            {'type':'error','message':'Selected model is at capacity. Please try a different model.'},
            {'type':'turn.failed','error':{'message':'Selected model is at capacity. Please try a different model.'}}]
    (output/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events),encoding='utf-8')
    (output/'stderr.log').write_text('isolated capacity fixture',encoding='utf-8')


@pytest.mark.parametrize('extra',[
    [{'type':'item.started','item':{'type':'command_execution'}}],
    [{'type':'item.completed','item':{'type':'agent_message'}}],
    [{'type':'turn.completed'}],[{'type':'unknown'}],
])
def test_capacity_after_any_work_or_unknown_event_is_not_retryable(tmp_path,extra):
    rejected(tmp_path,extra);assert capacity_rejection(tmp_path) is None


def test_capacity_requires_complete_terminal_log_and_no_response(tmp_path):
    rejected(tmp_path);assert capacity_rejection(tmp_path)['execution_started'] is False
    with (tmp_path/'events.jsonl').open('a') as f:f.write('\n{partial')
    assert capacity_rejection(tmp_path) is None
    rejected(tmp_path);(tmp_path/'response.json').write_text('{}')
    assert capacity_rejection(tmp_path) is None


@pytest.mark.parametrize('recover',[True,False])
def test_capacity_retry_preserves_failure_evidence_and_stays_bounded(tmp_path,monkeypatch,recover):
    image=tmp_path/'image.png';image.write_bytes(b'explicit fixture, not passed to a model')
    out=tmp_path/'call';calls=[];delays=[]
    monkeypatch.setattr(codex_review,'codex_command',lambda:['fixture-codex'])
    monkeypatch.setattr(codex_review.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='Logged in using ChatGPT',stderr=''))
    monkeypatch.setattr(codex_review.subprocess,'check_output',lambda *a,**k:'fixture version')
    monkeypatch.setattr(codex_review.time,'sleep',lambda seconds:delays.append(seconds))
    def execute(argv,prompt,output,env,timeout,max_tool_calls):
        calls.append(argv)
        assert 'OPENAI_API_KEY' not in env and 'CODEX_API_KEY' not in env
        if len(calls)<3 or not recover:rejected(output);return 1
        (output/'events.jsonl').write_text(json.dumps({'type':'turn.completed','usage':{}}))
        (output/'response.json').write_text('{"status":"complete"}')
        return 0
    monkeypatch.setattr(codex_review,'_execute_cli',execute)
    if recover:assert codex_review.invoke('fixture',[image],out)['status']=='complete'
    else:
        with pytest.raises(RuntimeError,match='at capacity'):codex_review.invoke('fixture',[image],out)
    assert len(calls)==3 and delays==[15,30]
    assert len(list((out/'preexecution_retries').glob('*/events.jsonl')))==2
    request=json.loads((out/'request.json').read_text())
    assert len(request['preexecution_capacity_retries'])==2
    assert request['status']==('complete' if recover else 'failed')


def test_non_capacity_or_executed_failure_is_never_retried(tmp_path,monkeypatch):
    image=tmp_path/'image.png';image.write_bytes(b'fixture');calls=[]
    monkeypatch.setattr(codex_review,'codex_command',lambda:['fixture-codex'])
    monkeypatch.setattr(codex_review.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='ChatGPT',stderr=''))
    monkeypatch.setattr(codex_review.subprocess,'check_output',lambda *a,**k:'fixture')
    def execute(argv,prompt,output,*args):
        calls.append(1);rejected(output,[{'type':'item.started','item':{'type':'command_execution'}}]);return 1
    monkeypatch.setattr(codex_review,'_execute_cli',execute)
    with pytest.raises(RuntimeError):codex_review.invoke('fixture',[image],tmp_path/'call')
    assert len(calls)==1
