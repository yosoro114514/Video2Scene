import asyncio
import json
from types import SimpleNamespace
import pytest

from test_rolling import rolling_h,start,row,wait_for
from test_service import h
from video2scene.contracts_v1 import ContractError
from video2scene.pi_workflow.job_recovery import verified_unstarted_failure


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.mark.parametrize('executed',[False,True])
def test_explicit_capacity_recovery_keeps_failed_attempt_and_uses_new_ids(rolling_h,executed,monkeypatch):
    h=rolling_h;original=h.actions.backend_factory
    class Capacity(original):
        async def refine(self,spec,*args):
            if spec['target_id']!='chair':return await super().refine(spec,*args)
            asset=h.service.runtime/h.pid/'assets/gpt-direct-capacity-fixture'
            author=asset/'author_00'
            save(author/'contract.json',{'instance_id':'chair','candidate':spec['candidate']})
            save(author/'request.json',{'status':'failed','authentication':'ChatGPT subscription'})
            save(asset/'author_00_control/invoke.log.execution.json',{'state':'failed','returncode':1,'owned_subprocess':True})
            (author/'events.jsonl').write_text(json.dumps({'type':'turn.failed','error':{'message':'Selected model is at capacity.'}}))
            if executed:(author/'object.py').write_text('raise RuntimeError("fixture already executed")')
            raise ContractError('BUILD_FAILED','Direct modeling subprocess failed')
    h.actions.backend_factory=Capacity
    start(h,3);wait_for(lambda:not h.service.rolling.active(h.pid))
    h.service._on_pi_event(h.pid,{'type':'settled','run_id':'fixture'})
    h.service.control(h.pid,{'action':'pause'})
    h.service._on_pi_event(h.pid,{'type':'aborted','run_id':'fixture'})
    before=row(h,'chair');old_op=before['operation_id'];calls=list(h.author_calls)
    assert before['state']=='needs_review'
    if executed:
        with pytest.raises(ContractError,match='不能重放'):h.service.control(h.pid,{'action':'resume_object','target_id':'chair'})
        assert row(h,'chair')['operation_id']==old_op
    else:
        original_put=h.service.put
        def lost_state_write(p):
            if next(r for r in p['batch']['targets'] if r['target_id']=='chair')['operation_id']!=old_op:
                raise OSError('fixture crash after durable recovery checkpoint')
            return original_put(p)
        monkeypatch.setattr(h.service,'put',lost_state_write)
        with pytest.raises(OSError,match='durable recovery'):h.service.control(h.pid,{'action':'resume_object','target_id':'chair'})
        assert row(h,'chair')['operation_id']==old_op
        monkeypatch.setattr(h.service,'put',original_put)
        h.service.control(h.pid,{'action':'resume_object','target_id':'chair'})
        after=row(h,'chair')
        assert after['state']=='pending' and after['operation_id']!=old_op
        assert after['attempt_history'][0]['previous']['operation_id']==old_op
        assert (h.actions.root/'packages'/old_op/'failure.json').exists()
        assert h.author_calls==calls
        # Repeated resume does not create another operation or model invocation.
        h.service.control(h.pid,{'action':'resume_object','target_id':'chair'})
        assert row(h,'chair')['operation_id']==after['operation_id']
        h.actions.backend_factory=original
        h.service.resume_batch(h.pid)
        wait_for(lambda:row(h,'chair')['state']=='package_ready')
        assert h.author_calls.count('chair')==1


def test_known_opening_preparation_failure_is_recoverable_before_any_model(tmp_path):
    actions=SimpleNamespace(root=tmp_path/'actions')
    batch={'acceptance_id':'accepted'};row={'operation_id':'door_old','target_id':'door'}
    job={'state':'failed',**row,'result':{'error_code':'EVIDENCE_MISMATCH','error':'Structure evidence must use original unmasked RGB and cameras'}}
    path=actions.root/'packages/door_old/started.json'
    save(path,{'acceptance_id':'accepted','allowed_object_ids':['door'],'before':{'snapshot':{'objects':[{'instance_id':'door','kind':'opening'}]}}})
    result=verified_unstarted_failure(None,actions,batch,row,job)
    assert 'before candidate preparation' in result['reason']
    job['result']['error']='Unknown preparation failure'
    with pytest.raises(ContractError):verified_unstarted_failure(None,actions,batch,row,job)
