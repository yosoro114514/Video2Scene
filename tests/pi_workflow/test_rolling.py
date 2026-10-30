"""Isolated model fixtures with real grants, packages, versions and save validation."""
import asyncio
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from test_service import h
from video2scene.contracts_v1 import ContractError
from video2scene.pi_workflow.resources import ResourceBudget
from video2scene.pi_workflow.rolling_jobs import POLICY


def wait_for(predicate,timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.01)
    raise AssertionError('Expected durable state did not arrive')


@pytest.fixture
def rolling_h(h,monkeypatch):
    # Add an independent third chair using explicit local evidence fixtures.
    item=copy.deepcopy(h.prior['objects'][0]);item['object_id']='chair_three';h.prior['objects'].append(item)
    item=copy.deepcopy(h.layout['objects'][0]);item['object_id']='chair_three';item['center_m'][0]=-1.5;h.layout['objects'].append(item)
    folder=h.sam/'chair_three/target_masks';folder.mkdir(parents=True)
    np.savez_compressed(folder/'00007.npz',mask=np.ones((40,60),dtype=bool),source_frame_index=np.int64(7))
    h.seed();monkeypatch.setattr(h.service,'_start',lambda *a,**k:None)
    # Save the common support first, so all three remaining chairs are independent.
    accepted=h.accept(('floor',));h.refine(accepted,'floor')
    h.service._on_pi_event(h.pid,{'type':'settled','run_id':'mock'})
    async def fixture_build(candidate,context):
        return {'input_fingerprint':candidate['candidate']['candidate_hash'],'artifact_refs':[],'asset_updates':[]}
    monkeypatch.setattr(h.actions.host.backend,'build',fixture_build)
    return h


def start(h,n=2):
    p=h.service.status(h.pid)
    p=h.service.control(h.pid,{'action':'start_refine_all','mode':'rolling','parallelism':n,'request_id':'rolling-start',**p['head']})
    args={'acceptance_id':p['batch']['acceptance_id'],'operation_id':'fixture_dispatch','target_ids':[r['target_id'] for r in p['batch']['targets']]}
    result,_=h.tool('start_refinement_jobs',args)
    return args,result


def row(h,target):return next(r for r in h.service.get(h.pid)['batch']['targets'] if r['target_id']==target)
def save_ready(h,target):
    package=row(h,target)['package']
    return h.tool('save_refinement',{k:package[k] for k in ('package_id','package_hash')})[0]


def test_fast_object_saves_and_refills_while_slow_peer_still_running(rolling_h):
    h=rolling_h;original=h.actions.backend_factory
    entered={name:threading.Event() for name in ('chair','chair_two','chair_three')}
    release={name:threading.Event() for name in entered}
    class Controlled(original):
        async def refine(self,spec,*args):
            target=spec['target_id'];entered[target].set()
            assert await asyncio.to_thread(release[target].wait,10),'fixture gate timeout'
            return await super().refine(spec,*args)
    h.actions.backend_factory=Controlled
    try:
        before=asyncio.run(h.actions.inspect_scene());args,_=start(h,2)
        assert entered['chair'].wait(5) and entered['chair_two'].wait(5)
        assert not entered['chair_three'].is_set()
        release['chair'].set()
        wait_for(lambda:row(h,'chair')['state']=='package_ready')
        assert entered['chair_three'].wait(5),'No rolling refill while slow peer runs'
        saved=save_ready(h,'chair')
        assert saved['head']['head_seq']==before['head']['head_seq']+1
        assert row(h,'chair_two')['state']=='generating' and not release['chair_two'].is_set()
        # A lost response / duplicate authorization returns existing jobs, never authors again.
        h.tool('start_refinement_jobs',args)
        release['chair_two'].set();release['chair_three'].set()
        for name in ('chair_two','chair_three'):
            wait_for(lambda:row(h,name)['state']=='package_ready');save_ready(h,name)
        wait_for(lambda:not h.service.rolling.active(h.pid))
        after=asyncio.run(h.actions.inspect_scene())
        assert after['head']['head_seq']==before['head']['head_seq']+3
        assert h.author_calls.count('chair')==h.author_calls.count('chair_two')==h.author_calls.count('chair_three')==1
        for obj in after['snapshot']['objects']:
            old=next(o for o in before['snapshot']['objects'] if o['instance_id']==obj['instance_id'])
            assert all(obj[k]==old[k] for k in ('transform','support_id','local_bounds_m'))
            if obj['instance_id']=='floor':assert obj==old
        h.tool('export_scene');result,_=h.tool('finish_workflow',{'summary':'Isolated fixture rolling loop; no live model'})
        assert result['status']=='complete' and h.service.get(h.pid)['batch']['status']=='complete'
    finally:
        for gate in release.values():gate.set()


@pytest.mark.parametrize('n',[1,2,3,4])
def test_configured_object_limit_and_lowering_drain_existing_tasks(rolling_h,n):
    h=rolling_h;original=h.actions.backend_factory;gate=threading.Event();lock=threading.Lock();started=[]
    class Controlled(original):
        async def refine(self,spec,*args):
            with lock:started.append(spec['target_id'])
            assert await asyncio.to_thread(gate.wait,10)
            return await super().refine(spec,*args)
    h.actions.backend_factory=Controlled
    try:
        start(h,n);wait_for(lambda:len(started)==min(n,3))
        h.service.control(h.pid,{'action':'set_refinement_concurrency','parallelism':1})
        assert len(started)==min(n,3)
        assert h.service.get(h.pid)['batch']['parallelism']==1
        gate.set();wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
        assert len(started)==3
    finally:gate.set()


def test_failure_isolated_pause_drains_and_export_waits(rolling_h):
    h=rolling_h;original=h.actions.backend_factory;gate=threading.Event();entered=threading.Event()
    class Controlled(original):
        async def refine(self,spec,*args):
            if spec['target_id']=='chair':raise RuntimeError('fixture fetch failed')
            if spec['target_id']=='chair_two':entered.set();assert await asyncio.to_thread(gate.wait,10)
            return await super().refine(spec,*args)
    h.actions.backend_factory=Controlled
    try:
        start(h,2);assert entered.wait(5)
        h.service.control(h.pid,{'action':'pause_object','target_id':'chair_two'})
        gate.set();wait_for(lambda:row(h,'chair_two')['state']=='paused')
        wait_for(lambda:row(h,'chair_three')['state']=='package_ready')
        save_ready(h,'chair_three')
        assert row(h,'chair')['state']=='needs_review'
        with pytest.raises(ContractError,match='every target'):h.tool('export_scene')
        h.service.control(h.pid,{'action':'resume_object','target_id':'chair_two'})
        save_ready(h,'chair_two')
        assert h.author_calls.count('chair_two')==1
        with pytest.raises(ContractError,match='不能重放'):h.service.control(h.pid,{'action':'resume_object','target_id':'chair'})
    finally:gate.set()


def test_bounded_backpressure_and_dependency_wake_after_save(rolling_h):
    h=rolling_h
    p=h.service.status(h.pid)
    p=h.service.control(h.pid,{'action':'start_refine_all','mode':'rolling','parallelism':1,'request_id':'rolling-start',**p['head']})
    p['batch']['ready_package_limit']=1
    p['batch']['targets'][2]['depends_on']=['chair_two'];h.service.put(p)
    h.tool('start_refinement_jobs',{'acceptance_id':p['batch']['acceptance_id'],'operation_id':'fixture_dispatch','target_ids':[r['target_id'] for r in p['batch']['targets']]})
    wait_for(lambda:row(h,'chair')['state']=='package_ready')
    assert row(h,'chair_two')['state']=='pending' and row(h,'chair_three')['state']=='pending'
    save_ready(h,'chair');wait_for(lambda:row(h,'chair_two')['state']=='package_ready')
    assert row(h,'chair_three')['state']=='pending'
    save_ready(h,'chair_two');wait_for(lambda:row(h,'chair_three')['state']=='package_ready')


def test_resource_groups_shared_model_limit_and_no_leak_on_error():
    budget=ResourceBudget(model=3,blender=1);entered=threading.Barrier(4);release=threading.Event()
    def critic():
        with budget.hold('model'):
            entered.wait(timeout=4);release.wait(timeout=4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(critic) for _ in range(3)];entered.wait(timeout=4)
        assert budget.snapshot()['used']['model']==3;release.set()
        for f in futures:f.result()
    with pytest.raises(RuntimeError):
        with budget.hold('model','blender'):raise RuntimeError('fixture cancelled')
    assert budget.snapshot()['used']=={'model':0,'blender':0}


def test_network_pressure_reduces_dispatch_without_replaying_jobs(rolling_h):
    h=rolling_h;p=h.service.status(h.pid)
    h.service.control(h.pid,{'action':'start_refine_all','mode':'rolling','parallelism':3,'request_id':'rolling-start',**p['head']})
    for _ in range(POLICY['failure_threshold']):
        h.service.rolling.reconnect_event(h.pid,{'type':'auto_retry_start','data':{'errorMessage':'fetch failed'}})
    b=h.service.get(h.pid)['batch'];assert b['effective_concurrency']==2 and b['cooldown_until']>time.time()
    assert not h.service.rolling.jobs(h.pid)


def test_atomic_save_replays_exact_commit_after_response_loss(rolling_h,monkeypatch):
    h=rolling_h;start(h,3)
    wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
    wait_for(lambda:not h.service.rolling.active(h.pid))
    saved=save_ready(h,'chair');package=row(h,'chair')['package'];calls=list(h.author_calls)
    replay=asyncio.run(h.actions.save_refinement(package['package_id'],package['package_hash']))
    assert replay['head']==saved['head'] and h.author_calls==calls
    assert asyncio.run(h.actions.inspect_scene())['head']==saved['head']
    transaction=json.loads((h.actions.root/'integrations'/package['package_id']/'transaction.json').read_text())
    # Simulate a crash after repo.commit but before the completion receipt is written.
    completion=h.actions.root/'operations'/transaction['operation_id']/'completion.json'
    completion.unlink()
    p=h.service.get(h.pid);p.update(phase='paused',active_run=None);p['batch']['status']='paused';h.service.put(p)
    resumed=h.service.resume_batch(h.pid)
    assert completion.exists() and resumed['head']==saved['head']
    assert row(h,'chair')['state']=='saved' and h.author_calls==calls


def test_restart_reuses_durable_packages_without_model_replay(rolling_h,monkeypatch):
    from video2scene.pi_workflow.service import WorkflowService
    h=rolling_h;start(h,3)
    wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
    wait_for(lambda:not h.service.rolling.active(h.pid))
    calls=list(h.author_calls);packages=[r['package'] for r in h.service.get(h.pid)['batch']['targets']]
    root=h.service.root;factory=h.actions.backend_factory
    h.service.close()
    h.service=WorkflowService(root);h.actions=h.service.action(h.pid);h.actions.backend_factory=factory
    monkeypatch.setattr(h.service,'_start',lambda *a,**k:None)
    assert h.service.get(h.pid)['phase']=='paused'
    resumed=h.service.resume_batch(h.pid)
    assert [r['package'] for r in resumed['batch']['targets']]==packages
    assert h.author_calls==calls and not h.service.rolling.active(h.pid)


@pytest.mark.parametrize('started',[False,True])
def test_registered_or_uncertain_task_reconciles_without_blind_replay(rolling_h,started):
    from video2scene.pi_workflow.pi_runtime import save
    h=rolling_h;p=h.service.status(h.pid)
    h.service.control(h.pid,{'action':'start_refine_all','mode':'rolling','parallelism':2,'request_id':'rolling-start',**p['head']})
    p=h.service.get(h.pid);p.update(phase='paused',active_run=None);p['batch']['status']='paused'
    target=p['batch']['targets'][0];target.update(state='generating' if started else 'registered',operation_id='interrupted_fixture',job_id='interrupted_fixture')
    if started:save(h.actions.root/'packages/interrupted_fixture/started.json',{'fixture':'unknown invocation outcome'})
    h.service.put(p);calls=list(h.author_calls);h.service.resume_batch(h.pid)
    assert row(h,'chair')['state']==('needs_review' if started else 'pending')
    assert h.author_calls==calls


def test_two_coordinators_cannot_adopt_the_same_root(rolling_h):
    from video2scene.pi_workflow.service import WorkflowService
    h=rolling_h;before=h.service.get(h.pid)
    with pytest.raises(ContractError,match='Another workflow service'):WorkflowService(h.service.root)
    assert h.service.get(h.pid)==before


def test_atomic_save_failure_preserves_other_packages_and_releases_lane(rolling_h,monkeypatch):
    h=rolling_h;start(h,3)
    wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
    original=h.actions.validate_candidate
    async def reject_once(**args):return {'committable':False,'validation_id':'fixture-rejected'}
    monkeypatch.setattr(h.actions,'validate_candidate',reject_once)
    with pytest.raises(ContractError,match='failed scene validation'):save_ready(h,'chair')
    assert row(h,'chair')['state']=='needs_review'
    assert not h.service.get(h.pid)['batch'].get('active_save_target')
    monkeypatch.setattr(h.actions,'validate_candidate',original)
    save_ready(h,'chair_two')
    assert row(h,'chair_three')['state']=='package_ready'


def test_whole_batch_pause_stops_refill_and_keeps_finished_package(rolling_h):
    h=rolling_h;original=h.actions.backend_factory;gate=threading.Event();entered=threading.Event()
    class Controlled(original):
        async def refine(self,spec,*args):
            entered.set();assert await asyncio.to_thread(gate.wait,10)
            return await super().refine(spec,*args)
    h.actions.backend_factory=Controlled
    try:
        start(h,1);assert entered.wait(5)
        h.service.control(h.pid,{'action':'pause'})
        gate.set();wait_for(lambda:not h.service.rolling.active(h.pid))
        p=h.service.get(h.pid)
        assert p['batch']['status']=='paused' and p['phase']=='paused'
        assert row(h,'chair_two')['state']==row(h,'chair_three')['state']=='pending'
        assert row(h,'chair')['state']=='needs_review'  # cancellation is not a complete package
    finally:gate.set()


def test_rejected_integration_rebases_on_peer_save_without_reauthoring(rolling_h,monkeypatch):
    h=rolling_h;start(h,3)
    wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
    original=h.actions.validate_candidate
    async def reject(**kw):return {'committable':False,'validation_id':'fixture-rejection'}
    monkeypatch.setattr(h.actions,'validate_candidate',reject)
    with pytest.raises(ContractError):save_ready(h,'chair')
    monkeypatch.setattr(h.actions,'validate_candidate',original)
    peer=save_ready(h,'chair_two');calls=list(h.author_calls)
    p=h.service.get(h.pid);p.update(phase='paused',active_run=None);p['batch']['status']='paused';h.service.put(p)
    h.service.resume_batch(h.pid)
    saved=save_ready(h,'chair')
    assert saved['head']['head_seq']==peer['head']['head_seq']+1 and h.author_calls==calls


@pytest.mark.parametrize('mode',['waves','rolling'])
def test_controlled_scheduler_benchmark(rolling_h,monkeypatch,mode):
    """Real production dispatch and saves, timed fake authors; not model throughput."""
    from pathlib import Path
    from video2scene.contracts_v1 import canonical_hash
    h=rolling_h;original=h.actions.backend_factory;timeline=[];mutex=threading.Lock()
    durations={'chair':.15,'chair_two':.8,'chair_three':.15};started=time.perf_counter()
    def record(target,stage):
        with mutex:timeline.append({'target':target,'stage':stage,'seconds':round(time.perf_counter()-started,4)})
    class Timed(original):
        async def refine(self,spec,*args):
            target=spec['target_id'];record(target,'author_start')
            await asyncio.sleep(durations[target]);value=await super().refine(spec,*args);record(target,'package_ready');return value
    h.actions.backend_factory=Timed
    build=h.actions.host.backend.build
    async def timed_build(*args):await asyncio.sleep(.1);return await build(*args)
    monkeypatch.setattr(h.actions.host.backend,'build',timed_build)
    before=asyncio.run(h.actions.inspect_scene());count=h.count;started=time.perf_counter()
    if mode=='rolling':
        start(h,2)
        while h.service.get(h.pid)['batch']['completed_count']<3:
            ready=wait_for(lambda:next((r for r in h.service.get(h.pid)['batch']['targets'] if r['state']=='package_ready'),None))
            record(ready['target_id'],'save_start');save_ready(h,ready['target_id']);record(ready['target_id'],'saved')
    else:
        p=h.service.status(h.pid);h.service.control(h.pid,{'action':'start_refine_all','mode':'waves','parallelism':2,'request_id':'wave-benchmark',**p['head']})
        for targets in (['chair','chair_two'],['chair_three']):
            wait_for(lambda:h.service.get(h.pid)['batch'].get('run_targets')==targets and h.service.get(h.pid).get('active_run'))
            b=h.service.get(h.pid)['batch']
            packages,_=h.tool('refine_objects_parallel',{'acceptance_id':b['acceptance_id'],'objects':[{'target_id':t,'operation_id':'bench_'+t} for t in targets]})
            for package in packages['objects']:
                target=package['target_id'];record(target,'save_start')
                candidate,_=h.tool('stage_refinement',{k:package[k] for k in ('package_id','package_hash')})
                ref={k:candidate['candidate'][k] for k in ('candidate_id','candidate_hash')}
                validation,_=h.tool('validate_candidate',ref);h.tool('commit_candidate',{**ref,'validation_id':validation['validation_id']});record(target,'saved')
            if targets!=['chair_three']:h.service._on_pi_event(h.pid,{'type':'settled','run_id':'fixture'})
    result={'mode':mode,'scope':'Isolated fixture model timings with real dispatcher, package guards and repository saves; no actual Codex calls',
        'object_concurrency':2,'author_seconds':durations,'save_build_seconds':.1,'seconds':round(time.perf_counter()-started,4),
        'tool_calls':h.count-count,'saved':3,'timeline':timeline,
        'geometry_fingerprint':canonical_hash([{k:o[k] for k in ('instance_id','transform','support_id','local_bounds_m')} for o in before['snapshot']['objects']])}
    folder=Path(__file__).resolve().parents[2]/'outputs/pi_workflow_v1/development/rolling';folder.mkdir(exist_ok=True,parents=True)
    (folder/f'benchmark_{mode}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


def test_save_checkpoint_cannot_be_substituted_with_another_objects_candidate(rolling_h,monkeypatch):
    from video2scene.pi_workflow.pi_runtime import save
    h=rolling_h;start(h,3)
    wait_for(lambda:all(r['state']=='package_ready' for r in h.service.get(h.pid)['batch']['targets']))
    save_ready(h,'chair');saved=save_ready(h,'chair_two')
    a=row(h,'chair')['package'];b=row(h,'chair_two')['package']
    path=h.actions.root/'integrations'/a['package_id']/'transaction.json'
    bad=json.loads((h.actions.root/'integrations'/b['package_id']/'transaction.json').read_text())
    bad.update(package_id=a['package_id'],package_hash=a['package_hash']);save(path,bad)
    with pytest.raises(ContractError,match='another package'):
        asyncio.run(h.actions.save_refinement(a['package_id'],a['package_hash']))
    assert asyncio.run(h.actions.inspect_scene())['head']==saved['head']
