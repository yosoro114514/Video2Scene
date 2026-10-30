"""Real repository/validation/merge guards, explicit fixture model and exporter."""
import asyncio
import copy
import json

import pytest

from test_service import h
from video2scene.contracts_v1 import ContractError, canonical_hash


@pytest.fixture
def package_h(h, monkeypatch):
    h.seed()
    monkeypatch.setattr(h.service, '_start', lambda *args, **kwargs: None)
    async def fixture_build(candidate, context):
        return {'input_fingerprint': candidate['candidate']['candidate_hash'], 'artifact_refs': [], 'asset_updates': []}
    monkeypatch.setattr(h.actions.host.backend, 'build', fixture_build)
    return h


def generate(h, acceptance):
    return asyncio.run(h.actions.refine_objects_parallel(acceptance['acceptance_id'], [
        {'target_id': 'chair', 'operation_id': 'parallel_chair'},
        {'target_id': 'chair_two', 'operation_id': 'parallel_chair_two'}]))


def save_package(h, package):
    candidate = asyncio.run(h.actions.stage_refinement(package['package_id'], package['package_hash']))
    ref = {k: candidate['candidate'][k] for k in ('candidate_id', 'candidate_hash')}
    report = asyncio.run(h.actions.validate_candidate(**ref))
    assert report['committable'], report
    return asyncio.run(h.actions.commit_candidate(**ref, validation_id=report['validation_id']))


def test_two_workers_overlap_then_save_on_successive_heads_without_overwrite(package_h, monkeypatch):
    h = package_h
    acceptance = h.accept()
    before = asyncio.run(h.actions.inspect_scene())
    original = h.actions.backend_factory
    entered = []
    class OverlapAuthor(original):
        async def refine(self, spec, *args):
            entered.append(spec['target_id'])
            for _ in range(100):
                if len(entered) == 2: break
                await asyncio.sleep(.01)
            assert len(entered) == 2, 'Authors did not run concurrently'
            return await super().refine(spec, *args)
    h.actions.backend_factory = OverlapAuthor
    result = generate(h, acceptance)
    assert all(r['state'] == 'package_ready' for r in result['objects']), result
    assert asyncio.run(h.actions.inspect_scene()) == before
    first, second = result['objects']
    a = save_package(h, first)
    b = save_package(h, second)
    assert (a['head']['head_seq'], b['head']['head_seq']) == (1, 2)
    assert len(h.author_calls) == 2
    original_records = {o['instance_id']: o for o in before['snapshot']['objects']}
    for obj in b['snapshot']['objects']:
        old = original_records[obj['instance_id']]
        if obj['instance_id'] == 'floor': assert obj == old
        else:
            assert obj['asset'] != old['asset']
            assert all(obj[k] == old[k] for k in ('transform', 'local_bounds_m', 'support_id'))
    a_object = next(o for o in a['snapshot']['objects'] if o['instance_id'] == first['target_id'])
    assert a_object == next(o for o in b['snapshot']['objects'] if o['instance_id'] == first['target_id'])
    export = asyncio.run(h.actions.export_scene())
    delivery = asyncio.run(h.actions.verify_delivery(acceptance['acceptance_id'], export))
    assert delivery['status'] == 'complete' and delivery['refined_objects'] == 2


def test_partial_failure_keeps_other_package_and_reuses_without_author_replay(package_h):
    h = package_h; acceptance = h.accept(); original = h.actions.backend_factory
    class OneFails(original):
        async def refine(self, spec, *args):
            if spec['target_id'] == 'chair': raise RuntimeError('fixture stream drop')
            return await super().refine(spec, *args)
    h.actions.backend_factory = OneFails
    result = generate(h, acceptance)
    assert [r['state'] for r in result['objects']] == ['failed', 'package_ready']
    replay = generate(h, acceptance)
    assert len(h.author_calls) == 1
    good = next(r for r in replay['objects'] if r['state'] == 'package_ready')
    saved = save_package(h, good)
    assert saved['head']['head_seq'] == 1


def test_worker_initialization_failure_is_isolated(package_h):
    h = package_h; accepted = h.accept(); original = h.actions.backend_factory
    def factory(*args, **kwargs):
        confirmation = json.loads(kwargs['confirmation_path'].read_text(encoding='utf-8'))
        if confirmation['target_id'] == 'chair': raise RuntimeError('fixture initialization failure')
        return original(*args, **kwargs)
    h.actions.backend_factory = factory
    result = generate(h, accepted)
    assert [r['state'] for r in result['objects']] == ['failed', 'package_ready']
    assert h.actions._worker_cancels == {}


def test_package_hash_and_current_layout_changes_are_rejected(package_h):
    h = package_h; acceptance = h.accept(); package = generate(h, acceptance)['objects'][0]
    with pytest.raises(ContractError, match='package changed'):
        asyncio.run(h.actions.stage_refinement(package['package_id'], '0' * 64))
    record = asyncio.run(h.actions._read_package(package['package_id'], package['package_hash']))
    changed = copy.deepcopy(record['before'])
    target = next(o for o in changed['snapshot']['objects'] if o['instance_id'] == 'chair')
    target['transform']['translation_m'][0] += .2
    with pytest.raises(ContractError, match='Target or support changed'):
        h.actions._compatible_package(record, changed)


def test_parallel_batch_tool_flow_restart_and_export(package_h):
    h = package_h
    # Existing fine floor is preserved. Both chairs are now independent pending tasks.
    floor_acceptance = h.accept(('floor',)); h.refine(floor_acceptance, 'floor')
    h.service._on_pi_event(h.pid, {'type': 'settled', 'run_id': h.service.get(h.pid)['active_run']})
    p = h.service.status(h.pid)
    p = h.service.control(h.pid, {'action': 'start_refine_all', 'mode':'waves','parallelism':2, 'request_id': 'parallel_start', **p['head']})
    assert p['batch']['parallelism'] == 2
    assert set(p['batch']['run_targets']) == {'chair', 'chair_two'}
    result, _ = h.tool('refine_objects_parallel', {'acceptance_id': p['acceptance']['acceptance_id'], 'objects': [
        {'target_id': 'chair', 'operation_id': 'worker_chair'}, {'target_id': 'chair_two', 'operation_id': 'worker_chair_two'}]})
    assert all(r['state'] == 'package_ready' for r in result['objects'])
    # Simulate a stopped coordinator after packages have been persisted.
    p = h.service.get(h.pid); p.update(active_run=None, phase='paused');p['batch']['status']='paused';h.service.put(p)
    resumed = h.service.resume_batch(h.pid)
    assert all(r['state'] == 'package_ready' for r in resumed['batch']['targets'])
    calls = len(h.author_calls)
    for package in result['objects']:
        candidate, _ = h.tool('stage_refinement', {k: package[k] for k in ('package_id', 'package_hash')})
        ref = {k: candidate['candidate'][k] for k in ('candidate_id', 'candidate_hash')}
        report, _ = h.tool('validate_candidate', ref)
        saved, _ = h.tool('commit_candidate', dict(ref, validation_id=report['validation_id']))
        assert 'snapshot' not in saved  # compact model response; full receipt stays in the repository
    assert len(h.author_calls) == calls
    h.tool('export_scene'); delivery, _ = h.tool('finish_workflow', {'summary': 'explicit concurrent fixture'})
    assert delivery['refined_objects'] == 2
    assert h.service.get(h.pid)['batch']['status'] == 'complete'


def test_pause_one_ready_package_preserves_it_and_does_not_pause_peer(package_h):
    h = package_h; accepted = h.accept()
    result = generate(h, accepted)
    p = h.service.get(h.pid)
    p.update(active_run=None, phase='idle', batch={'parallelism': 2, 'status': 'running',
        'acceptance_id': accepted['acceptance_id'], 'completed_count': 0, 'total': 2,
        'run_targets': ['chair', 'chair_two'], 'targets': [
            {'target_id': r['target_id'], 'state': 'package_ready', 'attempts': 1,
             'operation_id': r['operation_id'], 'package': r} for r in result['objects']]})
    h.service.put(p)
    p = h.service.control(h.pid, {'action': 'pause_object', 'target_id': 'chair'})
    assert [r['state'] for r in p['batch']['targets']] == ['paused', 'package_ready']
    package = result['objects'][0]
    with pytest.raises(ContractError, match='one reviewed assigned'):
        h.tool('stage_refinement', {k: package[k] for k in ('package_id', 'package_hash')})
    calls = len(h.author_calls)
    p = h.service.control(h.pid, {'action': 'resume_object', 'target_id': 'chair'})
    assert all(r['state'] == 'package_ready' for r in p['batch']['targets'])
    assert len(h.author_calls) == calls


def test_second_integration_waits_for_first_commit(package_h):
    h = package_h; accepted = h.accept(); result = generate(h, accepted)
    p = h.service.get(h.pid)
    p['batch'] = {'parallelism': 2, 'status': 'running', 'completed_count': 0, 'total': 2,
        'acceptance_id': accepted['acceptance_id'], 'run_targets': ['chair', 'chair_two'],
        'targets': [{'target_id': r['target_id'], 'state': 'package_ready', 'attempts': 1,
                     'package': r, 'operation_id': r['operation_id']} for r in result['objects']]}
    h.service.put(p)
    a, b = result['objects']
    h.tool('stage_refinement', {k: a[k] for k in ('package_id', 'package_hash')})
    with pytest.raises(ContractError, match='at a time'):
        h.tool('stage_refinement', {k: b[k] for k in ('package_id', 'package_hash')})
    p = h.service.get(h.pid)
    assert p['batch']['targets'][0]['state'] == 'candidate_ready'
    assert p['batch']['active_save_target'] == 'chair'
    assert asyncio.run(h.actions.inspect_scene())['head']['head_seq'] == 0
