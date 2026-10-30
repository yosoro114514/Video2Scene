"""Temporary fixtures: real grants/version gates, mocked Pi and author, no real user/model."""
import asyncio
import json
import time

import pytest

from test_service import h
from video2scene.agent_scene.planning import save
from video2scene.contracts_v1 import ContractError


def start(h, monkeypatch, key='batch-start'):
    monkeypatch.setattr(h.service, '_start', lambda *args, **kwargs: None)
    p = h.service.status(h.pid)
    request = {'action': 'start_refine_all', 'request_id': key, 'parallelism': 1, **p['head']}
    result = h.service.control(h.pid, request)
    return request, result


def settle(h):
    p = h.service.get(h.pid); before = p['active_run']
    h.service._on_pi_event(h.pid, {'type': 'settled', 'run_id': before})
    if p['batch']['completed_count'] > p['batch'].get('run_start_completed', 0) and p['batch']['status'] == 'running':
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if h.service.get(h.pid)['active_run']: return
            time.sleep(.01)
        raise AssertionError('Batch did not advance the Pi session')


def test_all_remaining_includes_structures_skips_refined_and_is_idempotent(h, monkeypatch):
    h.seed(); accepted = h.accept(('chair',))
    h.refine(accepted, 'chair')
    request, p = start(h, monkeypatch)
    assert p['batch']['skipped_ids'] == ['chair']
    assert [r['target_id'] for r in p['batch']['targets']] == ['chair_two', 'floor']
    assert h.service.control(h.pid, request)['batch']['batch_id'] == p['batch']['batch_id']
    assert h.author_calls == ['chair']
    token = json.loads((h.actions.root / 'acceptances' / (p['acceptance']['acceptance_id'] + '.json')).read_text(encoding='utf-8'))
    assert token['human_event']['payload']['include_structures'] is True
    assert token['head'] == p['head']


def test_batch_runs_each_item_then_exports_and_finishes_with_one_grant(h, monkeypatch):
    h.seed(); _, p = start(h, monkeypatch)
    accepted = p['acceptance']
    h.refine(accepted, 'chair'); settle(h)
    assert h.service.get(h.pid)['batch']['completed_count'] == 1
    h.refine(accepted, 'chair_two'); settle(h)
    h.refine(accepted, 'floor'); settle(h)
    assert h.author_calls == ['chair', 'chair_two', 'floor']
    h.tool('export_scene'); result, _ = h.tool('finish_workflow', {'summary': 'explicit fixture whole-room batch'})
    assert result['refined_objects'] == 3
    batch = h.service.get(h.pid)['batch']
    assert batch['status'] == 'complete' and batch['completed_count'] == 3
    assert all(row['state'] == 'saved' for row in batch['targets'])
    assert all(o['transform'] == next(x for x in json.loads((h.run/'snapshot.json').read_text())['objects'] if x['instance_id']==o['instance_id'])['transform'] for o in asyncio.run(h.actions.inspect_scene())['snapshot']['objects'])


def test_batch_pause_resume_preserves_candidate_without_reauthoring(h, monkeypatch):
    h.seed(); _, p = start(h, monkeypatch)
    built, _ = h.tool('refine_object', {'target_id': 'chair', 'acceptance_id': p['acceptance']['acceptance_id'], 'operation_id': 'resume_candidate'})
    h.service.control(h.pid, {'action': 'pause'})
    h.service._on_pi_event(h.pid, {'type': 'aborted'})
    resumed = h.service.control(h.pid, {'action': 'resume'})
    assert resumed['batch']['targets'][0]['candidate'] == built['candidate']
    assert h.author_calls == ['chair']
    with pytest.raises(ContractError, match='existing candidate'):
        h.tool('refine_object', {'target_id': 'chair', 'acceptance_id': p['acceptance']['acceptance_id'], 'operation_id': 'duplicate_author'})
    assert h.author_calls == ['chair']


def test_batch_refuses_stale_start_and_uncertain_author_replay(h, monkeypatch):
    h.seed(); p = h.service.status(h.pid)
    with pytest.raises(ContractError, match='当前保存'):
        h.service.control(h.pid, {'action': 'start_refine_all', 'request_id': 'stale', 'revision': 'stale', 'head_seq': 0})
    with h.service.db() as db: assert db.execute('SELECT count(*) FROM human_events').fetchone()[0] == 0
    _, p = start(h, monkeypatch)
    p['batch']['targets'][0].update(operation_id='uncertain_fixture', state='generating')
    p.update(active_run=None, phase='paused'); p['batch']['status']='paused'; h.service.put(p)
    save(h.actions.root/'operations/uncertain_fixture/started.json', {'explicit_fixture': True})
    with pytest.raises(ContractError, match='不会自动重放'):
        h.service.control(h.pid, {'action': 'resume'})
    assert h.author_calls == []


def test_structural_evidence_rejects_neighbor_mask_and_changed_camera(h):
    target={'instance_id':'wall', 'kind':'wall'}
    frame={'source_frame_id':'original', 'rgb':{'sha256':'fixture'}, 'mask':None, 'c2w':[[1]], 'K':[[2]]}
    capture={'frames':[frame]}
    h.actions._refinement_evidence({'frames':[frame]},target,capture,{})
    for changed in ({**frame,'mask':{'sha256':'other-object'}}, {**frame,'c2w':[[3]]}):
        with pytest.raises(ContractError, match='unmasked'):
            h.actions._refinement_evidence({'frames':[changed]},target,capture,{})


def test_segmented_opening_uses_only_its_accepted_original_mask_and_camera(h):
    target={'instance_id':'door','kind':'opening'}
    frame={'source_frame_id':'door_12','rgb':{'sha256':'rgb'},'mask':{'sha256':'door-mask'},'c2w':[[1]],'K':[[2]]}
    capture={'frames':[frame],'metadata':{'frame_targets':{'door_12':'door'},'original_source_indices':{'door_12':12}}}
    proposals={'objects':[{'object_id':'door','accepted_frame_indices':[12]}]}
    h.actions._refinement_evidence({'frames':[frame]},target,capture,proposals)
    for key,value in [('mask',{'sha256':'neighbor'}),('c2w',[[3]]),('rgb',{'sha256':'replacement'})]:
        with pytest.raises(ContractError,match='original target'):
            h.actions._refinement_evidence({'frames':[{**frame,key:value}]},target,capture,proposals)
    capture['metadata']['frame_targets']['door_12']='cabinet'
    with pytest.raises(ContractError,match='accepted spatial observation'):
        h.actions._refinement_evidence({'frames':[frame]},target,capture,proposals)
