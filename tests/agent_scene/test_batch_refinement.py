"""Batch authorization/layout/resume guards with explicit local fixtures."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from video2scene.agent_scene import batch_refinement as batch
from video2scene.contracts_v1 import ContractError, ExecutionContext, camera_hash, canonical_hash, file_hash
from video2scene.contracts_v1.mocks import example


@pytest.fixture
def scene():
    result = copy.deepcopy(example('model_SceneSnapshot'))
    template = result['objects'][1]
    objects = []
    for identity, kind in [('wall', 'wall'), ('chair', 'object'), ('rug', 'support'), ('coffee_table', 'object')]:
        obj = copy.deepcopy(template)
        obj.update(instance_id=identity, kind=kind, support_id=None)
        obj['asset']['asset_version'] = ('gpt-material-' if identity == 'coffee_table' else 'whitebox-') + identity
        objects.append(obj)
    result['objects'] = objects
    return result


def authorization(scene):
    return {'authorized': True, 'scene_id': scene['scene_id'], 'target_ids': ['chair', 'rug'],
            'authorization_source': 'current_user_chat', 'authorized_at': '2026-10-29T00:00:00Z',
            'user_instruction': 'Explicit test fixture authorization', 'batch_id': 'fixture_batch',
            'baseline_head': {'revision': scene['revision'], 'head_seq': 2}, 'baseline_snapshot_hash': canonical_hash(scene)}


def refined(scene):
    after = copy.deepcopy(scene)
    target = next(obj for obj in after['objects'] if obj['instance_id'] == 'chair')
    target['asset']['asset_version'] = 'gpt-direct-fixture-new'
    target['asset']['artifact']['sha256'] = 'f' * 64
    target['field_state']['asset.geometry'].update(source='model', decision_ref='fixture_decision')
    target['field_state']['asset.materials'].update(source='model', decision_ref='fixture_decision')
    after.update(revision='fixture_after', parent_revision=scene['revision'])
    return after


def test_authorized_batch_includes_support_furniture_and_rug(scene):
    assert batch.authorized_targets(authorization(scene), scene) == ['chair', 'rug']


@pytest.mark.parametrize('requested', [['wall'], ['coffee_table'], ['chair', 'chair']])
def test_room_refined_or_repeated_targets_are_not_started(scene, requested):
    auth = authorization(scene)
    auth['target_ids'] += ['wall', 'coffee_table']
    with pytest.raises(ContractError):
        batch.authorized_targets(auth, scene, requested)


def test_locked_layout_allows_only_planned_asset_and_provenance_changes(scene):
    after = refined(scene)
    assert batch.locked_layout(scene, ['chair', 'rug']) == batch.locked_layout(after, ['chair', 'rug'])
    assert batch.audit_transition(scene, after, 'chair') == {'locked_fields_unchanged': True, 'non_targets_unchanged': True}
    next(obj for obj in after['objects'] if obj['instance_id'] == 'coffee_table')['asset']['asset_version'] = 'tampered'
    assert batch.locked_layout(scene, ['chair', 'rug']) != batch.locked_layout(after, ['chair', 'rug'])
    with pytest.raises(ContractError):
        batch.audit_transition(scene, after, 'chair')


@pytest.mark.parametrize('field,value', [('category', 'other'), ('kind', 'wall'), ('support_id', 'wall'),
                                      ('transform', {'translation_m': [9, 9, 9]}),
                                      ('local_bounds_m', {'min': [-9, -9, -9], 'max': [9, 9, 9]})])
def test_single_target_cannot_change_locked_fields(scene, field, value):
    after = refined(scene)
    next(obj for obj in after['objects'] if obj['instance_id'] == 'chair')[field] = value
    with pytest.raises(ContractError) as error:
        batch.audit_transition(scene, after, 'chair')
    assert error.value.code == 'FORBIDDEN_SCOPE'


def test_one_real_accepted_frame_is_sufficient_but_other_instances_are_rejected():
    evidence = {'frames': [{'source_frame_id': 'frame1', 'mask': {'explicit_fixture': True}}]}
    capture = {'metadata': {'frame_targets': {'frame1': 'chair'}, 'original_source_indices': {'frame1': 1932}}}
    proposals = {'objects': [{'object_id': 'chair', 'accepted_frame_indices': [1932]}]}
    batch.check_accepted_frames(evidence, 'chair', capture, proposals)
    capture['metadata']['frame_targets']['frame1'] = 'other_chair'
    with pytest.raises(ContractError):
        batch.check_accepted_frames(evidence, 'chair', capture, proposals)


def test_evidence_outside_accepted_spatial_cluster_is_rejected():
    evidence = {'frames': [{'source_frame_id': 'frame1', 'mask': {'explicit_fixture': True}}]}
    capture = {'metadata': {'frame_targets': {'frame1': 'chair'}, 'original_source_indices': {'frame1': 99}}}
    proposals = {'objects': [{'object_id': 'chair', 'accepted_frame_indices': [1932]}]}
    with pytest.raises(ContractError):
        batch.check_accepted_frames(evidence, 'chair', capture, proposals)


def setup_resume(tmp_path, monkeypatch, scene, *, committed):
    root, batch_root = tmp_path / 'run', tmp_path / 'run/fixture_batch'
    batch_root.mkdir(parents=True)
    auth = authorization(scene)
    batch.immutable_json(batch_root / 'authorization.json', auth)
    auth_hash = file_hash(batch_root / 'authorization.json')
    evidence = copy.deepcopy(example('model_EvidenceBundle'))
    frame = evidence['frames'][0]
    frame['mask'] = copy.deepcopy(frame['rgb'])
    capture = {'metadata': {'frame_targets': {frame['source_frame_id']: 'chair'},
                            'original_source_indices': {frame['source_frame_id']: 1932}}}
    proposals = {'objects': [{'object_id': 'chair', 'accepted_frame_indices': [1932]}]}
    records = [('initial_snapshot', root / 'snapshot.json', scene), ('capture_record', root / 'capture.json', capture),
               ('geometry_proposals', root / 'geometry_proposals.json', proposals)]
    baseline = {'scene_id': scene['scene_id'], 'batch_id': batch_root.name, 'head': auth['baseline_head'],
                'snapshot_hash': canonical_hash(scene), 'layout_hash': canonical_hash(batch.locked_layout(scene, ['chair'])),
                'authorization_sha256': auth_hash, 'target_ids': ['chair']}
    for key, path, value in records:
        batch.immutable_json(path, value)
        baseline[key] = {'path': str(path), 'sha256': file_hash(path)}
    batch.immutable_json(batch_root / 'baseline_snapshot.json', scene)
    batch.immutable_json(batch_root / 'baseline.json', baseline)
    folder = batch_root / 'targets/001_chair'
    folder.mkdir(parents=True)
    batch.immutable_json(folder / 'started.json', {'explicit_local_fixture': True})
    after = refined(scene) if committed else scene
    head = {'revision': after['revision'], 'head_seq': 3 if committed else 2}
    if committed:
        asset = next(obj['asset'] for obj in after['objects'] if obj['instance_id'] == 'chair')
        completion = {'status': 'complete', 'target_id': 'chair', 'authorization_sha256': auth_hash,
                      'before_head': auth['baseline_head'], 'head': head, 'after_snapshot_hash': canonical_hash(after),
                      'asset': asset, 'model': 'fixture-model', 'receipt_records': [], 'provider_call_receipts': [],
                      'evidence_id': evidence['evidence_id'], 'camera_fingerprint': camera_hash(evidence['frames']),
                      'critic': {'status': 'accept', 'summary': 'Explicit fixture, not model acceptance', 'issues': [], 'limitations': []}}
        for name, value in [('before.json', scene), ('after.json', after), ('completion.json', completion)]:
            batch.immutable_json(folder / name, value)

    class Repo:
        async def inspect(self, request, context):
            return {'head': head, 'snapshot': copy.deepcopy(after)}

    class Evidence:
        def __init__(self, *args):
            pass

        async def get(self, *args):
            return copy.deepcopy(evidence)

    host = SimpleNamespace(repo=Repo(), validator=SimpleNamespace(),
                           context=lambda **kwargs: ExecutionContext(storage_root=root / 'runtime', **kwargs),
                           resolver=SimpleNamespace(resolve=lambda *args, **kwargs: tmp_path / 'fixture-asset'))
    monkeypatch.setattr(batch, 'create_host', lambda *args: host)
    monkeypatch.setattr(batch, 'MaskEvidenceService', Evidence)
    def forbidden_backend(*args, **kwargs):
        raise AssertionError('Resume must not create a new author/backend in this fixture')
    monkeypatch.setattr(batch, 'DirectAssetBackend', forbidden_backend)
    return root, batch_root


def test_resume_halts_uncertain_started_target_without_replaying(scene, tmp_path, monkeypatch):
    root, batch_root = setup_resume(tmp_path, monkeypatch, scene, committed=False)
    with pytest.raises(ContractError) as error:
        asyncio.run(batch.run_batch(root, batch_root, 'unused-blender', resume=True))
    assert error.value.code == 'RESOURCE_EXHAUSTED'
    status = batch.read(batch_root / 'status.json')
    assert status['phase'] == 'needs_reconciliation' and status['active_target_id'] == 'chair'
    assert not (batch_root / '.runner.lock').exists()


def test_resume_skips_verified_committed_target_without_new_author(scene, tmp_path, monkeypatch):
    root, batch_root = setup_resume(tmp_path, monkeypatch, scene, committed=True)
    result = asyncio.run(batch.run_batch(root, batch_root, 'unused-blender', resume=True))
    assert result['phase'] == 'complete'
    assert result['progress'] == {'completed': 1, 'total': 1}
    assert result['current_head']['head_seq'] == 3
    assert result['completed'][0]['asset']['asset_version'] == 'gpt-direct-fixture-new'
