"""Isolated Pi tool guards; model receipts are explicit fixtures, never live calls."""
import asyncio
import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from video2scene.artifacts import Frame
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash
from video2scene.agent_scene.planning import save, read
from video2scene.agent_scene import whitebox
from video2scene.pi_workflow.scene_actions import SceneActions, TOOL_SCHEMAS


def run(coro):
    return asyncio.run(coro)


def prior():
    return {'summary': 'fixture prior', 'room': {'shape_hypothesis': 'rectangle', 'visible_boundaries': [],
        'unseen_boundaries': [], 'approx_dimensions_m': [5, 3, 5], 'dimension_basis': 'hypothesis', 'layout_relations': []},
        'objects': [{'object_id': 'chair', 'label_zh': '椅子', 'category': 'furniture/chair', 'sam_prompt': 'chair',
        'seed_source_frame_index': 7, 'positive_points_normalized': [[.5, .5]], 'negative_points_normalized': [],
        'hint_box_normalized': [.2, .2, .6, .6], 'evidence_frame_indices': [7], 'primitive': 'box',
        'approx_dimensions_m': [.8, 1., .6], 'placement': 'floor', 'support_object_id': 'floor',
        'observed_features': ['seat'], 'assumptions': ['back']}],
        'floor_observations': [{'source_frame_index': 7, 'polygon_normalized': [[.1, .6], [.9, .6], [.9, .9]], 'reason': 'visible'}],
        'wall_observations': [], 'tool_plan': {'call_sam31': True, 'call_vipe': True, 'vipe_reason': 'depth',
        'requested_extra_frame_indices': [], 'segmentation_reason': 'instance'}, 'uncertainties': ['scale'],
        'suggested_refinement_target': 'chair'}


def layout():
    common = {'yaw_deg': 0., 'basis': 'mixed', 'reason': 'fixture'}
    return {'summary': 'fixture', 'objects': [{**common, 'object_id': 'chair', 'kind': 'object', 'category': 'furniture/chair',
        'center_m': [0., .5, 3.], 'dimensions_m': [.8, 1., .6], 'support_object_id': 'floor'}],
        'structures': [{**common, 'object_id': 'floor', 'kind': 'floor', 'category': 'structure/floor',
        'center_m': [0., -.025, 3.], 'dimensions_m': [5., .05, 5.], 'support_object_id': 'unknown'}],
        'rejected_objects': [], 'uncertainties': ['fixture approximate'], 'recommended_refinement_target': 'chair'}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path / 'projects' / 'project_a'
    runtime = tmp_path / 'runtime'
    events = {}
    image = project / 'frames' / '00000.jpg'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (60, 40), '#777777').save(image)
    receipt = {'scene_id': 'project_a', 'provider': 'openai-codex', 'model': 'fixture-not-real-model',
               'run_id': 'fixture_run', 'turn_id': 'fixture_turn', 'images_delivered': True,
               'frames': [{'source_frame_index': 7, 'path': str(image), 'sha256': file_hash(image)}]}
    actions = SceneActions('project_a', project, runtime, human_event_verifier=lambda token: copy.deepcopy(events[token]),
                           prior_observation_verifier=lambda token: copy.deepcopy(receipt))
    sam = project / 'tools/sam/fixture/output'
    save(sam / 'result.json', {'status': 'complete'})
    save(project / 'frames/manifest.json', {'prepared_fps': 10, 'frames': [{'frame_index': 0, 'source_frame_index': 7,
        'source_width': 120, 'source_height': 80, 'source_time_s': .7}]})
    mask_dir = sam / 'chair/target_masks'
    mask_dir.mkdir(parents=True)
    np.savez_compressed(mask_dir / '00007.npz', mask=np.ones((40, 60), dtype=bool), source_frame_index=np.int64(7))
    frame = Frame(0, np.full((40, 60, 3), 120, dtype=np.uint8), np.full((40, 60), 3., dtype=np.float32),
                  np.array([45., 45., 30., 20.]), np.eye(4))
    monkeypatch.setattr(whitebox, 'frames_by_source', lambda _: {7: frame})
    def measured(root, samroot):
        value = {'status': 'proposed', 'scene_from_vipe': np.eye(4).tolist(), 'objects': [{
            'object_id': 'chair', 'status': 'observed_proposal', 'center_m': [0., .5, 3.], 'accepted_frame_indices': [7]}],
            'prior_sha256': file_hash(root / 'prior.json'), 'sam_result_sha256': file_hash(samroot / 'result.json')}
        save(root / 'geometry_proposals.json', value)
        return value
    monkeypatch.setattr(whitebox, 'propose', measured)
    async def fake_export(snapshot, label):
        return {'snapshot_hash': canonical_hash(snapshot), 'artifact_refs': [], 'explicit_mock_export': True}
    monkeypatch.setattr(actions, '_export_snapshot', fake_export)
    yield actions, events, receipt, sam
    actions.host.jobs.shutdown(wait=True)


def seed(setup):
    actions, events, receipt, sam = setup
    run(actions.record_prior(prior(), 'fixture_observation'))
    run(actions.measure(str(sam)))
    return run(actions.build_whitebox(layout(), str(sam)))


def event(actions, kind, payload):
    view = run(actions.inspect_scene())
    return {'event_id': 'human_fixture_' + kind, 'scene_id': actions.scene_id, 'actor': 'user', 'source': 'browser',
            'kind': kind, 'head': view['head'], 'snapshot_hash': canonical_hash(view['snapshot']), 'payload': payload}


def test_prior_requires_host_delivered_images_and_uses_dynamic_indices(setup):
    actions, _, receipt, _ = setup
    receipt['images_delivered'] = False
    with pytest.raises(ContractError, match='observation receipt'):
        run(actions.record_prior(prior(), 'fixture'))
    receipt['images_delivered'] = True
    result = run(actions.record_prior(prior(), 'fixture'))
    assert result['objects'] == 1
    unseen = prior()
    unseen['objects'][0]['seed_source_frame_index'] = 8
    with pytest.raises(ValueError, match='supplied'):
        run(actions.record_prior(unseen, 'fixture'))


def test_dispatch_does_not_expose_acceptance(setup):
    actions, _, _, _ = setup
    assert 'accept_whitebox' not in TOOL_SCHEMAS
    with pytest.raises(ContractError, match='host-only'):
        run(actions.dispatch('accept_whitebox', {}))


def test_project_isolation_and_general_source_coordinates(setup):
    actions, _, _, _ = setup
    result = seed(setup)
    assert result['snapshot']['scene_id'] == 'project_a'
    capture = read(actions.runtime / 'project_a/captures' / (result['snapshot']['capture_revision'] + '.json'))
    assert capture['frames'][0]['source_to_image'] == [[.5, 0., 0.], [0., .5, 0.], [0., 0., 1.]]
    assert capture['frames'][0]['source_time_s'] == .7
    assert not (actions.runtime / 'redwood-agent').exists()
    with pytest.raises(ContractError, match='already exists'):
        run(actions.build_whitebox(layout(), str(setup[3])))


def test_old_redwood_identity_is_rejected(tmp_path):
    with pytest.raises(ContractError, match='historical Redwood'):
        SceneActions('redwood-agent', tmp_path / 'project', tmp_path / 'runtime')


def test_observed_image_and_measurement_tampering_are_rejected(setup):
    actions, _, receipt, sam = setup
    run(actions.record_prior(prior(), 'fixture'))
    run(actions.measure(str(sam)))
    save(actions.run / 'geometry_proposals.json', {'status': 'invented'})
    with pytest.raises(ContractError, match='Measured geometry changed'):
        run(actions.build_whitebox(layout(), str(sam)))
    Path(receipt['frames'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ContractError, match='observed frame changed'):
        run(actions.measure(str(sam)))


@pytest.mark.parametrize('bad', ['source', 'head', 'snapshot_hash', 'payload'])
def test_acceptance_is_bound_to_actual_human_event(setup, bad):
    actions, events, _, _ = setup
    seed(setup)
    payload = {'target_ids': ['chair'], 'user_instruction': '精修椅子'}
    events['feedback'] = event(actions, 'accept_whitebox', payload)
    if bad == 'source': events['feedback']['source'] = 'model'
    elif bad == 'head': events['feedback']['head']['head_seq'] = 1
    elif bad == 'snapshot_hash': events['feedback']['snapshot_hash'] = '0' * 64
    else: payload = {**payload, 'target_ids': ['floor']}
    with pytest.raises(ContractError):
        run(actions.accept_whitebox(payload, 'feedback'))


def test_acceptance_persists_and_failed_started_refinement_is_not_replayed(setup):
    actions, events, _, _ = setup
    seed(setup)
    payload = {'target_ids': ['chair'], 'user_instruction': '精修椅子', 'visual_policy': 'structure_first_relaxed_appearance'}
    events['feedback'] = event(actions, 'accept_whitebox', payload)
    accepted = run(actions.accept_whitebox(payload, 'feedback'))
    calls = []
    class FailingBackend:
        def __init__(self, *args, **kwargs): pass
        async def refine(self, *args):
            calls.append('explicit fixture, no provider')
            raise RuntimeError('fixture uncertain provider outcome')
    actions.backend_factory = FailingBackend
    with pytest.raises(RuntimeError, match='uncertain'):
        run(actions.refine_object('chair', accepted['acceptance_id'], 'one_attempt'))
    with pytest.raises(FileExistsError):
        run(actions.refine_object('chair', accepted['acceptance_id'], 'one_attempt'))
    assert len(calls) == 1
    assert read(actions.root / 'operations/one_attempt/failure.json')['automatic_replay'] is False
    assert run(actions.inspect_scene())['head']['head_seq'] == 0


def test_user_patch_cannot_broaden_grant(setup):
    actions, events, _, _ = setup
    seed(setup)
    events['edit'] = event(actions, 'scene_patch', {'allowed_object_ids': ['chair'],
        'allowed_fields': ['transform.translation_m'], 'user_instruction': '移动椅子'})
    with pytest.raises(ContractError, match='exceeds'):
        run(actions.apply_scene_patch('patch1', 'edit', [{'instance_id': 'floor', 'field': 'transform.translation_m', 'value': [0, 0, 0]}]))


def test_actual_cpu_patch_validate_save_export_and_stale_acceptance(setup, monkeypatch):
    actions, events, _, _ = setup
    if not actions.blender.is_file(): pytest.skip('Local Blender missing')
    seed(setup)
    payload = {'target_ids': ['chair'], 'user_instruction': '精修椅子'}
    events['accept'] = event(actions, 'accept_whitebox', payload)
    accepted = run(actions.accept_whitebox(payload, 'accept'))
    edits = [{'instance_id': 'chair', 'field': 'transform.translation_m', 'value': [.1, .5, 3.]}]
    events['edit'] = event(actions, 'scene_patch', {'allowed_object_ids': ['chair'],
        'allowed_fields': ['transform.translation_m'], 'user_instruction': '向右移动0.1米', 'edits': edits})
    staged = run(actions.apply_scene_patch('patch_cpu', 'edit', edits))
    candidate = staged['candidate']
    assert run(actions.inspect_scene())['head']['head_seq'] == 0
    report = run(actions.validate_candidate(candidate['candidate_id'], candidate['candidate_hash']))
    assert report['technical_pass'] and report['committable']
    saved = run(actions.commit_candidate(candidate['candidate_id'], candidate['candidate_hash'], report['validation_id']))
    assert saved['head']['head_seq'] == 1
    replay = run(actions.commit_candidate(candidate['candidate_id'], candidate['candidate_hash'], report['validation_id']))
    assert replay == saved
    with pytest.raises(ContractError, match='outside the accepted refinement chain'):
        run(actions.refine_object('chair', accepted['acceptance_id'], 'after_edit'))
    monkeypatch.setattr(actions, '_export_snapshot', SceneActions._export_snapshot.__get__(actions))
    export = run(actions.export_scene())
    assert {a['role'] for a in export['artifact_refs']} >= {'scene_glb', 'scene_blend', 'reopen_report'}
    for ref in export['artifact_refs']:
        published = actions.host.store.artifact(ref['uri'])
        assert published == ref
        assert file_hash(actions.host.resolver.resolve(published)) == ref['sha256']
    assert run(actions.inspect_scene())['head'] == saved['head']


def test_vipe_pointer_is_project_bound(tmp_path, monkeypatch):
    save(tmp_path / 'frames/manifest.json', {'frames': [{'frame_index': 0, 'source_frame_index': 11}]})
    save(tmp_path / 'tools/vipe/latest.json', {'vipe_dir': str(tmp_path.parent / 'unrelated')})
    with pytest.raises(ValueError, match='belong'):
        whitebox.frames_by_source(tmp_path)


def test_measure_archives_previous_proposals_and_binds_current_vipe_selection(setup, monkeypatch):
    actions, _, _, sam = setup
    run(actions.record_prior(prior(), 'fixture'))
    pointer = actions.run / 'tools/vipe/latest.json'
    save(pointer, {'vipe_dir': str(actions.run / 'fixture_sparse'), 'sampling_version': 'explicit_sparse_fixture'})
    first = run(actions.measure(str(sam)))
    old_bytes = (actions.run / 'geometry_proposals.json').read_bytes()
    original_propose = whitebox.propose
    save(pointer, {'vipe_dir': str(actions.run / 'fixture_dense'), 'sampling_version': 'explicit_dense_fixture'})
    with pytest.raises(ContractError, match='Selected ViPE output changed'):
        actions._measurements()
    def different(root, samroot):
        result = original_propose(root, samroot)
        result['comparison_fixture'] = 'dense'
        save(root / 'geometry_proposals.json', result)
        return result
    monkeypatch.setattr(whitebox, 'propose', different)
    second = run(actions.measure(str(sam)))
    assert Path(second['previous_measurement']['path']).read_bytes() == old_bytes
    assert first['measurement_record']['sha256'] == second['previous_measurement']['sha256']
    assert second['measurement_record']['sha256'] != second['previous_measurement']['sha256']
    assert actions._measurements()['comparison_fixture'] == 'dense'
    def changes_selection(root, samroot):
        result = original_propose(root, samroot)
        save(pointer, {'vipe_dir': str(actions.run / 'fixture_concurrent_change')})
        return result
    monkeypatch.setattr(whitebox, 'propose', changes_selection)
    with pytest.raises(ContractError, match='changed during measurement'):
        run(actions.measure(str(sam)))
