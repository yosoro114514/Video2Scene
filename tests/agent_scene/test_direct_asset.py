"""Local tests for direct object integration; no external model is invoked."""
import asyncio
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
from PIL import Image
import pytest
import trimesh

from video2scene.agent_scene.direct_asset import (
    AUTHOR_SCHEMA, DirectAssetBackend, author_prompt, critic_prompt, local_blender_bounds, read, write,
)
from video2scene.asset_backend.backend import DEFAULT_BLENDER
from video2scene.contracts_v1 import ContractError, ExecutionContext, file_hash, validate
from video2scene.contracts_v1.mocks import example


@pytest.fixture
def prepared(tmp_path):
    scene = copy.deepcopy(example('model_SceneSnapshot'))
    target = scene['objects'][1]
    target.update(category='furniture/chair', local_bounds_m={'min': [-.4, 0., -.25], 'max': [.6, 1.2, .35]})
    target['transform']['translation_m'] = [1., .75, 3.]
    target['transform']['scale'] = [1.1, 1.05, .9]
    target['transform']['rotation_xyzw'] = [0., .25881904510252074, 0., .9659258262890683]
    for obj in scene['objects']:
        path = tmp_path / scene['scene_id'] / 'seed' / (obj['instance_id'] + '.glb')
        path.parent.mkdir(parents=True, exist_ok=True)
        lo, hi = np.array(obj['local_bounds_m']['min']), np.array(obj['local_bounds_m']['max'])
        mesh = trimesh.creation.box(extents=hi-lo)
        mesh.apply_translation((hi+lo)/2)
        path.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
        obj['asset']['artifact'].update(uri=f'artifact://{scene["scene_id"]}/seed/{path.name}', sha256=file_hash(path))
    evidence = copy.deepcopy(example('model_EvidenceBundle'))
    evidence.update(scene_id=scene['scene_id'], scene_revision=scene['revision'], target_ids=[target['instance_id']])
    frame = evidence['frames'][0]
    path = tmp_path / scene['scene_id'] / 'capture/rgb.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (640, 480), '#adc7b7').save(path)
    frame['rgb'].update(uri=f'artifact://{scene["scene_id"]}/capture/rgb.png', sha256=file_hash(path))
    candidate = copy.deepcopy(example('model_CandidateRef'))
    spec = {'target_id': target['instance_id'], 'allowed_fields': ['asset.geometry', 'asset.materials'],
            'candidate': candidate, 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
            'budget': {'timeout_s': 180, 'max_iterations': 1, 'max_model_calls': 2}}
    confirmation_path = tmp_path / 'confirmation.json'
    confirmation = {'confirmed': True, 'scene_id': scene['scene_id'], 'revision': scene['revision'],
                    'head_seq': candidate['base_head_seq'], 'target_id': target['instance_id'],
                    'confirmed_at': '2026-10-29T00:00:00Z', 'user_instruction': 'Local fixture; preserve size and placement.'}
    write(confirmation_path, confirmation)
    context = ExecutionContext(storage_root=tmp_path, actor='user', decision_ref='local-fixture',
                               allowed_object_ids=frozenset({target['instance_id']}),
                               allowed_fields=frozenset({'asset.geometry', 'asset.materials'}))
    return scene, evidence, spec, confirmation_path, context


def test_asymmetric_bounds_axis_conversion():
    assert local_blender_bounds({'min': [-1, 2, -3], 'max': [4, 5, 6]}) == [[-1., -6., 2.], [4., 3., 5.]]


@pytest.mark.parametrize('mutation,code', [
    ({'confirmed': False}, 'STALE_REVISION'),
    ({'revision': 'other'}, 'STALE_REVISION'),
    ({'head_seq': 99}, 'STALE_REVISION'),
    ({'target_id': 'other'}, 'STALE_REVISION'),
    ({'scene_id': 'other'}, 'STALE_REVISION'),
    ({'confirmed_at': ''}, 'STALE_REVISION'),
])
def test_unconfirmed_or_stale_user_gate_never_calls_model(prepared, mutation, code):
    scene, evidence, spec, confirmation, context = prepared
    record = read(confirmation)
    record.update(mutation)
    write(confirmation, record)
    calls = []
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation,
                                 invoke_fn=lambda **kw: calls.append(kw))
    with pytest.raises(ContractError) as error:
        asyncio.run(backend.refine(spec, scene, evidence, context))
    assert error.value.code == code and calls == []


def test_missing_confirmation_never_creates_model_workspace(prepared):
    scene, evidence, spec, confirmation, context = prepared
    missing = confirmation.parent / 'not-confirmed.json'
    backend = DirectAssetBackend(context.storage_root, confirmation_path=missing)
    with pytest.raises(ContractError) as error:
        asyncio.run(backend.refine(spec, scene, evidence, context))
    assert error.value.code == 'FORBIDDEN_SCOPE'
    assert not (context.storage_root / scene['scene_id'] / 'assets').exists()


def test_uncertain_invocation_is_not_replayed(prepared, tmp_path):
    _, evidence, _, confirmation, context = prepared
    calls = []
    def failing(**kw):
        calls.append(kw)
        raise TimeoutError('Local fixture simulates uncertain remote response')
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation, invoke_fn=failing)
    output = tmp_path / 'uncertain'
    image = backend.resolver.resolve(evidence['frames'][0]['rgb'], scene_id=evidence['scene_id'])
    for expected in (TimeoutError, ContractError):
        with pytest.raises(expected):
            backend._invoke('fixture', [image], output, context, timeout=5, schema=AUTHOR_SCHEMA,
                            sandbox='workspace-write', max_tool_calls=1)
    assert len(calls) == 1
    assert read(output / 'invocation_outcome.json')['automatic_retry'] is False


def test_insufficient_call_budget_and_structure_rejected_before_provider(prepared):
    scene, evidence, spec, confirmation, context = prepared
    calls = []
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation,
                                 invoke_fn=lambda **kw: calls.append(kw))
    spec['budget']['max_model_calls'] = 1
    with pytest.raises(ContractError) as error:
        asyncio.run(backend.refine(spec, scene, evidence, context))
    assert error.value.code == 'RESOURCE_EXHAUSTED'
    scene['objects'][1]['kind'] = 'wall'
    with pytest.raises(ContractError) as error:
        asyncio.run(backend.refine(spec, scene, evidence, context))
    assert error.value.code == 'FORBIDDEN_SCOPE' and calls == []


@pytest.mark.parametrize('kind', ['wall', 'floor', 'opening', 'ceiling'])
def test_explicit_structure_grant_reaches_bounded_author(prepared, monkeypatch, kind):
    scene, evidence, spec, confirmation, context = prepared
    scene['objects'][1]['kind'] = kind
    record = read(confirmation); record['include_structures'] = True; write(confirmation, record)
    # The generator is intercepted below; this authorization test starts no Blender.
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation, blender_path=sys.executable)
    def reached(*args):
        raise RuntimeError('explicit fixture reached generation without calling any provider')
    monkeypatch.setattr(backend, '_generate', reached)
    with pytest.raises(RuntimeError, match='explicit fixture reached'):
        asyncio.run(backend.refine(spec, scene, evidence, context))


def test_cancel_waits_for_owned_child(prepared, tmp_path):
    *_, confirmation, context = prepared
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation)
    timer = threading.Timer(.3, context.cancel_event.set)
    timer.start()
    try:
        with pytest.raises(ContractError) as error:
            backend._process([sys.executable, '-c', 'import time; time.sleep(20)'], tmp_path,
                             context, 25, log_name='owned.log')
        assert error.value.code == 'CANCELLED'
        execution = read(tmp_path / 'owned.log.execution.json')
        assert execution['state'] == 'cancelled' and execution['returncode'] is not None
    finally:
        timer.cancel()
        timer.join()


def test_observed_mask_localizes_reference_without_changing_camera_or_accepted_pose(prepared, tmp_path):
    scene, evidence, spec, confirmation, context = prepared
    target = scene['objects'][1]
    observation = copy.deepcopy(target)
    observation['transform']['translation_m'] = [0., 0., 3.]
    frame = evidence['frames'][0]
    mask = np.zeros((frame['height'], frame['width']), dtype=np.uint8)
    mask[200:300, 100:150] = 255
    mask_path = context.storage_root / scene['scene_id'] / 'capture/observed-mask.png'
    Image.fromarray(mask).save(mask_path)
    frame['mask'] = dict(frame['rgb'], uri=f'artifact://{scene["scene_id"]}/capture/observed-mask.png',
                         sha256=file_hash(mask_path), role='observed_mask')
    before = copy.deepcopy(frame)
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation,
                                 observation_objects={target['instance_id']: observation})
    folder = tmp_path / 'prepared-mask'
    contract, images = backend._prepare(folder, scene, target, evidence, spec['candidate'], read(confirmation))
    assert contract['crop_rectangles'] == [[90, 188, 160, 312]]
    assert contract['frames'][0]['crop_origin'] == 'observed_target_mask'
    assert contract['frames'][0]['c2w'] == before['c2w']
    assert contract['frames'][0]['K'] == before['K']
    assert contract['accepted_transform'] == target['transform']
    assert contract['observation_transform'][0][3] == 0.
    assert [path.name for path in images] == ['ref_00_rgb.png', 'ref_00_crop.png', 'ref_00_mask.png']
    assert frame == before


@pytest.mark.parametrize('status,issues', [
    ('revise', []), ('unclear', []),
    ('accept', [{'severity': 'major', 'evidence': 'Fixture defect', 'requested_fix': 'Correct fixture defect'}]),
])
def test_independent_critic_failure_exhausts_bound_without_candidate(prepared, tmp_path, monkeypatch, status, issues):
    scene, evidence, spec, confirmation, context = prepared
    authors = []
    def provider(**kwargs):
        if kwargs['schema'] == AUTHOR_SCHEMA:
            authors.append([Path(path).name for path in kwargs['images']])
            return {'status': 'complete', 'instance_id': spec['target_id'], 'asset_blend': 'fixture.blend',
                    'script_path': 'fixture.py', 'parts': [], 'observed_features': [], 'assumptions': []}
        return {'status': status, 'summary': 'Explicit local critic-gate fixture', 'issues': issues, 'limitations': []}
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation, invoke_fn=provider)
    # This unit test isolates orchestration; the real CPU export has a separate integration test.
    monkeypatch.setattr(backend, '_author_proof', lambda *args: {})
    monkeypatch.setattr(backend, '_validate_authored', lambda *args: {})
    def montages(work, output, contract):
        output.mkdir(exist_ok=True)
        paths = [output / name for name in ('turntable.jpg', 'comparison_00.jpg')]
        for path in paths:
            Image.new('RGB', (8, 8), '#82593d').save(path)
        return paths
    monkeypatch.setattr(backend, '_montages', montages)
    root = tmp_path / 'critic-gate'
    root.mkdir()
    with pytest.raises(ContractError) as error:
        backend._generate(root, scene, scene['objects'][1], evidence, spec['candidate'], read(confirmation),
                          file_hash(confirmation), {'max_iterations': 2, 'max_model_calls': 4, 'timeout_s': 10}, context)
    assert error.value.code == 'BUILD_FAILED'
    ledger = read(root / 'run.json')
    assert ledger['provider_calls'] == 4 and ledger['model_calls'] == 0
    assert ledger['status'] == 'failed' and len(ledger['iterations']) == 2
    assert 'accepted_asset' not in ledger
    assert not any(name.startswith('previous_') for name in authors[0])
    assert {'previous_turntable.jpg', 'previous_comparison_00.jpg'} <= set(authors[1])


def test_explicit_relaxed_appearance_reaches_both_roles(prepared, tmp_path):
    scene, evidence, spec, confirmation, context = prepared
    value = read(confirmation)
    value['visual_policy'] = 'structure_first_relaxed_appearance'
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation)
    work = tmp_path / 'appearance-policy'
    contract, _ = backend._prepare(work, scene, scene['objects'][1], evidence, spec['candidate'], value)
    assert contract['visual_policy'] == value['visual_policy']
    for prompt in (author_prompt(work, contract, backend.blender_path, None), critic_prompt(contract)):
        assert 'reasonable differences in color' in prompt
        assert 'These differences alone are minor and must not cause rejection' in prompt
        assert 'Never relax saved placement, bounds or support' in prompt


@pytest.mark.parametrize('tamper', [False, True])
def test_terminal_failed_author_can_retry_but_never_tampered_input(prepared, tmp_path, monkeypatch, tamper):
    scene, evidence, spec, confirmation, context = prepared
    calls = []
    def provider(**kwargs):
        calls.append(kwargs['schema'])
        if len(calls) == 1 and tamper:
            (Path(kwargs['output']) / 'ref_00_rgb.png').write_bytes(b'corrupt')
        if kwargs['schema'] == AUTHOR_SCHEMA:
            return {'status': 'failed' if len(calls) == 1 else 'complete', 'instance_id': spec['target_id'],
                    'asset_blend': 'fixture.blend', 'script_path': 'fixture.py', 'parts': [], 'observed_features': [], 'assumptions': []}
        return {'status': 'accept', 'summary': 'Isolated local fixture', 'issues': [], 'limitations': []}
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation, invoke_fn=provider)
    def proof(work, response, *args):
        if response['status'] == 'failed':
            raise ContractError('BUILD_FAILED', 'Fixture Blender error')
        return {}
    monkeypatch.setattr(backend, '_author_proof', proof)
    monkeypatch.setattr(backend, '_validate_authored', lambda *args: {})
    monkeypatch.setattr(backend, '_montages', lambda *args: [])
    root = tmp_path / 'technical-repair'
    root.mkdir()
    args = (root, scene, scene['objects'][1], evidence, spec['candidate'], read(confirmation),
            file_hash(confirmation), {'max_iterations': 2, 'max_model_calls': 4, 'timeout_s': 10}, context)
    if tamper:
        with pytest.raises(ContractError) as error:
            backend._generate(*args)
        assert error.value.code == 'STALE_ARTIFACT' and len(calls) == 1
    else:
        backend._generate(*args)
        assert len(calls) == 3
        assert read(root / 'run.json')['status'] == 'accepted_asset'
        assert read(root / 'run.json')['iterations'][0]['feedback']['author_response']['status'] == 'failed'


def make_local_provider():
    def provider(**kwargs):
        output = Path(kwargs['output'])
        if kwargs['schema'] != AUTHOR_SCHEMA:
            return {'status': 'accept', 'summary': 'Local fixture acceptance, not a model judgment.',
                    'issues': [], 'limitations': ['Fixture does not assess reconstruction fidelity.']}
        contract = read(output / 'contract.json')
        lo, hi = np.array(contract['local_bounds_blender'][0]), np.array(contract['local_bounds_blender'][1])
        script = output / 'object.py'
        script.write_text(
            'import bpy\n'
            'bpy.ops.wm.read_factory_settings(use_empty=True)\n'
            f'bpy.ops.mesh.primitive_cube_add(size=1, location={tuple(float(v) for v in (lo+hi)/2)!r})\n' +
            f'bpy.context.object.dimensions={tuple(float(v) for v in hi-lo)!r}\n'
            'bpy.context.object.name="Explicit local fixture part"\n'
            'bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\n'
            'material=bpy.data.materials.new("Fixture green")\n'
            'material.diffuse_color=(.2,.5,.3,1)\n'
            'bpy.context.object.data.materials.append(material)\n'
            f'bpy.ops.wm.save_as_mainfile(filepath={str(output / "asset.blend")!r})\n', encoding='utf-8')
        command = [str(DEFAULT_BLENDER), '--background', '--factory-startup', '--threads', '2',
                   '--python-exit-code', '1', '--python', str(script)]
        with (output / 'fixture_author.log').open('w', encoding='utf-8') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=35,
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        assert completed.returncode == 0
        event = {'type': 'item.completed', 'item': {'type': 'command_execution',
                 'command': ' '.join(command), 'exit_code': 0, 'test_fixture': True}}
        (output / 'events.jsonl').write_text(json.dumps(event) + '\n', encoding='utf-8')
        return {'status': 'complete', 'instance_id': contract['instance_id'],
                'asset_blend': str(output / 'asset.blend'), 'script_path': str(script),
                'parts': ['fixture'], 'observed_features': ['Local CPU integration fixture'], 'assumptions': ['No model invoked']}
    return provider


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Existing Blender unavailable')
def test_real_cpu_export_preserves_bounds_root_support_and_non_targets(prepared):
    scene, evidence, spec, confirmation, context = prepared
    original = copy.deepcopy(scene)
    observation = copy.deepcopy(scene['objects'][1])
    observation['transform']['translation_m'] = [0., .75, 3.]
    observation['transform']['scale'] = [1., 1., 1.]
    backend = DirectAssetBackend(context.storage_root, confirmation_path=confirmation,
                                 observation_objects={observation['instance_id']: observation},
                                 invoke_fn=make_local_provider(), samples=1, timeout_s=120)
    output = asyncio.run(backend.refine(spec, scene, evidence, context))
    validate(output, 'BuildOutput')
    assert scene == original
    assert len(output['asset_updates']) == 1
    update = output['asset_updates'][0]
    assert update['asset']['native_frame'] == 'asset_y_up'
    assert update['asset']['asset_version'].startswith('fixture-direct-')
    assert update['local_bounds_m'] == original['objects'][1]['local_bounds_m']
    assert update['asset']['artifact']['sha256'] != original['objects'][1]['asset']['artifact']['sha256']
    run_path = next((context.storage_root / scene['scene_id'] / 'assets').glob('gpt-direct-*/run.json'))
    run = read(run_path)
    assert run['origin'] == 'local_fixture' and run['model_calls'] == 0 and run['provider_calls'] == 2
    asset_reopen = next(ref for ref in output['artifact_refs'] if ref['role'] == 'asset_reopen')
    reopened = read(backend.resolver.resolve(asset_reopen, scene_id=scene['scene_id']))
    assert reopened['native_reopened'] and reopened['glb_reopened']
    np.testing.assert_allclose(reopened['glb_bounds_y_up'], [update['local_bounds_m']['min'], update['local_bounds_m']['max']], atol=2e-5)
    scene_reopen = next(ref for ref in output['artifact_refs'] if ref['role'] == 'reopen_report')
    assembled = read(backend.resolver.resolve(scene_reopen, scene_id=scene['scene_id']))
    # The existing worker's own reopening checks verify world/root placement.
    assert assembled['passed']
    proof = read(run_path.parent / 'author_00/execution_proof.json')
    assert proof['actual_author_execution'] is False and proof['origin'] == 'local_fixture'
    result = read(run_path.parent / 'validated_00/result.json')
    assert result['render_source'] == 'reopened_canonical_glb'
    assert result['rendered_asset_sha256'] == update['asset']['artifact']['sha256']
    assert all(view['camera_projection_max_error_px'] < .01
               for view in result['renders'] if view['kind'] == 'source_pose')
    contract = read(run_path.parent / 'author_00/contract.json')
    assert contract['accepted_transform'] == original['objects'][1]['transform']
    np.testing.assert_allclose(np.asarray(contract['observation_transform'])[:3, 3],
                               observation['transform']['translation_m'])


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Existing Blender unavailable')
def test_package_only_avoids_discarded_scene_build_but_keeps_real_object_review(prepared,monkeypatch):
    from video2scene.pi_workflow.resources import ResourceBudget
    scene,evidence,spec,confirmation,context=prepared
    original=copy.deepcopy(scene)
    backend=DirectAssetBackend(context.storage_root,confirmation_path=confirmation,
        invoke_fn=make_local_provider(),samples=1,timeout_s=120)
    backend.package_only=True;backend.resource_budget=ResourceBudget(model=3,blender=1)
    async def unwanted_scene_build(*args):raise AssertionError('An unmerged package must not rebuild the old whole scene')
    monkeypatch.setattr(backend,'build',unwanted_scene_build)
    output=asyncio.run(backend.refine(spec,scene,evidence,context))
    validate(output,'BuildOutput');assert scene==original
    roles={ref['role']:ref for ref in output['artifact_refs']}
    assert 'scene_blend' not in roles and 'reopen_report' not in roles
    assert {'object_geometry','object_blend','object_visual_review','asset_reopen','author_execution'}<=roles.keys()
    reopened=read(backend.resolver.resolve(roles['asset_reopen'],scene_id=scene['scene_id']))
    assert reopened['native_reopened'] and reopened['glb_reopened']
    assert backend.resource_budget.snapshot()['used']=={'model':0,'blender':0}
    assert backend.resource_budget.snapshot()['peak']=={'model':1,'blender':1}


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Existing Blender unavailable')
def test_review_save_confirm_job_and_commit_with_explicit_local_provider(prepared, tmp_path):
    """Full local host integration; confirmation and author are explicit test fixtures."""
    from video2scene.agent_scene.evidence import MaskEvidenceService
    from video2scene.agent_scene.review_server import create_review_host
    from video2scene.capture_evidence import register_capture, transform_to_matrix
    scene, evidence, _, _, context = prepared
    scene = copy.deepcopy(scene)
    target = scene['objects'][1]
    # Keep the fixture independent of the mock scene's old tabletop support geometry.
    for obj in scene['objects']:
        obj['support_id'] = None
    target['category'] = 'coffee_table'
    run_root = tmp_path / 'fixture-review-run'
    runtime = run_root / 'runtime'
    shutil.copytree(context.storage_root / scene['scene_id'], runtime / scene['scene_id'])
    write(run_root / 'snapshot.json', scene)
    frame = copy.deepcopy(evidence['frames'][0])
    mask_path = runtime / scene['scene_id'] / 'capture/observed-mask.png'
    mask = np.zeros((frame['height'], frame['width']), dtype=np.uint8)
    mask[200:350, 350:550] = 255
    Image.fromarray(mask).save(mask_path)
    frame['mask'] = dict(frame['rgb'], uri=f'artifact://{scene["scene_id"]}/capture/observed-mask.png',
                         sha256=file_hash(mask_path), role='observed_mask')
    register_capture(runtime, scene['scene_id'], scene['capture_revision'], scene['geometry_revision'], [frame],
                     metadata={'frame_targets': {frame['source_frame_id']: target['instance_id']}})
    backend = DirectAssetBackend(runtime, confirmation_path=run_root / 'confirmation.json', samples=1,
                                 observation_objects={o['instance_id']: o for o in scene['objects']},
                                 invoke_fn=make_local_provider(), timeout_s=120)
    host = create_review_host(run_root, backend_override=backend)
    assert isinstance(host.evidence, MaskEvidenceService)
    queued = []
    host._start_command = lambda command: queued.append(copy.deepcopy(command))
    scale = copy.deepcopy(target['transform']['scale'])
    scale[1] *= 1.05
    command = {'contract_version': 'video2scene/1.0', 'command_id': 'fixture_saved_numeric_edit',
               'session_id': 'fixture_browser', 'scene_id': scene['scene_id'], 'base_revision': scene['revision'],
               'base_head_seq': 0, 'selected_ids': [target['instance_id']], 'intent': 'design_edit', 'kind': 'patch',
               'edits': [{'instance_id': target['instance_id'], 'field': 'transform.scale', 'value': scale}]}
    try:
        host.submit_command(command)
        asyncio.run(host.run_command(queued.pop(0)))
        record = host.store.command(scene['scene_id'], command['command_id'])
        assert record['state'] == 'succeeded', record
        accepted = host.current_view()
        assert accepted['head']['head_seq'] == 1 and host.review_status()['human_edit_saved']
        receipt = host.confirm({'scene_id': scene['scene_id'], **accepted['head'], 'target_id': target['instance_id']})
        asyncio.run(host.run_command(queued.pop(0)))
        record = host.store.command(scene['scene_id'], receipt['command_id'])
        assert record['state'] == 'succeeded', record
        final = host.current_view()
        assert final['head']['head_seq'] == 2
        before_by_id = {o['instance_id']: o for o in accepted['snapshot']['objects']}
        after_by_id = {o['instance_id']: o for o in final['snapshot']['objects']}
        for identity, before in before_by_id.items():
            after = after_by_id[identity]
            if identity != target['instance_id']:
                assert after == before
            else:
                assert all(after[k] == before[k] for k in ('transform', 'local_bounds_m', 'support_id'))
                assert after['asset'] != before['asset']
                assert after['asset']['asset_version'].startswith('fixture-direct-')
        run_path = next((runtime / scene['scene_id'] / 'assets').glob('gpt-direct-*/run.json'))
        contract = read(run_path.parent / 'author_00/contract.json')
        np.testing.assert_allclose(contract['observation_transform'], transform_to_matrix(target['transform']))
        assert contract['accepted_transform']['scale'] == scale
        assert contract['frames'][0]['c2w'] == frame['c2w']
        assert contract['frames'][0]['crop_origin'] == 'observed_target_mask'
        result = read(run_path.parent / 'validated_00/result.json')
        assert result['rendered_asset_sha256'] == after_by_id[target['instance_id']]['asset']['artifact']['sha256']
        assert read(run_path)['model_calls'] == 0
    finally:
        host.jobs.shutdown(wait=True)
