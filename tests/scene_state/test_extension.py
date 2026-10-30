"""Scene extension uses real repository/evidence/validation with explicit assets.

Most tests use an explicitly labelled mock build receipt. The CPU Blender test
assembles real GLBs. No model is called and no reconstruction quality is claimed.
"""
import asyncio
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from video2scene.api_v1.__main__ import create_host
from video2scene.asset_backend.backend import BlenderAssetBackend, DEFAULT_BLENDER
from video2scene.capture_evidence import register_capture
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash, validate
from video2scene.contracts_v1.mocks import example


def run(coro):
    return asyncio.run(coro)


class ExtensionHarness:
    def __init__(self, root):
        self.root = Path(root)
        self.host = create_host(root)
        self.scene = copy.deepcopy(example('model_SceneSnapshot'))
        self.scene_id = self.scene['scene_id']
        template = self.scene['objects'][1]
        for obj in self.scene['objects']:
            obj['asset'] = self.asset(obj, 'original-' + obj['instance_id'])
        for identity, kind, location, low, high in [
            ('floor', 'floor', [0, 0, 0], [-5, -.1, -5], [5, 0, 5]),
            ('wall_a', 'wall', [0, 0, 2.4], [-2.5, 0, -.05], [2.5, 2.5, .05]),
            ('wall_b', 'wall', [2.4, 0, 0], [-.05, 0, -2.5], [.05, 2.5, 2.5]),
            ('wall_c', 'wall', [-2.4, 0, 0], [-.05, 0, -2.5], [.05, 2.5, 2.5]),
        ]:
            obj = copy.deepcopy(template)
            obj.update(instance_id=identity, kind=kind, category='structure/' + kind, support_id=None,
                       local_bounds_m={'min': low, 'max': high})
            obj['transform']['translation_m'] = location
            obj['asset'] = self.asset(obj, 'original-' + identity)
            self.scene['objects'].append(obj)
        self.new_ids = ['interior_door', 'balcony_door', 'vase_tv_stand']
        self.replace_ids = ['floor', 'wall_a', 'wall_b', 'wall_c']
        self.ctx = self.host.context(actor='user', decision_ref='extension_fixture_decision',
                    allowed_object_ids=frozenset(self.new_ids + self.replace_ids),
                    allowed_fields=frozenset({'scene.objects.add', 'asset.geometry', 'asset.materials'}))
        run(self.host.repo.import_scene(self.scene, self.ctx))
        self.frames = []
        for i in range(2):
            frame = copy.deepcopy(example('model_EvidenceBundle')['frames'][0])
            frame['source_frame_id'] = 'source_' + str(i)
            frame['c2w'][0][3], frame['c2w'][1][3], frame['c2w'][2][3] = float(i), 1., -4.
            path = self.root / self.scene_id / 'capture' / f'{i}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (640, 480), (60 + i, 70, 80)).save(path)
            frame['rgb'] = {'uri': f'artifact://{self.scene_id}/capture/{i}.png', 'sha256': file_hash(path),
                            'media_type': 'image/png', 'role': 'source_rgb'}
            self.frames.append(frame)
        masked = copy.deepcopy(self.frames[0])
        masked['source_frame_id'] = 'existing_object_mask'
        masked['mask'] = copy.deepcopy(masked['rgb'])
        register_capture(self.root, self.scene_id, self.scene['capture_revision'], self.scene['geometry_revision'], self.frames + [masked])
        self.capture_path = self.root / self.scene_id / 'captures' / f"{self.scene['capture_revision']}.json"
        self.capture_hash = file_hash(self.capture_path)
        self.additions = []
        for identity, kind, position in [('interior_door', 'opening', [1.5, 0., 2.4]),
                                         ('balcony_door', 'opening', [-2.4, 0., 1.5]),
                                         ('vase_tv_stand', 'object', [.3, .75, 0.])]:
            obj = copy.deepcopy(template)
            obj.update(instance_id=identity, kind=kind, category='structure/door' if kind == 'opening' else 'vase',
                       support_id='floor' if kind == 'opening' else 'table_00', geometry_source='inferred')
            obj['transform']['translation_m'] = position
            obj['local_bounds_m'] = {'min': [-.4, 0., -.04], 'max': [.4, 2., .04]} if kind == 'opening' else {'min': [-.07, 0., -.07], 'max': [.07, .22, .07]}
            obj['asset'] = self.asset(obj, 'new-' + identity)
            self.additions.append(obj)
        self.replacements = [{'instance_id': obj['instance_id'], 'asset': self.asset(obj, 'finished-' + obj['instance_id'])}
                             for obj in self.scene['objects'] if obj['instance_id'] in self.replace_ids]

    def asset(self, obj, version):
        value = copy.deepcopy(obj['asset'])
        low, high = np.array(obj['local_bounds_m']['min']), np.array(obj['local_bounds_m']['max'])
        mesh = trimesh.creation.box(extents=high - low)
        mesh.apply_translation((low + high) / 2)
        mesh.visual.vertex_colors = [150, 100, 60, 255] if version.startswith(('new-', 'finished-')) else [220, 220, 220, 255]
        path = self.root / self.scene_id / 'assets' / (version + '.glb')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
        value.update(asset_version=version, artifact={'uri': f'artifact://{self.scene_id}/assets/{path.name}',
                    'sha256': file_hash(path), 'media_type': 'model/gltf-binary', 'role': 'object_geometry'})
        return value

    def request(self, identity='extension_1'):
        return {'contract_version': 'video2scene/1.0', 'request_id': identity, 'scene_id': self.scene_id,
                'tool': 'extend_scene', 'base_revision': self.scene['revision'], 'base_head_seq': 0,
                'params': {'intent': 'design_edit', 'decision_ref': self.ctx.decision_ref,
                           'additions': copy.deepcopy(self.additions), 'asset_replacements': copy.deepcopy(self.replacements)}}

    def view(self):
        return run(self.host.repo.inspect({'contract_version': 'video2scene/1.0', 'request_id': 'inspect',
                   'scene_id': self.scene_id, 'tool': 'inspect_scene', 'params': {'revision': 'head'}}, self.ctx))

    def prepare(self, request=None, context=None):
        return run(self.host.repo.prepare_extension(request or self.request(), context or self.ctx))

    def evidence_request(self):
        return {'contract_version': 'video2scene/1.0', 'request_id': 'extension_evidence', 'scene_id': self.scene_id,
                'tool': 'select_evidence', 'params': {'revision': self.scene['revision'], 'target_ids': self.new_ids + self.replace_ids,
                                                   'purpose': 'validation', 'n': 3}}

    def evidence(self, ref, request=None, frame_ids=None, context=None):
        return run(self.host.evidence.select_for_extension(request or self.evidence_request(), ref['candidate_id'], ref['candidate_hash'],
                                                         context or self.ctx, frame_ids=frame_ids))

    def build(self, ref, *, real_blender=False):
        snapshot = run(self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], self.ctx))
        if real_blender:
            output = run(BlenderAssetBackend(self.root, samples=1).build(snapshot, self.ctx))
        else:
            path = self.root / self.scene_id / 'fixture_build' / (ref['candidate_id'] + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"explicit_mock_build":true}', encoding='utf-8')
            output = {'input_fingerprint': ref['candidate_hash'], 'asset_updates': [], 'artifact_refs': [
                {'uri': f'artifact://{self.scene_id}/fixture_build/{path.name}', 'sha256': file_hash(path),
                 'media_type': 'application/json', 'role': 'explicit_fixture_build'}]}
        built = run(self.host.repo.attach_build(ref['candidate_id'], ref['candidate_hash'], output, self.ctx))
        return built, output

    def report(self, ref, evidence):
        snapshot = run(self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], self.ctx))
        return run(self.host.validator.validate(snapshot, evidence, 'tabletop_design_v1', self.ctx))

    def commit(self, ref, report):
        return run(self.host.repo.commit({'contract_version': 'video2scene/1.0', 'request_id': 'commit_' + ref['candidate_id'],
                   'scene_id': self.scene_id, 'tool': 'commit_candidate', 'base_revision': ref['base_revision'],
                   'base_head_seq': ref['base_head_seq'], 'params': {'candidate_id': ref['candidate_id'], 'candidate_hash': ref['candidate_hash'],
                   'validation_id': report['validation_id'], 'decision_ref': self.ctx.decision_ref}}, self.ctx))


@pytest.fixture
def h(tmp_path):
    return ExtensionHarness(tmp_path)


def assert_code(code, function):
    with pytest.raises(ContractError) as error:
        function()
    assert error.value.code == code


def test_extension_is_internal_and_does_not_expand_public_wire_tools(h):
    assert_code('INVALID_ARGUMENT', lambda: validate(h.request(), 'ToolRequest'))
    ref = h.prepare()
    assert ref['state'] == 'staged' and h.view()['head']['head_seq'] == 0


@pytest.mark.parametrize('missing', ['scene.objects.add', 'asset.geometry', 'asset.materials'])
def test_extension_requires_explicit_identity_and_field_grants(h, missing):
    context = replace(h.ctx, allowed_fields=h.ctx.allowed_fields - {missing})
    assert_code('FORBIDDEN_SCOPE', lambda: h.prepare(context=context))
    context = replace(h.ctx, allowed_object_ids=h.ctx.allowed_object_ids - {'vase_tv_stand'})
    assert_code('FORBIDDEN_SCOPE', lambda: h.prepare(context=context))
    assert h.view()['head']['head_seq'] == 0


@pytest.mark.parametrize('mutation,code', [('existing_id', 'INVALID_ARGUMENT'), ('duplicate', 'INVALID_ARGUMENT'),
                                        ('missing_support', 'CONSTRAINT_VIOLATION'), ('cycle', 'CONSTRAINT_VIOLATION'),
                                        ('replace_transform', 'INVALID_ARGUMENT'), ('stale_base', 'STALE_REVISION')])
def test_invalid_extension_is_atomic(h, mutation, code):
    request = h.request()
    if mutation == 'existing_id':
        request['params']['additions'][0]['instance_id'] = 'floor'
    elif mutation == 'duplicate':
        request['params']['additions'].append(copy.deepcopy(request['params']['additions'][0]))
    elif mutation == 'missing_support':
        request['params']['additions'][0]['support_id'] = 'absent'
    elif mutation == 'cycle':
        request['params']['additions'][0]['support_id'] = 'balcony_door'
        request['params']['additions'][1]['support_id'] = 'interior_door'
    elif mutation == 'replace_transform':
        request['params']['asset_replacements'][0]['transform'] = h.scene['objects'][0]['transform']
    else:
        request['base_head_seq'] = 99
    assert_code(code, lambda: h.prepare(request))
    assert h.view()['snapshot'] == h.scene


def test_prepare_replay_is_exact_and_still_requires_current_scope(h):
    request = h.request()
    ref = h.prepare(request)
    assert h.prepare(request) == ref
    changed = copy.deepcopy(request)
    changed['params']['additions'][0]['category'] = 'another_door'
    assert_code('IDEMPOTENCY_CONFLICT', lambda: h.prepare(changed))
    assert_code('FORBIDDEN_SCOPE', lambda: h.prepare(request, replace(h.ctx, allowed_fields=frozenset())))


def test_extension_evidence_preserves_raw_capture_and_supports_new_ids(h):
    ref = h.prepare()
    bundle = h.evidence(ref, frame_ids=['source_0', 'source_1'])
    assert set(bundle['target_ids']) == set(h.new_ids + h.replace_ids)
    assert bundle['scene_revision'] == h.scene['revision']
    for delivered, original in zip(bundle['frames'], h.frames):
        assert all(delivered[key] == original[key] for key in original if key != 'selection_reason')
        assert delivered['mask'] is None
    assert run(h.host.evidence.get(bundle['evidence_id'], h.ctx)) == bundle
    assert file_hash(h.capture_path) == h.capture_hash


def test_extension_auto_evidence_uses_only_existing_unmasked_frames(h):
    bundle = h.evidence(h.prepare())
    assert 1 <= len(bundle['frames']) <= 2
    assert all(frame['mask'] is None and frame['source_frame_id'].startswith('source_') for frame in bundle['frames'])


def test_extension_evidence_rejects_wrong_scope_masked_and_fabricated_frames(h):
    ref = h.prepare()
    assert_code('FRAME_MISMATCH', lambda: h.evidence(ref, frame_ids=['existing_object_mask']))
    assert_code('FRAME_MISMATCH', lambda: h.evidence(ref, frame_ids=['made_up_frame']))
    request = h.evidence_request()
    request['params']['target_ids'] = ['vase_tv_stand']
    assert_code('EVIDENCE_MISMATCH', lambda: h.evidence(ref, request))
    context = replace(h.ctx, decision_ref='forged_decision')
    assert_code('FORBIDDEN_SCOPE', lambda: h.evidence(ref, context=context))


def test_builder_cannot_rewrite_any_frozen_extension_asset(h):
    ref = h.prepare()
    addition = h.additions[-1]
    output = {'input_fingerprint': ref['candidate_hash'], 'artifact_refs': [addition['asset']['artifact']],
              'asset_updates': [{'instance_id': addition['instance_id'], 'asset': addition['asset'], 'local_bounds_m': addition['local_bounds_m']}]}
    assert_code('FORBIDDEN_SCOPE', lambda: run(h.host.repo.attach_build(ref['candidate_id'], ref['candidate_hash'], output, h.ctx)))
    assert h.view()['snapshot'] == h.scene


@pytest.mark.parametrize('mutation', ['addition_transform', 'existing_transform', 'non_target_asset', 'capture', 'missing_addition'])
def test_validator_rejects_changes_beyond_exact_extension(h, mutation):
    ref = h.prepare()
    snapshot = run(h.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], h.ctx))
    metadata = run(h.host.repo.candidate_metadata(ref['candidate_id'], h.ctx))
    scene = snapshot['scene']
    if mutation == 'addition_transform':
        scene['objects'][-1]['transform']['translation_m'][0] += 1.
    elif mutation == 'existing_transform':
        scene['objects'][2]['transform']['translation_m'][0] += 1.
    elif mutation == 'non_target_asset':
        scene['objects'][1]['asset'] = copy.deepcopy(scene['objects'][-1]['asset'])
    elif mutation == 'capture':
        scene['capture_revision'] = 'other_capture'
    else:
        scene['objects'].pop()
    checks = h.host.validator._scope_checks(scene, h.scene, metadata)
    assert any(check['passed'] is False for check in checks)


def test_new_object_collision_is_rejected_without_old_object_lookup_failure(h):
    request = h.request()
    request['params']['additions'][-1]['transform']['translation_m'][0] = 0.
    ref = h.prepare(request)
    evidence = h.evidence(ref, frame_ids=['source_0'])
    built, _ = h.build(ref)
    report = h.report(built, evidence)
    assert not report['technical_pass'] and not report['committable']
    assert any(check['name'] == 'support_and_box_collisions' and check['passed'] is False
               and 'collision:' in check['details'] for check in report['checks'])


def test_full_extension_commit_keeps_existing_layout_and_capture(h):
    ref = h.prepare()
    evidence = h.evidence(ref, frame_ids=['source_0'])
    built, _ = h.build(ref)
    report = h.report(built, evidence)
    assert report['technical_pass'] and report['committable'], report
    result = h.commit(built, report)
    assert result['head_seq'] == 1
    after = h.view()['snapshot']
    assert len(after['objects']) == len(h.scene['objects']) + 3
    for old, current in zip(h.scene['objects'], after['objects']):
        for field in ('instance_id', 'kind', 'category', 'transform', 'local_bounds_m', 'support_id'):
            assert old[field] == current[field]
        if old['instance_id'] not in h.replace_ids:
            assert current == old
    assert after['objects'][-3:] == h.additions
    assert after['capture_revision'] == h.scene['capture_revision']
    assert after['geometry_revision'] == h.scene['geometry_revision']
    assert file_hash(h.capture_path) == h.capture_hash
    assert h.commit(built, report) == result
    assert_code('STALE_REVISION', lambda: h.prepare(h.request('another_request')))
    assert_code('STALE_REVISION', lambda: h.evidence(ref, frame_ids=['source_0']))


def test_tampered_new_asset_prevents_commit_after_validation(h):
    ref = h.prepare()
    evidence = h.evidence(ref, frame_ids=['source_0'])
    built, _ = h.build(ref)
    report = h.report(built, evidence)
    path = h.host.resolver.resolve(h.additions[-1]['asset']['artifact'], scene_id=h.scene_id)
    path.write_bytes(b'explicit tampered fixture asset')
    assert_code('STALE_ARTIFACT', lambda: h.commit(built, report))
    assert h.view()['snapshot'] == h.scene


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Existing Blender unavailable')
def test_real_cpu_blender_build_extension_then_normal_validate_commit(h):
    ref = h.prepare()
    evidence = h.evidence(ref, frame_ids=['source_0', 'source_1'])
    built, output = h.build(ref, real_blender=True)
    assert any(row['role'] == 'reopen_report' for row in output['artifact_refs'])
    report = h.report(built, evidence)
    assert report['technical_pass'] and report['committable'], report
    assert h.commit(built, report)['head_seq'] == 1
