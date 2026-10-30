import asyncio
import copy
import json

import numpy as np
import pytest
import trimesh

from video2scene.asset_backend import ObservedSurfaceBackend
from video2scene.asset_backend.backend import DEFAULT_BLENDER
from video2scene.asset_backend.observed_surface import crop_observed_mesh
from video2scene.capture_evidence import transform_points, transform_to_matrix
from video2scene.contracts_v1 import ContractError, ExecutionContext, canonical_hash, file_hash
from video2scene.contracts_v1.mocks import example


def setup_source(tmp_path):
    scene = copy.deepcopy(example('model_SceneSnapshot'))
    target = scene['objects'][0]
    scene['objects'] = [target]
    target['support_id'] = None
    source = tmp_path / 'source'; source.mkdir()
    bounds = target['local_bounds_m']
    lo, hi = np.asarray(bounds['min']), np.asarray(bounds['max'])
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(np.min(hi-lo)) * .3)
    mesh.vertices += (lo+hi)/2
    mesh.vertices = transform_points(transform_to_matrix(target['transform']), mesh.vertices)
    mesh.visual.vertex_colors = np.tile([65, 132, 201, 255], (len(mesh.vertices), 1))
    mesh.export(source / 'scan.glb')
    (source / 'scene.json').write_text(json.dumps({'coordinate_system': 'right-handed Y-up',
        'objects': [{'id': target['instance_id']}]}))
    (source / 'snapshot.json').write_text(json.dumps(scene))
    root = tmp_path / 'runtime'
    backend = ObservedSurfaceBackend(root, source/'scan.glb', source/'scene.json', source/'snapshot.json', samples=1)
    evidence = copy.deepcopy(example('model_EvidenceBundle'))
    for key in ('scene_id', 'capture_revision', 'geometry_revision'):
        evidence[key] = scene[key]
    evidence['scene_revision'] = scene['revision']; evidence['target_ids'] = [target['instance_id']]
    for frame in evidence['frames']:
        path = root / scene['scene_id'] / (frame['source_frame_id'] + '.png')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'offline RGB fixture; not decoded in asset build')
        frame['rgb'] = backend._artifact(path, scene['scene_id'], 'evidence_rgb', 'image/png')
    evidence['evidence_hash'] = canonical_hash({k:v for k,v in evidence.items() if k != 'evidence_hash'})
    candidate = copy.deepcopy(example('model_CandidateSnapshot')['candidate'])
    candidate.update(scene_id=scene['scene_id'], base_revision=scene['revision'], candidate_hash=canonical_hash(scene))
    context = ExecutionContext(storage_root=root, actor='user', decision_ref='observed-cpu-test',
        allowed_object_ids=frozenset([target['instance_id']]),
        allowed_fields=frozenset(['asset.geometry', 'asset.materials']))
    return backend, scene, target, evidence, candidate, context, source


def test_crop_preserves_actual_vertices_colors_and_original_pose(tmp_path):
    backend, scene, target, evidence, candidate, ctx, source = setup_source(tmp_path)
    payload = backend._observed_payload(target, scene, evidence, candidate, ctx)
    moved = copy.deepcopy(target); moved['transform']['translation_m'][0] += 7
    again = backend._observed_payload(moved, scene, evidence, candidate, ctx)
    assert payload['observed_mesh'] == again['observed_mesh']
    assert np.asarray(payload['observed_mesh']['colors_rgba']).tolist() == [[65,132,201,255]] * len(payload['observed_mesh']['vertices'])
    assert payload['observed_record']['source_object_transform'] != again['observed_record']['current_user_transform']
    assert not payload['observed_mesh']['stats']['geometry_rescaled']


def test_no_observed_faces_fails_without_template(tmp_path):
    backend, scene, target, evidence, candidate, ctx, source = setup_source(tmp_path)
    mesh = trimesh.load(source/'scan.glb', force='mesh', process=False)
    mesh.vertices += 100
    with pytest.raises(ContractError, match='Insufficient observed triangles'):
        crop_observed_mesh(mesh, target)


def test_source_hash_and_geometry_revision_are_bound(tmp_path):
    backend, scene, target, evidence, candidate, ctx, source = setup_source(tmp_path)
    changed = copy.deepcopy(scene); changed['geometry_revision'] = 'another-geometry'
    with pytest.raises(ContractError) as error:
        backend._observed_payload(target, changed, evidence, candidate, ctx)
    assert error.value.code == 'EVIDENCE_MISMATCH'
    with (source/'scan.glb').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ContractError) as error:
        backend._observed_payload(target, scene, evidence, candidate, ctx)
    assert error.value.code == 'STALE_ARTIFACT'


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Configured Blender unavailable')
def test_real_partial_asset_containment_reopen_and_outside_rejection(tmp_path):
    backend, scene, target, evidence, candidate, ctx, source = setup_source(tmp_path)
    spec = {'candidate': candidate, 'target_id': target['instance_id'],
            'allowed_fields': ['asset.geometry','asset.materials'],
            'constraints': {'keep_root_transform':True,'keep_support':True,'keep_bounds':True},
            'budget': {'timeout_s':120,'max_model_calls':0,'max_iterations':1}}
    result = asyncio.run(backend.refine(spec, scene, evidence, ctx))
    update = result['asset_updates'][0]
    assert update['local_bounds_m'] == target['local_bounds_m']
    assert update['asset']['asset_version'].startswith('observed-partial-')
    reopen_ref = next(ref for ref in result['artifact_refs'] if ref['role'] == 'reopen_report')
    report = json.loads(backend.resolver.resolve(reopen_ref).read_text())
    row = report['objects'][target['instance_id']]
    assert report['passed'] and row['partial_observed']
    assert row['declared_local_bounds_m'] == target['local_bounds_m']
    assert row['local_bounds_m'] != row['declared_local_bounds_m']
    payload = backend._observed_payload(target, scene, evidence, candidate, ctx)
    payload['observed_mesh']['vertices'][0][0] += 100
    payload['observed_record']['cropped_mesh_hash'] = canonical_hash(payload['observed_mesh'])
    payload.update(target=target, evidence=evidence, candidate=candidate, input_fingerprint=candidate['candidate_hash'])
    with pytest.raises(ContractError) as error:
        asyncio.run(backend._run('refine_observed', scene['scene_id'], payload, ctx))
    assert error.value.code == 'BUILD_FAILED'
