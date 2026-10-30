"""Material-only guards; local CPU Blender fixtures, never a model invocation."""
import copy
from pathlib import Path

import pytest

from video2scene.agent_scene.direct_asset import DirectAssetBackend, write
from video2scene.agent_scene.material_refinement import (
    claim_material_run, fingerprint, provider_receipt, require_same_geometry,
)
from video2scene.asset_backend.backend import DEFAULT_BLENDER
from video2scene.contracts_v1 import ContractError, ExecutionContext, canonical_hash, file_hash


def report(geometry):
    return {'geometry': geometry, 'geometry_sha256': canonical_hash(geometry)}


@pytest.mark.parametrize('field,value', [('vertices', [[.1, 0., 0.]]), ('faces', [[1, 0, 2]]),
                                      ('transform', [1., 0., 0.]), ('bounds', [[-1., -1., -1.], [2., 1., 1.]])])
def test_geometry_changes_fail_even_with_valid_recomputed_hash(field, value):
    before = {'vertices': [[0., 0., 0.]], 'faces': [[0, 1, 2]], 'transform': [0., 0., 0.],
              'bounds': [[-1., -1., -1.], [1., 1., 1.]]}
    after = dict(before, **{field: value})
    with pytest.raises(ContractError) as error:
        require_same_geometry(report(before), report(after))
    assert error.value.code == 'FORBIDDEN_SCOPE'


def test_geometry_report_cannot_forge_unchanged_hash():
    before = report({'vertices': [[0., 0., 0.]]})
    after = copy.deepcopy(before)
    after['geometry']['vertices'][0][0] = 1.
    with pytest.raises(ContractError) as error:
        require_same_geometry(before, after)
    assert error.value.code == 'STALE_ARTIFACT'


def test_existing_material_attempt_cannot_be_replayed(tmp_path):
    root = claim_material_run(tmp_path)
    write(root / 'invocation_started.json', {'explicit_local_fixture': True})
    with pytest.raises(ContractError) as error:
        claim_material_run(tmp_path)
    assert error.value.code == 'IDEMPOTENCY_CONFLICT'
    assert (root / 'invocation_started.json').is_file()


def make_receipt(tmp_path):
    response = {'status': 'accept', 'explicit_local_fixture': True}
    write(tmp_path / 'response.json', response)
    write(tmp_path / 'request.json', {'backend': 'codex-cli', 'status': 'complete', 'requested_model': 'fixture-model',
                                    'response_sha256': file_hash(tmp_path / 'response.json')})
    write(tmp_path / 'invocation_started.json', {'origin': 'codex_image', 'explicit_local_fixture': True})
    write(tmp_path / 'invocation_outcome.json', {'origin': 'codex_image', 'status': 'complete',
                                               'response_hash': canonical_hash(response)})
    (tmp_path / 'events.jsonl').write_text('{"explicit_local_fixture":true}\n', encoding='utf-8')


def test_provider_receipt_binds_real_directory_and_all_files(tmp_path):
    make_receipt(tmp_path)
    receipt = provider_receipt(tmp_path, 'fixture-model')
    assert receipt['directory'] == str(tmp_path.resolve())
    assert len(receipt['files']) == 5


@pytest.mark.parametrize('change', ['response', 'model', 'outcome'])
def test_provider_receipt_rejects_hash_model_or_uncertain_outcome(tmp_path, change):
    make_receipt(tmp_path)
    if change == 'response':
        write(tmp_path / 'response.json', {'status': 'accept', 'tampered_fixture': True})
    elif change == 'outcome':
        write(tmp_path / 'invocation_outcome.json', {'origin': 'codex_image', 'status': 'failed_or_uncertain'})
    with pytest.raises(ContractError) as error:
        provider_receipt(tmp_path, 'different-model' if change == 'model' else 'fixture-model')
    assert error.value.code == 'STALE_ARTIFACT'


@pytest.mark.skipif(not DEFAULT_BLENDER.is_file(), reason='Existing Blender unavailable')
def test_cpu_blender_material_edit_preserves_geometry_and_vertex_edit_fails(tmp_path):
    backend = DirectAssetBackend(tmp_path, confirmation_path=tmp_path / 'unused-confirmation.json')
    context = ExecutionContext(storage_root=tmp_path)
    script = tmp_path / 'fixture_create.py'
    input_path, material_path, changed_path = (tmp_path / name for name in ('input.blend', 'material.blend', 'changed.blend'))
    script.write_text(
        'import bpy\n'
        'bpy.ops.wm.read_factory_settings(use_empty=True)\n'
        'bpy.ops.mesh.primitive_cube_add(size=1)\n'
        'ob=bpy.context.object\n'
        'ob.location=(1.,2.,3.)\n'
        'ob.rotation_euler=(.1,.3,.2)\n'
        'ob.scale=(1.,2.,.5)\n'
        'bevel=ob.modifiers.new("Geometry fixture bevel","BEVEL")\n'
        'bevel.width=.03\n'
        'bevel.segments=2\n'
        'material=bpy.data.materials.new("Fixture material")\n'
        'material.use_nodes=True\n'
        'ob.data.materials.append(material)\n'
        f'bpy.ops.wm.save_as_mainfile(filepath={str(input_path)!r})\n'
        'material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value=(.01,.02,.03,1.)\n'
        f'bpy.ops.wm.save_as_mainfile(filepath={str(material_path)!r})\n'
        'ob.data.vertices[0].co.x += .1\n'
        f'bpy.ops.wm.save_as_mainfile(filepath={str(changed_path)!r})\n', encoding='utf-8')
    backend._process([str(DEFAULT_BLENDER), '--background', '--factory-startup', '--threads', '2',
                      '--python-exit-code', '1', '--python', str(script)], tmp_path, context, 45, log_name='fixture.log')
    before = fingerprint(backend, input_path, tmp_path / 'before', context, 45)
    material = fingerprint(backend, material_path, tmp_path / 'material', context, 45)
    changed = fingerprint(backend, changed_path, tmp_path / 'changed', context, 45)
    assert before['blend_sha256'] != material['blend_sha256']
    require_same_geometry(before, material)
    with pytest.raises(ContractError) as error:
        require_same_geometry(before, changed)
    assert error.value.code == 'FORBIDDEN_SCOPE'
