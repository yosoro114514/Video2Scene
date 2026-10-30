import asyncio

import pytest

from video2scene.config import blender_path
from video2scene.contracts_v1 import file_hash, validate
from video2scene.demo import SCENE_ID, create_sample, create_demo_host, export_scene, request


def test_sample_is_self_contained_and_does_not_overwrite(tmp_path):
    from video2scene.api_v1.__main__ import create_host
    sample = create_sample(tmp_path)
    validate(sample, 'SceneSnapshot')
    host = create_host(tmp_path)
    asyncio.run(host.repo.import_scene(sample, host.context()))
    bundle = asyncio.run(host.evidence.select(request('select_evidence', {
        'revision': 'head', 'target_ids': ['t001'], 'purpose': 'review', 'n': 1}), host.context()))
    assert bundle['frames'][0]['source_frame_id'] == 'synthetic-000'
    for obj in sample['objects']:
        assert host.resolver.resolve(obj['asset']['artifact']).is_file()
    with pytest.raises(FileExistsError):
        create_sample(tmp_path)


@pytest.mark.skipif(not blender_path().is_file(), reason='Blender is optional for CPU-only tests')
def test_edit_persists_other_object_and_exports_saved_revision(tmp_path):
    host = create_demo_host(tmp_path)
    try:
        before = asyncio.run(host.repo.inspect(request(), host.context()))
        cup = next(obj for obj in before['snapshot']['objects'] if obj['instance_id'] == 't001')
        position = cup['transform']['translation_m'].copy();position[0] += .05
        command = {'contract_version': 'video2scene/1.0', 'command_id': 'demo_move', 'session_id': 'demo_browser',
                   'scene_id': SCENE_ID, 'base_revision': before['head']['revision'], 'base_head_seq': before['head']['head_seq'],
                   'selected_ids': ['t001'], 'intent': 'design_edit', 'kind': 'patch',
                   'edits': [{'instance_id': 't001', 'field': 'transform.translation_m', 'value': position}]}
        host.store.accept_command(command)
        asyncio.run(host.run_command(command))
        status = host.store.command(SCENE_ID, 'demo_move')
        assert status['state'] == 'succeeded', status
        after = asyncio.run(host.repo.inspect(request(), host.context()))
        assert after['head']['head_seq'] == 1
        assert after['snapshot']['objects'][0] == before['snapshot']['objects'][0]
        record = asyncio.run(export_scene(host))
        assert record['head'] == after['head']
        assert {f['role'] for f in record['files']} == {'scene_blend', 'scene_glb'}
        for ref in record['files']:
            stored = host.store.artifact(ref['uri'])
            assert file_hash(host.resolver.resolve(stored)) == ref['sha256']
        assert asyncio.run(host.repo.inspect(request(), host.context()))['head'] == after['head']
    finally:
        host.jobs.shutdown(wait=True)
    resumed = create_demo_host(tmp_path)
    try:
        assert asyncio.run(resumed.repo.inspect(request(), resumed.context()))['head'] == after['head']
    finally:
        resumed.jobs.shutdown(wait=True)
