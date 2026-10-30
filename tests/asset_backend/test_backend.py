import asyncio
import copy
import json
import threading

import pytest

import video2scene.asset_backend.backend as implementation
from video2scene.asset_backend import BlenderAssetBackend
from video2scene.contracts_v1 import (
    ContractError,
    ExecutionContext,
    canonical_hash,
    file_hash,
    validate,
)
from video2scene.contracts_v1.mocks import example


def context(root):
    return ExecutionContext(storage_root=root, actor='user', decision_ref='authorized',
                            allowed_object_ids=frozenset({'cup-1'}),
                            allowed_fields=frozenset({'asset.geometry', 'asset.materials'}))


def candidate():
    return copy.deepcopy(example('model_CandidateSnapshot'))


def test_B09_changed_input_asset_rejected_before_build(tmp_path):
    snapshot = candidate()
    obj = snapshot['scene']['objects'][0]
    scene_id = snapshot['scene']['scene_id']
    asset_path = tmp_path / scene_id / 'asset.blend'
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b'previous valid bytes')
    obj['asset']['artifact'] = {'uri': f'artifact://{scene_id}/asset.blend', 'sha256': file_hash(asset_path),
                               'media_type': 'application/x-blender', 'role': 'object_geometry'}
    asset_path.write_bytes(b'changed bytes after source edit')
    with pytest.raises(ContractError) as error:
        asyncio.run(BlenderAssetBackend(tmp_path).build(snapshot, context(tmp_path)))
    assert error.value.code == 'STALE_ARTIFACT'
    assert not (tmp_path / scene_id / 'assets').exists()


def test_box_collision_contract_and_explicit_mesh_unsupported(tmp_path):
    snapshot = candidate()
    backend = BlenderAssetBackend(tmp_path)
    result = asyncio.run(backend.collisions(snapshot, 'box', context(tmp_path)))
    validate(result, 'CollisionReport')
    assert result['method'] == 'box'
    with pytest.raises(ContractError) as error:
        asyncio.run(backend.collisions(snapshot, 'mesh', context(tmp_path)))
    assert error.value.code == 'TOOL_UNAVAILABLE'


@pytest.mark.skipif(not implementation.DEFAULT_BLENDER.is_file(), reason='Configured local Blender unavailable')
def test_B09_real_failed_blender_never_reuses_old_glb(tmp_path, monkeypatch):
    old = tmp_path / 'fault-scene/assets/previous/scene.glb'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'previous GLB marker must not be returned')
    old_hash = file_hash(old)
    script = tmp_path / 'fault_worker.py'
    script.write_text("raise RuntimeError('Injected fresh build failure before export')", encoding='utf-8')
    monkeypatch.setattr(implementation, 'WORKER', script)
    with pytest.raises(ContractError) as error:
        asyncio.run(BlenderAssetBackend(tmp_path)._run('build', 'fault-scene',
                    {'input_fingerprint': canonical_hash({'fault': True})}, context(tmp_path)))
    assert error.value.code == 'BUILD_FAILED'
    assert file_hash(old) == old_hash
    runs = list((tmp_path / 'fault-scene/assets').glob('build-*'))
    assert len(runs) == 1 and not (runs[0] / 'scene.glb').exists()
    record = json.loads((runs[0] / 'execution.json').read_text())
    assert record['returncode'] != 0
    assert record['state'] == 'failed'


@pytest.mark.skipif(not implementation.DEFAULT_BLENDER.is_file(), reason='Configured local Blender unavailable')
def test_cancel_waits_for_own_blender_process_to_stop(tmp_path, monkeypatch):
    script = tmp_path / 'slow_worker.py'
    script.write_text('import time\ntime.sleep(30)\n', encoding='utf-8')
    monkeypatch.setattr(implementation, 'WORKER', script)
    ctx = context(tmp_path)
    timer = threading.Timer(.5, ctx.cancel_event.set)
    timer.start()
    try:
        with pytest.raises(ContractError) as error:
            asyncio.run(BlenderAssetBackend(tmp_path)._run('build', 'cancel-scene',
                        {'input_fingerprint': canonical_hash({'cancel': True})}, ctx))
        assert error.value.code == 'CANCELLED'
    finally:
        timer.cancel()
        timer.join()
    record = json.loads(next((tmp_path / 'cancel-scene/assets').glob('*/execution.json')).read_text())
    assert record['state'] == 'cancelled'
    assert record['returncode'] is not None
