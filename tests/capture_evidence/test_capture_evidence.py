import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from video2scene.capture_evidence import (
    BLENDER_TO_SCENE,
    SCENE_TO_BLENDER,
    EvidenceService,
    bounds_corners,
    camera_to_blender,
    check_frame,
    legacy_object_matrix,
    migrate_prototype,
    project,
    transform_points,
    transform_to_matrix,
    transformed_intrinsics,
    unproject,
    warp_image_and_camera,
)
from video2scene.capture_evidence.storage import file_hash, read_json, resolve_artifact
from video2scene.contracts_v1 import ContractError, validate

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/uw_whitebox_loop_20261028'


@pytest.fixture(scope='module')
def migrated(tmp_path_factory):
    if not (SOURCE / 'revisions/r005/scene.json').is_file():
        pytest.skip('Actual desktop prototype not installed')
    root = tmp_path_factory.mktemp('capture-real')
    original = file_hash(SOURCE / 'revisions/r005/scene.blend')
    draft = migrate_prototype(SOURCE, root, 'capture-test')
    assert file_hash(SOURCE / 'revisions/r005/scene.blend') == original
    snapshot = dict(draft, revision='import-r005', parent_revision=None)
    validate(snapshot, 'SceneSnapshot')
    return root, draft, snapshot


class Repository:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    async def inspect(self, request, context):
        return {'head': {'revision': self.snapshot['revision'], 'head_seq': 0}, 'snapshot': copy.deepcopy(self.snapshot)}


def request(snapshot, **params):
    return {'contract_version': 'video2scene/1.0', 'request_id': 'evidence-test', 'scene_id': snapshot['scene_id'],
            'tool': 'select_evidence', 'params': {'revision': 'head', 'target_ids': ['t001'],
                                                 'purpose': 'refine', 'n': 8, **params}}


def test_B07_real_object_and_camera_roundtrip(migrated):
    root, draft, _ = migrated
    old = read_json(SOURCE / 'revisions/r005/scene.json')
    compiled = read_json(SOURCE / 'revisions/r005/compiled.json')
    table_rotation = np.asarray(old['objects']['table_00']['base_matrix'])[:3, :3]
    for obj in draft['objects']:
        original = old['objects'][obj['instance_id']]
        legacy_matrix = legacy_object_matrix(original, table_rotation)
        np.testing.assert_allclose(legacy_matrix, compiled['objects'][obj['instance_id']]['matrix_world'], atol=1e-7)
        standard_matrix = transform_to_matrix(obj['transform'])
        native_points = bounds_corners(original['local_bounds'])
        standard_world = transform_points(standard_matrix @ np.asarray(obj['asset']['native_to_object']), native_points)
        expected = transform_points(BLENDER_TO_SCENE @ legacy_matrix, native_points)
        np.testing.assert_allclose(standard_world, expected, atol=1e-7)
        restored = SCENE_TO_BLENDER @ standard_matrix @ BLENDER_TO_SCENE
        np.testing.assert_allclose(restored, legacy_matrix, atol=1e-7)
    capture = read_json(root / 'capture-test/captures' / f"{draft['capture_revision']}.json")
    cameras = {f"frame-{c['frame']:06d}": c for c in read_json(SOURCE / 'cameras.json')}
    for frame in capture['frames']:
        np.testing.assert_allclose(frame['c2w'], cameras[frame['source_frame_id']]['c2w'])
        matrix = np.asarray(frame['c2w'])
        np.testing.assert_allclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-7)
        local = np.array([[0., 0., 1.], [.1, -.2, 2.]])
        world = transform_points(matrix, local)
        pixels, depth = project(world, frame['K'], matrix)
        assert np.all(depth > 0)
        np.testing.assert_allclose(unproject(pixels, depth, frame['K'], matrix), world, atol=1e-10)
        blender = camera_to_blender(matrix)
        np.testing.assert_allclose(transform_points(blender, [[0, 0, -1]]),
                                   transform_points(SCENE_TO_BLENDER, world[:1]), atol=1e-10)


@pytest.mark.parametrize('mapping,size', [
    ([[.5, 0, -10], [0, .5, -8], [0, 0, 1]], (160, 120)),
    ([[0, -1, 239], [1, 0, 0], [0, 0, 1]], (240, 320)),
])
def test_B07_noncenter_crop_scale_portrait_mapping(mapping, size):
    K = np.array([[281., 0, 145.3], [0, 299., 101.7], [0, 0, 1]])
    points = np.array([[.1, .2, 2.], [.2, -.1, 1.]])
    old_pixels, _ = project(points, K, np.eye(4))
    new_pixels, _ = project(points, transformed_intrinsics(K, mapping), np.eye(4))
    projected = np.c_[old_pixels, np.ones(2)] @ np.asarray(mapping).T
    np.testing.assert_allclose(new_pixels, projected[:, :2] / projected[:, 2:])
    rgb = np.zeros((240, 320, 3), np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    mask[100:120, 140:160] = 255
    warped, updated_K, warped_mask = warp_image_and_camera(rgb, K, mapping, size, mask)
    assert warped.shape[:2] == (size[1], size[0]) == warped_mask.shape
    assert set(np.unique(warped_mask)) == {0, 255}
    np.testing.assert_allclose(updated_K, transformed_intrinsics(K, mapping))


def test_B08_real_missing_rgb_mask_depth_are_explicit(migrated):
    root, _draft, snapshot = migrated
    service = EvidenceService(root, Repository(snapshot))
    bundle = asyncio.run(service.select(request(snapshot), None))
    validate(bundle, 'EvidenceBundle')
    assert 1 <= len(bundle['frames']) <= 3
    assert all(f['source_time_s'] is None and f['mask'] is None for f in bundle['frames'])
    assert asyncio.run(service.get(bundle['evidence_id'])) == bundle
    assert asyncio.run(EvidenceService(root, Repository(snapshot)).get(bundle['evidence_id'])) == bundle


def test_B08_wrong_frame_id_is_rejected(migrated):
    root, draft, _ = migrated
    frame = read_json(root / 'capture-test/captures' / f"{draft['capture_revision']}.json")['frames'][0]
    with pytest.raises(ContractError) as error:
        check_frame(frame, root, camera_frame_id='frame-other')
    assert error.value.code == 'FRAME_MISMATCH'
    frame = copy.deepcopy(frame)
    frame['mask'] = frame['rgb']
    with pytest.raises(ContractError) as error:
        check_frame(frame, root, mask_frame_id='frame-other')
    assert error.value.code == 'FRAME_MISMATCH'


def test_B08_wrong_dimensions_rejected(migrated):
    root, draft, _ = migrated
    frame = read_json(root / 'capture-test/captures' / f"{draft['capture_revision']}.json")['frames'][0]
    frame['width'] += 1
    with pytest.raises(ContractError) as error:
        check_frame(frame, root)
    assert error.value.code == 'FRAME_MISMATCH'


def test_evidence_detects_tampering_and_wrong_geometry(migrated):
    root, _draft, snapshot = migrated
    changed = copy.deepcopy(snapshot)
    changed['geometry_revision'] = 'different'
    with pytest.raises(ContractError) as error:
        asyncio.run(EvidenceService(root, Repository(changed)).select(request(snapshot), None))
    assert error.value.code == 'EVIDENCE_MISMATCH'
    service = EvidenceService(root, Repository(snapshot))
    bundle = asyncio.run(service.select(request(snapshot, purpose='validation'), None))
    evidence_path = root / 'capture-test/evidence' / f"{bundle['evidence_id']}.json"
    original = evidence_path.read_bytes()
    changed = copy.deepcopy(bundle)
    changed['frames'][0]['K'][0][0] += 1
    evidence_path.write_text(json.dumps(changed), encoding='utf-8')
    try:
        with pytest.raises(ContractError) as error:
            asyncio.run(service.get(bundle['evidence_id']))
        assert error.value.code == 'EVIDENCE_MISMATCH'
    finally:
        evidence_path.write_bytes(original)


def test_assets_and_snapshot_import_are_idempotent(migrated):
    root, draft, _ = migrated
    assert migrate_prototype(SOURCE, root, 'capture-test') == draft
    for obj in draft['objects']:
        path = resolve_artifact(root, obj['asset']['artifact'])
        assert path.is_file() and obj['asset']['native_frame'] == 'blender_z_up'
        assert obj['field_state']['transform.translation_m']['locked']


def test_no_import_into_original_output_tree():
    with pytest.raises(ContractError) as error:
        migrate_prototype(SOURCE, SOURCE / 'forbidden', 'unsafe')
    assert error.value.code == 'FORBIDDEN_SCOPE'
