"""Read-only import of an existing whitebox-loop revision into independent v1 storage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video2scene.contracts_v1 import ContractError, canonical_hash, validate

from .coordinates import (
    BLENDER_TO_SCENE,
    convert_bounds,
    intrinsics_matrix,
    legacy_to_standard,
    rigid,
)
from .service import register_capture
from .storage import copy_artifact, file_hash, immutable_json, read_json

FIELDS = ('category', 'transform.translation_m', 'transform.rotation_xyzw', 'transform.scale',
          'support_id', 'asset.geometry', 'asset.materials')
LEGACY_LOCKS = {'offset_m': 'transform.translation_m', 'yaw_deg': 'transform.rotation_xyzw',
                'scale': 'transform.scale', 'support_id': 'support_id'}


def migrate_prototype(source_root, storage_root, scene_id, revision='r005', *, preview_fps=10.):
    """Return a SceneDraft; never creates a head or rewrites original prototype files.

    Prototype cameras are already OpenCV-to-Y-up. Only object/native asset axes
    need conversion. ``preview_fps`` describes playback, not acquisition time.
    """
    source_root, storage_root = Path(source_root).resolve(), Path(storage_root).resolve()
    if storage_root == source_root or storage_root.is_relative_to(source_root):
        raise ContractError('FORBIDDEN_SCOPE', 'Migration output must be separate from the original prototype')
    if not np.isfinite(preview_fps) or preview_fps <= 0:
        raise ContractError('INVALID_ARGUMENT', 'Preview FPS must be positive')
    scene_file = source_root / 'revisions' / revision / 'scene.json'
    camera_file = source_root / 'cameras.json'
    scene, cameras = read_json(scene_file), read_json(camera_file)
    if scene.get('schema_version') != 'whitebox-loop/1':
        raise ContractError('UNSUPPORTED_CONTRACT', 'Expected whitebox-loop/1 source')
    table_rotation = np.asarray(scene['objects']['table_00']['base_matrix'])[:3, :3]
    source_hashes = {str(scene_file.relative_to(source_root)): file_hash(scene_file), 'cameras.json': file_hash(camera_file)}
    objects = []
    for identity, obj in scene['objects'].items():
        asset_file = source_root / obj.get('asset_file', 'assets/assets.blend')
        if not asset_file.resolve().is_relative_to(source_root):
            raise ContractError('FORBIDDEN_SCOPE', 'Prototype asset path escapes the source root')
        artifact = copy_artifact(asset_file, storage_root, scene_id, 'object_geometry', relative_folder='imports/assets')
        expected = obj.get('asset_sha256')
        if expected and expected != artifact['sha256']:
            raise ContractError('STALE_ARTIFACT', 'Prototype asset differs from its recorded hash', details={'instance_id': identity})
        source_hashes[str(asset_file.relative_to(source_root))] = artifact['sha256']
        locked_fields = {LEGACY_LOCKS[key] for key in obj['locks'] if key in LEGACY_LOCKS}
        field_state = {field: {'source': 'derived' if field.startswith('transform.') else 'model',
                               'locked': field in locked_fields, 'decision_ref': None} for field in FIELDS}
        objects.append({'instance_id': identity, 'kind': 'support' if identity == 'table_00' else 'object',
                        'category': obj['semantic_class'], 'transform': legacy_to_standard(obj, table_rotation),
                        'local_bounds_m': convert_bounds(obj['local_bounds']), 'support_id': obj['support_id'],
                        'asset': {'asset_version': 'import-' + artifact['sha256'][:24], 'artifact': artifact,
                                  'native_frame': 'blender_z_up', 'native_to_object': BLENDER_TO_SCENE.tolist()},
                        'field_state': field_state, 'evidence_ids': [], 'geometry_source': 'mixed'})
    frames, skipped = [], []
    seen = set()
    for camera in cameras:
        frame_id = int(camera['frame'])
        if frame_id in seen:
            raise ContractError('FRAME_MISMATCH', 'Duplicate source camera frame ID')
        seen.add(frame_id)
        source = source_root / 'source_frames' / f'{frame_id:03d}.png'
        if not source.is_file():
            skipped.append(frame_id)
            continue
        width, height = camera['resolution']
        image = cv2.imread(str(source))
        if image is None or image.shape[:2] != (height, width):
            raise ContractError('FRAME_MISMATCH', 'Prototype source image size differs from camera')
        artifact = copy_artifact(source, storage_root, scene_id, 'source_rgb')
        source_hashes[str(source.relative_to(source_root))] = artifact['sha256']
        frames.append({'source_frame_id': f'frame-{frame_id:06d}', 'source_time_s': None,
                       'preview_time_s': frame_id / preview_fps, 'width': width, 'height': height,
                       'rgb': artifact, 'mask': None, 'K': intrinsics_matrix(camera['intrinsics']).tolist(),
                       'c2w': rigid(camera['c2w']).tolist(), 'source_to_image': np.eye(3).tolist(),
                       'selection_reason': 'Existing prototype RGB and camera share the same stable source frame ID.'})
    capture_revision = 'cap-' + canonical_hash(frames)[:24]
    geometry_revision = 'geom-' + canonical_hash({'scene_hash': source_hashes[str(scene_file.relative_to(source_root))],
                                                'camera_hash': source_hashes['cameras.json']})[:24]
    draft = {'contract_version': 'video2scene/1.0', 'scene_id': scene_id,
             'capture_revision': capture_revision, 'geometry_revision': geometry_revision,
             'world_frame': 'scene_y_up', 'object_frame': 'object_y_up', 'unit': 'm',
             'scale_status': 'estimated', 'scope': 'partial_scene', 'objects': objects}
    validate(draft, 'SceneDraft')
    register_capture(storage_root, scene_id, capture_revision, geometry_revision, frames,
                     metadata={'adapter': 'whitebox-loop/1', 'source_revision': revision,
                               'camera_world': 'already scene_y_up; no additional axis conversion',
                               'source_time': 'unknown', 'preview_fps': preview_fps,
                               'skipped_camera_ids_without_rgb': skipped, 'source_hashes': source_hashes,
                               'observed_masks': False, 'depth': 'not present in prototype capture'})
    manifest = {'source_root': str(source_root), 'source_revision': revision, 'source_hashes': source_hashes,
                'scene_id': scene_id, 'capture_revision': capture_revision, 'geometry_revision': geometry_revision,
                'object_count': len(objects), 'delivered_frame_count': len(frames),
                'notes': ['Original files are read only.', 'Imported assets are prior results, not newly refined assets.',
                          'Legacy table-local offsets are evaluated before axis conversion.',
                          'No observed mask/depth is fabricated; source time remains null.']}
    immutable_json(storage_root / scene_id / 'imports' / 'prototype_manifest.json', manifest)
    immutable_json(storage_root / scene_id / 'imports' / 'scene_draft.json', draft)
    return draft
