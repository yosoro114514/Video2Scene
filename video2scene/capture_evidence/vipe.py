"""Read existing ViPE outputs by stable frame ID; no inference or GT inputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video2scene.contracts_v1 import ContractError, canonical_hash

from .coordinates import intrinsics_matrix, rigid, transformed_intrinsics
from .service import register_capture
from .storage import copy_artifact, file_hash


def warp_image_and_camera(rgb, K, source_to_image, output_size, mask=None):
    """Apply the same pixel map to RGB, nearest-neighbour mask, and K."""
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ContractError('INVALID_ARGUMENT', 'Output image dimensions must be positive')
    mapping = np.asarray(source_to_image, float)
    updated_K = transformed_intrinsics(K, mapping)
    image = cv2.warpPerspective(rgb, mapping, (width, height), flags=cv2.INTER_LINEAR)
    transformed_mask = None
    if mask is not None:
        if mask.shape[:2] != rgb.shape[:2]:
            raise ContractError('FRAME_MISMATCH', 'Source mask and RGB dimensions differ')
        transformed_mask = cv2.warpPerspective(mask, mapping, (width, height), flags=cv2.INTER_NEAREST)
    return image, updated_K, transformed_mask


def import_vipe_capture(artifacts, storage_root, scene_id, geometry_revision, *, scene_from_vipe,
                        stride=1, max_frames=12, scale=1., masks=None, source_times=None):
    """Import an existing ViPEArtifacts instance with an explicit world alignment.

    Estimated depth stays in internal capture metadata; EvidenceFrame has no depth
    field in v1. Intrinsics from ViPEArtifacts already match delivered RGB/depth.
    """
    from video2scene.artifacts import ViPEArtifacts
    if not isinstance(artifacts, ViPEArtifacts):
        artifacts = ViPEArtifacts(artifacts)
    scene_from_vipe = rigid(scene_from_vipe)
    if max_frames < 1:
        raise ContractError('INVALID_ARGUMENT', 'max_frames must be positive')
    root = Path(storage_root) / scene_id / 'capture' / 'vipe_export'
    root.mkdir(parents=True, exist_ok=True)
    reader = cv2.VideoCapture(str(artifacts.rgb_path))
    try:
        source_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(reader.get(cv2.CAP_PROP_FPS))
    finally:
        reader.release()
    if source_width < 1 or source_height < 1 or not np.isfinite(fps) or fps <= 0:
        raise ContractError('FRAME_MISMATCH', 'Cannot establish the RGB preview dimensions/clock')
    frames, depth_records = [], []
    for frame in artifacts.frames(stride=stride, scale=scale):
        height, width = frame.rgb.shape[:2]
        path = root / f'frame-{frame.index:06d}.png'
        if path.exists():
            existing = cv2.imread(str(path))
            if existing is None or not np.array_equal(existing, cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise ContractError('EVIDENCE_MISMATCH', 'ViPE extraction cannot overwrite an existing frame')
        elif not cv2.imwrite(str(path), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
            raise ContractError('INTERNAL_ERROR', 'Failed to write ViPE RGB frame')
        rgb = copy_artifact(path, storage_root, scene_id, 'source_rgb')
        mask_ref = None
        if masks and frame.index in masks:
            mask_path = Path(masks[frame.index])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape[:2] != (height, width):
                raise ContractError('FRAME_MISMATCH', 'ViPE mask is not aligned to delivered RGB')
            mask_ref = copy_artifact(mask_path, storage_root, scene_id, 'observed_mask')
        depth_path = root / f'depth-{frame.index:06d}.npy'
        if depth_path.exists():
            if not np.array_equal(np.load(depth_path, allow_pickle=False), frame.depth, equal_nan=True):
                raise ContractError('EVIDENCE_MISMATCH', 'ViPE extraction cannot overwrite an existing depth')
        else:
            np.save(depth_path, frame.depth, allow_pickle=False)
        depth_records.append({'source_frame_id': f'frame-{frame.index:06d}', 'source': 'ViPE estimated depth',
                              'artifact': copy_artifact(depth_path, storage_root, scene_id, 'estimated_depth')})
        frames.append({'source_frame_id': f'frame-{frame.index:06d}',
                       'source_time_s': (source_times or {}).get(frame.index), 'preview_time_s': frame.index / fps,
                       'width': width, 'height': height, 'rgb': rgb, 'mask': mask_ref,
                       'K': intrinsics_matrix(frame.intrinsics).tolist(),
                       'c2w': rigid(scene_from_vipe @ frame.c2w).tolist(),
                       'source_to_image': np.diag([width / source_width, height / source_height, 1.]).tolist(),
                       'selection_reason': 'ViPE RGB/pose/intrinsics/depth joined by stable source frame index.'})
        if len(frames) >= max_frames:
            break
    capture_revision = 'cap-' + canonical_hash(frames)[:24]
    return register_capture(storage_root, scene_id, capture_revision, geometry_revision, frames,
                            metadata={'adapter': 'ViPEArtifacts', 'sequence': artifacts.sequence,
                                      'rgb_video_sha256': file_hash(artifacts.rgb_path),
                                      'scene_from_vipe': scene_from_vipe.tolist(), 'scale_status': 'estimated',
                                      'scale_multiplier': scale, 'depth': depth_records,
                                      'source_time': 'unknown unless supplied explicitly', 'preview_fps': fps})
