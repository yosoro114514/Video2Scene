"""Explicit, reversible coordinate and pixel transforms for the v1 capture boundary."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from video2scene.contracts_v1 import ContractError

# Blender +Z becomes scene +Y; Blender +Y becomes scene -Z (right handed).
BLENDER_TO_SCENE = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                             [0., -1., 0., 0.], [0., 0., 0., 1.]])
SCENE_TO_BLENDER = BLENDER_TO_SCENE.T
OPENCV_TO_BLENDER_CAMERA = np.diag([1., -1., -1., 1.])


def affine(matrix):
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all() or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise ContractError('FRAME_MISMATCH', 'Expected a finite affine 4x4 matrix')
    return value


def rigid(matrix):
    value = affine(matrix)
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ContractError('FRAME_MISMATCH', 'Camera pose must have a right-handed rigid rotation')
    return value


def transform_points(matrix, points):
    value = np.asarray(points, dtype=float)
    return (affine(matrix) @ np.c_[value.reshape(-1, 3), np.ones(value.size // 3)].T).T[:, :3].reshape(value.shape)


def bounds_corners(bounds):
    low, high = (bounds['min'], bounds['max']) if isinstance(bounds, dict) else bounds
    low, high = np.asarray(low, float), np.asarray(high, float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ContractError('INVALID_ARGUMENT', 'Invalid local bounds')
    return np.array(list(itertools.product(*zip(low, high))))


def convert_bounds(bounds, matrix=BLENDER_TO_SCENE):
    points = transform_points(matrix, bounds_corners(bounds))
    return {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()}


def matrix_to_transform(matrix):
    matrix = affine(matrix)
    scales = np.linalg.norm(matrix[:3, :3], axis=0)
    if np.any(scales <= 0):
        raise ContractError('INVALID_ARGUMENT', 'Non-positive object scale')
    rotation = matrix[:3, :3] / scales
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ContractError('FRAME_MISMATCH', 'Shear or reflection is not supported by the object contract')
    return {'translation_m': matrix[:3, 3].tolist(),
            'rotation_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(), 'scale': scales.tolist()}


def transform_to_matrix(transform):
    matrix = np.eye(4)
    quaternion = np.asarray(transform['rotation_xyzw'], dtype=float)
    if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6):
        raise ContractError('INVALID_ARGUMENT', 'Quaternion must be normalized')
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix() @ np.diag(transform['scale'])
    matrix[:3, 3] = transform['translation_m']
    return affine(matrix)


def legacy_object_matrix(obj, table_rotation):
    """Mirror the original compiler, without importing Blender or mutating its scene."""
    base = affine(obj['base_matrix']).copy()
    base[:3, 3] += np.asarray(table_rotation) @ np.array([*obj['offset_m'], 0.])
    yaw = np.eye(4)
    yaw[:3, :3] = Rotation.from_euler('z', obj['yaw_deg'], degrees=True).as_matrix()
    return base @ yaw @ np.diag([*obj['scale'], 1.])


def legacy_to_standard(obj, table_rotation):
    return matrix_to_transform(BLENDER_TO_SCENE @ legacy_object_matrix(obj, table_rotation) @ SCENE_TO_BLENDER)


def camera_to_blender(c2w):
    """Standard OpenCV camera to a Blender camera object matrix; apply exactly once."""
    return SCENE_TO_BLENDER @ rigid(c2w) @ OPENCV_TO_BLENDER_CAMERA


def intrinsics_matrix(intrinsics):
    values = np.asarray(intrinsics, float)
    if values.shape == (4,):
        fx, fy, cx, cy = values
        values = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    if values.shape != (3, 3) or not np.isfinite(values).all() or abs(np.linalg.det(values)) < 1e-12:
        raise ContractError('FRAME_MISMATCH', 'Invalid camera intrinsics')
    return values


def transformed_intrinsics(K, source_to_image):
    mapping = np.asarray(source_to_image, float)
    if mapping.shape != (3, 3) or not np.isfinite(mapping).all() or abs(np.linalg.det(mapping)) < 1e-12:
        raise ContractError('FRAME_MISMATCH', 'Invalid source-to-image mapping')
    return mapping @ intrinsics_matrix(K)


def project(points_world, K, c2w):
    points = transform_points(np.linalg.inv(rigid(c2w)), points_world).reshape(-1, 3)
    projected = points @ intrinsics_matrix(K).T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = projected[:, :2] / projected[:, 2:]
    return pixels, points[:, 2]


def unproject(pixels, depth, K, c2w):
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsics_matrix(K)).T
    rays *= np.asarray(depth).reshape(-1, 1) / rays[:, 2:]
    return transform_points(rigid(c2w), rays)
