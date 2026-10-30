"""Room edits must distinguish oriented box overlap from inflated world AABBs."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video2scene.validation_v1.validator import (
    _DEFAULT_CRITERIA, _geometry_issues, _obb_penetrates, _world_bounds, _world_obb,
)


CRITERIA = _DEFAULT_CRITERIA[0]


def box(identity, center, size, rotation=None, *, kind='object', scale=(1., 1., 1.), local_center=(0., 0., 0.)):
    rotation = rotation or Rotation.identity()
    scale, size, local_center = np.asarray(scale), np.asarray(size), np.asarray(local_center)
    return {'instance_id': identity, 'kind': kind, 'support_id': None,
            'local_bounds_m': {'min': (local_center - size / 2).tolist(), 'max': (local_center + size / 2).tolist()},
            'transform': {'translation_m': (np.asarray(center) - rotation.apply(local_center * scale)).tolist(),
                          'rotation_xyzw': rotation.as_quat().tolist(), 'scale': scale.tolist()}}


def aabb_would_reject(left, right):
    alo, ahi = _world_bounds(left)
    blo, bhi = _world_bounds(right)
    overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
    smaller = min(np.prod(ahi - alo), np.prod(bhi - blo))
    return overlap.min() > CRITERIA['collision_penetration_m'] and np.prod(overlap) / smaller > CRITERIA['collision_fraction']


def issues(*objects):
    return _geometry_issues({'objects': list(objects)}, CRITERIA)


def test_rotated_thin_wall_aabb_overlaps_separate_furniture():
    wall = box('wall', [0., 1.25, 0.], [4., 2.5, .1], Rotation.from_euler('y', 45, degrees=True), kind='wall')
    table = box('coffee_table', [1., .5, 1.], [.3, .8, .3])
    assert aabb_would_reject(wall, table)
    assert not _obb_penetrates(_world_obb(wall), _world_obb(table))
    assert issues(wall, table) == {}
    assert issues(table, wall) == {}


@pytest.mark.parametrize('angles', [(0., 45., 0.), (23., -31., 47.), (89., 72., -133.)])
def test_rotated_wall_butt_join_is_contact_not_penetration(angles):
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    origin = np.array([4., -3., 2.])
    left = box('wall_left', origin + rotation.apply([-1., 0., 0.]), [2., 2.5, .2], rotation, kind='wall')
    right = box('wall_right', origin + rotation.apply([1., 0., 0.]), [2., 2.5, .2], rotation, kind='wall')
    assert not _obb_penetrates(_world_obb(left), _world_obb(right))
    assert issues(left, right) == {}


def test_perpendicular_wall_corner_butt_join_needs_no_wall_exemption():
    left = box('wall_left', [0., 1.25, 0.], [4., 2.5, .2], kind='wall')
    right = box('wall_right', [2.1, 1.25, -1.4], [3., 2.5, .2],
                Rotation.from_euler('y', 90., degrees=True), kind='wall')
    assert not _obb_penetrates(_world_obb(left), _world_obb(right))
    assert issues(left, right) == {}
    # Moving the end wall inside produces positive volume instead of contact.
    right['transform']['translation_m'] = [1.9, 1.25, 0.]
    assert _obb_penetrates(_world_obb(left), _world_obb(right))


def test_wall_wall_overlap_still_uses_existing_collision_threshold():
    left = box('wall_left', [0., 1.25, 0.], [3., 2.5, .2], kind='wall')
    right = box('wall_right', [.5, 1.25, 0.], [3., 2.5, .2], kind='wall')
    assert aabb_would_reject(left, right)
    assert set(issues(left, right)) == {'collision:wall_left:wall_right'}


@pytest.mark.parametrize('angles', [(0., 0., 0.), (23., -31., 47.), (89., 72., -133.)])
def test_true_penetration_with_scale_asymmetric_bounds_and_arbitrary_quaternion(angles):
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    center = [2., 1., -3.]
    left = box('wall', center, [4., 1., .4], rotation, kind='wall',
               scale=(.5, 2., 1.), local_center=(1., -.2, .7))
    right = box('object', center, [1., 1., 1.], rotation, scale=(.6, .6, .6), local_center=(.5, .5, .5))
    # Quaternion signs encode the same orientation; accepted serialization error
    # must not turn the SAT axes into a scaled/skewed basis.
    right['transform']['rotation_xyzw'] = (-np.asarray(right['transform']['rotation_xyzw']) * (1. + 1e-8)).tolist()
    assert aabb_would_reject(left, right)
    np.testing.assert_allclose(_world_obb(left)[0], center, atol=1e-12)
    np.testing.assert_allclose(_world_obb(left)[2], [1., 1., .2], atol=1e-12)
    assert _obb_penetrates(_world_obb(left), _world_obb(right))
    report = issues(left, right)
    assert set(report) == {'collision:wall:object'}
    assert 'AABB intersection fraction' in report['collision:wall:object']
    assert 'not mesh contact' in report['collision:wall:object']


def test_edge_cross_axis_separates_when_all_six_face_axes_overlap():
    # Skew thin rods are separated along an edge cross-product axis. Testing only
    # the six face normals would falsely report a collision for this construction.
    direction = np.ones(3) / np.sqrt(3.)
    normal = np.array([0., 1., -1.]) / np.sqrt(2.)
    tangent = np.cross(direction, normal)
    axes = np.column_stack([direction, (normal + tangent) / np.sqrt(2.), (tangent - normal) / np.sqrt(2.)])
    left = box('rod_a', [0., 0., 0.], [4., .2, .2])
    right = box('rod_b', normal * .35, [4., .2, .2], Rotation.from_matrix(axes))
    a, b = _world_obb(left), _world_obb(right)
    for axis in [a[1][:, i] for i in range(3)] + [b[1][:, i] for i in range(3)]:
        radii = np.dot(a[2], np.abs(a[1].T @ axis)) + np.dot(b[2], np.abs(b[1].T @ axis))
        assert abs(np.dot(b[0] - a[0], axis)) < radii
    assert aabb_would_reject(left, right)
    assert not _obb_penetrates(a, b)
    assert issues(left, right) == {}


@pytest.mark.parametrize('distance,penetrates', [(1.0001, False), (1., False), (.7, True)])
def test_nearly_parallel_axes_handle_gap_contact_and_overlap(distance, penetrates):
    rotation = Rotation.from_euler('xyz', [31., -22., 77.], degrees=True)
    left = box('a', [0., 0., 0.], [1., 1., 1.], rotation)
    right = box('b', rotation.apply([distance, 0., 0.]), [1., 1., 1.],
                rotation * Rotation.from_rotvec([1e-13, -2e-13, 1e-13]))
    assert _obb_penetrates(_world_obb(left), _world_obb(right)) is penetrates
    assert bool(issues(left, right)) is penetrates
