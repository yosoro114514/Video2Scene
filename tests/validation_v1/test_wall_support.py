"""Wall attachment must permit mounted objects and connected lintels, not floating ones."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video2scene.validation_v1.validator import _DEFAULT_CRITERIA, _geometry_issues


def object_at(identity, center, size, rotation, *, kind='object', support=None):
    half = np.asarray(size) / 2
    return {'instance_id': identity, 'kind': kind, 'support_id': support,
            'local_bounds_m': {'min': (-half).tolist(), 'max': half.tolist()},
            'transform': {'translation_m': rotation.apply(center).tolist(),
                          'rotation_xyzw': rotation.as_quat().tolist(), 'scale': [1., 1., 1.]}}


def issues(*objects):
    return _geometry_issues({'objects': list(objects)}, _DEFAULT_CRITERIA[0])


@pytest.mark.parametrize('yaw', [0, -12, 78])
def test_wall_mounted_tv_and_edge_connected_lintel(yaw):
    rot = Rotation.from_euler('y', yaw, degrees=True)
    wall = object_at('wall', [0, 1.25, 0], [3, 2.5, .12], rot, kind='wall')
    tv = object_at('tv', [0, 1.08, .095], [1.05, .68, .07], rot, support='wall')
    lintel = object_at('lintel', [1.95, 2.28, 0], [.9, .44, .12], rot, kind='wall', support='wall')
    assert issues(wall, tv, lintel) == {}


@pytest.mark.parametrize('center', [[0, 1.08, .3], [2.5, 1.08, .095], [0, 3.2, .095]])
def test_wall_support_cannot_hide_detached_or_outside_tv(center):
    rot = Rotation.from_euler('y', -12, degrees=True)
    wall = object_at('wall', [0, 1.25, 0], [3, 2.5, .12], rot, kind='wall')
    tv = object_at('tv', center, [1.05, .68, .07], rot, support='wall')
    assert set(issues(wall, tv)) == {'support:tv:wall'}


def test_wall_support_rejects_deep_embed_and_disconnected_lintel():
    rot = Rotation.from_euler('y', 78, degrees=True)
    wall = object_at('wall', [0, 1.25, 0], [3, 2.5, .12], rot, kind='wall')
    embedded = object_at('embedded', [0, 1.08, 0], [.5, .4, .1], rot, support='wall')
    lintel = object_at('lintel', [2.1, 2.28, 0], [.9, .44, .12], rot, kind='wall', support='wall')
    assert set(issues(wall, embedded)) == {'support:embedded:wall'}
    assert set(issues(wall, lintel)) == {'support:lintel:wall'}


def test_floor_support_still_requires_top_contact_not_edge_contact():
    rot = Rotation.identity()
    floor = object_at('floor', [0, -.06, 0], [5, .12, 5], rot, kind='floor')
    grounded = object_at('box', [0, .5, 0], [1, 1, 1], rot, support='floor')
    assert issues(floor, grounded) == {}
    side = object_at('box', [3, -.5, 0], [1, 1, 1], rot, support='floor')
    assert set(issues(floor, side)) == {'support:box:floor'}
