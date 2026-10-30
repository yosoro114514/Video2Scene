import numpy as np
import pytest

from video2scene.spatiallm_adapter import (
    corners,
    parse_layout,
    similarity,
    transform_camera,
    transform_points,
)


def test_similarity_camera_and_box_preserve_projection():
    matrix = similarity(2.3, 0.47, [1, -2, 3])
    inverse = np.linalg.inv(matrix)
    camera = np.eye(4)
    camera[:3, 3] = [2, 1, -0.5]
    spatial_camera = transform_camera(camera, matrix)
    assert np.linalg.det(spatial_camera[:3, :3]) == pytest.approx(1)
    local = np.array([[0.2, -0.4, 2], [-0.1, 0.2, 1]])
    world = transform_points(local, camera)
    spatial = transform_points(world, matrix)
    back = transform_points(spatial, np.linalg.inv(spatial_camera))
    # Non-central principal point, portrait/landscape and cropped/resized K.
    for fx, fy, cx, cy in [
        (450, 470, 203, 310),
        (700, 650, 301, 170),
        (225, 235, 91.5, 145),
    ]:
        before_uv = local[:, :2] / local[:, 2:] * [fx, fy] + [cx, cy]
        after_uv = back[:, :2] / back[:, 2:] * [fx, fy] + [cx, cy]
        np.testing.assert_allclose(before_uv, after_uv, atol=1e-10)
    np.testing.assert_allclose(transform_points(spatial, inverse), world, atol=1e-12)
    box = parse_layout("bbox_2=Bbox(desk,1,2,3,0.5,2,1,0.8)", inverse)[0]
    points = transform_points(corners(box, [box]), matrix)
    np.testing.assert_allclose(points.mean(0), [1, 2, 3], atol=1e-12)
    assert np.linalg.det(box["rotation"]) == pytest.approx(1)


def test_layout_structures_and_no_double_translation():
    layout = "wall_0=Wall(-2,3,1,2,3,1,2.4,0.1)\ndoor_0=Door(wall_0,0,3,2,0.8,2)\nwindow_0=Window(wall_0,1,3,2.5,0.8,0.5)\nbbox_0=Bbox(computer,-2,3,1,0,0.5,0.2,0.4)"
    entries = parse_layout(layout, np.linalg.inv(similarity()))
    np.testing.assert_allclose(entries[-1]["center"], [-2, 1, -3])
    for item in entries:
        assert corners(item, entries).shape == (8, 3)
    assert entries[1]["wall_id"] == "spatiallm:wall_0"
    assert parse_layout("", np.eye(4)) == []
    wall = parse_layout("wall_0=Wall(0,0,0,2,0,0,2.5,0)", np.eye(4))[0]
    assert wall["thickness"] == 0
    assert np.isfinite(corners(wall, [wall])).all()


@pytest.mark.parametrize(
    "text",
    [
        "bbox_0=Bbox(chair,0,0,0,0,-1,1,1)",
        "bbox_0=Bbox(chair,nan,0,0,0,1,1,1)",
        "door_0=Door(wall_9,0,0,1,1,2)",
        "wall_0=Wall(0,0,0,0,0,0,2,0.1)",
        "bbox_0=Bbox(chair,0,0,0,0,1,1,1)\nbbox_0=Bbox(chair,0,0,0,0,1,1,1)",
        '__import__("os").system("anything")',
        "wall_0=Bbox(chair,0,0,0,0,1,1,1)",
    ],
)
def test_invalid_layout_is_not_silently_accepted(text):
    with pytest.raises(ValueError):
        parse_layout(text, np.eye(4))


def test_control_iou_known_geometry():
    from scripts.assess_spatiallm_control import iou

    a = {'center': [0, 0, 0], 'dimensions': [2, 2, 2], 'rotation': np.eye(3)}
    assert iou(a, a) == pytest.approx(1)
    b = {**a, 'center': [1, 0, 0]}
    assert iou(a, b) == pytest.approx(1 / 3)
    b = {**a, 'center': [0, 0, 3]}
    assert iou(a, b) == pytest.approx(0)
