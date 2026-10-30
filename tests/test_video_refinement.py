"""Geometric recovery check for the video-guided rectangle optimizer."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refine_scene_from_video import fit_rectangle, flat_rotation, project, quad


def test_multiview_fit_recovers_known_flat_object_and_support_height():
    truth_center = np.array([0.10, 0.765, 0.15])
    truth_rotation = flat_rotation(0.25)
    truth_dims = np.array([0.45, 0.16, 0.03])
    corners = quad(truth_center, truth_rotation, truth_dims)
    frames = {}
    observations = []
    for i, eye in enumerate([[0.8, 1.6, 1.3], [-0.7, 1.4, 1.0], [0.6, 1.7, -0.9]]):
        eye = np.array(eye)
        forward = truth_center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.column_stack([right, down, forward])
        c2w[:3, 3] = eye
        frames[i] = SimpleNamespace(c2w=c2w, intrinsics=np.array([550, 550, 320, 240]))
        observations.append(
            dict(frame=i, quad=project(corners, frames[i]).tolist(), fixed_order=True)
        )
    obj = dict(
        id="test",
        center=(truth_center + [0.05, 0, -0.04]).tolist(),
        dimensions=[0.52, 0.21, 0.03],
        rotation=flat_rotation(0.30).tolist(),
        assumptions=[],
    )
    report = fit_rectangle(obj, observations, frames, flat=True, support=0.75)
    assert report["before_px"] > 5
    assert report["after_px"] < 0.5
    np.testing.assert_allclose(obj["center"], truth_center, atol=0.003)
    np.testing.assert_allclose(obj["dimensions"], truth_dims, atol=0.003)
