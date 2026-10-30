"""Contract tests use known geometry, independently of ViPE's learned predictions."""

import json
import zipfile
import cv2
import Imath
import numpy as np
import OpenEXR
import pytest
import trimesh

from video2scene.artifacts import ViPEArtifacts, clean_depth, unproject
from video2scene.geometry import proxy_mesh
from video2scene.reconstruction import reconstruct, export_models
from video2scene.segmentation import spatial_overlap


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "vipe"
    for name in ["rgb", "pose", "depth", "intrinsics"]:
        (root / name).mkdir(parents=True)
    w, h = 96, 72
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.arange(5) * 0.08
    ids = np.arange(5)
    np.savez(root / "pose/test.npz", inds=ids, data=poses)
    np.savez(
        root / "intrinsics/test.npz",
        inds=ids,
        data=np.tile([80, 80, w / 2, h / 2], (5, 1)),
    )
    (root / "intrinsics/test_camera.txt").write_text(
        "\n".join(f"{i}: PINHOLE" for i in ids)
    )
    writer = cv2.VideoWriter(
        str(root / "rgb/test.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
    )
    with zipfile.ZipFile(root / "depth/test.zip", "w") as archive:
        for i in ids:
            writer.write(np.full((h, w, 3), [80, 140, 210], np.uint8))
            header = OpenEXR.Header(w, h)
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
            }
            file = tmp_path / "depth.exr"
            exr = OpenEXR.OutputFile(str(file), header)
            exr.writePixels({"Z": np.full((h, w), 2, np.float16).tobytes()})
            exr.close()
            archive.write(file, f"{i:05d}.exr")
    writer.release()
    return root


def test_backprojection_has_camera_translation_and_correct_axes():
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 5]
    points = unproject(np.full((3, 3), 2.0), [2, 2, 1, 1], pose)
    np.testing.assert_allclose(points[4], [3, 4, 7])
    np.testing.assert_allclose(points[0], [2, 3, 7])


def test_adapter_matches_ids_instead_of_array_order(artifacts):
    path = artifacts / "pose/test.npz"
    with np.load(path) as data:
        indices, values = data["inds"].copy(), data["data"].copy()
    np.savez(path, inds=indices[::-1], data=values[::-1])
    frames = list(ViPEArtifacts(artifacts).frames(stride=2, scale=2))
    assert [f.index for f in frames] == [0, 2, 4]
    assert frames[-1].c2w[0, 3] == pytest.approx(0.64)
    assert frames[-1].depth.mean() == pytest.approx(4)


def test_adapter_decodes_only_requested_depth_indices(artifacts, monkeypatch):
    calls = []
    original = OpenEXR.InputFile
    def counted(stream):
        calls.append(stream.name)
        return original(stream)
    monkeypatch.setattr(OpenEXR, 'InputFile', counted)
    result = list(ViPEArtifacts(artifacts).frames(indices=[4, 1], scale=2))
    assert [item.index for item in result] == [1, 4]
    assert len(calls) == 2 and set(calls) == {'00001.exr', '00004.exr'}
    assert result[-1].depth.mean() == pytest.approx(4.)
    calls.clear()
    assert list(ViPEArtifacts(artifacts).frames(indices=[])) == []
    assert calls == []
    for invalid in ([1, 1], [6], [True], [-1]):
        with pytest.raises(ValueError, match='indices'):
            list(ViPEArtifacts(artifacts).frames(indices=invalid))


def test_non_pinhole_fails_explicitly(artifacts):
    (artifacts / "intrinsics/test_camera.txt").write_text("0: FISHEYE")
    with pytest.raises(ValueError, match="pinhole"):
        ViPEArtifacts(artifacts)


def test_tsdf_export_preserves_known_plane_and_camera_scale(artifacts, tmp_path):
    output = tmp_path / "scene"
    spec = reconstruct(
        artifacts,
        output,
        stride=1,
        scale=2,
        voxel=0.04,
        segment="none",
        max_faces=10000,
    )
    assert spec["stats"]["frames_integrated"] == 5
    scene = trimesh.load(output / "scan.glb", force="scene")
    mesh = scene.to_geometry()
    inverse = np.linalg.inv(spec["vipe_to_scene_rigid"])
    world = mesh.vertices @ inverse[:3, :3].T + inverse[:3, 3]
    assert np.median(world[:, 2]) == pytest.approx(4, abs=0.06)
    assert np.quantile(abs(world[:, 2] - 4), 0.95) < 0.08
    assert len(mesh.faces) > 1000
    camera_positions = [np.array(c["c2w"])[:3, 3] for c in spec["cameras"]]
    assert np.linalg.norm(camera_positions[-1] - camera_positions[0]) == pytest.approx(
        0.64
    )
    reread = json.loads((output / "scene.json").read_text())
    export_models(reread, tmp_path / "rebuilt")
    assert (tmp_path / "rebuilt/model.glb").is_file()


def test_invalid_depth_is_not_fused():
    result = clean_depth(
        np.array([[np.nan, np.inf], [-1, 2]], np.float32), edge_relative=0
    )
    np.testing.assert_array_equal(result, [[0, 0], [0, 2]])


def test_cylinder_proxy_extents_and_closed_solid():
    spec = dict(
        id="bottle",
        label="bottle",
        dimensions=[0.2, 0.5, 0.3],
        center=[1, 2, 3],
        rotation=np.eye(3).tolist(),
        primitive="cylinder",
        color=[255, 0, 0, 255],
        source="test",
    )
    mesh = proxy_mesh(spec)
    np.testing.assert_allclose(mesh.extents, [0.2, 0.5, 0.3], atol=1e-6)
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [1, 2, 3], atol=1e-6)
    assert mesh.is_watertight


def test_spatial_association_rejects_distant_identical_class():
    points = np.random.default_rng(5).uniform(0, 0.3, (100, 3))
    assert spatial_overlap(points, points + 0.01) > 0.7
    assert spatial_overlap(points, points + 1) == 0


def test_plane_levelling_uses_minimum_rotation_without_heading_flip():
    from video2scene.geometry import level_plane_rotation

    for normal in [[0.01, 1, 0.02], [0.01, -1, 0.02], [0.2, 1, -0.3], [0, 1, 0]]:
        normal = np.asarray(normal, float)
        normal /= np.linalg.norm(normal)
        upward = normal if normal[1] >= 0 else -normal
        rotation = level_plane_rotation(normal)
        np.testing.assert_allclose(rotation @ upward, [0, 1, 0], atol=1e-10)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        assert np.linalg.det(rotation) == pytest.approx(1)
        angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
        assert angle == pytest.approx(np.arccos(upward[1]), abs=1e-8)
def test_observed_plane_footprint_does_not_fill_rectangle_corners():
    import numpy as np
    from video2scene.geometry import plane_proxy, proxy_mesh
    points=np.array([[x,0,z] for x in np.linspace(0,1,60) for z in np.linspace(0,1,60) if x+z<1])
    spec=plane_proxy({'points':points,'normal':np.array([0,1,0])},np.eye(4),0)
    mesh=proxy_mesh(spec)
    assert mesh.is_watertight and mesh.volume>0
    assert mesh.volume<np.prod(spec['dimensions'])*.7

