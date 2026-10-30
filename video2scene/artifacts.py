"""Read ViPE's actual on-disk contract without importing its CUDA package."""

from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
import zipfile

import cv2
import Imath
import numpy as np
import OpenEXR


@dataclass
class Frame:
    index: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray  # fx, fy, cx, cy, in depth-image pixels
    c2w: np.ndarray  # OpenCV camera (right, down, forward) to ViPE world


def indexed_npz(path: Path, tail_shape: tuple) -> dict:
    with np.load(path, allow_pickle=False) as data:
        indices, values = data["inds"], data["data"]
        if indices.ndim != 1 or values.shape != (len(indices), *tail_shape):
            raise ValueError(f"Unexpected shape in {path}: {values.shape}")
        if len(set(indices.tolist())) != len(indices) or np.any(indices < 0):
            raise ValueError(f"Duplicate or negative frame indices in {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {path}")
        return dict(zip(indices.astype(int).tolist(), values))


class ViPEArtifacts:
    def __init__(self, root, sequence=None):
        self.root = Path(root)
        if sequence is None:
            names = sorted(p.stem for p in (self.root / "pose").glob("*.npz"))
            if len(names) != 1:
                raise ValueError(f"Specify --sequence; found sequences: {names}")
            sequence = names[0]
        self.sequence = sequence
        self.poses = indexed_npz(self.root / "pose" / f"{sequence}.npz", (4, 4))
        self.intrinsics = indexed_npz(
            self.root / "intrinsics" / f"{sequence}.npz", (4,)
        )
        camera_file = self.root / "intrinsics" / f"{sequence}_camera.txt"
        if camera_file.exists():
            types = {
                s.split(":", 1)[1].strip().lower()
                for s in camera_file.read_text().splitlines()
                if s.strip()
            }
            if types != {"pinhole"}:
                raise ValueError(
                    f"This mesh adapter supports pinhole only, got {types}. Undistort first."
                )
        self.rgb_path = self.root / "rgb" / f"{sequence}.mp4"
        self.depth_path = self.root / "depth" / f"{sequence}.zip"
        for path in [self.rgb_path, self.depth_path]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}; run ViPE with save_artifacts=true and dense depth enabled."
                )
        with zipfile.ZipFile(self.depth_path) as archive:
            self.depth_names = {
                int(Path(n).stem): n for n in archive.namelist() if n.endswith(".exr")
            }
        self.indices = sorted(
            self.poses.keys() & self.intrinsics.keys() & self.depth_names.keys()
        )
        if not self.indices:
            raise ValueError("No matching pose, intrinsics and depth frame IDs")
        for i in self.indices:
            pose = self.poses[i]
            if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4):
                raise ValueError(f"Invalid homogeneous pose at frame {i}")
            if (
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=0.02)
                or np.linalg.det(pose[:3, :3]) < 0.98
            ):
                raise ValueError(f"Invalid pose rotation at frame {i}")
            if np.any(self.intrinsics[i][:2] <= 0):
                raise ValueError(f"Invalid focal length at frame {i}")

    def frames(self, stride=1, scale=1.0, *, indices=None):
        """Yield requested frames in video order, decoding depth only when needed.

        ``indices`` names actual ViPE indices, not positions in ``self.indices``.
        The optional stride is applied to that selected sorted sequence. Other
        RGB frames are grabbed for codec progression but are not materialized.
        """
        if not isinstance(stride, Integral) or isinstance(stride, bool) or stride < 1 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Frame stride and scale must be positive")
        selected = self.indices
        if indices is not None:
            requested = list(indices)
            if (any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in requested)
                    or len(set(requested)) != len(requested)):
                raise ValueError('Requested ViPE indices must be distinct nonnegative integers')
            if set(requested) - set(self.indices):
                raise ValueError('Requested ViPE indices are missing from pose/intrinsics/depth artifacts')
            selected = sorted(requested)
        wanted = set(selected[::stride])
        if not wanted:
            return
        capture = cv2.VideoCapture(str(self.rgb_path))
        seen = set()
        try:
            with zipfile.ZipFile(self.depth_path) as archive:
                index = 0
                while index <= max(wanted):
                    ok = capture.grab()
                    if not ok:
                        break
                    if index in wanted:
                        ok, bgr = capture.retrieve()
                        if not ok:
                            raise ValueError(f'Could not decode requested RGB frame {index}')
                        with archive.open(self.depth_names[index]) as stream:
                            exr = OpenEXR.InputFile(stream)
                            dw = exr.header()["dataWindow"]
                            w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
                            depth = (
                                np.frombuffer(
                                    exr.channel(
                                        "Z", Imath.PixelType(Imath.PixelType.FLOAT)
                                    ),
                                    np.float32,
                                )
                                .reshape(h, w)
                                .copy()
                            )
                            exr.close()
                        depth *= scale
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        rh, rw = rgb.shape[:2]
                        intrinsics = self.intrinsics[index].astype(np.float64).copy()
                        if (h, w) != (rh, rw):
                            intrinsics *= [w / rw, h / rh, w / rw, h / rh]
                            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                        pose = self.poses[index].astype(np.float64).copy()
                        pose[:3, 3] *= scale
                        seen.add(index)
                        yield Frame(
                            index, np.ascontiguousarray(rgb), depth, intrinsics, pose
                        )
                    index += 1
        finally:
            capture.release()
        if wanted - seen:
            raise ValueError(
                f"RGB video ended before frame IDs {sorted(wanted - seen)[:8]}"
            )


def clean_depth(depth, max_depth=8.0, edge_relative=0.08):
    """Suppress invalid samples and foreground/background discontinuities."""
    depth = np.asarray(depth, np.float32).copy()
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < max_depth)
    depth[~valid] = 0
    if edge_relative > 0:
        hi = cv2.dilate(depth, np.ones((3, 3), np.uint8))
        lo = cv2.erode(depth, np.ones((3, 3), np.uint8))
        valid &= (hi - lo) < np.maximum(depth * edge_relative, 0.03)
    depth[~valid] = 0
    return depth


def unproject(depth, intrinsics, c2w, mask=None, step=1):
    v, u = np.mgrid[0 : depth.shape[0] : step, 0 : depth.shape[1] : step]
    z = depth[::step, ::step]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask[::step, ::step].astype(bool)
    fx, fy, cx, cy = intrinsics
    xyz = np.column_stack(
        ((u[valid] - cx) * z[valid] / fx, (v[valid] - cy) * z[valid] / fy, z[valid])
    )
    return xyz @ c2w[:3, :3].T + c2w[:3, 3]
