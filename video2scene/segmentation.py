"""Optional COCO instance masks, fused by class and 3D spatial agreement."""

from dataclasses import dataclass, field
import cv2
import numpy as np

from .artifacts import unproject
from .geometry import point_cloud, transform_points, fit_box


@dataclass
class Track:
    label: str
    points: np.ndarray
    colors: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    scores: list = field(default_factory=list)


def spatial_overlap(a, b):
    alo, ahi = np.quantile(a, [0.03, 0.97], axis=0)
    blo, bhi = np.quantile(b, [0.03, 0.97], axis=0)
    # A small tolerance covers depth noise on otherwise near-planar objects.
    pad = 0.04
    alo, ahi, blo, bhi = alo - pad, ahi + pad, blo - pad, bhi + pad
    intersection = np.prod(np.maximum(np.minimum(ahi, bhi) - np.maximum(alo, blo), 0))
    return intersection / max(min(np.prod(ahi - alo), np.prod(bhi - blo)), 1e-9)


class Segmenter:
    def __init__(self, weights="yolo11s-seg.pt", device="cpu", confidence=0.35):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device, self.confidence = device, confidence
        self.tracks = []
        self.detections = []

    def observe(self, frame, depth):
        result = self.model.predict(
            frame.rgb[..., ::-1].copy(),
            device=self.device,
            conf=self.confidence,
            retina_masks=True,
            verbose=False,
            imgsz=640,
        )[0]
        if result.masks is None:
            return
        masks = result.masks.data.cpu().numpy()
        assigned = set()
        for mask, box in zip(masks, result.boxes):
            label = result.names[int(box.cls.item())]
            if label in {"person", "cat", "dog", "bird"}:
                continue
            mask = cv2.resize(
                mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.erode(
                (mask > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)
            ).astype(bool)
            points = unproject(depth, frame.intrinsics, frame.c2w, mask=mask, step=2)
            if len(points) < 40:
                continue
            cloud = point_cloud(points).voxel_down_sample(0.015)
            labels = np.asarray(cloud.cluster_dbscan(eps=0.07, min_points=6))
            if not np.any(labels >= 0):
                continue
            points = np.asarray(cloud.points)[
                labels == np.bincount(labels[labels >= 0]).argmax()
            ]
            if len(points) < 30:
                continue
            score = float(box.conf.item())
            color = (
                np.median(frame.rgb[mask & (depth > 0)], axis=0).astype(int).tolist()
            )
            candidates = [
                (spatial_overlap(points, t.points), i)
                for i, t in enumerate(self.tracks)
                if t.label == label and i not in assigned
            ]
            overlap, tid = max(candidates, default=(0, -1))
            if overlap < 0.22:
                tid = len(self.tracks)
                self.tracks.append(Track(label, points.copy()))
            else:
                self.tracks[tid].points = np.asarray(
                    point_cloud(np.vstack([self.tracks[tid].points, points]))
                    .voxel_down_sample(0.015)
                    .points
                )
            track = self.tracks[tid]
            track.frames.append(frame.index)
            track.scores.append(score)
            track.colors.append(color)
            assigned.add(tid)
            self.detections.append(
                dict(
                    frame=frame.index,
                    track=tid,
                    label=label,
                    confidence=score,
                    bbox_xyxy=box.xyxy.cpu().numpy()[0].tolist(),
                    points=len(points),
                )
            )

    def proxies(self, transform, min_views=2):
        specs = []
        for i, track in enumerate(self.tracks):
            if len(track.frames) < min_views:
                continue
            points = transform_points(track.points, transform)
            center, dims, rotation = fit_box(points)
            primitive = (
                "cylinder"
                if track.label in {"bottle", "cup", "vase", "bowl"}
                else "box"
            )
            specs.append(
                dict(
                    id=f"object_{i:03d}_{track.label.replace(' ', '_')}",
                    label=track.label,
                    primitive=primitive,
                    center=center.tolist(),
                    dimensions=dims.tolist(),
                    rotation=rotation.tolist(),
                    color=[*np.median(track.colors, axis=0).astype(int).tolist(), 255],
                    source="YOLO masks lifted through ViPE depth and associated in 3D",
                    observations=len(points),
                    frame_indices=track.frames,
                    mean_detection_confidence=float(np.mean(track.scores)),
                    assumptions=[
                        "Closed primitive approximates observed surfaces; hidden sides and full dimensions are not measured.",
                        "COCO classes and 3D overlap can miss objects, split one object, or merge nearby instances.",
                    ],
                )
            )
        return specs
