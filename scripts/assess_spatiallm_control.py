"""Single official control: exact-category oriented 3D IoU, not benchmark mAP."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video2scene.spatiallm_adapter import corners, parse_layout, save_json, sha256


def iou(a, b):
    pa, pb = corners(a, [a]), corners(b, [b])
    xy_a = cv2.convexHull(pa[:, :2].astype(np.float32))
    xy_b = cv2.convexHull(pb[:, :2].astype(np.float32))
    area, _ = cv2.intersectConvexConvex(xy_a, xy_b)
    height = max(
        0, min(pa[:, 2].max(), pb[:, 2].max()) - max(pa[:, 2].min(), pb[:, 2].min())
    )
    intersection = max(0, float(area)) * height
    union = np.prod(a["dimensions"]) + np.prod(b["dimensions"]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def main():
    root = Path(sys.argv[1])
    gt_file = root / "official_control/ground_truth.txt"
    pred_file = root / "control_raw/official_control/layout.txt"
    gt = parse_layout(gt_file.read_text(), np.eye(4))
    pred = parse_layout(pred_file.read_text(), np.eye(4))
    boxes_gt = [e for e in gt if e["type"] == "bbox"]
    boxes_pred = [e for e in pred if e["type"] == "bbox"]
    overlaps = np.array(
        [
            [iou(a, b) if a["label"] == b["label"] else 0.0 for b in boxes_gt]
            for a in boxes_pred
        ]
    )
    pairs = []
    rows, cols = linear_sum_assignment(-overlaps)
    for r, c in zip(rows, cols):
        if overlaps[r, c] > 0:
            pairs.append(
                {
                    "prediction": boxes_pred[r]["id"],
                    "truth": boxes_gt[c]["id"],
                    "label": boxes_pred[r]["label"],
                    "iou": overlaps[r, c],
                }
            )
    walls_gt = [e for e in gt if e["type"] == "wall"]
    walls_pred = [e for e in pred if e["type"] == "wall"]
    costs = []
    for a in walls_pred:
        ap = np.array(a["endpoints"])
        costs.append(
            [
                min(
                    np.linalg.norm(ap - np.array(b["endpoints"]), axis=1).mean(),
                    np.linalg.norm(ap - np.array(b["endpoints"])[::-1], axis=1).mean(),
                )
                for b in walls_gt
            ]
        )
    wr, wc = linear_sum_assignment(costs)
    result = {
        "dataset": "manycore-research/SpatialLM-Testset",
        "dataset_revision": "60bab76febb0c9635f58e156f32e906afc91a499",
        "scene": "scene0000_00",
        "pointcloud_sha256": sha256(root / "official_control/points.ply"),
        "ground_truth_sha256": sha256(gt_file),
        "prediction_sha256": sha256(pred_file),
        "scope": "Single control scene; exact category matching and oriented 3D IoU. Not official category-remapped benchmark or mAP.",
        "gt_counts": {
            k: sum(e["type"] == k for e in gt)
            for k in ["bbox", "wall", "door", "window"]
        },
        "pred_counts": {
            k: sum(e["type"] == k for e in pred)
            for k in ["bbox", "wall", "door", "window"]
        },
        "overlapping_same_category_pairs": pairs,
        "wall_endpoint_mean_distance_m": float(np.mean(np.asarray(costs)[wr, wc])),
        "wall_mean_height_error_m": float(
            np.mean(
                [
                    abs(walls_pred[r]["height"] - walls_gt[c]["height"])
                    for r, c in zip(wr, wc)
                ]
            )
        ),
    }
    for threshold in [0.25, 0.5]:
        # Maximize number of matches above threshold, then IoU as tie breaker.
        r, c = linear_sum_assignment(
            -((overlaps >= threshold).astype(float) * 100 + overlaps)
        )
        tp = int(np.sum(overlaps[r, c] >= threshold))
        result[f"exact_class_iou_{threshold}"] = {
            "tp": tp,
            "fp": len(boxes_pred) - tp,
            "fn": len(boxes_gt) - tp,
            "precision": tp / len(boxes_pred),
            "recall": tp / len(boxes_gt),
        }
    save_json(root / "official_control/assessment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
