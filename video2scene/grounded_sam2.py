"""Open-vocabulary masks, bounded video tracking, and static 3D identity checks.

The inventory contains category prompts only. Detection/tracking never reads the
review polygons used by the comparison script. Model imports are lazy so cached
reconstruction works without torch or SAM 2 installed.
"""

from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .artifacts import clean_depth, unproject
from .geometry import point_cloud
from .segmentation import Segmenter, Track, spatial_overlap


def mask_points(frame, depth, mask):
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    xyz = unproject(depth, frame.intrinsics, frame.c2w, mask, step=2)
    if len(xyz) < 40:
        return np.empty((0, 3)), mask
    cloud = point_cloud(xyz).voxel_down_sample(0.015)
    groups = np.asarray(cloud.cluster_dbscan(eps=0.07, min_points=6))
    if not np.any(groups >= 0):
        return np.empty((0, 3)), mask
    xyz = np.asarray(cloud.points)[groups == np.bincount(groups[groups >= 0]).argmax()]
    return xyz, mask


def mask_iou(a, b):
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def associate(candidates, tracks, previous):
    """One-to-one association; distant same-class objects cannot steal an ID."""
    ids = sorted(tracks)
    scores = np.zeros((len(candidates), len(ids)))
    for row, candidate in enumerate(candidates):
        for col, tid in enumerate(ids):
            track = tracks[tid]
            if candidate["label"] != track["label"]:
                continue
            a, b = candidate["points"], track["points"]
            overlap = spatial_overlap(a, b) if len(a) >= 30 and len(b) >= 30 else 0
            iou = mask_iou(candidate["mask"], previous[tid]) if tid in previous else 0
            if len(a) >= 30 and len(b) >= 30:
                distance = np.linalg.norm(np.median(a, axis=0) - np.median(b, axis=0))
                extent = np.linalg.norm(
                    np.quantile(b, 0.95, axis=0) - np.quantile(b, 0.05, axis=0)
                )
                if distance > max(0.20, extent * 1.25):
                    continue
            if iou >= 0.25 or overlap >= 0.30:
                scores[row, col] = 0.65 * iou + 0.35 * min(overlap, 1) + 0.01
    matched = {}
    if scores.size:
        rows, cols = linear_sum_assignment(-scores)
        matched = {int(r): ids[c] for r, c in zip(rows, cols) if scores[r, c] > 0}
    return matched


def save_masks(path, masks):
    ids = sorted(masks)
    if ids:
        stack = np.stack([masks[i] for i in ids]).astype(bool)
        shape = stack.shape[1:]
        packed = np.packbits(stack.reshape(len(ids), -1), axis=1)
    else:
        shape, packed = (0, 0), np.empty((0, 0), np.uint8)
    np.savez_compressed(path, ids=np.asarray(ids, np.int32), shape=shape, packed=packed)


def load_masks(path):
    with np.load(path, allow_pickle=False) as data:
        ids, shape = data["ids"], tuple(data["shape"])
        if not len(ids):
            return {}
        stack = np.unpackbits(data["packed"], axis=1, count=int(np.prod(shape)))
        return {
            int(tid): mask.reshape(shape).astype(bool) for tid, mask in zip(ids, stack)
        }


def fingerprint(artifacts, inventory, settings):
    files = [
        artifacts.rgb_path,
        artifacts.depth_path,
        artifacts.root / "pose" / f"{artifacts.sequence}.npz",
        artifacts.root / "intrinsics" / f"{artifacts.sequence}.npz",
    ]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[str(path.resolve())] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return dict(version=1, inputs=hashes, inventory=inventory, settings=settings)


def prepare_masks(
    artifacts,
    inventory_path,
    cache,
    device="cuda",
    detect_every=20,
    threshold=0.25,
    sam_model="facebook/sam2.1-hiera-small",
    scale=1.0,
    max_depth=8.0,
):
    if detect_every < 1 or not 0 < threshold < 1:
        raise ValueError(
            "detect_every must be positive and grounding threshold in (0, 1)"
        )
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    categories = inventory["categories"]
    if not categories or len({c["label"] for c in categories}) != len(categories):
        raise ValueError("Inventory must contain unique, nonempty category labels")
    settings = dict(
        device=device,
        detect_every=detect_every,
        threshold=threshold,
        sam_model=sam_model,
        scale=scale,
        max_depth=max_depth,
        detector="IDEA-Research/grounding-dino-tiny",
    )
    signature = fingerprint(artifacts, inventory, settings)
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") == "complete"
            and manifest.get("signature") == signature
        ):
            if all((cache / f"{i:06d}.npz").is_file() for i in artifacts.indices):
                print(f"Reusing verified mask cache {cache}", flush=True)
                return manifest
        if manifest.get("signature") != signature or manifest.get("status") != "failed":
            raise ValueError(
                f"Mask cache belongs to different settings: {cache}. Choose a fresh --mask-cache."
            )
        print(
            "Retrying interrupted run with identical inputs; regenerating all tracking masks",
            flush=True,
        )
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()) and not manifest_path.exists():
        raise ValueError(f"Mask cache must be empty: {cache}")
    start = time.perf_counter()
    manifest = dict(status="running", signature=signature, tracks={}, events=[])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        _generate(artifacts, categories, cache, manifest, settings)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - start)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _generate(artifacts, categories, cache, manifest, settings):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = settings["device"]
    torch.set_num_threads(4)
    frames = list(artifacts.frames(scale=settings["scale"]))
    key_positions = list(range(0, len(frames), settings["detect_every"]))
    processor = AutoProcessor.from_pretrained(settings["detector"])
    detector = (
        AutoModelForZeroShotObjectDetection.from_pretrained(settings["detector"])
        .to(device)
        .eval()
    )
    manifest["detector_revision"] = getattr(detector.config, "_commit_hash", None)
    # Small prompt batches avoid unrelated phrases competing in one long caption.
    batches = [categories[i : i + 4] for i in range(0, len(categories), 4)]
    detections = {}
    with torch.inference_mode():
        for pos in key_positions:
            frame = frames[pos]
            candidates = []
            for batch in batches:
                text = ". ".join(c["prompt"].lower().strip(" .") for c in batch) + "."
                inputs = processor(
                    images=Image.fromarray(frame.rgb), text=text, return_tensors="pt"
                ).to(device)
                output = detector(**inputs)
                result = processor.post_process_grounded_object_detection(
                    output,
                    inputs.input_ids,
                    threshold=settings["threshold"],
                    text_threshold=0.20,
                    target_sizes=[frame.rgb.shape[:2]],
                )[0]
                phrases = result.get("text_labels", result.get("labels"))
                for box, score, phrase in zip(
                    result["boxes"], result["scores"], phrases
                ):
                    phrase = phrase.lower().strip(" .")
                    matches = [c for c in batch if c["prompt"].lower() == phrase]
                    if not matches:
                        matches = [
                            c
                            for c in batch
                            if phrase
                            and (
                                phrase in c["prompt"].lower()
                                or c["prompt"].lower() in phrase
                            )
                        ]
                    if len(matches) != 1:
                        continue
                    candidates.append(
                        dict(
                            label=matches[0]["label"],
                            phrase=phrase,
                            score=float(score),
                            box=box.cpu().tolist(),
                        )
                    )
            # Class-aware NMS, followed by mask deduplication after SAM.
            from torchvision.ops import nms

            kept = []
            for label in sorted({c["label"] for c in candidates}):
                group = [c for c in candidates if c["label"] == label]
                ids = nms(
                    torch.tensor([c["box"] for c in group]),
                    torch.tensor([c["score"] for c in group]),
                    0.5,
                )
                kept.extend(group[int(i)] for i in ids)
            detections[pos] = kept
            print(f"Grounding DINO frame {frame.index}: {len(kept)} boxes", flush=True)
        del inputs, output, result
    (cache / "keyframe_detections.json").write_text(
        json.dumps(
            {str(frames[p].index): items for p, items in detections.items()}, indent=2
        )
    )
    del detector, processor
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    predictor = SAM2VideoPredictor.from_pretrained(
        settings["sam_model"], device=device, apply_postprocessing=False
    )
    from huggingface_hub import try_to_load_from_cache
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    checkpoint_name = HF_MODEL_ID_TO_FILENAMES[settings["sam_model"]][1]
    checkpoint_path = try_to_load_from_cache(settings["sam_model"], checkpoint_name)
    manifest["sam2_revision"] = (
        Path(checkpoint_path).parent.name if isinstance(checkpoint_path, str) else None
    )
    manifest["runtime_versions"] = {
        p: importlib.metadata.version(p)
        for p in ["torch", "transformers", "SAM-2", "numpy"]
    }
    image_predictor = SAM2ImagePredictor(predictor)

    # Use only one copy of SAM weights. Disable optional CUDA connected components;
    # geometry filtering and masks are still computed by the standard model.
    def amp():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(device).startswith("cuda")
            else nullcontext()
        )

    tracks, previous, next_id = {}, {}, 1
    with tempfile.TemporaryDirectory(prefix="video2scene-sam2-") as tmp:
        for p, frame in enumerate(frames):
            cv2.imwrite(
                str(Path(tmp) / f"{p:06d}.jpg"),
                cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
            )
        with torch.inference_mode(), amp():
            state = predictor.init_state(
                tmp, offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            for pos in key_positions:
                frame = frames[pos]
                depth = clean_depth(frame.depth, settings["max_depth"])
                image_predictor.set_image(frame.rgb)
                candidates = []
                for item in sorted(detections[pos], key=lambda x: -x["score"]):
                    masks, quality, _ = image_predictor.predict(
                        box=np.asarray(item["box"]), multimask_output=False
                    )
                    mask = masks[0].astype(bool)
                    if mask.sum() < 40 or any(
                        mask_iou(mask, c["mask"]) > 0.8 for c in candidates
                    ):
                        continue
                    xyz, _ = mask_points(frame, depth, mask)
                    candidates.append(
                        dict(
                            item, mask=mask, points=xyz, mask_quality=float(quality[0])
                        )
                    )
                matched = associate(candidates, tracks, previous)
                predictor.reset_state(state)
                seeded = {}
                for row, c in enumerate(candidates):
                    tid = matched.get(row)
                    if tid is None:
                        tid, next_id = next_id, next_id + 1
                        tracks[tid] = dict(label=c["label"], points=c["points"])
                        manifest["tracks"][str(tid)] = dict(
                            label=c["label"], seed_frames=[], detection_scores=[]
                        )
                    elif len(c["points"]) >= 30:
                        old = tracks[tid]["points"]
                        tracks[tid]["points"] = np.asarray(
                            point_cloud(np.vstack([old, c["points"]]))
                            .voxel_down_sample(0.02)
                            .points
                        )
                    rec = manifest["tracks"][str(tid)]
                    rec["seed_frames"].append(frame.index)
                    rec["detection_scores"].append(c["score"])
                    predictor.add_new_mask(state, pos, tid, c["mask"])
                    seeded[tid] = c["mask"]
                manifest["events"].append(
                    dict(frame=frame.index, seeds=list(seeded), reused=len(matched))
                )
                end = min(pos + settings["detect_every"], len(frames) - 1)
                previous = {}
                if seeded:
                    for out_pos, ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=pos, max_frame_num_to_track=end - pos
                    ):
                        masks = {
                            int(tid): (logits[k, 0] > 0).cpu().numpy()
                            for k, tid in enumerate(ids)
                        }
                        masks = {tid: m for tid, m in masks.items() if m.sum() >= 40}
                        save_masks(cache / f"{frames[out_pos].index:06d}.npz", masks)
                        if out_pos == end:
                            previous = masks
                else:
                    for out_pos in range(pos, end + 1):
                        save_masks(cache / f"{frames[out_pos].index:06d}.npz", {})
                print(
                    f"SAM 2.1 frames {frame.index}–{frames[end].index}: {len(seeded)} active, {len(tracks)} total IDs",
                    flush=True,
                )
                image_predictor.reset_predictor()
    manifest["frame_indices"] = [f.index for f in frames]
    manifest["sam2_model"] = settings["sam_model"]
    del state, predictor, image_predictor
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


class GroundedSAM2Segmenter(Segmenter):
    """Read masks by exact ViPE frame ID and fit the existing box/cylinder proxies."""

    def __init__(self, cache, evidence_dir=None):
        self.cache = Path(cache)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        if self.manifest["status"] != "complete":
            raise ValueError("Cannot reconstruct from incomplete masks")
        self.tracks, self.detections, self.track_ids = [], [], {}

    def observe(self, frame, depth):
        for tid, mask in load_masks(self.cache / f"{frame.index:06d}.npz").items():
            if mask.shape != depth.shape:
                raise ValueError("Mask resolution differs from ViPE depth")
            xyz, interior = mask_points(frame, depth, mask)
            if len(xyz) < 30:
                continue
            meta = self.manifest["tracks"][str(tid)]
            if tid not in self.track_ids:
                self.track_ids[tid] = len(self.tracks)
                self.tracks.append(Track(meta["label"], xyz.copy()))
            track = self.tracks[self.track_ids[tid]]
            distance = np.linalg.norm(
                np.median(xyz, axis=0) - np.median(track.points, axis=0)
            )
            extent = np.linalg.norm(
                np.quantile(track.points, 0.95, axis=0)
                - np.quantile(track.points, 0.05, axis=0)
            )
            if distance > max(0.20, extent * 1.25):
                self.detections.append(
                    dict(
                        frame=frame.index,
                        track=tid,
                        label=meta["label"],
                        rejected="3d_disagreement",
                    )
                )
                continue
            track.points = np.asarray(
                point_cloud(np.vstack([track.points, xyz]))
                .voxel_down_sample(0.015)
                .points
            )
            track.frames.append(frame.index)
            # Detector score is seed evidence, not a calibrated video-mask probability.
            track.scores.append(float(np.mean(meta["detection_scores"])))
            track.colors.append(
                np.median(frame.rgb[interior & (depth > 0)], axis=0).tolist()
            )
            ys, xs = np.where(mask)
            self.detections.append(
                dict(
                    frame=frame.index,
                    track=tid,
                    label=meta["label"],
                    bbox_xyxy=[
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ],
                    points=len(xyz),
                )
            )

    def proxies(self, transform, min_views=2):
        specs = super().proxies(transform, min_views)
        reverse = {v: k for k, v in self.track_ids.items()}
        for spec in specs:
            index = int(spec["id"].split("_")[1])
            tid = reverse[index]
            spec["tracking_id"] = tid
            spec["source"] = (
                "Grounding DINO + SAM 2.1 masks lifted through ViPE depth; static 3D identity checks"
            )
            spec["seed_frames"] = self.manifest["tracks"][str(tid)]["seed_frames"]
            spec["independent_detections"] = len(spec["seed_frames"])
            spec["review_status"] = "unreviewed_candidate"
            if "semantic_review" in self.manifest:
                spec["review_status"] = "vlm_reviewed_candidate"
                spec["raw_tracking_ids"] = self.manifest["tracks"][str(tid)]["raw_ids"]
                spec["semantic_review"] = self.manifest["semantic_review"]
            if self.evidence_dir:
                import open3d as o3d
                from .geometry import transform_points

                self.evidence_dir.mkdir(parents=True, exist_ok=True)
                path = self.evidence_dir / (spec["id"] + ".ply")
                o3d.io.write_point_cloud(
                    str(path),
                    point_cloud(transform_points(self.tracks[index].points, transform)),
                )
                spec["point_cloud"] = str(path)
            spec["assumptions"][1] = (
                "Open-vocabulary detections and video identities can miss, split or merge objects; closed dimensions are estimates."
            )
        return specs
