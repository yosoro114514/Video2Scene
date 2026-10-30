"""Offline SAM 3.1 worker; imported by CPU tests without importing torch/SAM."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType
import shutil
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

try:
    from .sam_tool import CHECKPOINT_SHA256, read_json, sha256, validate_inputs, write_json
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sam_tool import CHECKPOINT_SHA256, read_json, sha256, validate_inputs, write_json


def install_uncached_refinement_merge(model):
    """Keep real tracker predictions when interactivity precedes VG propagation.

    The pinned multiplex implementation returns {} for an uncached frame before
    merging its freshly inferred refinement masks. This process-local adapter
    changes only that branch; it neither creates masks nor changes tracking.
    """
    original = model._build_sam2_output
    if list(inspect.signature(original).parameters) != [
        "inference_state", "frame_idx", "refined_obj_id_to_mask"
    ]:
        raise RuntimeError("Unrecognized multiplex output-merge API; compatibility adapter refused")
    source = inspect.getsource(original)
    source_file = Path(inspect.getsourcefile(original))

    def merge(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        if frame_idx not in inference_state["cached_frame_outputs"] and refined_obj_id_to_mask is not None:
            if not isinstance(refined_obj_id_to_mask, dict):
                raise TypeError("Refined SAM masks must be an object-ID mapping")
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                if not isinstance(obj_id, (int, np.integer)) or refined_mask is None:
                    raise ValueError("A refined tracker object ID and actual mask are required")
            return refined_obj_id_to_mask.copy()
        return original(inference_state, frame_idx, refined_obj_id_to_mask)

    model._build_sam2_output = MethodType(merge, model)
    return {"name": "sam31_uncached_refinement_merge", "version": 1,
            "original_source_path": str(source_file), "original_source_sha256": sha256(source_file),
            "original_method_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "scope": "Only missing-cache merge with supplied tracker masks; cached frames delegate unchanged",
            "upstream_files_modified": False, "synthetic_masks": False}


def initial_propagation_quality(row, frame_count):
    """Fail the first multi-frame object if propagation never leaves its seed."""
    if row["status"] == "complete" and frame_count > 1 and row["frames_with_target"] <= 1:
        raise RuntimeError("First SAM target has no non-seed observations; propagation quality gate failed")


def propagation_direction(seed, frame_count):
    """Avoid the multiplex cache-only second pass for endpoint prompts.

    Its action-history parser assumes a completed pass starting at either end
    covered the whole video. ``both`` always runs forward first, which covers
    only the seed when it is the final frame; backward would then fetch absent
    cached frames. The explicit prompt already supplies the seed's mask.
    """
    if not 0 <= seed < frame_count:
        raise ValueError("Seed sample index is outside the frame manifest")
    if seed == 0:
        return "forward"
    if seed == frame_count - 1:
        return "backward"
    return "both"


def arrays(outputs, height, width):
    def cpu(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if outputs is None:
        raise RuntimeError("SAM returned no output object")
    ids = cpu(outputs["out_obj_ids"]).astype(np.int64)
    masks = cpu(outputs["out_binary_masks"]).astype(bool)
    scores = cpu(outputs.get("out_probs", np.full(len(ids), np.nan))).astype(np.float32)
    if ids.ndim != 1 or masks.shape != (len(ids), height, width) or scores.shape != ids.shape:
        raise ValueError("SAM output does not match source frame dimensions")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("SAM returned duplicate tracking IDs in one frame")
    return ids, masks, scores


def choose_anchor(ids, masks, scores, request):
    """Select exactly one seed track; never silently union same-category instances."""
    positive = request["positive_points_normalized"]
    negative = request["negative_points_normalized"]
    box = request.get("hint_box_normalized")
    height, width = masks.shape[1:]
    rows = []
    for identity, mask, score in zip(ids, masks, scores):
        def hits(points):
            return sum(bool(mask[min(int(y * height), height - 1), min(int(x * width), width - 1)]) for x, y in points)
        y, x = np.nonzero(mask)
        iou = 0.0
        if box and len(x):
            actual = [x.min() / width, y.min() / height, (x.max() + 1) / width, (y.max() + 1) / height]
            expected = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            intersection = max(0., min(actual[2], expected[2]) - max(actual[0], expected[0])) * max(0., min(actual[3], expected[3]) - max(actual[1], expected[1]))
            union = (actual[2] - actual[0]) * (actual[3] - actual[1]) + box[2] * box[3] - intersection
            iou = intersection / union if union > 0 else 0.
        rows.append({"sam_id": int(identity), "positive_hits": hits(positive), "negative_hits": hits(negative),
                     "box_iou": float(iou), "area_px": int(mask.sum()), "score": float(score) if np.isfinite(score) else None})
    if positive:
        eligible = [r for r in rows if r["positive_hits"] > 0 and r["negative_hits"] == 0]
        if not eligible:
            return None, {"method": "point_new_target", "candidates": rows, "reason": "No text seed mask satisfies the supplied target points"}
        best_hits = max(r["positive_hits"] for r in eligible)
        eligible = [r for r in eligible if r["positive_hits"] == best_hits]
    elif box:
        eligible = [r for r in rows if r["box_iou"] > 0]
    else:
        eligible = [r for r in rows if r["area_px"] > 0]
        if len(eligible) != 1:
            return None, {"method": "unresolved", "candidates": rows, "reason": "Text-only request needs exactly one seed mask or a target point/box"}
    if not eligible:
        return None, {"method": "unresolved", "candidates": rows, "reason": "No seed candidate overlaps the supplied hint"}
    # A hint box resolves overlapping seed candidates. Without such evidence a
    # tie remains ambiguous rather than using an arbitrary confidence ranking.
    if len(eligible) > 1:
        if not box:
            return None, {"method": "ambiguous", "candidates": rows, "reason": "Multiple masks satisfy identical target point evidence"}
        eligible.sort(key=lambda r: (-r["box_iou"], r["sam_id"]))
        if abs(eligible[0]["box_iou"] - eligible[1]["box_iou"]) < 1e-6:
            return None, {"method": "ambiguous", "candidates": rows, "reason": "Hint-box match is tied"}
    target = eligible[0]["sam_id"]
    return target, {"method": "positive_points" if positive else "hint_box" if box else "single_text_candidate", "selected_sam_id": target, "candidates": rows}


def _save_raw(path, values):
    ids, masks, scores = values
    np.savez_compressed(path, ids=ids, masks=masks, scores=scores)


def _overlay(source, mask, destination, label):
    image = Image.open(source).convert("RGB")
    pixels = np.asarray(image).copy()
    pixels[mask] = (pixels[mask] * .55 + np.array([235, 128, 58]) * .45).astype(np.uint8)
    result = Image.fromarray(pixels)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 26), fill="white")
    draw.text((7, 7), label, fill="#244239")
    result.save(destination)


def make_preview(folder, inputs, frames, rows, target, seed):
    present = [r for r in rows if r["target_area_px"] > 0]
    chosen = {seed}
    for collection in (present, rows):
        if collection:
            for i in np.linspace(0, len(collection) - 1, min(5, len(collection)), dtype=int):
                chosen.add(collection[int(i)]["sample_index"])
    chosen = sorted(chosen)[:10]
    previews = folder / "previews"
    previews.mkdir()
    tiles = []
    for index in chosen:
        source = frames[index]["source_frame_index"]
        with np.load(folder / "target_masks" / f"{source:05d}.npz", allow_pickle=False) as npz:
            mask = npz["mask"]
        file = previews / f"{source:05d}.jpg"
        _overlay(inputs / f"{index:06d}.jpg", mask, file, f"{target} / source {source:05d} / mask {int(mask.sum())} px")
        with Image.open(file) as image:
            tile = image.copy()
            tile.thumbnail((480, 360))
        tiles.append(tile)
    if not tiles:
        return []
    contact = Image.new("RGB", (960, 360 * ((len(tiles) + 1) // 2)), "#edf0ea")
    for i, image in enumerate(tiles):
        contact.paste(image, ((i % 2) * 480, (i // 2) * 360))
    contact.save(previews / "contact.jpg", quality=90)
    return [str((previews / "contact.jpg").relative_to(folder.parent))]


def run_object(predictor, request, frames, inputs, output, *, deadline):
    """Pure predictor API orchestration; used with a fake predictor by CPU tests."""
    folder = Path(output) / request["object_id"]
    folder.mkdir(exist_ok=False)
    (folder / "raw").mkdir()
    (folder / "target_masks").mkdir()
    seed = next(i for i, frame in enumerate(frames) if frame["source_frame_index"] == request["seed_source_frame_index"])
    direction = propagation_direction(seed, len(frames))
    height, width = frames[0]["height"], frames[0]["width"]
    record = {"object_id": request["object_id"], "category": request["category"], "request": request,
              "seed_sample_index": seed, "propagation_direction": direction,
              "status": "running", "frames": [], "duplicates": []}
    by_index = {}
    started = time.monotonic()
    session = predictor.handle_request({"type": "start_session", "resource_path": str(inputs),
                                        "offload_video_to_cpu": True, "offload_state_to_cpu": False})["session_id"]
    try:
        response = predictor.handle_request({"type": "add_prompt", "session_id": session,
                                            "frame_index": seed, "text": request["prompt"], "output_prob_thresh": .5})
        values = arrays(response["outputs"], height, width)
        _save_raw(folder / "seed_text.npz", values)
        target, association = choose_anchor(*values, request)
        record["anchor_association"] = association
        positive, negative = request["positive_points_normalized"], request["negative_points_normalized"]
        if target is None and association["method"] == "point_new_target":
            target = max(values[0].tolist(), default=-1) + 1
            association["selected_sam_id"] = target
        elif target is None:
            record.update(status="needs_prompt", reason=association["reason"], seconds=time.monotonic() - started)
            write_json(folder / "result.json", record)
            return record
        record["selected_sam_id"] = target
        # Remove distractor tracks programmatically. Raw text detections remain
        # archived; is_user_action=False avoids the known propagation shortcut.
        for identity in values[0].tolist():
            if identity != target:
                predictor.handle_request({"type": "remove_object", "session_id": session,
                                          "frame_index": seed, "obj_id": int(identity), "is_user_action": False})
        if positive:
            response = predictor.handle_request({"type": "add_prompt", "session_id": session,
                                                "frame_index": seed, "points": positive + negative,
                                                "point_labels": [1] * len(positive) + [0] * len(negative),
                                                "obj_id": target, "rel_coordinates": True, "output_prob_thresh": .5})
            values = arrays(response["outputs"], height, width)
            _save_raw(folder / "seed_corrected.npz", values)
            if target not in values[0]:
                raise RuntimeError("Point correction did not return the anchored object ID")
        def save_frame(index, data, origin):
            if not isinstance(index, int) or not 0 <= index < len(frames):
                raise ValueError("SAM returned an out-of-manifest sample index")
            ids, masks, scores = arrays(data, height, width)
            locations = np.flatnonzero(ids == target)
            mask = masks[int(locations[0])] if len(locations) else np.zeros((height, width), dtype=bool)
            score = float(scores[int(locations[0])]) if len(locations) and np.isfinite(scores[int(locations[0])]) else None
            source = frames[index]["source_frame_index"]
            destination = folder / "raw" / f"{source:05d}.npz"
            if index in by_index:
                # The explicit seed is authoritative when both propagation
                # directions re-emit it. Retain duplicate output separately.
                duplicate = folder / "raw" / f"{source:05d}_duplicate_{len(record['duplicates']):02d}.npz"
                _save_raw(duplicate, (ids, masks, scores))
                record["duplicates"].append({"sample_index": index, "origin": origin, "file": str(duplicate.relative_to(folder)),
                                             "target_area_px": int(mask.sum())})
                return
            _save_raw(destination, (ids, masks, scores))
            np.savez_compressed(folder / "target_masks" / f"{source:05d}.npz", mask=mask,
                                source_frame_index=np.int64(source), sample_index=np.int64(index),
                                sam_id=np.int64(target), target_present=np.bool_(len(locations) > 0))
            by_index[index] = {"sample_index": index, "source_frame_index": source,
                               "source_sha256": frames[index]["sha256"], "raw_ids": ids.tolist(),
                               "target_present": bool(len(locations)), "target_area_px": int(mask.sum()), "target_score": score,
                               "target_mask": f"{request['object_id']}/target_masks/{source:05d}.npz"}
        # Preserve prompt-conditioned seed rather than losing it to propagation.
        save_frame(seed, {"out_obj_ids": values[0], "out_binary_masks": values[1], "out_probs": values[2]}, "prompt")
        stream_request = {"type": "propagate_in_video", "session_id": session, "propagation_direction": direction,
                          "start_frame_index": seed, "max_frame_num_to_track": None,
                          "output_prob_thresh": .5, "evict_cached_frame_outputs": True}
        for response in predictor.handle_stream_request(stream_request):
            if time.monotonic() > deadline:
                raise TimeoutError("SAM tool exceeded the shared runtime budget")
            save_frame(int(response["frame_index"]), response["outputs"], direction + "_propagation")
            if len(by_index) % 10 == 0:
                print(json.dumps({"event": "object_progress", "object_id": request["object_id"], "frames": len(by_index)}), flush=True)
        if set(by_index) != set(range(len(frames))):
            raise RuntimeError("SAM propagation did not cover every manifest frame")
        record["frames"] = [by_index[i] for i in sorted(by_index)]
        record.update(status="complete", frames_with_target=sum(r["target_area_px"] > 0 for r in record["frames"]),
                      seconds=time.monotonic() - started,
                      identity_scope="Only the seed-anchored SAM ID is assigned to this scene object; later IDs remain unassociated",
                      previews=make_preview(folder, Path(inputs), frames, record["frames"], request["object_id"], seed))
    except Exception as exc:
        context = (f"SAM object {request['object_id']} seed source {request['seed_source_frame_index']} "
                   f"(sample {seed}/{len(frames)-1}), direction={direction}, "
                   f"saved_frames={len(by_index)}: {type(exc).__name__}: {exc}")
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), error_context=context,
                      frames=[by_index[i] for i in sorted(by_index)], seconds=time.monotonic() - started)
        raise RuntimeError(context) from exc
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session})
        write_json(folder / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    request, frames = validate_inputs(read_json(args.request), args.request.parent)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists() or (output / "input_frames").exists():
        raise ValueError("Worker refuses to overwrite any recorded invocation")
    started = time.monotonic()
    deadline = started + request["timeout_s"] - 5
    record = {"status": "running", "started": time.time(), "input_count": len(frames), "objects": [],
              "request_sha256": sha256(args.request), "input_modalities": ["RGB", "text", "points"], "geometry_inputs": [],
              "model": "SAM 3.1 Object Multiplex", "configuration": {"precision": "bfloat16", "compile": False,
              "use_fa3": False, "detection_batch_size": 4, "max_objects_per_session": 8,
              "multiplex_count": 16, "offload_video_to_cpu": True, "offload_state_to_cpu": False}}
    write_json(output / "result.json", record)
    try:
        inputs = output / "input_frames"
        inputs.mkdir()
        for frame in frames:
            source = Path(request["frames_dir"]) / frame["file"]
            target = inputs / f"{frame['sample_index']:06d}.jpg"
            shutil.copyfile(source, target)
            if sha256(target) != frame["sha256"]:
                raise ValueError("Source changed while preparing the SAM input")
        write_json(output / "input_manifest.json", {"frames": frames, "input_modalities": ["RGB", "text", "points"]})
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        free, total = torch.cuda.mem_get_info()
        record["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(), "free_gib": free / 2**30, "total_gib": total / 2**30}
        if free < 12.5 * 2**30:
            raise RuntimeError("SAM requires at least 12.5 GiB free GPU memory for the verified configuration")
        checkpoint_hash = sha256(args.checkpoint)
        if checkpoint_hash != CHECKPOINT_SHA256:
            raise ValueError("SAM checkpoint differs from the verified local 3.1 weight")
        record["checkpoint_sha256"] = checkpoint_hash
        torch.cuda.reset_peak_memory_stats()
        loading = time.monotonic()
        predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(args.checkpoint),
                                                        use_fa3=False, compile=False, max_num_objects=8, multiplex_count=16)
        predictor.model.batched_grounding_batch_size = 4
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = state.get("model", state)
        parameters = dict(predictor.model.named_parameters())
        missing = sorted(set(parameters) - set(state))
        wrong = [key for key in parameters if key in state and parameters[key].shape != state[key].shape]
        record["checkpoint_parameter_coverage"] = {"parameter_tensors": len(parameters), "missing": missing, "shape_errors": wrong}
        if missing or wrong:
            raise RuntimeError("SAM checkpoint parameter coverage failed")
        del parameters, state
        init_state = predictor.model.init_state
        if "offload_state_to_cpu" not in inspect.signature(init_state).parameters:
            def compatible_init_state(*a, offload_state_to_cpu=False, **kw):
                if offload_state_to_cpu:
                    raise NotImplementedError("CPU state offload unsupported by pinned multiplex code")
                return init_state(*a, **kw)
            predictor.model.init_state = compatible_init_state
            record["compatibility_adapter"] = "Ignore only unsupported offload_state_to_cpu=False, matching verified previous run"
        record["output_merge_compatibility"] = install_uncached_refinement_merge(predictor.model)
        record["model"] = "SAM 3.1 Object Multiplex + uncached refinement merge compatibility v1"
        write_json(output / "result.json", record)
        torch.cuda.synchronize()
        record["load_seconds"] = time.monotonic() - loading
        print(json.dumps({"event": "model_loaded", "seconds": record["load_seconds"]}), flush=True)
        for item in request["requests"]:
            if time.monotonic() > deadline:
                raise TimeoutError("SAM request runtime budget expired")
            row = run_object(predictor, item, frames, inputs, output, deadline=deadline)
            if not record["objects"]:
                try:
                    initial_propagation_quality(row, len(frames))
                    row["initial_propagation_quality_gate"] = "passed" if row["status"] == "complete" else "not_resolved"
                except RuntimeError as exc:
                    row.update(status="failed", error=str(exc), initial_propagation_quality_gate="failed")
                    write_json(output / item["object_id"] / "result.json", row)
                    record["objects"].append(row)
                    raise
            record["objects"].append(row)
            write_json(output / "result.json", record)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        record.update(status="complete", all_targets_resolved=all(r["status"] == "complete" for r in record["objects"]),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      limitations=["SAM tracking IDs are not verified physical identity across occlusion.",
                                   "Masks are predictions, not ground truth or three-dimensional geometry.",
                                   "Raw additional tracks are retained and never silently merged into the anchored target."])
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_json(output / "result.json", record)


if __name__ == "__main__":
    main()
