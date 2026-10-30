"""Bounded SAM 3.1 tool contract and Windows/WSL launcher.

Input JSON::

    {"version": 1, "frames_dir": "...", "manifest": "...", "timeout_s": 900,
     "requests": [{"object_id": "sofa_01", "category": "furniture/sofa",
       "prompt": "sofa", "seed_source_frame_index": 1330,
       "positive_points_normalized": [[0.5, 0.6]],
       "negative_points_normalized": [],
       "hint_box_normalized": [0.2, 0.3, 0.6, 0.5]}]}

Manifest: {"frames": [{"source_frame_index": 0, "file": "00000.jpg",
"sha256": "optional but checked when present"}, ...]}. Frames must be strictly
increasing, original RGB JPEGs of one size. All coordinates are x/y in [0, 1].
An independent session is used for each scene object. Only its anchored SAM ID
is exported as the target mask; additional category IDs never become that
scene object's observations. The raw outputs remain available for inspection.

CLI: python -m video2scene.agent_scene.sam_tool REQUEST.json NEW_OUTPUT
No inference happens on import or during request validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from video2scene.config import load_settings, gpu_environment


CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
DEFAULT_CHECKPOINT = load_settings().sam_checkpoint
DEFAULT_PYTHON = load_settings().sam_python
MAX_FRAMES = 160
MAX_OBJECTS = 12
MAX_TIMEOUT = 1800


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _points(value, name):
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError(f"{name} must contain at most 16 normalized points")
    for point in value:
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or not 0 <= x <= 1 for x in point)):
            raise ValueError(f"{name} requires finite normalized [x,y] coordinates")
    return value


def validate_request(value):
    """Validate the data-only contract without loading models or accessing paths."""
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("SAM tool request requires version=1")
    allowed = {"version", "frames_dir", "manifest", "requests", "timeout_s"}
    if set(value) - allowed:
        raise ValueError("Unsupported SAM tool request fields: " + ", ".join(sorted(set(value) - allowed)))
    result = copy.deepcopy(value)
    for name in ("frames_dir", "manifest"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"{name} must be a nonempty path")
    timeout = result.setdefault("timeout_s", 900)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 30 <= timeout <= MAX_TIMEOUT:
        raise ValueError(f"timeout_s must be an integer between 30 and {MAX_TIMEOUT}")
    requests = result.get("requests")
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_OBJECTS:
        raise ValueError(f"requests must contain 1–{MAX_OBJECTS} objects")
    identities = set()
    item_fields = {"object_id", "category", "prompt", "seed_source_frame_index",
                   "positive_points_normalized", "negative_points_normalized", "hint_box_normalized"}
    for request in requests:
        if not isinstance(request, dict) or set(request) - item_fields:
            raise ValueError("Unsupported object request fields or prompt mode")
        identity = request.get("object_id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", identity):
            raise ValueError("object_id must be a path-safe stable identifier")
        if identity in identities:
            raise ValueError("Duplicate scene object_id")
        identities.add(identity)
        for name in ("category", "prompt"):
            if not isinstance(request.get(name), str) or not 1 <= len(request[name].strip()) <= 160:
                raise ValueError(f"{name} must be a nonempty string up to 160 characters")
        seed = request.get("seed_source_frame_index")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed_source_frame_index must be a nonnegative integer")
        _points(request.setdefault("positive_points_normalized", []), "positive_points_normalized")
        _points(request.setdefault("negative_points_normalized", []), "negative_points_normalized")
        if request["negative_points_normalized"] and not request["positive_points_normalized"]:
            raise ValueError("Negative-only point prompts cannot identify a target")
        box = request.get("hint_box_normalized")
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in box)
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0] + box[2] > 1 or box[1] + box[3] > 1):
                raise ValueError("hint_box_normalized must be [x,y,width,height] inside the image")
    return result


def validate_inputs(value, request_dir):
    """Freeze the checked temporal frame mapping and verify optional source hashes."""
    from PIL import Image

    result = validate_request(value)
    request_dir = Path(request_dir).resolve()
    for name in ("frames_dir", "manifest"):
        result[name] = str((request_dir / result[name]).resolve())
    directory = Path(result["frames_dir"])
    manifest = read_json(result["manifest"])
    frames = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"Manifest must contain 1–{MAX_FRAMES} frames")
    previous = -1
    shape = None
    frozen = []
    files = set()
    for i, frame in enumerate(frames):
        index = frame.get("source_frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index <= previous:
            raise ValueError("source_frame_index must be unique and strictly increasing")
        previous = index
        name = frame.get("file")
        if not isinstance(name, str):
            raise ValueError("Each frame requires file")
        source = (directory / name).resolve()
        if not source.is_relative_to(directory) or source.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("Each source must be an RGB JPEG within frames_dir")
        if source in files:
            raise ValueError("Each source JPEG must occur only once")
        files.add(source)
        digest = sha256(source)
        if frame.get("sha256") and frame["sha256"] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        with Image.open(source) as image:
            image.load()
            if image.mode != "RGB" or image.format != "JPEG":
                raise ValueError("Sources must be unchanged RGB JPEGs")
            if shape is None:
                shape = image.size
            if image.size != shape:
                raise ValueError("All source frames must have identical dimensions")
        frozen.append({"sample_index": i, "source_frame_index": index,
                       "file": name, "sha256": digest, "width": shape[0], "height": shape[1]})
    indices = {f["source_frame_index"] for f in frozen}
    for request in result["requests"]:
        if request["seed_source_frame_index"] not in indices:
            raise ValueError(f"Seed is not in the supplied frame manifest: {request['object_id']}")
    return result, frozen


def to_wsl(path):
    value = str(Path(path).resolve())
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return value


def launch(request_path, output, *, python=None, distro=None, checkpoint=None):
    """Execute one explicit invocation. Existing output directories are rejected."""
    settings = load_settings()
    python, distro = python or settings.sam_python, distro or settings.wsl_distro
    checkpoint = checkpoint or settings.sam_checkpoint
    if not checkpoint:
        raise ValueError('Set V2S_SAM_CHECKPOINT or sam_checkpoint in video2scene.local.toml')
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    value, frames = validate_inputs(read_json(request_path), request_path.parent)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", value)
    write_json(output / "input_manifest.json", {"frames": frames, "input_modalities": ["RGB", "text", "points"]})
    worker_request = copy.deepcopy(value)
    worker_request.update(frames_dir=to_wsl(value["frames_dir"]), manifest=to_wsl(output / "input_manifest.json"))
    write_json(output / "worker_request.json", worker_request)
    worker = Path(__file__).with_name("sam_worker.py")
    argv = (["wsl", "-d", distro, "--exec"] if os.name == 'nt' else []) + ["env", *gpu_environment(settings),
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
            "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "PYTHONUNBUFFERED=1", "timeout",
            "--signal=TERM", "--kill-after=20s", f"{value['timeout_s']}s", python, "-u", to_wsl(worker),
            "--request", to_wsl(output / "worker_request.json"), "--output", to_wsl(output),
            "--checkpoint", checkpoint]
    record = {"status": "running", "started": time.time(), "command": argv,
              "source_request_sha256": sha256(request_path), "worker_sha256": sha256(worker),
              "launcher_sha256": sha256(__file__), "source_frame_count": len(frames),
              "requested_objects": len(value["requests"]), "network_downloads_allowed": False,
              "geometry_inputs": [], "checkpoint_expected_sha256": CHECKPOINT_SHA256}
    write_json(output / "execution.json", record)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        record["pid"] = process.pid
        write_json(output / "execution.json", record)
        try:
            code = process.wait(timeout=value["timeout_s"] + 90)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            record.update(status="failed", error="WSL wrapper exceeded timeout; inspect worker process state before retry",
                          seconds=time.time() - record["started"])
            write_json(output / "execution.json", record)
            raise
    record.update(status="complete" if code == 0 else "failed", returncode=code,
                  seconds=time.time() - record["started"])
    write_json(output / "execution.json", record)
    if code:
        raise RuntimeError(f"SAM worker failed ({code}); see {output / 'worker.log'}")
    result = read_json(output / "result.json")
    if result.get("status") != "complete":
        raise RuntimeError("SAM worker exited without a complete result")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(launch(args.request, args.output), ensure_ascii=False))
