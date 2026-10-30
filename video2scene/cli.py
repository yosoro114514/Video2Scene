import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def local_path(value):
    # Accept Windows paths through the PowerShell WSL launcher.
    if sys.platform != "win32" and len(value) > 2 and value[1:3] in {":\\", ":/"}:
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return Path(value).expanduser().resolve()


def model_options(parser):
    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=120000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--up", choices=["auto", "x", "y", "z", "-x", "-y", "-z"], default="auto"
    )
    parser.add_argument(
        "--segment", choices=["none", "yolo", "grounded_sam2"], default="yolo"
    )
    parser.add_argument(
        "--inventory",
        type=local_path,
        help="Category-only VLM inventory JSON for grounded_sam2",
    )
    parser.add_argument(
        "--mask-cache",
        type=local_path,
        help="Reusable Grounded SAM 2 masks; default OUTPUT/masks",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=20,
        help="Grounding DINO keyframe interval in available ViPE frames",
    )
    parser.add_argument("--grounding-threshold", type=float, default=0.25)
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    parser.add_argument(
        "--semantic-review", choices=["none", "qwen", "file"], default="none"
    )
    parser.add_argument(
        "--review-decisions",
        type=local_path,
        help="Source-bound GPT-6 review JSON for --semantic-review file",
    )
    parser.add_argument("--review-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--review-revision", help="Pin a reviewer checkpoint revision")
    parser.add_argument(
        "--review-cache",
        type=local_path,
        help="Semantic evidence and reviewed masks; default OUTPUT/semantic_review",
    )
    parser.add_argument("--weights", default="yolo11s-seg.pt")
    parser.add_argument(
        "--segment-every", type=int, default=5, help="Segment every N integrated frames"
    )
    parser.add_argument("--segment-device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.35)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'workbench':
        from .pi_workflow.server import main as run_workbench
        sys.argv.pop(1)
        return run_workbench()
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        from .preflight import main as run_check
        return run_check(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        from .demo import main as run_demo
        return run_demo(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'spatiallm':
        from .spatiallm_adapter import main as run_spatiallm
        return run_spatiallm(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='pipeline':
        from .pipeline import main as run_pipeline
        return run_pipeline(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="Video2Scene: use 'demo', 'check' or 'workbench'; legacy geometry commands below"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="Complete pipeline; requires a fresh output directory"
    )
    run.add_argument("--video", type=local_path, required=True)
    run.add_argument("--output", type=local_path, required=True)
    run.add_argument("--fps", type=float, default=10)
    run.add_argument("--frames", type=int, default=160)
    run.add_argument(
        "--max-width", type=int, default=640, help="Maximum image side in pixels"
    )
    run.add_argument(
        "--depth-model",
        choices=["unidepth-s", "unidepth-b", "unidepth-l", "moge2-l"],
        default="unidepth-l",
    )
    run.add_argument(
        "--dynamic-masks",
        action="store_true",
        help="Enable ViPE's additional moving-object mask models",
    )
    model_options(run)
    rec = sub.add_parser(
        "reconstruct",
        help="Build meshes from existing ViPE artifacts, no ViPE/CUDA import",
    )
    rec.add_argument("--vipe-dir", type=local_path, required=True)
    rec.add_argument("--sequence")
    rec.add_argument("--output", type=local_path, required=True)
    model_options(rec)
    infer = sub.add_parser(
        "infer", help="ViPE inference only, on an already resized short video"
    )
    infer.add_argument("--video", type=local_path, required=True)
    infer.add_argument("--output", type=local_path, required=True)
    infer.add_argument("--depth-model", default="unidepth-l")
    infer.add_argument("--buffer", type=int, default=192)
    infer.add_argument("--dynamic-masks", action="store_true")
    build = sub.add_parser("build", help="Rebuild edited scene.json geometric proxies")
    build.add_argument("--scene-json", type=local_path, required=True)
    build.add_argument("--output", type=local_path, required=True)
    view = sub.add_parser("view", help="Serve an exported scene on localhost")
    view.add_argument("directory", type=local_path)
    view.add_argument("--port", type=int, default=8080)
    sub.add_parser("doctor", help="Report runtime packages and CUDA status")
    args = parser.parse_args()
    if args.command == "doctor":
        record = dict(python=sys.version, executable=sys.executable)
        for package in [
            "nvidia-vipe",
            "torch",
            "torchvision",
            "open3d",
            "trimesh",
            "ultralytics",
            "OpenEXR",
        ]:
            try:
                record[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                record[package] = "not installed"
        try:
            import torch

            record["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                record["gpu"] = torch.cuda.get_device_name()
        except ImportError:
            record["cuda_available"] = False
        print(json.dumps(record, indent=2))
        return
    if args.command == "view":
        from functools import partial
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

        if not (args.directory / "index.html").is_file():
            parser.error("No exported viewer at this directory")
        server = ThreadingHTTPServer(
            ("127.0.0.1", args.port),
            partial(SimpleHTTPRequestHandler, directory=str(args.directory)),
        )
        print(f"Open http://localhost:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
        return
    if args.command == "infer":
        from .inference import infer

        infer(
            args.video, args.output, args.depth_model, args.buffer, args.dynamic_masks
        )
        return
    if args.command == "build":
        from .reconstruction import export_models

        spec = json.loads(args.scene_json.read_text(encoding="utf-8"))
        export_models(spec, args.output)
        return
    if args.command == "run":
        from .inference import prepare_video

        if args.output.exists() and any(args.output.iterdir()):
            parser.error(
                "Output directory is not empty. Choose a new output, or use reconstruct for existing ViPE output."
            )
        args.output.mkdir(parents=True, exist_ok=True)
        with args.video.open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        manifest = dict(
            status="preparing",
            source=str(args.video),
            source_sha256=source_hash,
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
        )
        manifest_path = args.output / "run.json"

        def save_status(status):
            manifest["status"] = status
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        start = time.perf_counter()
        try:
            video = args.output / "input.mp4"
            manifest["preparation"] = prepare_video(
                args.video, video, args.fps, args.frames, args.max_width
            )
            save_status("vipe_inference")
            command = [
                sys.executable,
                "-m",
                "video2scene",
                "infer",
                "--video",
                str(video),
                "--output",
                str(args.output / "vipe"),
                "--depth-model",
                args.depth_model,
                "--buffer",
                str(args.frames + 16),
            ]
            if args.dynamic_masks:
                command.append("--dynamic-masks")
            manifest["vipe_command"] = command
            save_status("vipe_inference")
            with (args.output / "vipe.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"ViPE failed; see {args.output / 'vipe.log'}")
            args.vipe_dir, args.sequence = args.output / "vipe", "input"
            save_status("reconstruction")
            reconstruct_args(args, args.output / "scene")
            manifest["elapsed_seconds"] = time.perf_counter() - start
            save_status("complete")
        except Exception as exc:
            manifest["error"] = str(exc)
            save_status("failed")
            raise
    else:
        reconstruct_args(args, args.output)


def reconstruct_args(args, output):
    from .reconstruction import reconstruct

    return reconstruct(
        args.vipe_dir,
        output,
        sequence=args.sequence,
        voxel=args.voxel,
        max_depth=args.max_depth,
        stride=args.stride,
        max_faces=args.max_faces,
        scale=args.scale,
        up=args.up,
        segment=args.segment,
        weights=args.weights,
        segment_every=args.segment_every,
        segment_device=args.segment_device,
        confidence=args.confidence,
        inventory=args.inventory,
        mask_cache=args.mask_cache,
        detect_every=args.detect_every,
        grounding_threshold=args.grounding_threshold,
        sam2_model=args.sam2_model,
        semantic_review=args.semantic_review,
        review_model=args.review_model,
        review_revision=args.review_revision,
        review_cache=args.review_cache,
        review_decisions=args.review_decisions,
    )
