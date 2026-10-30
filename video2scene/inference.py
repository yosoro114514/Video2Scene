"""Video preparation and an isolated ViPE inference worker."""

import json
from pathlib import Path

import cv2


def prepare_video(source, target, fps=10.0, max_frames=160, max_width=640):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    if fps <= 0 or max_frames < 12 or max_width < 128:
        raise ValueError("Require fps > 0, max_frames >= 12 and max_width >= 128")
    capture = cv2.VideoCapture(str(source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or source_fps <= 0:
        raise ValueError(f"Cannot decode video: {source}")
    step = max(1, round(source_fps / fps))
    output_fps = source_fps / step
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    sampled_ids = []
    try:
        index = 0
        while len(sampled_ids) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                h, w = frame.shape[:2]
                factor = min(1.0, max_width / max(h, w))
                size = (
                    max(16, int(w * factor) // 16 * 16),
                    max(16, int(h * factor) // 16 * 16),
                )
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(target), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create {target}")
                writer.write(frame)
                sampled_ids.append(index)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if len(sampled_ids) < 12:
        raise ValueError("At least 12 decodable sampled frames are required for SLAM")
    record = dict(
        source=str(source.resolve()),
        source_fps=source_fps,
        output_fps=output_fps,
        sampled_source_indices=sampled_ids,
        frames=len(sampled_ids),
        resolution=list(size),
    )
    target.with_suffix(".json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def infer(video, output, depth_model="unidepth-l", buffer=192, dynamic_masks=False):
    # Keep CUDA dependencies out of the CPU-only reconstruction path.
    import torch
    from vipe import make_pipeline
    from vipe.config import parse_typed_config
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.logging import configure_logging

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ViPE inference requires an NVIDIA CUDA GPU; use reconstruct for existing artifacts"
        )
    configure_logging()
    overrides = [
        "pipeline=no_vda",
        f"pipeline.output.path={Path(output).resolve()}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        "pipeline.output.save_slam_map=true",
        f"pipeline.slam.buffer={buffer}",
        f"pipeline.slam.keyframe_depth={depth_model}",
        f"pipeline.post.depth_align_model=adaptive_{depth_model}",
        "pipeline.init.async_prefetch=false",
    ]
    if not dynamic_masks:
        overrides.append("pipeline.init.instance=null")
    args = parse_typed_config("default", hydra_args=overrides)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "inference_config.json").write_text(
        json.dumps({"overrides": overrides}, indent=2)
    )
    make_pipeline(args.pipeline).run(RawMp4Stream(Path(video)))
