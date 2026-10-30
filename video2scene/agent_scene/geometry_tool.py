"""RGB preparation and explicitly recorded ViPE tool execution."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .planning import ROOT, KEYFRAMES, read, save
from video2scene.codex_review import digest


def prepare(run_root):
    run = Path(run_root).resolve()
    manifest_path = run / 'frames/manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        for frame in manifest['frames']:
            if digest(run / 'frames' / frame['file']) != frame['sha256']:
                raise ValueError('Prepared RGB frame changed')
        return manifest
    # Include every GPT-observed frame exactly; no nearest-frame depth substitution.
    indices = sorted(set(np.linspace(0, 2869, 148).round().astype(int).tolist()) | set(KEYFRAMES))
    if len(indices) > 160:
        raise ValueError('Prepared sequence exceeds bounded frame count')
    folder = run / 'frames'
    folder.mkdir(parents=True, exist_ok=True)
    video = run / 'prepared/input.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    if video.exists():
        raise ValueError('Incomplete preparation exists; inspect before rewriting')
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (640, 480))
    if not writer.isOpened():
        raise RuntimeError('Could not open prepared RGB video writer')
    frames = []
    try:
        for index, source_index in enumerate(indices):
            source = ROOT / 'data/redwood_livingroom1_rgb_20261028/rgb' / f'{source_index:05d}.jpg'
            image = cv2.imread(str(source))
            if image is None or image.shape != (480,640,3):
                raise ValueError('Unexpected source RGB dimensions')
            name = f'{index:06d}.jpg'
            shutil.copyfile(source, folder / name)
            frames.append({'frame_index': index, 'source_frame_index': source_index,
                           'file': name, 'source': str(source), 'sha256': digest(source)})
            writer.write(image)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != len(frames):
        raise ValueError('Prepared video frame count mismatch')
    manifest = {'frames': frames, 'source': 'original RGB only',
                'preview_fps': 10, 'source_timestamps': 'unknown',
                'prepared_video': str(video), 'prepared_video_sha256': digest(video),
                'ground_truth_inputs': []}
    save(manifest_path, manifest)
    return manifest


def linux(path):
    value = str(Path(path).resolve())
    return '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')


def run_vipe(run_root):
    run = Path(run_root).resolve()
    manifest = prepare(run)
    prior = read(run / 'prior.json')
    if not prior['tool_plan']['call_vipe']:
        raise ValueError('GPT did not request ViPE; do not silently execute it')
    result_path = run / 'tools/vipe.json'
    if result_path.exists():
        result = read(result_path)
        if result['status'] != 'complete':
            raise RuntimeError('ViPE tool already attempted; inspect before rerun')
        return result
    command = ['wsl', '-d', 'Ubuntu-22.04', '--exec', 'bash',
               linux(ROOT / 'experiments/redwood_v1/run_rgb_only.sh'), linux(run), 'inference']
    result = {'tool': 'estimate_geometry_vipe', 'requested_by': 'calls/prior/response.json',
              'decision_sha256': digest(run / 'prior.json'), 'reason': prior['tool_plan']['vipe_reason'],
              'mode': 'fresh_inference', 'command': command, 'started': time.time(), 'status': 'running',
              'input_sha256': manifest['prepared_video_sha256'], 'frames': len(manifest['frames']),
              'ground_truth_inputs': [], 'source_modalities': ['RGB']}
    save(result_path, result)
    log = run / 'tools/vipe.log'
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
        result['pid'] = process.pid
        save(result_path, result)
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=15)
            result.update(status='failed', error='ViPE launcher timeout; verify WSL child outcome before retry')
            save(result_path, result)
            raise
    result.update(status='complete' if code == 0 else 'failed', returncode=code, seconds=time.time()-result['started'])
    if code == 0:
        from video2scene.artifacts import ViPEArtifacts
        artifacts = ViPEArtifacts(run / 'vipe')
        if artifacts.indices != list(range(len(manifest['frames']))):
            result.update(status='failed', error='ViPE frame-index coverage differs from prepared RGB')
        result['artifacts'] = [{'path': str(p.relative_to(run)), 'sha256': digest(p)} for p in sorted((run/'vipe').rglob('*')) if p.is_file() and p.suffix in {'.npz','.zip','.mp4','.txt'}]
    save(result_path, result)
    if result['status'] != 'complete':
        raise RuntimeError('Actual ViPE inference failed; see tool log')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    result = prepare(args.run_root) if args.prepare_only else run_vipe(args.run_root)
    print(json.dumps({'status': result.get('status','prepared'), 'frames': len(result['frames']) if isinstance(result['frames'],list) else result['frames']}, ensure_ascii=False), flush=True)
