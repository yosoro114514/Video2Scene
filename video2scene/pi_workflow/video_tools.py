"""Project-scoped RGB import and explicit, bounded ViPE/SAM tools for Pi.

No model planner runs here. JPEG coordinates are the resized source image;
normalized prompts therefore refer to the same view as original video RGB.
Completed tool jobs are immutable. ``latest.json`` files are pointers only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw

from video2scene.agent_scene.sam_tool import (
    CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, DEFAULT_PYTHON, to_wsl,
    validate_inputs, write_json,
)
from video2scene.config import load_settings, gpu_environment, vipe_environment

ROOT = Path(__file__).resolve().parents[2]
DISTRO = 'Ubuntu-22.04'
PREPARED_FPS = 10.0
MAX_IMAGE_EDGE = 960
IMPORT_VERSION = 1
VIPE_SAMPLING_VERSION = 'vipe_dense_v1'
MAX_DENSE_FRAMES = 2048
_GPU_TOOL_LOCK = threading.Lock()


class VideoToolCancelled(RuntimeError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _cancelled(cancel):
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    if hasattr(cancel, 'is_set'):
        return bool(cancel.is_set())
    raise TypeError('cancel must be a callable or threading.Event')


def _job_id(value=None):
    value = value or ('job_' + uuid.uuid4().hex)
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('Invalid tool job_id')
    return value


def _files(paths, base):
    return [{'path': str(Path(path).resolve().relative_to(base)), 'sha256': _hash(path)} for path in paths]


def _verify_files(rows, base):
    for row in rows:
        path = (base / row['path']).resolve()
        if not path.is_relative_to(base) or not path.is_file() or _hash(path) != row['sha256']:
            raise ValueError('Recorded tool artifact changed or is missing: ' + row['path'])


def _contact(rows, output, *, title, image_key='path', columns=3):
    width, height = 320, 266
    sheet = Image.new('RGB', (columns * width, max(1, math.ceil(len(rows) / columns)) * height + 28), '#edf1ee')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill='#203a31')
    for i, row in enumerate(rows):
        with Image.open(row[image_key]) as source:
            tile = source.convert('RGB')
            tile.thumbnail((width, height - 28))
        x, y = (i % columns) * width, 28 + (i // columns) * height
        sheet.paste(tile, (x + (width - tile.width) // 2, y))
        label = f"source {row['source_frame_index']} | {row['source_time_s']:.3f} s"
        draw.text((x + 8, y + height - 22), label, fill='#203a31')
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def _prepared(run_root):
    run = Path(run_root).resolve()
    path = run / 'frames/manifest.json'
    manifest = _read(path)
    if manifest.get('preparation_version') != IMPORT_VERSION:
        raise ValueError('Project has no compatible video preparation manifest')
    _verify_files([{'path': 'frames/' + row['file'], 'sha256': row['sha256']} for row in manifest['frames']], run)
    video = run / 'prepared/input.mp4'
    if _hash(video) != manifest['prepared_video_sha256']:
        raise ValueError('Prepared RGB video changed')
    return run, manifest


def _import_result(run, manifest, *, cached):
    return {'status': 'complete', 'cached': cached, 'tool': 'prepare_video',
            'manifest_path': str(run / 'frames/manifest.json'), 'frames_dir': str(run / 'frames'),
            'prepared_video': str(run / 'prepared/input.mp4'), 'receipt_path': str(run / 'tools/import/receipt.json'),
            'frame_count': len(manifest['frames']), 'source_frame_count': manifest['source_frame_count'],
            'source_fps': manifest['source_fps'], 'duration_s': manifest['duration_s'],
            'keyframes': copy.deepcopy(manifest['keyframes']), 'previews': [str(run / 'prepared/keyframes.jpg')]}


def prepare_video(run_root, source_path, frame_budget=96, keyframe_count=12):
    """Freeze uniformly sampled frames from a real local video, without inference."""
    if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or not 2 <= frame_budget <= 160:
        raise ValueError('frame_budget must be an integer in [2,160]')
    if isinstance(keyframe_count, bool) or not isinstance(keyframe_count, int) or not 1 <= keyframe_count <= min(32, frame_budget):
        raise ValueError('keyframe_count must be in [1,min(32,frame_budget)]')
    run, source = Path(run_root).resolve(), Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError('Video input is not a file: ' + str(source))
    source_hash = _hash(source)
    config = {'frame_budget': frame_budget, 'keyframe_count': keyframe_count,
              'max_image_edge': MAX_IMAGE_EDGE, 'prepared_fps': PREPARED_FPS}
    manifest_path = run / 'frames/manifest.json'
    if manifest_path.exists():
        _, manifest = _prepared(run)
        if manifest['source_sha256'] != source_hash or manifest['configuration'] != config:
            raise ValueError('Project input/configuration is already frozen; create a new project')
        _verify_files(_read(run / 'tools/import/receipt.json')['artifacts'], run)
        return _import_result(run, manifest, cached=True)
    if (run / 'frames').exists() or (run / 'prepared').exists():
        raise ValueError('Incomplete video preparation exists; do not overwrite its evidence')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Input could not be decoded as a video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no usable frame-rate metadata')
    count = 0
    try:
        while cap.grab():
            count += 1
    finally:
        cap.release()
    if count < 2:
        raise ValueError('Video must contain at least two decodable frames')
    indices = np.linspace(0, count - 1, min(frame_budget, count)).round().astype(int).tolist()
    wanted = set(indices)
    (run / 'frames').mkdir(parents=True)
    (run / 'prepared').mkdir()
    receipt_path = run / 'tools/import/receipt.json'
    receipt = {'tool': 'prepare_video', 'status': 'running', 'started': time.time(),
               'source_path': str(source), 'source_sha256': source_hash, 'configuration': config,
               'actual_inference': False, 'source_modalities': ['RGB'], 'ground_truth_inputs': []}
    write_json(receipt_path, receipt)
    cap, writer, rows = cv2.VideoCapture(str(source)), None, []
    try:
        for source_index in range(count):
            if source_index not in wanted:
                if not cap.grab():
                    raise ValueError('Video changed or failed during decoding')
                continue
            ok, image = cap.read()
            if not ok:
                raise ValueError('Could not decode sampled source frame ' + str(source_index))
            source_height, source_width = image.shape[:2]
            ratio = min(1., MAX_IMAGE_EDGE / max(source_height, source_width))
            width, height = max(2, int(round(source_width * ratio / 2)) * 2), max(2, int(round(source_height * ratio / 2)) * 2)
            if rows and (source_width, source_height, width, height) != (rows[0]['source_width'], rows[0]['source_height'], rows[0]['width'], rows[0]['height']):
                raise ValueError('Changing video frame dimensions are unsupported')
            if (height, width) != (source_height, source_width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(run / 'prepared/input.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), PREPARED_FPS, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Prepared video encoder unavailable')
            sample_index = len(rows)
            name = f'{sample_index:06d}.jpg'
            frame_path = run / 'frames' / name
            if not cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                raise RuntimeError('Could not write prepared RGB frame')
            # Encode the exact JPEG decoded pixels so both tools consume aligned views.
            writer.write(cv2.imread(str(frame_path)))
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            valid_time = math.isfinite(timestamp) and timestamp >= 0 and (not rows or timestamp > rows[-1]['source_time_s'])
            basis = 'decoder_timestamp' if valid_time else 'source_frame_index / nominal_fps'
            if not valid_time:
                timestamp = source_index / fps
            rows.append({'frame_index': sample_index, 'sample_index': sample_index,
                         'source_frame_index': source_index, 'source_time_s': timestamp, 'time_basis': basis,
                         'source_fps': fps, 'source_width': source_width, 'source_height': source_height,
                         'width': width, 'height': height, 'preview_time_s': sample_index / PREPARED_FPS,
                         'file': name, 'path': str(frame_path), 'sha256': _hash(frame_path),
                         'source_to_prepared': [[width/source_width,0,0],[0,height/source_height,0],[0,0,1]]})
        writer.release()
        writer = None
        if len(rows) != len(indices) or _hash(source) != source_hash:
            raise ValueError('Source video changed or sampled frame count differs')
        verify = cv2.VideoCapture(str(run / 'prepared/input.mp4'))
        decoded = 0
        try:
            while verify.grab():
                decoded += 1
        finally:
            verify.release()
        if decoded != len(rows):
            raise ValueError('Prepared video frame count differs from manifest')
        keys = np.linspace(0, len(rows)-1, min(keyframe_count,len(rows))).round().astype(int).tolist()
        keyframes = [rows[i] for i in keys]
        _contact(keyframes, run / 'prepared/keyframes.jpg', title='Original video keyframes')
        manifest = {'preparation_version': IMPORT_VERSION, 'source_path': str(source), 'source_sha256': source_hash,
                    'source_frame_count': count, 'source_fps': fps, 'duration_s': count/fps,
                    'source_width': rows[0]['source_width'], 'source_height': rows[0]['source_height'],
                    'prepared_fps': PREPARED_FPS, 'preview_fps': PREPARED_FPS,
                    'prepared_video': str(run/'prepared/input.mp4'), 'prepared_video_sha256': _hash(run/'prepared/input.mp4'),
                    'configuration': config, 'frames': rows, 'keyframes': keyframes,
                    'source_modalities': ['RGB'], 'ground_truth_inputs': []}
        write_json(manifest_path, manifest)
        artifacts = [manifest_path, run/'prepared/input.mp4', run/'prepared/keyframes.jpg'] + [run/'frames'/r['file'] for r in rows]
        receipt.update(status='complete', seconds=time.time()-receipt['started'], frame_count=len(rows),
                       artifacts=_files(artifacts, run), manifest_path=str(manifest_path))
        write_json(receipt_path, receipt)
        return _import_result(run, manifest, cached=False)
    except BaseException as exc:
        receipt.update(status='failed', error=str(exc), seconds=time.time()-receipt['started'])
        write_json(receipt_path, receipt)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


# The supervisor lives in the immutable job directory. It owns exactly one
# Linux process group, including child GPU workers, and records its exit.
_SUPERVISOR = '''import json,os,signal,subprocess,sys,time
from pathlib import Path
spec_path=Path(sys.argv[1]);spec=json.loads(spec_path.read_text());root=spec_path.parent
def save(name,value):
 p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)
started=time.monotonic();p=subprocess.Popen(spec['argv'],start_new_session=True)
save('linux_process.json',{'pid':p.pid,'process_group':p.pid,'job_token':spec['job_token']})
status='complete'
try:
 while p.poll() is None:
  if (root/'cancel.request').exists():status='cancelled';break
  if time.monotonic()-started>spec['timeout_s']:status='timeout';break
  time.sleep(.2)
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
finally:
 code=p.poll()
 save('process_result.json',{'status':status if status!='complete' or code==0 else 'failed','returncode':code,'seconds':time.monotonic()-started,'job_token':spec['job_token']})
sys.exit(130 if status=='cancelled' else 124 if status=='timeout' else code)
'''


def _run_wsl(argv, job, timeout_s, cancel):
    """Serialize GPU tools across projects served by this host process."""
    queued = time.monotonic()
    while not _GPU_TOOL_LOCK.acquire(timeout=.2):
        if _cancelled(cancel):
            raise VideoToolCancelled('Tool cancelled while waiting for the GPU tool slot')
        if time.monotonic()-queued > timeout_s:
            raise TimeoutError('GPU tool slot was busy for the request timeout')
    try:
        return _run_wsl_unlocked(argv, job, timeout_s, cancel)
    finally:
        _GPU_TOOL_LOCK.release()


def _run_wsl_unlocked(argv, job, timeout_s, cancel):
    if _cancelled(cancel):
        raise VideoToolCancelled('Tool cancelled before execution')
    token = uuid.uuid4().hex
    source = job / 'supervisor.py'
    source.write_text(_SUPERVISOR, encoding='utf-8')
    spec_path = job / 'process.json'
    write_json(spec_path, {'argv': argv, 'timeout_s': timeout_s, 'job_token': token})
    command = ['wsl', '-d', load_settings().wsl_distro, '--exec', '/usr/bin/python3', '-u', to_wsl(source), to_wsl(spec_path)]
    if os.name != 'nt':
        command = ['/usr/bin/python3', '-u', str(source), str(spec_path)]
    started = time.monotonic()
    with (job/'worker.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write_json(job/'launcher.json', {'command': command, 'pid': process.pid, 'job_token': token,
                                         'supervisor_sha256': _hash(source)})
        try:
            while process.poll() is None:
                if _cancelled(cancel) and not (job/'cancel.request').exists():
                    write_json(job/'cancel.request', {'reason':'caller_cancelled','time':time.time()})
                if time.monotonic()-started > timeout_s+45:
                    write_json(job/'cancel.request', {'reason':'supervisor_timeout','time':time.time()})
                    raise TimeoutError('WSL supervisor did not return; inspect this job process receipt before retry')
                time.sleep(.2)
        except BaseException:
            if not (job/'cancel.request').exists():
                write_json(job/'cancel.request', {'reason':'launcher_interrupted','time':time.time()})
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            raise
    result = _read(job/'process_result.json') if (job/'process_result.json').exists() else {'status':'failed','returncode':process.returncode}
    if result.get('job_token') not in (None, token):
        raise RuntimeError('Worker process receipt belongs to another invocation')
    if result['status']=='cancelled':
        raise VideoToolCancelled('Tool worker cancelled and its process group stopped')
    if process.returncode or result['status']!='complete':
        raise RuntimeError('Tool worker failed: '+json.dumps(result))
    return result


def _begin_job(run, tool, identity, binding, request):
    job = run/'tools'/tool/_job_id(identity)
    receipt_path = job/'receipt.json'
    if job.exists():
        if not receipt_path.exists():
            raise ValueError('Incomplete tool job exists; use a new job_id')
        receipt = _read(receipt_path)
        if receipt.get('status')!='complete' or receipt.get('binding')!=binding:
            raise ValueError(f'Tool job_id {job.name} is already bound to another request or incomplete attempt; '
                             f'use a new job_id for an explicit retry. Existing receipt: {receipt_path}')
        _verify_files(receipt['artifacts'], run)
        result=copy.deepcopy(receipt['result']);result['cached']=True
        return job, receipt, result
    job.mkdir(parents=True)
    receipt={'tool':tool,'job_id':job.name,'status':'running','started':time.time(),'binding':binding,
             'request':request,'actual_inference':True,'source_modalities':['RGB'],'ground_truth_inputs':[]}
    write_json(receipt_path,receipt)
    return job,receipt,None


def _finish_job(run, job, receipt, result):
    paths=sorted(p for p in job.rglob('*') if p.is_file() and p.name not in {'receipt.json','worker.log'})
    receipt.update(status='complete',seconds=time.time()-receipt['started'],result=result,artifacts=_files(paths,run))
    write_json(job/'receipt.json',receipt)
    return result


def _fail_job(job,receipt,exc):
    receipt.update(status='cancelled' if isinstance(exc,VideoToolCancelled) else 'failed',error=str(exc),
                   seconds=time.time()-receipt['started'])
    write_json(job/'receipt.json',receipt)


def _sam_failure_details(output):
    """Expose the worker cause without returning its potentially huge frame list."""
    path=output/'result.json'
    if not path.exists():return None
    try:raw=_read(path)
    except (OSError,ValueError):return None
    if raw.get('status')!='failed':return None
    return {'result_path':str(path),'error_type':raw.get('error_type','WorkerError'),
            'error':str(raw.get('error','Unknown SAM worker failure'))[:2000],
            'traceback_tail':str(raw.get('traceback',''))[-1600:],
            'completed_object_ids':[row['object_id'] for row in raw.get('objects',[]) if row.get('status')=='complete']}


def _geometry_preview(run, job, artifacts, manifest):
    wanted=set(np.linspace(0,len(artifacts.indices)-1,min(6,len(artifacts.indices))).round().astype(int))
    rows=[];depth_ranges=[]
    for frame in artifacts.frames(indices=sorted(wanted)):
        if frame.index not in wanted:continue
        source=manifest['frames'][frame.index]
        valid=frame.depth[np.isfinite(frame.depth)&(frame.depth>0)]
        if not len(valid):raise ValueError('ViPE returned no valid depth on a preview frame')
        lo,hi=np.quantile(valid,[.02,.98]);depth_ranges.append([float(lo),float(hi)])
        normalized=np.clip((frame.depth-lo)/max(float(hi-lo),.001),0,1)
        normalized=np.nan_to_num(normalized,nan=0,posinf=1,neginf=0)
        colored=cv2.cvtColor(cv2.applyColorMap((normalized*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
        tile=Image.fromarray(np.concatenate([frame.rgb,colored],axis=1))
        path=job/'previews'/f"frame-{source['source_frame_index']:06d}.jpg";path.parent.mkdir(exist_ok=True)
        tile.save(path,quality=88)
        rows.append({**source,'path':str(path)})
    contact=job/'previews/geometry.jpg';_contact(rows,contact,title='ViPE RGB / estimated depth',columns=2)
    return [str(contact)],{'preview_depth_ranges_m':depth_ranges,'scale_status':'estimated','camera_frame':'OpenCV right/down/forward in ViPE world'}


def _dense_plan(manifest, target_fps=10.):
    if isinstance(target_fps,bool) or not isinstance(target_fps,(int,float)) or not math.isfinite(target_fps) or not 1<=target_fps<=30:
        raise ValueError('dense_fps must be finite and in [1,30]')
    source_fps=float(manifest['source_fps']);count=manifest['source_frame_count']
    # Floor the stride so a fractional nominal FPS never yields source gaps
    # larger than the requested interval (e.g. 29.97 fps uses stride 2).
    stride=max(1,int(math.floor(source_fps/target_fps+1e-8)))
    observations={row['source_frame_index']:row for row in manifest['frames']}
    if len(observations)!=len(manifest['frames']) or any(not 0<=i<count for i in observations):
        raise ValueError('Observation manifest source indices are duplicated or outside the video')
    indices=sorted(set(range(0,count,stride))|set(observations)|{0,count-1})
    if len(indices)>MAX_DENSE_FRAMES:
        raise ValueError(f'Dense ViPE sequence needs {len(indices)} frames; this standard full-graph tool is bounded to {MAX_DENSE_FRAMES}. Use a shorter source clip, not sparse observation frames.')
    required_buffer=len(indices)+16  # configured infill_chunk_size=16
    buffer=max(176,int(math.ceil(required_buffer/32))*32)
    return {'sampling_version':VIPE_SAMPLING_VERSION,'target_fps':float(target_fps),'source_stride':stride,
            'selection':'uniform_source_time_grid_union_required_observations','source_indices':indices,
            'required_observation_source_indices':sorted(observations),'dense_frame_count':len(indices),
            'required_observation_count':len(observations),'encoded_fps':len(indices)/(count/source_fps),
            'source_gap_limit_s':max(1/float(target_fps),1/source_fps),'slam_buffer':buffer,
            'infill_chunk_size':16,'graph_buffer_only_estimate_gib':buffer*(384*512*6+(384//8)*(512//8)*(768+9))/(2**30),
            'buffer_basis':'Worst case all dense frames become keyframes, plus 16 infill slots; graph-memory estimate excludes model and BA allocations'}


def _prepare_dense_clip(run, manifest, destination, plan, cancel=None):
    """Decode original RGB into a separate dense clip; observation files stay frozen."""
    source=Path(manifest['source_path']).resolve()
    if not source.is_file() or _hash(source)!=manifest['source_sha256']:
        raise ValueError('Original source video is missing or changed; dense ViPE cannot use the sparse observation preview')
    observation_path=run/'frames/manifest.json'
    observation_hash=_hash(observation_path)
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        existing=_read(manifest_path)
        if existing.get('sampling')!=plan or existing.get('observation_manifest_sha256')!=observation_hash or existing.get('source_sha256')!=manifest['source_sha256']:
            raise ValueError('Existing dense clip is bound to different inputs')
        if _hash(destination/'input.mp4')!=existing['prepared_video_sha256']:
            raise ValueError('Existing dense ViPE input changed')
        return existing
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Incomplete dense preparation exists; do not overwrite it')
    destination.mkdir(parents=True,exist_ok=True)
    observations={row['source_frame_index']:row for row in manifest['frames']}
    wanted=set(plan['source_indices']);rows=[]
    width,height=manifest['frames'][0]['width'],manifest['frames'][0]['height']
    cap=cv2.VideoCapture(str(source));writer=None
    try:
        if not cap.isOpened():raise ValueError('Original video cannot be decoded for dense ViPE')
        writer=cv2.VideoWriter(str(destination/'input.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),plan['encoded_fps'],(width,height))
        if not writer.isOpened():raise RuntimeError('Dense ViPE video encoder unavailable')
        for source_index in range(manifest['source_frame_count']):
            if _cancelled(cancel):raise VideoToolCancelled('Dense video preparation cancelled')
            if source_index not in wanted:
                if not cap.grab():raise ValueError('Dense source video ended before its recorded frame count')
                continue
            ok,image=cap.read()
            if not ok:raise ValueError('Dense source frame could not be decoded: '+str(source_index))
            source_height,source_width=image.shape[:2]
            if (source_width,source_height)!=(manifest['source_width'],manifest['source_height']):
                raise ValueError('Dense source dimensions differ from the frozen observation video')
            if (width,height)!=(source_width,source_height):image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
            timestamp=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.
            valid=math.isfinite(timestamp) and timestamp>=0 and (not rows or timestamp>rows[-1]['source_time_s'])
            basis='decoder_timestamp' if valid else 'source_frame_index / nominal_fps'
            if not valid:timestamp=source_index/manifest['source_fps']
            observation=observations.get(source_index)
            if observation and abs(timestamp-observation['source_time_s'])>1e-5:
                raise ValueError('Dense/observation timestamps disagree for original frame '+str(source_index))
            dense_index=len(rows)
            rows.append({'frame_index':dense_index,'dense_frame_index':dense_index,'source_frame_index':source_index,
                         'source_time_s':timestamp,'time_basis':basis,'source_fps':manifest['source_fps'],
                         'source_width':source_width,'source_height':source_height,'width':width,'height':height,
                         'encoded_time_s':dense_index/plan['encoded_fps'],
                         'is_observation_frame':observation is not None,
                         'observation_frame_index':observation['frame_index'] if observation else None,
                         'observation_rgb_sha256':observation['sha256'] if observation else None,
                         'decoded_rgb_sha256':hashlib.sha256(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()})
            writer.write(image)
        writer.release();writer=None
    finally:
        cap.release()
        if writer is not None:writer.release()
    if [row['source_frame_index'] for row in rows]!=plan['source_indices']:
        raise ValueError('Dense input frame mapping does not match its planned source indices')
    max_gap=max(np.diff([row['source_time_s'] for row in rows]))
    if max_gap>plan['source_gap_limit_s']+1e-5:
        raise ValueError('Decoded source timestamps exceed the dense maximum gap; increase dense_fps for this variable-rate video')
    verify=cv2.VideoCapture(str(destination/'input.mp4'));decoded=0
    try:
        while verify.grab():decoded+=1
    finally:verify.release()
    if decoded!=len(rows) or _hash(source)!=manifest['source_sha256']:
        raise ValueError('Dense encoded frame count or original source hash changed')
    result={'sampling_version':VIPE_SAMPLING_VERSION,'observation_manifest_path':str(observation_path),
            'observation_manifest_sha256':observation_hash,'source_path':str(source),'source_sha256':manifest['source_sha256'],
            'source_frame_count':manifest['source_frame_count'],'source_fps':manifest['source_fps'],
            'source_duration_s':manifest['source_frame_count']/manifest['source_fps'],
            'sampling':plan,'prepared_video':str(destination/'input.mp4'),'prepared_video_sha256':_hash(destination/'input.mp4'),
            'prepared_fps':plan['encoded_fps'],'source_max_gap_s':float(max_gap),'frames':rows,
            'timeline_note':'Playback FPS preserves approximate source duration; exact source timestamps and frame IDs, not playback times, join SAM observations to ViPE.',
            'source_modalities':['RGB'],'ground_truth_inputs':[]}
    write_json(manifest_path,result)
    return result


def prepare_vipe_dense_video(run_root, output_dir, dense_fps=10., cancel=None):
    """CPU-only dense preparation for inspection; never executes ViPE or GPT."""
    run,manifest=_prepared(run_root)
    destination=Path(output_dir).resolve()
    if destination.is_relative_to(run/'prepared') or destination.is_relative_to(run/'frames'):
        raise ValueError('Dense input must not overwrite observation preparation')
    return _prepare_dense_clip(run,manifest,destination,_dense_plan(manifest,dense_fps),cancel)


def run_vipe_tool(run_root, request=None, cancel=None):
    """Run ViPE only for this explicit Pi tool call; request may name job_id."""
    request = copy.deepcopy(request or {})
    if not isinstance(request,dict) or set(request)-{'job_id','timeout_s','reason','reuse_completed','dense_fps','slam_buffer'}:
        raise ValueError('Unsupported ViPE request fields')
    timeout=request.get('timeout_s',1800)
    if isinstance(timeout,bool) or not isinstance(timeout,int) or not 30<=timeout<=1800:
        raise ValueError('ViPE timeout_s must be in [30,1800]')
    if 'reuse_completed' in request and not isinstance(request['reuse_completed'],bool):
        raise ValueError('reuse_completed must be boolean')
    if _cancelled(cancel):raise VideoToolCancelled('ViPE cancelled before execution')
    run,manifest=_prepared(run_root)
    source=Path(manifest['source_path'])
    if not source.is_file() or _hash(source)!=manifest['source_sha256']:
        raise ValueError('Original source video is required and must match its frozen hash for dense ViPE')
    plan=_dense_plan(manifest,request.get('dense_fps',10.))
    buffer=request.get('slam_buffer',plan['slam_buffer'])
    if isinstance(buffer,bool) or not isinstance(buffer,int) or not plan['dense_frame_count']+16<=buffer<=4096:
        raise ValueError('slam_buffer must accommodate all dense keyframes plus 16 infill slots, and be <=4096')
    plan['slam_buffer']=buffer
    plan['graph_buffer_only_estimate_gib']=buffer*(384*512*6+(384//8)*(512//8)*(768+9))/(2**30)
    script=ROOT/'scripts/run_vipe_workflow.sh'
    settings=load_settings()
    binding=_fingerprint({'observation_manifest_sha256':_hash(run/'frames/manifest.json'),'script_sha256':_hash(script),
                          'mode':'rgb_only_unidepth_l_dense','source_video_sha256':manifest['source_sha256'],
                          'sampling_version':VIPE_SAMPLING_VERSION,'dense_sampling':plan,
                          'worker_environment':vipe_environment(settings)})
    latest=run/'tools/vipe/latest.json'
    identity=request.get('job_id')
    previous=None
    if latest.exists():previous=_read(latest)
    if identity is None and request.get('reuse_completed',True) and previous:
        previous_receipt=Path(previous['receipt_path']).resolve()
        if not previous_receipt.is_relative_to(run/'tools/vipe'):
            raise ValueError('Invalid ViPE result pointer')
        old=_read(previous_receipt)
        if old.get('binding')==binding:identity=old['job_id']
    job,receipt,cached=_begin_job(run,'vipe',identity,binding,request)
    def publish(result):
        write_json(latest,{key:result[key] for key in ['vipe_dir','result_path','receipt_path','job_id',
                   'vipe_manifest_path','vipe_manifest_sha256','observation_manifest_sha256','sampling_version']})
    if cached:
        if cached.get('sampling_version')!=VIPE_SAMPLING_VERSION:raise ValueError('Sparse ViPE output cannot be reused as dense geometry')
        publish(cached)
        return cached
    try:
        dense=_prepare_dense_clip(run,manifest,job/'prepared',plan,cancel)
        dense_path=job/'prepared/manifest.json'
        argv=['env',f'VIPE_BUFFER={buffer}',*vipe_environment(settings),'bash',to_wsl(script),to_wsl(job),'inference']
        receipt.update(command=argv,source_script_sha256=_hash(script),input_manifest_sha256=_hash(dense_path),
                       sampling_version=VIPE_SAMPLING_VERSION,dense_sampling=plan,
                       observation_manifest_sha256=_hash(run/'frames/manifest.json'),
                       previous_vipe_result={'job_id':previous.get('job_id'),'sampling_version':previous.get('sampling_version','sparse_observation_v0'),
                                             'receipt_path':previous.get('receipt_path')} if previous else None)
        write_json(job/'receipt.json',receipt)
        _run_wsl(argv,job,timeout,cancel)
        from video2scene.artifacts import ViPEArtifacts
        artifacts=ViPEArtifacts(job/'vipe')
        if artifacts.indices!=list(range(len(dense['frames']))):raise ValueError('ViPE indices do not match dense prepared frames')
        previews,summary=_geometry_preview(run,job,artifacts,dense)
        summary.update(base_sampling_fps=plan['target_fps'],encoded_fps=plan['encoded_fps'],
                       source_max_gap_s=dense['source_max_gap_s'],slam_buffer=buffer)
        result={'tool':'run_vipe','status':'complete','cached':False,'job_id':job.name,'actual_inference':True,
                'vipe_dir':str(job/'vipe'),'result_path':str(job/'receipt.json'),'receipt_path':str(job/'receipt.json'),
                'manifest_path':str(run/'frames/manifest.json'),'observation_manifest_sha256':_hash(run/'frames/manifest.json'),
                'vipe_manifest_path':str(dense_path),'vipe_manifest_sha256':_hash(dense_path),'sampling_version':VIPE_SAMPLING_VERSION,
                'frame_count':len(artifacts.indices),'observation_frame_count':len(manifest['frames']),'previews':previews,'summary':summary}
        _finish_job(run,job,receipt,result)
        publish(result)
        return result
    except BaseException as exc:
        _fail_job(job,receipt,exc);raise


def run_sam_tool(run_root, requests, job_id, cancel=None):
    """Execute explicit SAM object prompts against this project's prepared RGB."""
    if _cancelled(cancel):raise VideoToolCancelled('SAM cancelled before execution')
    run,manifest=_prepared(run_root)
    settings=load_settings()
    if not settings.sam_checkpoint:
        raise ValueError('Configure sam_checkpoint in video2scene.local.toml or V2S_SAM_CHECKPOINT before segmentation')
    value={'version':1,'frames_dir':str(run/'frames'),'manifest':str(run/'frames/manifest.json'),
           'timeout_s':1200,'requests':requests}
    value,frames=validate_inputs(value,run)
    worker=ROOT/'video2scene/agent_scene/sam_worker.py'
    binding=_fingerprint({'request':value,'manifest_sha256':_hash(run/'frames/manifest.json'),
                          'worker_sha256':_hash(worker),'checkpoint_sha256':CHECKPOINT_SHA256,
                          'sam_python':settings.sam_python,'sam_checkpoint':settings.sam_checkpoint,
                          'worker_environment':gpu_environment(settings)})
    job,receipt,cached=_begin_job(run,'sam',job_id,binding,value)
    if cached:
        write_json(run/'tools/sam/latest.json',{'sam_root':cached['sam_root'],'result_path':cached['result_path'],
                                             'receipt_path':cached['receipt_path'],'job_id':job.name})
        return cached
    output=job/'output'
    try:
        output.mkdir()
        write_json(output/'input_manifest.json',{'frames':frames,'input_modalities':['RGB','text','points']})
        worker_request=copy.deepcopy(value)
        worker_request.update(frames_dir=to_wsl(value['frames_dir']),manifest=to_wsl(output/'input_manifest.json'))
        write_json(output/'worker_request.json',worker_request)
        write_json(output/'request.json',value)
        argv=['env',*gpu_environment(settings),
              'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','PYTHONUNBUFFERED=1',
              settings.sam_python,'-u',to_wsl(worker),'--request',to_wsl(output/'worker_request.json'),
              '--output',to_wsl(output),'--checkpoint',settings.sam_checkpoint]
        receipt.update(command=argv,worker_sha256=_hash(worker),checkpoint_expected_sha256=CHECKPOINT_SHA256,
                       input_manifest_sha256=_hash(run/'frames/manifest.json'),network_downloads_allowed=False)
        write_json(job/'receipt.json',receipt)
        _run_wsl(argv,job,value['timeout_s'],cancel)
        raw=_read(output/'result.json')
        if raw['status']!='complete':raise RuntimeError('SAM output is incomplete')
        summaries=[];previews=[]
        for row in raw['objects']:
            summaries.append({key:row[key] for key in ['object_id','category','status','seed_sample_index','selected_sam_id','frames_with_target','reason','error'] if key in row})
            previews.extend(str(output/name) for name in row.get('previews',[]))
        result={'tool':'run_sam','status':'complete','cached':False,'job_id':job.name,'actual_inference':True,
                'sam_root':str(output),'result_path':str(output/'result.json'),'receipt_path':str(job/'receipt.json'),
                'manifest_path':str(run/'frames/manifest.json'),'frame_count':len(frames),
                'all_targets_resolved':raw.get('all_targets_resolved',False),'objects':summaries,'previews':previews}
        _finish_job(run,job,receipt,result)
        write_json(run/'tools/sam/latest.json',{'sam_root':str(output),'result_path':result['result_path'],
                                             'receipt_path':result['receipt_path'],'job_id':job.name})
        return result
    except BaseException as exc:
        detail=_sam_failure_details(output)
        if detail:receipt['worker_failure']=detail
        if detail and isinstance(exc,Exception) and not isinstance(exc,VideoToolCancelled):
            explained=RuntimeError(f"SAM {detail['error_type']}: {detail['error']}. "
                                   f"Failed attempt retained at {job}; retry requires a new job_id.")
            _fail_job(job,receipt,explained)
            raise explained from exc
        _fail_job(job,receipt,exc);raise
