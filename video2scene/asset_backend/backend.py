"""Fresh-output Blender subprocesses. No repository/head or model-provider writes."""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from video2scene.capture_evidence import (
    bounds_corners,
    transform_points,
    transform_to_matrix,
)
from video2scene.contracts_v1 import (
    ContractError,
    camera_hash,
    file_hash,
    validate,
)
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.config import blender_path as configured_blender

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER = configured_blender()
WORKER = ROOT / 'scripts/asset_backend/blender_worker.py'
RECIPE_COMPILER = Path(__file__).with_name('recipe_geometry.py')


class BlenderAssetBackend:
    def __init__(self, storage_root, blender_path=None, *, timeout_s=300, samples=8,
                 recipe_provider=None, refinement_mode='procedural_fallback'):
        self.storage_root = Path(storage_root).resolve()
        self.blender_path = configured_blender(blender_path)
        self.resolver = ArtifactResolver(self.storage_root)
        self.timeout_s = timeout_s
        self.samples = samples
        if refinement_mode not in {'procedural_fallback','image_recipe','observed_surface'}:
            raise ValueError('Unknown refinement mode')
        self.recipe_provider = recipe_provider
        self.refinement_mode = refinement_mode

    def _artifact(self, path, scene_id, role, media_type):
        key = Path(path).resolve().relative_to(self.storage_root / scene_id).as_posix()
        return {'uri': f'artifact://{scene_id}/{key}', 'sha256': file_hash(path),
                'media_type': media_type, 'role': role}

    def _scene_spec(self, scene):
        spec = copy.deepcopy(scene)
        for obj in spec['objects']:
            obj['_asset_path'] = str(self.resolver.resolve(obj['asset']['artifact'], scene_id=scene['scene_id']))
        return spec

    def _execute(self, action, scene_id, payload, context, timeout_s=None):
        budget=getattr(self,'resource_budget',None)
        if budget:
            with budget.hold('blender',context=context):
                return self._execute_unlimited(action,scene_id,payload,context,timeout_s)
        return self._execute_unlimited(action,scene_id,payload,context,timeout_s)

    def _execute_unlimited(self, action, scene_id, payload, context, timeout_s=None):
        context.check_cancelled()
        if not self.blender_path.is_file():
            raise ContractError('TOOL_UNAVAILABLE', 'Configured Blender executable is unavailable')
        run_dir = self.storage_root / scene_id / 'assets' / f'{action}-{uuid.uuid4().hex}'
        run_dir.mkdir(parents=True, exist_ok=False)
        script = run_dir / 'source.py'
        script.write_bytes(WORKER.read_bytes())
        spec = {'action': action, 'output_dir': str(run_dir), 'samples': self.samples,
                'worker_sha256': file_hash(script), **payload}
        if action == 'refine_recipe':
            compiler = run_dir / 'recipe_geometry.py'
            compiler.write_bytes(RECIPE_COMPILER.read_bytes())
            spec['compiler_sha256'] = file_hash(compiler)
            bundle = Path(payload['recipe_record']['bundle_path'])
            if file_hash(bundle) != payload['recipe_record']['bundle_file_sha256']:
                raise ContractError('STALE_ARTIFACT', 'Authored recipe bundle changed before build')
            (run_dir / 'author_recipe_bundle.json').write_bytes(bundle.read_bytes())
        if action == 'refine_observed':
            for name, row in payload['observed_record']['source_files'].items():
                source = Path(row['path'])
                if file_hash(source) != row['sha256']:
                    raise ContractError('STALE_ARTIFACT', 'Observed surface source changed before build')
                (run_dir / name).write_bytes(source.read_bytes())
                if file_hash(run_dir / name) != row['sha256']:
                    raise ContractError('STALE_ARTIFACT', 'Observed source copy differs from its binding')
        manifest = run_dir / 'input.json'
        manifest.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        spec_hash = file_hash(manifest)
        command = [str(self.blender_path), '--background', '--factory-startup', '--threads', '4',
                   '--python-exit-code', '1', '--python', str(script), '--', str(manifest)]
        started = time.monotonic()
        state = 'running'
        log = run_dir / 'blender.log'
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                while process.poll() is None:
                    if context.cancel_event.is_set():
                        state = 'cancelled'
                        raise ContractError('CANCELLED', 'Blender job cancellation requested')
                    if time.monotonic() - started > (timeout_s or self.timeout_s):
                        state = 'timeout'
                        raise ContractError('TIMEOUT', 'Blender subprocess exceeded its timeout', retryable=True)
                    time.sleep(.1)
                state = 'failed' if process.returncode else 'completed_unverified'
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                raise
            finally:
                execution = {'command': command, 'pid': process.pid, 'returncode': process.poll(),
                             'state': state, 'duration_s': round(time.monotonic() - started, 3),
                             'input_sha256': spec_hash, 'worker_sha256': spec['worker_sha256'], 'device': 'CPU'}
                (run_dir / 'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
        if process.returncode:
            raise ContractError('BUILD_FAILED', 'Blender subprocess failed; existing assets were not reused',
                                details={'returncode': process.returncode, 'log': self._artifact(log, scene_id, 'build_log', 'text/plain')})
        context.check_cancelled()
        result_path = run_dir / 'result.json'
        if not result_path.is_file():
            raise ContractError('BUILD_FAILED', 'Blender exited without producing a result manifest')
        if file_hash(manifest) != spec_hash or file_hash(script) != spec['worker_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Build inputs changed during execution')
        if action == 'refine_recipe' and file_hash(run_dir / 'recipe_geometry.py') != spec['compiler_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Recipe compiler changed during execution')
        if action == 'refine_observed':
            for name, row in payload['observed_record']['source_files'].items():
                if file_hash(run_dir / name) != row['sha256'] or file_hash(row['path']) != row['sha256']:
                    raise ContractError('STALE_ARTIFACT', 'Observed surface source changed during build')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['input_fingerprint'] != payload['input_fingerprint'] or result['worker_sha256'] != spec['worker_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Output manifest is not bound to this input/source script')
        for filename, expected in result['files'].items():
            path = (run_dir / filename).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file() or file_hash(path) != expected:
                raise ContractError('STALE_ARTIFACT', 'Produced artifact is missing or stale', details={'name': filename})
        execution['state'] = 'succeeded'
        (run_dir / 'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
        return run_dir, result

    async def _run(self, action, scene_id, payload, context, timeout_s=None):
        return await asyncio.to_thread(self._execute, action, scene_id, payload, context, timeout_s)

    def _build_refs(self, run_dir, scene_id):
        entries = [('scene.blend', 'scene_blend', 'application/x-blender'),
                   ('scene.glb', 'scene_glb', 'model/gltf-binary'),
                   ('reopen.json', 'reopen_report', 'application/json'),
                   ('source.py', 'build_source', 'text/x-python'),
                   ('input.json', 'build_input', 'application/json'),
                   ('result.json', 'build_manifest', 'application/json'),
                   ('blender.log', 'build_log', 'text/plain')]
        return [self._artifact(run_dir / filename, scene_id, role, mime) for filename, role, mime in entries]

    async def build(self, candidate_snapshot, job_context):
        validate(candidate_snapshot, 'CandidateSnapshot')
        candidate, scene = candidate_snapshot['candidate'], candidate_snapshot['scene']
        payload = {'scene': self._scene_spec(scene), 'candidate': candidate,
                   'input_fingerprint': candidate['candidate_hash']}
        run_dir, _ = await self._run('build', scene['scene_id'], payload, job_context)
        for obj in scene['objects']:
            self.resolver.resolve(obj['asset']['artifact'], scene_id=scene['scene_id'])
        output = {'artifact_refs': self._build_refs(run_dir, scene['scene_id']), 'asset_updates': [],
                  'input_fingerprint': candidate['candidate_hash']}
        return validate(output, 'BuildOutput')

    async def refine(self, refine_spec, scene_snapshot, evidence, job_context):
        validate(scene_snapshot, 'SceneSnapshot')
        validate(evidence, 'EvidenceBundle')
        candidate = refine_spec['candidate']
        validate(candidate, 'CandidateRef')
        if refine_spec.get('input_fingerprint', candidate['candidate_hash']) != candidate['candidate_hash']:
            raise ContractError('STALE_ARTIFACT', 'Refinement fingerprint differs from frozen candidate')
        identity = refine_spec['target_id']
        fields = set(refine_spec['allowed_fields'])
        if fields != {'asset.geometry', 'asset.materials'}:
            raise ContractError('TOOL_UNAVAILABLE', 'Asset replacement currently requires geometry and materials authorization')
        for field in fields:
            job_context.authorize(identity, field)
        target = next((obj for obj in scene_snapshot['objects'] if obj['instance_id'] == identity), None)
        if target is None or identity not in evidence['target_ids']:
            raise ContractError('EVIDENCE_MISMATCH', 'Refinement target is missing from scene/evidence')
        if evidence['scene_revision'] != candidate['base_revision'] or evidence['scene_id'] != scene_snapshot['scene_id']:
            raise ContractError('EVIDENCE_MISMATCH', 'Refinement evidence belongs to another base snapshot')
        for frame in evidence['frames']:
            self.resolver.resolve(frame['rgb'], scene_id=scene_snapshot['scene_id'])
        constraints = refine_spec['constraints']
        if not constraints['keep_root_transform'] or not constraints['keep_support']:
            raise ContractError('TOOL_UNAVAILABLE', 'Asset refinement preserves root and support')
        payload = {'target': target, 'evidence': evidence, 'candidate': candidate,
                   'input_fingerprint': candidate['candidate_hash']}
        if self.refinement_mode == 'image_recipe':
            if self.recipe_provider is None:
                raise ContractError('TOOL_UNAVAILABLE', 'Image-driven mode requires an approved authored recipe bundle; no procedural fallback is permitted')
            record = self.recipe_provider.get_recipe(target, scene_snapshot, evidence, job_context)
            if inspect.isawaitable(record):
                record = await record
            payload.update(authored_recipe=record['recipe'], recipe_record=record)
            action = 'refine_recipe'
            version_prefix = 'image-recipe-' if record['provenance']['origin'] == 'codex_image' else 'fixture-recipe-'
        elif self.refinement_mode == 'observed_surface':
            if not hasattr(self, '_observed_payload'):
                raise ContractError('TOOL_UNAVAILABLE', 'Observed mode requires a bound ObservedSurfaceBackend')
            observed = await asyncio.to_thread(self._observed_payload, target, scene_snapshot, evidence, candidate, job_context)
            payload.update(observed)
            action, version_prefix = 'refine_observed', 'observed-partial-'
        else:
            category = target['category'].split('/')[-1].lower()
            if category not in {'mug', 'cup', 'bowl', 'can', 'carton', 'table'}:
                raise ContractError('TOOL_UNAVAILABLE', 'No procedural fallback recipe for this category', details={'category': category})
            payload.update(recipe=category, refinement_mode='procedural_fallback')
            action, version_prefix = 'refine', 'procedural-'
        run_dir, result = await self._run(action, scene_snapshot['scene_id'], payload, job_context,
                                        refine_spec['budget']['timeout_s'])
        # The canonical asset must be directly consumable by the browser. Blender's
        # exporter has already converted the file to actual glTF Y-up coordinates.
        asset_artifact = self._artifact(run_dir / 'asset.glb', scene_snapshot['scene_id'], 'object_geometry', 'model/gltf-binary')
        update = {'instance_id': identity, 'asset': {'asset_version': version_prefix + asset_artifact['sha256'][:24],
                  'artifact': asset_artifact, 'native_frame': 'asset_y_up',
                  'native_to_object': np.eye(4).tolist()},
                  'local_bounds_m': result['local_bounds_m']}
        draft = {key: copy.deepcopy(value) for key, value in scene_snapshot.items() if key not in {'revision', 'parent_revision'}}
        for obj in draft['objects']:
            if obj['instance_id'] == identity:
                obj['asset'] = update['asset']
                obj['local_bounds_m'] = update['local_bounds_m']
        assembled = await self.build({'candidate': candidate, 'scene': draft}, job_context)
        assembled['asset_updates'] = [update]
        assembled['artifact_refs'].extend([
            asset_artifact, self._artifact(run_dir / 'asset.blend', scene_snapshot['scene_id'], 'object_blend', 'application/x-blender'),
            self._artifact(run_dir / 'source.py', scene_snapshot['scene_id'], 'refinement_source', 'text/x-python'),
            self._artifact(run_dir / 'result.json', scene_snapshot['scene_id'], 'refinement_manifest', 'application/json'),
            self._artifact(run_dir / 'input.json', scene_snapshot['scene_id'], 'refinement_input', 'application/json'),
            self._artifact(run_dir / 'execution.json', scene_snapshot['scene_id'], 'refinement_execution', 'application/json')])
        if action == 'refine_recipe':
            assembled['artifact_refs'].extend([
                self._artifact(run_dir / 'recipe_geometry.py', scene_snapshot['scene_id'], 'data_recipe_compiler', 'text/x-python'),
                self._artifact(run_dir / 'author_recipe_bundle.json', scene_snapshot['scene_id'], 'authored_recipe_bundle', 'application/json')])
        if action == 'refine_observed':
            assembled['artifact_refs'].extend(self._artifact(run_dir / name, scene_snapshot['scene_id'],
                'observed_source', 'model/gltf-binary' if name.endswith('.glb') else 'application/json')
                for name in payload['observed_record']['source_files'])
        return validate(assembled, 'BuildOutput')

    async def render(self, candidate_snapshot, evidence, view_mode, job_context, *, frame_ids=None):
        validate(candidate_snapshot, 'CandidateSnapshot')
        validate(evidence, 'EvidenceBundle')
        if view_mode not in {'normal', 'visible_mask', 'xray'}:
            raise ContractError('INVALID_ARGUMENT', 'Unknown render mode')
        candidate, scene = candidate_snapshot['candidate'], candidate_snapshot['scene']
        if evidence['scene_id'] != scene['scene_id'] or evidence['scene_revision'] != candidate['base_revision']:
            raise ContractError('EVIDENCE_MISMATCH', 'Render evidence does not match the candidate base')
        available = {frame['source_frame_id']: frame for frame in evidence['frames']}
        requested = frame_ids or list(available)
        if not requested or len(set(requested)) != len(requested) or any(key not in available for key in requested):
            raise ContractError('FRAME_MISMATCH', 'Requested render frame is absent or duplicated')
        frames = [available[key] for key in requested]
        for frame in frames:
            self.resolver.resolve(frame['rgb'], scene_id=scene['scene_id'])
        payload = {'scene': self._scene_spec(scene), 'candidate': candidate, 'evidence': evidence,
                   'render_frame_ids': requested, 'view_mode': view_mode, 'input_fingerprint': candidate['candidate_hash']}
        run_dir, result = await self._run('render', scene['scene_id'], payload, job_context)
        for obj in scene['objects']:
            self.resolver.resolve(obj['asset']['artifact'], scene_id=scene['scene_id'])
        report = {'candidate_id': candidate['candidate_id'], 'candidate_hash': candidate['candidate_hash'],
                  'evidence_id': evidence['evidence_id'], 'evidence_hash': evidence['evidence_hash'],
                  'frame_ids': requested, 'view_mode': view_mode,
                  'camera_hash': camera_hash(evidence['frames'], requested),
                  'artifact_refs': [self._artifact(run_dir / row['filename'], scene['scene_id'], 'render_' + view_mode, 'image/png')
                                    for row in result['renders']]}
        report['artifact_refs'].append(self._artifact(run_dir / 'result.json', scene['scene_id'], 'render_manifest', 'application/json'))
        return validate(report, 'RenderReport')

    async def collisions(self, candidate_snapshot, method, job_context):
        validate(candidate_snapshot, 'CandidateSnapshot')
        job_context.check_cancelled()
        if method != 'box':
            raise ContractError('TOOL_UNAVAILABLE', 'Only conservative oriented-bounds collision checks are implemented')
        candidate, scene = candidate_snapshot['candidate'], candidate_snapshot['scene']
        boxes = {}
        for obj in scene['objects']:
            points = transform_points(transform_to_matrix(obj['transform']), bounds_corners(obj['local_bounds_m']))
            boxes[obj['instance_id']] = (points.min(axis=0), points.max(axis=0))
        checks = []
        for i, a in enumerate(scene['objects']):
            for b in scene['objects'][i + 1:]:
                if a['support_id'] == b['instance_id'] or b['support_id'] == a['instance_id']:
                    continue
                alo, ahi = boxes[a['instance_id']]
                blo, bhi = boxes[b['instance_id']]
                overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
                collides = bool(np.all(overlap > .002))
                checks.append({'name': f"bounds_overlap:{a['instance_id']}:{b['instance_id']}", 'passed': not collides,
                               'severity': 'error' if collides else 'info', 'artifact_refs': [],
                               'details': json.dumps({'method': 'conservative world AABB, not mesh intersection', 'overlap_m': overlap.tolist()})})
        report = {'candidate_id': candidate['candidate_id'], 'candidate_hash': candidate['candidate_hash'],
                  'method': 'box', 'passed': all(check['passed'] for check in checks), 'checks': checks}
        return validate(report, 'CollisionReport')
