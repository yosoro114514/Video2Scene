"""GPT-authored single-object assets behind the existing v1 candidate contract.

The host records explicit whitebox confirmation. This backend checks that record,
invokes a bounded author/critic loop, assembles a candidate and returns BuildOutput.
It never changes repository head. Authoring and deterministic export have separate
execution records; injected providers are always labelled local fixtures.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from video2scene.asset_backend.backend import BlenderAssetBackend, DEFAULT_BLENDER
from video2scene.capture_evidence import bounds_corners, transform_points, transform_to_matrix
from video2scene.capture_evidence.coordinates import SCENE_TO_BLENDER
from video2scene.codex_review import obj, STR
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash, validate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERIFIER = HERE / 'direct_validate.py'
INVOKER = HERE / 'direct_invoke.py'

AUTHOR_SCHEMA = obj({
    'status': {'type': 'string', 'enum': ['complete', 'failed']},
    'instance_id': STR, 'asset_blend': STR, 'script_path': STR,
    'parts': {'type': 'array', 'items': STR},
    'observed_features': {'type': 'array', 'items': STR},
    'assumptions': {'type': 'array', 'items': STR},
})
CRITIC_SCHEMA = obj({
    'status': {'type': 'string', 'enum': ['accept', 'revise', 'unclear']},
    'summary': STR,
    'issues': {'type': 'array', 'items': obj({
        'severity': {'type': 'string', 'enum': ['major', 'minor']},
        'evidence': STR, 'requested_fix': STR,
    })},
    'limitations': {'type': 'array', 'items': STR},
})


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_blender_bounds(bounds):
    points = transform_points(SCENE_TO_BLENDER, bounds_corners(bounds))
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()]


def crop_rect(target, frame):
    """Projected coarse ROI is localization, not a claim of visible segmentation."""
    points = transform_points(transform_to_matrix(target['transform']), bounds_corners(target['local_bounds_m']))
    camera = transform_points(np.linalg.inv(np.asarray(frame['c2w'])), points)
    width, height = frame['width'], frame['height']
    if np.any(camera[:, 2] <= .01):
        return [0, 0, width, height]
    projected = camera @ np.asarray(frame['K']).T
    pixels = projected[:, :2] / projected[:, 2:]
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    padding = np.maximum((high - low) * .12, 10)
    low = np.maximum(np.floor(low - padding), [0, 0]).astype(int)
    high = np.minimum(np.ceil(high + padding), [width, height]).astype(int)
    if (high - low).min() < 12:
        return [0, 0, width, height]
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def observed_crop_rect(mask):
    """Use the observed instance for RGB localization when SAM supplied a mask."""
    ys, xs = np.nonzero(np.asarray(mask.convert('L')))
    if not len(xs):
        return None
    low, high = np.array([xs.min(), ys.min()]), np.array([xs.max() + 1, ys.max() + 1])
    padding = np.maximum((high - low) * .12, 10)
    low = np.maximum(np.floor(low - padding), [0, 0]).astype(int)
    high = np.minimum(np.ceil(high + padding), mask.size).astype(int)
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def author_prompt(work, contract, blender_path, feedback):
    return f'''You are the GPT single-object modeller for a real RGB-only room reconstruction.
Actually write and execute a fresh Blender object, {contract['instance_id']} ({contract['category']}).
Workspace: {work}
Blender executable: {blender_path}
Required script: {work / 'object.py'}
Required output: {work / 'asset.blend'}
Read contract.json. Attached full source images, crops and optional observed masks describe this object. Crops use observed masks when available and otherwise projected coarse bounds; they may include neighbors. Do not model those neighbors.

Start object.py with bpy.ops.wm.read_factory_settings(use_empty=True), then construct NEW named editable meshes/curves, modifiers and materials and save the required asset.blend. Actually execute with Blender --background --factory-startup --threads 4 --python-exit-code 1 --python object.py. No plan-only completion. Use CPU rendering if any. The host independently renders and reviews your result; prioritize one complete object, not a presentation. One technical execution correction is allowed. Do not run your own iterative visual-review loop.
An empty factory scene may have scene.world=None. Do not access scene.world.color or node_tree unless you first create/assign a world. The host supplies inspection lighting, so no author lights/cameras are required. Avoid unsupported modifier API attributes; verify the script succeeds before returning complete.

LOCAL COORDINATES: Blender XYZ, Z up. Canonical local bounds converted into Blender coordinates are min/max {contract['local_bounds_blender']}. Build around exactly that origin and extent. Never apply world position, object rotation or user's scale; the host applies the locked root transform exactly once. Overall bounds are fixed; the host permits at most 12% size/origin error and explicitly records any small fit. Do not use stray vertices to fake bounds. Preserve visually observed front/back orientation using the supplied observation transform and cameras. The JSON scene coordinates are Y-up; Blender local = (scene.x, -scene.z, scene.y).

Keep locked world transform, category, support and local bounds. User instruction: {contract['user_instruction']}
Source photographs show the capture pose. The user's accepted layout may intentionally differ. Improve intrinsic object shape and material; never undo accepted user edits. Build meaningful visible parts, openings, connected supports and curved profiles where evidenced. For chairs/sofas/tables use the observed instance's structure rather than a generic category cube. Hidden backs and undersides may receive conservative completion, recorded as assumptions. No whole-room meshes, other objects, camera-facing photo planes or billboards. {structural_policy(contract)}

The only provided geometry is numeric coarse bounds; do not import or append older assets or inspect sibling/parent projects. Use only this workspace's reference files and installed libraries. New code, meshes and textures must be created here. Use glTF-compatible Principled BSDF materials with explicit base color/roughness/metallic or packed image textures; source-derived local crops/rectification are permitted. Bake any procedural appearance into packed image textures if it is essential, because the independent critic renders the reopened exported GLB, not the native Blender shader graph. No downloads, network calls, other agents, nested Codex or credentials. No fake brands or invented unreadable labels. Preserve contract.json and all ref_* inputs.

This author turn has a bounded tool/time budget managed by the host. Return complete only after a successful Blender command and actual asset.blend exists. Return the exact output/script paths, stable instance_id, named parts, observed features, and assumptions.
Previous independent feedback: {json.dumps(feedback, ensure_ascii=False) if feedback else 'First pass.'}
If previous_turntable.jpg or previous_comparison_*.jpg are supplied, they show this run's preceding candidate after GLB export under the host's inspection lighting. Compare those actual renders with the source before revising; changes to native material values alone may not survive export or look as expected.
Appearance policy: {appearance_policy(contract)}
'''


def appearance_policy(contract):
    if contract.get('visual_policy') == 'structure_first_relaxed_appearance':
        return ('The user explicitly permits reasonable differences in color and appearance. Prioritize object identity, '
                'silhouette, correct front/back, major visible parts, openings, proportions and connected supports. '
                'Use plausible category-appropriate materials. Hue, brightness, wood grain, upholstery pattern, '
                'fine surface detail and roughness need not match photographs exactly. These differences alone '
                'are minor and must not cause rejection. Lighting and estimated capture alignment also limit comparison. '
                'Retain major rejection for missing/wrong structural parts, disconnected supports, clearly inverted '
                'front/back, severe disproportion, a category cube where meaningful structure is visible, or '
                'background/neighbor contamination or photo billboards. For naturally thin flat objects such as rugs '
                'or panels, an appropriate thin surface with material is valid. Never relax saved placement, bounds or support.')
    return 'Match visible shape and materials with reasonable allowance for low-resolution source images and different illumination.'


def structural_policy(contract):
    if contract.get('kind') in {'floor', 'wall', 'opening', 'ceiling'}:
        return ('This target is ONE explicitly authorized room component. Model only this existing segment within its fixed local bounds. '
                'Preserve its planar rectangular boundary, straight edges and thickness. Never bridge adjacent door gaps, extend a wall, '
                'change the room footprint or include neighboring segments. Wall and lintel segments use consistent warm off-white paint; '
                'floors use plausible wood finishes; door leaves/frames may use wood or glass as evidenced. '
                'Keep frames, trim and thresholds inside this component bounds. A correctly detailed flat wall/floor panel is valid; '
                'do not invent furniture-like protrusions merely to avoid a box. A door/window may have a verified target SAM mask; '
                'when no mask is supplied, projected RGB crops are approximate, not instance segmentation. Record uncertain or unseen details as assumptions.')
    return 'Do not include any floor or wall geometry in a furniture asset.'


def critic_prompt(contract):
    return f'''Independently inspect the newly authored single object {contract['instance_id']} ({contract['category']}).
Attached: four-view final GLB render, then RAW RGB / GENERATED OBJECT comparisons, with OBSERVED TARGET MASK when available, at frozen original capture cameras with identical pixel crop. The mask identifies the target; it is not a render. Only this object is rendered; ignore unrelated source furniture, background, and occlusion by other objects. Also inspect full source images if supplied. User instruction: {contract['user_instruction']}.
The final assembly preserves accepted user position, orientation, scale and support; those edits are intentional. Evaluate intrinsic silhouette, parts, openings, thickness, connectivity and materials. Do not demand moving fixed cameras or undoing user layout edits. Do not judge detail quality from polygon count. Reject a generic box where clear visible structure is missing, obvious disconnected parts, wrong front/back, major disproportion or dominant texture contamination. Allow minor uncertainty from low source resolution and unseen surfaces, recording it. Return accept only when there is no clear major defect; revise with a specific view and actionable change, or unclear if even object identity cannot be established. Do not claim metric accuracy or scene-wide visual validation.
Binding user appearance policy: {appearance_policy(contract)}
Structural constraints, when applicable: {structural_policy(contract)}
Return Chinese JSON and use no tools.'''


class DirectAssetBackend(BlenderAssetBackend):
    def __init__(self, storage_root, blender_path=DEFAULT_BLENDER, *, confirmation_path,
                 observation_objects=None, model='gpt-6-astra', effort='high',
                 invoke_fn=None, samples=8, timeout_s=300, author_timeout_s=720,
                 critic_timeout_s=180, max_author_tools=32, bounds_fit_tolerance=.12,
                 status_callback=None):
        super().__init__(storage_root, blender_path, timeout_s=timeout_s, samples=samples)
        self.refinement_mode = 'gpt_direct'
        self.confirmation_path = Path(confirmation_path).resolve()
        self.observation_objects = copy.deepcopy(observation_objects or {})
        self.model, self.effort = model, effort
        self.invoke_fn = invoke_fn
        self.origin = 'codex_image' if invoke_fn is None else 'local_fixture'
        self.author_timeout_s, self.critic_timeout_s = author_timeout_s, critic_timeout_s
        self.max_author_tools = max_author_tools
        if not 0 <= bounds_fit_tolerance <= .12:
            raise ValueError('Bounds tolerance must be between 0 and 12%')
        self.bounds_fit_tolerance = bounds_fit_tolerance
        self.status_callback = status_callback

    def _notify(self, stage, **values):
        if self.status_callback:
            self.status_callback({'stage': stage, **values})

    def _confirmation(self, scene, candidate, target_id):
        if not self.confirmation_path.is_file():
            raise ContractError('FORBIDDEN_SCOPE', 'Fine modeling requires an actual user-confirmed whitebox revision')
        confirmation = read(self.confirmation_path)
        expected = {'confirmed': True, 'scene_id': scene['scene_id'], 'revision': scene['revision'],
                    'head_seq': candidate['base_head_seq'], 'target_id': target_id}
        if any(confirmation.get(key) != value for key, value in expected.items()) or not confirmation.get('confirmed_at'):
            raise ContractError('STALE_REVISION', 'Whitebox confirmation does not match this target and frozen base version')
        return confirmation, file_hash(self.confirmation_path)

    @staticmethod
    def _stop_owned(process):
        if process.poll() is not None:
            return
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if os.name != 'nt':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=8)

    def _process(self, command, folder, context, timeout_s, *, log_name):
        budget=getattr(self,'resource_budget',None)
        if budget and Path(command[0]).resolve()==self.blender_path.resolve():
            with budget.hold('blender',context=context,notify=self._notify):
                return self._process_unlimited(command,folder,context,timeout_s,log_name=log_name)
        return self._process_unlimited(command,folder,context,timeout_s,log_name=log_name)

    def _process_unlimited(self, command, folder, context, timeout_s, *, log_name):
        context.check_cancelled()
        started = time.monotonic()
        state = 'running'
        with (folder / log_name).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       start_new_session=os.name != 'nt')
            try:
                while process.poll() is None:
                    if context.cancel_event.is_set():
                        state = 'cancelled'
                        raise ContractError('CANCELLED', 'Direct modeling child cancellation requested')
                    if time.monotonic() - started > timeout_s:
                        state = 'timeout'
                        raise ContractError('TIMEOUT', 'Direct modeling child exceeded its bounded timeout')
                    time.sleep(.1)
                state = 'completed_unverified' if process.returncode == 0 else 'failed'
            except BaseException:
                self._stop_owned(process)
                raise
            finally:
                write(folder / (log_name + '.execution.json'), {
                    'command': command, 'pid': process.pid, 'state': state,
                    'returncode': process.poll(), 'seconds': round(time.monotonic() - started, 3),
                    'owned_subprocess': True,
                })
        if process.returncode:
            raise ContractError('BUILD_FAILED', 'Direct modeling subprocess failed',
                                {'returncode': process.returncode, 'log_path': str(folder / log_name)})
        context.check_cancelled()

    def _invoke(self, prompt, images, output, context, *, timeout, schema, sandbox, max_tool_calls):
        budget=getattr(self,'resource_budget',None)
        if budget:
            names=('model','blender') if sandbox=='workspace-write' else ('model',)
            with budget.hold(*names,context=context,notify=self._notify):
                return self._invoke_unlimited(prompt,images,output,context,timeout=timeout,schema=schema,sandbox=sandbox,max_tool_calls=max_tool_calls)
        return self._invoke_unlimited(prompt,images,output,context,timeout=timeout,schema=schema,sandbox=sandbox,max_tool_calls=max_tool_calls)

    def _invoke_unlimited(self, prompt, images, output, context, *, timeout, schema, sandbox, max_tool_calls):
        if (output / 'invocation_started.json').exists():
            raise ContractError('IDEMPOTENCY_CONFLICT', 'A model invocation already started here; uncertain calls are not replayed')
        output.mkdir(parents=True, exist_ok=True)
        specification = {'prompt': prompt, 'images': [str(path.resolve()) for path in images],
                         'output': str(output.resolve()), 'model': self.model, 'effort': self.effort,
                         'timeout': timeout, 'schema': schema, 'sandbox': sandbox,
                         'max_tool_calls': max_tool_calls}
        write(output / 'invocation_started.json', {'origin': self.origin, 'started': time.time(),
                                                 'specification_hash': canonical_hash(specification)})
        try:
            if self.invoke_fn is not None:
                result = self.invoke_fn(**specification)
            else:
                # Keep host control files outside the author-writable workspace.
                control = output.parent / (output.name + '_control')
                control.mkdir(exist_ok=False)
                path = control / 'invoke.json'
                write(path, specification)
                self._process([sys.executable, str(INVOKER), str(path)], control, context,
                              timeout + 15, log_name='invoke.log')
                result = read(control / 'invoke_result.json')
            context.check_cancelled()
            from jsonschema import Draft202012Validator
            Draft202012Validator(schema).validate(result)
            write(output / 'invocation_outcome.json', {'status': 'complete', 'origin': self.origin,
                                                     'response_hash': canonical_hash(result)})
            return result
        except BaseException as error:
            from video2scene.codex_capacity import capacity_rejection
            capacity=capacity_rejection(output) if isinstance(error,ContractError) and error.code=='BUILD_FAILED' else None
            if capacity:error=ContractError('MODEL_CAPACITY',capacity['message'],capacity)
            write(output / 'invocation_outcome.json', {'status': 'failed_or_uncertain', 'origin': self.origin,
                                                     'error': str(error), 'error_code':getattr(error,'code',None), 'automatic_retry': False})
            if capacity:raise error
            raise

    def _prepare(self, folder, scene, target, evidence, candidate, confirmation):
        folder.mkdir(parents=True, exist_ok=False)
        observation = self.observation_objects.get(target['instance_id'], target)
        images, frames, rectangles = [], [], []
        for index, frame in enumerate(evidence['frames'][:5]):
            source = self.resolver.resolve(frame['rgb'], scene_id=scene['scene_id'])
            image = Image.open(source).convert('RGB')
            if image.size != (frame['width'], frame['height']):
                raise ContractError('FRAME_MISMATCH', 'Evidence image size differs from frozen camera')
            full = folder / f'ref_{index:02d}_rgb.png'
            image.save(full)
            rectangle = crop_rect(observation, frame)
            frame_copy = copy.deepcopy(frame)
            frame_copy['local_rgb'] = full.name
            frame_copy['crop_origin'] = 'projected_initial_object_bounds'
            destination = None
            if frame.get('mask'):
                mask = self.resolver.resolve(frame['mask'], scene_id=scene['scene_id'])
                destination = folder / f'ref_{index:02d}_mask.png'
                with Image.open(mask) as observed_mask:
                    if observed_mask.size != image.size:
                        raise ContractError('FRAME_MISMATCH', 'Observed mask resolution differs from RGB')
                    observed_mask.save(destination)
                    observed_rectangle = observed_crop_rect(observed_mask)
                    if observed_rectangle is not None:
                        rectangle = observed_rectangle
                        frame_copy['crop_origin'] = 'observed_target_mask'
                frame_copy['local_mask'] = destination.name
            crop = folder / f'ref_{index:02d}_crop.png'
            image.crop(rectangle).save(crop)
            images.extend([full, crop])
            if destination is not None:
                images.append(destination)
            rectangles.append(rectangle)
            frame_copy['crop_rectangle'] = rectangle
            frames.append(frame_copy)
        contract = {'instance_id': target['instance_id'], 'category': target['category'],
                    'kind': target['kind'], 'scene_id': scene['scene_id'], 'revision': scene['revision'],
                    'candidate': candidate, 'evidence_id': evidence['evidence_id'],
                    'evidence_hash': evidence['evidence_hash'], 'local_bounds_m': target['local_bounds_m'],
                    'local_bounds_blender': local_blender_bounds(target['local_bounds_m']),
                    'accepted_transform': target['transform'], 'support_id': target['support_id'],
                    'observation_transform': transform_to_matrix(observation['transform']).tolist(),
                    'user_instruction': confirmation.get('user_instruction') or 'Keep the confirmed size, placement, orientation and support; refine intrinsic shape and materials.',
                    'visual_policy': confirmation.get('visual_policy', 'match_observed_appearance'),
                    'frames': frames, 'crop_rectangles': rectangles,
                    'input_origin': 'Raw RGB evidence and coarse numeric bounds; no old fine asset provided.'}
        write(folder / 'contract.json', contract)
        return contract, images

    def _author_proof(self, folder, response, identity, protected):
        if response['status'] != 'complete' or response['instance_id'] != identity:
            raise ContractError('BUILD_FAILED', 'Model author did not report completion for the requested instance')
        for key, required in [('asset_blend', 'asset.blend'), ('script_path', 'object.py')]:
            value = Path(response[key])
            value = value.resolve() if value.is_absolute() else (folder / value).resolve()
            if value != folder / required or not value.is_file():
                raise ContractError('BUILD_FAILED', 'Authored output path is absent or outside its required workspace')
        if (folder / 'asset.blend').stat().st_size < 10000:
            raise ContractError('BUILD_FAILED', 'Author Blender output is empty or unusable')
        for name, digest in protected.items():
            if not (folder / name).is_file() or file_hash(folder / name) != digest:
                raise ContractError('STALE_ARTIFACT', 'Author modified a protected evidence input', {'name': name})
        events_path = folder / 'events.jsonl'
        if not events_path.is_file():
            raise ContractError('BUILD_FAILED', 'Actual author command events are missing')
        events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        executed = [e['item'] for e in events if e.get('type') == 'item.completed'
                    and e.get('item', {}).get('type') == 'command_execution'
                    and e['item'].get('exit_code') == 0
                    and 'blender' in e['item'].get('command', '').lower()
                    and '--python' in e['item'].get('command', '')
                    and 'object.py' in e['item'].get('command', '')]
        if not executed:
            raise ContractError('BUILD_FAILED', 'No successful author-issued Blender execution of object.py')
        if self.origin == 'codex_image':
            request = read(folder / 'request.json')
            if request.get('status') != 'complete' or request.get('requested_model') != self.model:
                raise ContractError('BUILD_FAILED', 'Author provider record is incomplete or bound to another model')
        proof = {'origin': self.origin, 'model': self.model if self.origin == 'codex_image' else None,
                 'effort': self.effort, 'script_sha256': file_hash(folder / 'object.py'),
                 'authored_blend_sha256': file_hash(folder / 'asset.blend'),
                 'protected_inputs': protected, 'successful_blender_commands': executed,
                 'actual_author_execution': self.origin == 'codex_image',
                 'host_generated_geometry': False}
        write(folder / 'execution_proof.json', proof)
        return proof

    def _validate_authored(self, work, output, contract, proof, context, timeout_s):
        output.mkdir(exist_ok=False)
        script = output / 'direct_validate.py'
        shutil.copyfile(VERIFIER, script)
        specification = {**contract, 'output_dir': str(output), 'authored_blend': str(work / 'asset.blend'),
                         'authored_sha256': proof['authored_blend_sha256'], 'worker_sha256': file_hash(script),
                         'input_fingerprint': contract['candidate']['candidate_hash'], 'samples': self.samples,
                         'bounds_fit_tolerance': self.bounds_fit_tolerance}
        manifest = output / 'input.json'
        write(manifest, specification)
        manifest_hash = file_hash(manifest)
        self._process([str(self.blender_path), '--background', '--factory-startup', '--threads', '4',
                       '--python-exit-code', '1', '--python', str(script), '--', str(manifest)],
                      output, context, timeout_s, log_name='validation.log')
        if file_hash(manifest) != manifest_hash or file_hash(script) != specification['worker_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Independent verifier input/source was modified')
        result = read(output / 'result.json')
        if result['input_fingerprint'] != contract['candidate']['candidate_hash'] or result['authored_sha256'] != proof['authored_blend_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Independent export belongs to another candidate')
        for name, digest in result['files'].items():
            file = (output / name).resolve()
            if not file.is_relative_to(output) or not file.is_file() or file_hash(file) != digest:
                raise ContractError('STALE_ARTIFACT', 'Independent output hash mismatch')
        if not result.get('native_reopened') or not result.get('glb_reopened'):
            raise ContractError('BUILD_FAILED', 'Independent native/glTF reopening was not completed')
        if (result.get('render_source') != 'reopened_canonical_glb'
                or result.get('rendered_asset_sha256') != file_hash(output / 'asset.glb')):
            raise ContractError('STALE_ARTIFACT', 'Independent renders do not bind the final canonical GLB')
        return result

    @staticmethod
    def _montages(work, output, contract):
        turntable = Image.new('RGB', (768, 800), '#eef2ee')
        draw = ImageDraw.Draw(turntable)
        draw.text((12, 10), 'NEW OBJECT / FOUR FIXED INSPECTION VIEWS', fill='#244239')
        for index in range(4):
            with Image.open(output / f'turn_{index}.png') as image:
                turntable.paste(image.convert('RGB').resize((384, 384)), ((index % 2) * 384, 32 + (index // 2) * 384))
        path = output / 'turntable.jpg'
        turntable.save(path, quality=94)
        images = [path]
        for index, rectangle in enumerate(contract['crop_rectangles']):
            with Image.open(work / f'ref_{index:02d}_rgb.png') as source, Image.open(output / f'source_{index:02d}.png') as rendered:
                # Identical pixel rectangle and uniform common resize retain projection errors.
                crops = [im.convert('RGB').crop(rectangle) for im in (source, rendered)]
                labels = ['RAW RGB', 'GENERATED OBJECT']
                if contract['frames'][index].get('local_mask'):
                    with Image.open(work / contract['frames'][index]['local_mask']) as mask:
                        crops.append(mask.convert('RGB').crop(rectangle))
                    labels.append('OBSERVED TARGET MASK')
                width, height = crops[0].size
                scale = min(1., 500 / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                sheet = Image.new('RGB', (size[0] * len(crops), size[1] + 30), '#eef2ee')
                draw = ImageDraw.Draw(sheet)
                for k, (crop, label) in enumerate(zip(crops, labels)):
                    sheet.paste(crop.resize(size), (k * size[0], 30))
                    draw.text((k * size[0] + 5, 8), label, fill='#244239')
                pair = output / f'comparison_{index:02d}.jpg'
                sheet.save(pair, quality=94)
                images.append(pair)
        return images

    def _generate(self, root, scene, target, evidence, candidate, confirmation, confirmation_hash, budget, context):
        deadline = time.monotonic() + budget['timeout_s']
        allowed_calls = min(4, budget.get('max_model_calls', 2))
        iterations = min(2, budget['max_iterations'])
        calls = 0
        feedback = None
        ledger = {'origin': self.origin, 'scene_id': scene['scene_id'], 'instance_id': target['instance_id'],
                  'candidate': candidate, 'confirmation_sha256': confirmation_hash, 'model_calls': 0,
                  'provider_calls': 0, 'iterations': [], 'status': 'running'}
        write(root / 'run.json', ledger)

        def remaining(cap):
            value = min(cap, deadline - time.monotonic())
            if value < 1:
                raise ContractError('TIMEOUT', 'Direct modeling total budget exhausted')
            return value

        try:
            for iteration in range(iterations):
                context.check_cancelled()
                if calls + 2 > allowed_calls:
                    raise ContractError('RESOURCE_EXHAUSTED', 'Author plus independent critic requires two remaining model calls')
                work = root / f'author_{iteration:02d}'
                contract, images = self._prepare(work, scene, target, evidence, candidate, confirmation)
                # Only this run's preceding fresh candidate may be offered for a bounded correction.
                if iteration:
                    previous = root / f'author_{iteration - 1:02d}'
                    if (previous / 'object.py').is_file():
                        shutil.copyfile(previous / 'object.py', work / 'previous_object.py')
                    rendered = root / f'validated_{iteration - 1:02d}'
                    for path in [rendered / 'turntable.jpg', *sorted(rendered.glob('comparison_*.jpg'))]:
                        if path.is_file():
                            reference = work / ('previous_' + path.name)
                            shutil.copyfile(path, reference)
                            images.append(reference)
                protected = {p.name: file_hash(p) for p in work.iterdir() if p.is_file()}
                row = {'iteration': iteration, 'author': str(work), 'status': 'authoring'}
                ledger['iterations'].append(row)
                calls += 1
                ledger.update(provider_calls=calls, model_calls=calls if self.origin == 'codex_image' else 0)
                write(root / 'run.json', ledger)
                self._notify('authoring', target_id=target['instance_id'], iteration=iteration)
                response = self._invoke(author_prompt(work, contract, self.blender_path, feedback), images, work, context,
                                        timeout=remaining(self.author_timeout_s), schema=AUTHOR_SCHEMA,
                                        sandbox='workspace-write', max_tool_calls=self.max_author_tools)
                try:
                    proof = self._author_proof(work, response, target['instance_id'], protected)
                except ContractError as error:
                    if error.code != 'BUILD_FAILED' or iteration + 1 >= iterations:
                        raise
                    for name, digest in protected.items():
                        if not (work / name).is_file() or file_hash(work / name) != digest:
                            raise ContractError('STALE_ARTIFACT', 'Failed author modified a protected input', {'name': name})
                    # A terminal provider response with a failed Blender execution
                    # may consume the next bounded author turn. Uncertain provider
                    # calls and modified protected inputs never reach this branch.
                    events = work / 'events.jsonl'
                    failures = []
                    if events.is_file():
                        for line in events.read_text(encoding='utf-8').splitlines():
                            event = json.loads(line)
                            item = event.get('item', {})
                            if event.get('type') == 'item.completed' and item.get('type') == 'command_execution' and item.get('exit_code') not in (None, 0):
                                failures.append(item.get('aggregated_output', '')[-5000:])
                    feedback = {'technical_error': str(error), 'author_response': response, 'failed_execution_outputs': failures[-2:]}
                    row.update(status='author_technical_repair_requested', feedback=feedback)
                    write(root / 'run.json', ledger)
                    continue
                output = root / f'validated_{iteration:02d}'
                self._notify('validating_geometry', target_id=target['instance_id'], iteration=iteration)
                try:
                    result = self._validate_authored(work, output, contract, proof, context,
                                                    remaining(self.timeout_s))
                except ContractError as error:
                    if error.code not in {'BUILD_FAILED'} or iteration + 1 >= iterations:
                        raise
                    log = output / 'validation.log'
                    feedback = {'technical_error': log.read_text(encoding='utf-8')[-9000:] if log.is_file() else str(error)}
                    row.update(status='technical_repair_requested', feedback=feedback)
                    write(root / 'run.json', ledger)
                    continue
                critic_images = self._montages(work, output, contract)
                critic_dir = root / f'critic_{iteration:02d}'
                calls += 1
                ledger.update(provider_calls=calls, model_calls=calls if self.origin == 'codex_image' else 0)
                write(root / 'run.json', ledger)
                self._notify('independent_review', target_id=target['instance_id'], iteration=iteration)
                critic = self._invoke(critic_prompt(contract), critic_images, critic_dir, context,
                                      timeout=remaining(self.critic_timeout_s), schema=CRITIC_SCHEMA,
                                      sandbox='read-only', max_tool_calls=0)
                write(critic_dir / 'critic.json', critic)
                row.update(status='reviewed', critic=critic, result=str(output / 'result.json'))
                feedback = critic
                write(root / 'run.json', ledger)
                if critic['status'] == 'accept' and not any(item['severity'] == 'major' for item in critic['issues']):
                    if file_hash(self.confirmation_path) != confirmation_hash:
                        raise ContractError('STALE_REVISION', 'User confirmation changed while authoring')
                    for frame in evidence['frames']:
                        self.resolver.resolve(frame['rgb'], scene_id=scene['scene_id'])
                        if frame.get('mask'):
                            self.resolver.resolve(frame['mask'], scene_id=scene['scene_id'])
                    ledger.update(status='accepted_asset', accepted_iteration=iteration,
                                  accepted_asset=str(output / 'asset.glb'), visual_scope='Single-object shape review at capture pose; not whole-room metric accuracy.')
                    write(root / 'run.json', ledger)
                    return output, work, critic_dir, result
            raise ContractError('BUILD_FAILED', 'Bounded object refinement ended without independent acceptance', {'feedback': feedback})
        except BaseException as error:
            ledger.update(status='failed', error=str(error), automatic_retry=False)
            write(root / 'run.json', ledger)
            raise

    async def refine(self, refine_spec, scene_snapshot, evidence, job_context):
        validate(scene_snapshot, 'SceneSnapshot')
        validate(evidence, 'EvidenceBundle')
        candidate = refine_spec['candidate']
        validate(candidate, 'CandidateRef')
        if refine_spec.get('input_fingerprint', candidate['candidate_hash']) != candidate['candidate_hash']:
            raise ContractError('STALE_ARTIFACT', 'Candidate fingerprint mismatch')
        if candidate['scene_id'] != scene_snapshot['scene_id'] or candidate['base_revision'] != scene_snapshot['revision']:
            raise ContractError('STALE_REVISION', 'Candidate does not belong to the supplied base scene')
        identity = refine_spec['target_id']
        if set(refine_spec['allowed_fields']) != {'asset.geometry', 'asset.materials'}:
            raise ContractError('FORBIDDEN_SCOPE', 'Direct modeling requires geometry and materials fields only')
        for field in refine_spec['allowed_fields']:
            job_context.authorize(identity, field)
        target = next((o for o in scene_snapshot['objects'] if o['instance_id'] == identity), None)
        if target is None or identity not in evidence['target_ids']:
            raise ContractError('EVIDENCE_MISMATCH', 'Target absent from scene/evidence')
        if evidence['scene_id'] != scene_snapshot['scene_id'] or evidence['scene_revision'] != scene_snapshot['revision']:
            raise ContractError('EVIDENCE_MISMATCH', 'Evidence belongs to another scene revision')
        if any(not refine_spec['constraints'].get(key) for key in ('keep_root_transform', 'keep_support', 'keep_bounds')):
            raise ContractError('FORBIDDEN_SCOPE', 'Direct object refinement requires root/support/bounds preservation')
        confirmation, confirmation_hash = self._confirmation(scene_snapshot, candidate, identity)
        permitted = {'object', 'support'} | ({'floor', 'wall', 'opening', 'ceiling'} if confirmation.get('include_structures') is True else set())
        if target['kind'] not in permitted:
            raise ContractError('FORBIDDEN_SCOPE', 'Room component refinement requires explicit structure authorization')
        if refine_spec['budget'].get('max_model_calls', 2) < 2:
            raise ContractError('RESOURCE_EXHAUSTED', 'Direct author plus independent critic needs at least two model calls')
        if not self.blender_path.is_file():
            raise ContractError('TOOL_UNAVAILABLE', 'Configured Blender executable is unavailable')
        # Hash-check all existing scene inputs before any external call.
        self._scene_spec(scene_snapshot)
        original_hash = canonical_hash(scene_snapshot)
        root = self.storage_root / scene_snapshot['scene_id'] / 'assets' / ('gpt-direct-' + uuid.uuid4().hex)
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'frozen_scene.json', scene_snapshot)
        write(root / 'frozen_evidence.json', evidence)
        write(root / 'confirmation.json', confirmation)
        output, author, critic, result = await asyncio.to_thread(
            self._generate, root, scene_snapshot, target, evidence, candidate,
            confirmation, confirmation_hash, refine_spec['budget'], job_context)
        job_context.check_cancelled()
        if canonical_hash(scene_snapshot) != original_hash or result['local_bounds_m'] != target['local_bounds_m']:
            raise ContractError('STALE_ARTIFACT', 'Frozen scene or locked local bounds changed')
        artifact = self._artifact(output / 'asset.glb', scene_snapshot['scene_id'], 'object_geometry', 'model/gltf-binary')
        prefix = 'gpt-direct-' if self.origin == 'codex_image' else 'fixture-direct-'
        update = {'instance_id': identity, 'asset': {'asset_version': prefix + artifact['sha256'][:24],
                  'artifact': artifact, 'native_frame': 'asset_y_up', 'native_to_object': np.eye(4).tolist()},
                  'local_bounds_m': copy.deepcopy(target['local_bounds_m'])}
        draft = {key: copy.deepcopy(value) for key, value in scene_snapshot.items() if key not in {'revision', 'parent_revision'}}
        for item in draft['objects']:
            if item['instance_id'] == identity:
                item['asset'] = update['asset']
        # A reviewed package has no scene head. Building its entire old scene here
        # would be discarded when it is merged onto the latest head later.
        assembled = ({'input_fingerprint':candidate['candidate_hash'],'artifact_refs':[],'asset_updates':[]}
                     if getattr(self,'package_only',False) else await self.build({'candidate': candidate, 'scene': draft}, job_context))
        assembled['asset_updates'] = [update]
        assembled['artifact_refs'].append(artifact)
        refs = [(output / 'asset.blend', 'object_blend', 'application/x-blender'),
                (author / 'asset.blend', 'authored_native_object', 'application/x-blender'),
                (author / 'object.py', 'refinement_source', 'text/x-python'),
                (author / 'execution_proof.json', 'author_execution', 'application/json'),
                (author / 'contract.json', 'author_contract', 'application/json'),
                (output / 'result.json', 'refinement_manifest', 'application/json'),
                (output / 'reopen.json', 'asset_reopen', 'application/json'),
                (output / 'direct_validate.py', 'independent_export_source', 'text/x-python'),
                (output / 'validation.log.execution.json', 'independent_export_execution', 'application/json'),
                (output / 'turntable.jpg', 'object_turntable', 'image/jpeg'),
                (critic / 'critic.json', 'object_visual_review', 'application/json'),
                (root / 'run.json', 'refinement_run', 'application/json')]
        assembled['artifact_refs'].extend(self._artifact(path, scene_snapshot['scene_id'], role, mime) for path, role, mime in refs)
        self._notify('candidate_built', target_id=identity, artifact=artifact)
        return validate(assembled, 'BuildOutput')
