"""One bounded material correction after the rejected empty-world recovery.

Old failures remain immutable. A new author must execute its material edit, an
independent Blender process proves geometry unchanged, and a fresh critic must
accept the exported GLB before the ordinary repository commit path is used.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from video2scene.api_v1.__main__ import create_host
from video2scene.capture_evidence import transform_to_matrix
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, utc_now, validate
from .direct_asset import AUTHOR_SCHEMA, CRITIC_SCHEMA, DirectAssetBackend, critic_prompt, read, write
from .evidence import MaskEvidenceService
from .recovery import FIELDS, _atomic_json, _hashes, _original_execution, _unchanged, inspect_source, require_critic_accept


MATERIAL_NAME = 'recovery_material_v1'

# This host-owned worker runs outside the author's writable directory. Materials,
# image pixels, material assignments and UV coordinates deliberately are not part
# of the geometry signature. Raw and evaluated geometry both are included.
GEOMETRY_WORKER = r'''import bpy, hashlib, json, math, sys
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def matrix(value):
    return [[float(x) for x in row] for row in value]

def mesh_data(mesh):
    return {'vertices': [[float(x) for x in v.co] for v in mesh.vertices],
            'edges': [list(e.vertices) for e in mesh.edges],
            'faces': [list(p.vertices) for p in mesh.polygons]}

spec = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
if digest(__file__) != spec['worker_sha256'] or digest(spec['blend']) != spec['blend_sha256']:
    raise ValueError('Geometry fingerprint input/source changed')
bpy.ops.wm.open_mainfile(filepath=spec['blend'])
bpy.context.view_layer.update()
deps = bpy.context.evaluated_depsgraph_get()
objects, points = [], []
for ob in sorted(bpy.context.scene.objects, key=lambda x: x.name):
    row = {'name': ob.name, 'type': ob.type, 'parent': ob.parent.name if ob.parent else None,
           'parent_type': ob.parent_type, 'parent_bone': ob.parent_bone, 'hide_render': bool(ob.hide_render),
           'matrix_world': matrix(ob.matrix_world), 'matrix_local': matrix(ob.matrix_local),
           'matrix_parent_inverse': matrix(ob.matrix_parent_inverse), 'instance_type': ob.instance_type}
    if ob.type == 'MESH':
        row['raw_mesh'] = mesh_data(ob.data)
    if ob.type in {'MESH', 'CURVE', 'SURFACE', 'FONT'}:
        evaluated = ob.evaluated_get(deps)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
        row['evaluated_mesh'] = mesh_data(mesh)
        row['evaluated_matrix_world'] = matrix(evaluated.matrix_world)
        for v in mesh.vertices:
            points.append([float(x) for x in evaluated.matrix_world @ v.co])
        evaluated.to_mesh_clear()
    objects.append(row)
if not points or not all(math.isfinite(x) for point in points for x in point):
    raise ValueError('Missing or nonfinite evaluated geometry')
geometry = {'objects': objects, 'evaluated_bounds_blender':
            [[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]}
result = {'geometry': geometry, 'geometry_sha256': hashlib.sha256(encoded(geometry)).hexdigest(),
          'blend_sha256': spec['blend_sha256'], 'worker_sha256': spec['worker_sha256'],
          'blender_version': bpy.app.version_string}
Path(spec['output']).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
'''


def provider_receipt(folder, expected_model):
    """Bind a single actual provider call; copied archive folders are never counted."""
    folder = Path(folder).resolve()
    request = read(folder / 'request.json')
    outcome = read(folder / 'invocation_outcome.json')
    started = read(folder / 'invocation_started.json')
    response = read(folder / 'response.json')
    if (request.get('backend') != 'codex-cli' or request.get('status') != 'complete'
            or request.get('requested_model') != expected_model
            or request.get('response_sha256') != file_hash(folder / 'response.json')
            or outcome.get('origin') != 'codex_image' or outcome.get('status') != 'complete'
            or started.get('origin') != 'codex_image' or outcome.get('response_hash') != canonical_hash(response)):
        raise ContractError('STALE_ARTIFACT', 'Provider receipt is incomplete, uncertain or inconsistent', {'folder': str(folder)})
    names = ('request.json', 'response.json', 'events.jsonl', 'invocation_started.json', 'invocation_outcome.json')
    return {'directory': str(folder), 'requested_model': expected_model,
            'files': _hashes([folder / name for name in names]), 'response_status': response.get('status')}


def claim_material_run(source_run):
    root = Path(source_run) / MATERIAL_NAME
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ContractError('IDEMPOTENCY_CONFLICT', 'Material correction already started; no model call may be replayed') from exc
    return root


def require_same_geometry(before, after):
    for report in (before, after):
        if canonical_hash(report['geometry']) != report['geometry_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Independent geometry report hash is invalid')
    if before['geometry_sha256'] != after['geometry_sha256'] or before['geometry'] != after['geometry']:
        raise ContractError('FORBIDDEN_SCOPE', 'Material author changed geometry, transforms, visibility or evaluated bounds')


def fingerprint(backend, blend, folder, context, timeout_s):
    folder.mkdir(exist_ok=False)
    script, output = folder / 'geometry_worker.py', folder / 'geometry.json'
    script.write_text(GEOMETRY_WORKER, encoding='utf-8')
    spec = {'blend': str(blend.resolve()), 'blend_sha256': file_hash(blend),
            'worker_sha256': file_hash(script), 'output': str(output)}
    manifest = folder / 'input.json'
    write(manifest, spec)
    protected = _hashes([script, manifest, blend])
    backend._process([str(backend.blender_path), '--background', '--factory-startup', '--threads', '4',
                      '--python-exit-code', '1', '--python', str(script), '--', str(manifest)],
                     folder, context, timeout_s, log_name='geometry.log')
    _unchanged(protected)
    result = read(output)
    if (result['blend_sha256'] != spec['blend_sha256'] or result['worker_sha256'] != spec['worker_sha256']
            or result['geometry_sha256'] != canonical_hash(result['geometry'])):
        raise ContractError('STALE_ARTIFACT', 'Geometry report does not bind its input and independent worker')
    return result


def material_prompt(work, contract, blender):
    return f'''You are the material-only Blender author for {contract['instance_id']} ({contract['category']}).
Required script: {work / 'object.py'}
Required output: {work / 'asset.blend'}
Blender executable: {blender}

Actually read contract.json, color_measurements.json and previous_critic.json. The attached previous_comparison_00/01.jpg and previous_turntable.jpg show the EXPORTED GLB rejected by the independent critic. The ref_* images are original RGB/crops/observed masks. Inspect both the prior rendered failure and the original references before editing.

Open the supplied packed input.blend in object.py with bpy.ops.wm.open_mainfile. Preserve EVERY mesh, object, name, parent, modifier, transform, vertex, face, evaluated shape and local bounds. Do not use factory reset, rebuild geometry, add objects or change lighting/cameras/world/exposure. This is exclusively material/texture/UV editing on the copied input. An independent process compares raw and evaluated geometry hashes exactly before and after.

The preceding two independent critics found the wood far too pale/gray. Source table median encoded RGB is about [53,41,31], while the previous GLB render is about [146,125,106], despite its texture median around [72,56,40]. These are DISPLAY RGB bytes under different illumination, NOT measured albedo or a direct texture target. Simply copying source RGB to a texture was insufficient. Make the rendered wood substantially darker and warm brown under the UNCHANGED host verifier. Correct excessive bright/specular contribution as appropriate and add broad longitudinal wood variation with moderate contrast; avoid uniformly fine noisy grain. Preserve appropriate source uncertainty. Do not bake the decorative plate, shadows, other furniture or background into the wood.

Use glTF-compatible Principled BSDF materials and packed image textures. Procedural shaders must be baked if essential to exported appearance. Source display RGB, Blender linear socket values and sRGB texture pixels have different meanings. Inspect the measurements and prior images rather than claiming a radiometric calibration. Name any new texture differently from protected inputs. Save a self-contained packed asset.blend; the host renders the reopened final GLB for the critic.

Write and EXECUTE object.py with Blender --background --threads 4 --python-exit-code 1 --python object.py. Use CPU if rendering. One technical command correction is allowed within this author call; do not run your own iterative visual-review loop. Return AUTHOR_SCHEMA complete only after actual successful execution and the required asset exists. The instance_id must remain {contract['instance_id']}.
Do not modify input.blend, contract.json, color_measurements.json, previous_critic.json, ref_* or previous_* inputs. Use this workspace only. No parent/sibling file access, downloads, network calls, other agents, nested Codex or credentials. No new model calls. Return observed material corrections and assumptions honestly.'''


async def refine_material(run_root, source_run, blender, color_measurements, timeout_s=1200):
    run_root, source_run = Path(run_root).resolve(), Path(source_run).resolve()
    measurement_path = Path(color_measurements).resolve()
    measurement = read(measurement_path)
    if not isinstance(measurement, dict) or not 0 < timeout_s <= 1200:
        raise ValueError('Expected measurement JSON object and timeout in (0, 1200] seconds')
    confirmation = read(run_root / 'confirmation.json')
    confirmation_hash = file_hash(run_root / 'confirmation.json')
    scene_id, target_id = confirmation['scene_id'], confirmation['target_id']
    runtime = run_root / 'runtime'
    if not source_run.is_relative_to((runtime / scene_id / 'assets').resolve()):
        raise ContractError('FORBIDDEN_SCOPE', 'Source run must belong to the confirmed scene')
    original_run, provider, source_hashes = inspect_source(source_run)
    model = provider['requested_model']
    prior = source_run / 'recovery_world_init_v1'
    previous_status, previous_critic = read(prior / 'status.json'), read(prior / 'critic/critic.json')
    previous_contract, previous_result = read(prior / 'repaired/contract.json'), read(prior / 'validated/result.json')
    confirmed_base = {'revision': confirmation['revision'], 'head_seq': confirmation['head_seq']}
    if (previous_status.get('phase') != 'failed' or previous_status.get('committed_head') is not None
            or previous_critic.get('status') != 'revise' or previous_status.get('total_fine_calls') != 4
            or previous_status.get('confirmed_base') != confirmed_base
            or previous_status.get('confirmation_sha256') != confirmation_hash
            or original_run['confirmation_sha256'] != confirmation_hash or confirmation.get('confirmed') is not True):
        raise ContractError('FORBIDDEN_SCOPE', 'Material continuation requires the confirmed, rejected fourth-call recovery')
    folders = [source_run / 'author_00', source_run / 'critic_00', source_run / 'author_01', prior / 'critic']
    receipts = [provider_receipt(folder, model) for folder in folders]
    for receipt in receipts:
        source_hashes.update(receipt['files'])
    prior_inputs = [prior / name for name in ('status.json', 'repair_proof.json', 'source_hashes.json',
                    'original_execution_records.json', 'repaired/asset.blend', 'repaired/contract.json',
                    'validated/result.json', 'validated/asset.glb', 'validated/turntable.jpg', 'critic/critic.json')]
    prior_inputs += sorted((prior / 'validated').glob('comparison_*.jpg'))
    prior_inputs += sorted((prior / 'repaired').glob('ref_*.png'))
    source_hashes.update(_hashes(prior_inputs + [measurement_path]))
    if (file_hash(prior / 'repaired/asset.blend') != previous_result['authored_sha256']
            or file_hash(prior / 'validated/asset.glb') != previous_result['rendered_asset_sha256']):
        raise ContractError('STALE_ARTIFACT', 'Previous material rejection is not bound to its source/native GLB')
    frozen, evidence = read(source_run / 'frozen_scene.json'), read(source_run / 'frozen_evidence.json')
    if canonical_hash(frozen) != confirmation['snapshot_hash'] or read(source_run / 'confirmation.json') != confirmation:
        raise ContractError('STALE_REVISION', 'Confirmed scene differs from the failed run')
    host = create_host(runtime)
    host.evidence = MaskEvidenceService(runtime, host.repo)
    host.validator.evidence_service = host.evidence
    host.repo.evidence_getter = host.evidence.get
    decision = 'decision_' + canonical_hash([scene_id, confirmation['command_id']])[:32]
    context = host.context(actor='user', session_id=confirmation['session_id'], decision_ref=decision,
                           allowed_object_ids=frozenset({target_id}), allowed_fields=FIELDS)
    token = canonical_hash([str(source_run), confirmation_hash, file_hash(measurement_path), MATERIAL_NAME])[:28]

    def request(tool, params, suffix, mutate=False):
        value = {'contract_version': CONTRACT_VERSION, 'request_id': 'material_' + token + '_' + suffix,
                 'scene_id': scene_id, 'tool': tool, 'params': params}
        if mutate:
            value.update(base_revision=confirmed_base['revision'], base_head_seq=confirmed_base['head_seq'])
        return validate(value, 'ToolRequest')

    async def check_base():
        view = await host.repo.inspect(request('inspect_scene', {'revision': 'head'}, 'inspect'), context)
        if view['head'] != confirmed_base or view['snapshot'] != frozen or file_hash(run_root / 'confirmation.json') != confirmation_hash:
            raise ContractError('STALE_REVISION', 'Confirmed base changed; material correction cannot be rebased')

    await check_base()
    if await host.evidence.get(evidence['evidence_id'], context) != evidence:
        raise ContractError('EVIDENCE_MISMATCH', 'Registered source evidence changed')
    original_execution = _original_execution(host, confirmation)
    if original_execution != read(prior / 'original_execution_records.json'):
        raise ContractError('STALE_ARTIFACT', 'Original failed command/job changed after recovery')
    target = next(obj for obj in frozen['objects'] if obj['instance_id'] == target_id)
    initial = read(run_root / 'snapshot.json')
    observation = next(obj for obj in initial['objects'] if obj['instance_id'] == target_id)
    if (previous_contract['instance_id'] != target_id or previous_contract['evidence_id'] != evidence['evidence_id']
            or previous_contract['evidence_hash'] != evidence['evidence_hash']
            or previous_contract['accepted_transform'] != target['transform']
            or previous_contract['local_bounds_m'] != target['local_bounds_m']
            or previous_contract['support_id'] != target['support_id']
            or previous_contract['observation_transform'] != transform_to_matrix(observation['transform']).tolist()
            or len(previous_contract['frames']) != len(evidence['frames'])):
        raise ContractError('EVIDENCE_MISMATCH', 'Material source contract changed locked geometry or observation pose')
    for frame, observed in zip(previous_contract['frames'], evidence['frames']):
        if any(frame[key] != value for key, value in observed.items()):
            raise ContractError('FRAME_MISMATCH', 'Material source camera/RGB no longer matches capture')
    backend = DirectAssetBackend(runtime, blender_path=blender, confirmation_path=run_root / 'confirmation.json',
                                 model=model, effort=provider['reasoning_effort'])
    if not backend.blender_path.is_file():
        raise ContractError('TOOL_UNAVAILABLE', 'Configured Blender is absent')
    backend._scene_spec(frozen)
    root = claim_material_run(source_run)
    deadline = time.monotonic() + timeout_s
    timer = threading.Timer(timeout_s, context.cancel_event.set)
    timer.daemon = True
    timer.start()
    calls, committed_head = 0, None
    previous_call_count = len(receipts)
    base = {'confirmed_base': confirmed_base, 'target_id': target_id, 'confirmation_sha256': confirmation_hash,
            'source_run': str(source_run), 'recovery_root': str(root), 'scene_id': scene_id, 'mode': 'material_only'}
    ledger = dict(base, status='running', previous_fine_calls=len(receipts), new_call_limit=2, new_model_calls=0,
                  provider_call_receipts=[r['directory'] for r in receipts], receipt_records=copy.deepcopy(receipts))

    def status(phase, message, **details):
        ledger.update(new_model_calls=calls, total_fine_calls=previous_call_count + calls,
                      provider_call_receipts=[r['directory'] for r in receipts], receipt_records=receipts)
        _atomic_json(root / 'run.json', ledger)
        value = dict(base, phase=phase, message=message, updated_at=utc_now(), total_fine_calls=previous_call_count + calls, **details)
        _atomic_json(root / 'status.json', value)
        _atomic_json(run_root / 'recovery_status.json', value)

    def remaining(cap=1200):
        context.check_cancelled()
        value = min(cap, deadline - time.monotonic())
        if value <= 0:
            raise ContractError('TIMEOUT', 'Material correction total timeout exhausted')
        return value

    async def call_model(prompt, images, folder, schema, sandbox, cap, tools):
        nonlocal calls
        if calls >= 2 or (folder / 'invocation_started.json').exists():
            raise ContractError('RESOURCE_EXHAUSTED', 'Material call budget exhausted or invocation already started')
        calls += 1
        status('refining' if schema == AUTHOR_SCHEMA else 'validating',
               'GPT 正在依据上轮渲染和源图修正材质。' if schema == AUTHOR_SCHEMA else '正在独立检查修正后的最终 GLB。',
               stage='material_author' if schema == AUTHOR_SCHEMA else 'material_critic')
        result = await asyncio.to_thread(backend._invoke, prompt, images, folder, context, timeout=remaining(cap),
                                        schema=schema, sandbox=sandbox, max_tool_calls=tools)
        receipts.append(provider_receipt(folder, model))
        return result

    try:
        status('refining', '正在锁定几何、归档前四次调用，并准备材质修正。', stage='preparing')
        write(root / 'raw_source_hashes.json', source_hashes)
        write(root / 'original_execution_records.json', original_execution)
        write(root / 'prior_provider_receipts.json', receipts)
        archive = root / 'prior_records'
        archive.mkdir()
        shutil.copyfile(prior / 'status.json', archive / 'world_recovery_failed.json')
        shutil.copyfile(source_run / 'run.json', archive / 'original_run_failed.json')
        prepare = request('refine_asset', {'target_id': target_id, 'intent': 'design_edit', 'allowed_fields': sorted(FIELDS),
                          'evidence_id': evidence['evidence_id'], 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                          'budget': {'max_iterations': 1, 'max_model_calls': 2, 'timeout_s': int(timeout_s)},
                          'decision_ref': decision}, 'prepare', True)
        candidate = await host.repo.prepare_refinement(prepare, context)
        write(root / 'prepare_request.json', prepare)
        write(root / 'candidate.json', candidate)
        work = root / 'author'
        work.mkdir()
        contract = copy.deepcopy(previous_contract)
        contract.update(candidate=candidate, input_origin='Material-only correction of the copied prior native asset; all geometry is immutable.')
        write(work / 'contract.json', contract)
        write(work / 'previous_critic.json', previous_critic)
        shutil.copyfile(measurement_path, work / 'color_measurements.json')
        shutil.copyfile(prior / 'repaired/asset.blend', work / 'input.blend')
        images = []
        for path in sorted((prior / 'validated').glob('comparison_*.jpg')) + [prior / 'validated/turntable.jpg']:
            destination = work / ('previous_' + path.name)
            shutil.copyfile(path, destination)
            images.append(destination)
        for path in sorted((prior / 'repaired').glob('ref_*.png')):
            destination = work / path.name
            shutil.copyfile(path, destination)
            images.append(destination)
        before = await asyncio.to_thread(fingerprint, backend, work / 'input.blend', root / 'geometry_before', context, remaining(120))
        protected = {path.name: file_hash(path) for path in work.iterdir() if path.is_file()}
        _unchanged(source_hashes)
        author = await call_model(material_prompt(work, contract, blender), images, work, AUTHOR_SCHEMA, 'workspace-write', 720, 32)
        proof = backend._author_proof(work, author, target_id, protected)
        status('validating', '正在逐项核对网格、变换和计算后的边界完全未变。', stage='geometry_comparison')
        after = await asyncio.to_thread(fingerprint, backend, work / 'asset.blend', root / 'geometry_after', context, remaining(120))
        require_same_geometry(before, after)
        geometry_proof = {'unchanged': True, 'before_geometry_sha256': before['geometry_sha256'],
                          'after_geometry_sha256': after['geometry_sha256'], 'source_blend_sha256': before['blend_sha256'],
                          'authored_blend_sha256': after['blend_sha256'], 'scope': 'Raw/evaluated mesh, transforms, hierarchy, visibility and bounds; materials/texture/UV excluded.'}
        write(root / 'geometry_unchanged.json', geometry_proof)
        output = root / 'validated'
        result = await asyncio.to_thread(backend._validate_authored, work, output, contract, proof, context, remaining(300))
        if result['local_bounds_m'] != target['local_bounds_m']:
            raise ContractError('FORBIDDEN_SCOPE', 'Material correction changed locked local bounds')
        critic_images = backend._montages(work, output, contract)
        await check_base()
        critic_dir = root / 'critic'
        critic = await call_model(critic_prompt(contract), critic_images, critic_dir, CRITIC_SCHEMA, 'read-only', 180, 0)
        write(critic_dir / 'critic.json', critic)
        require_critic_accept(critic)
        critic_request = read(critic_dir / 'request.json')
        if critic_request.get('sandbox') != 'read-only' or critic_request.get('max_tool_calls') != 0:
            raise ContractError('FORBIDDEN_SCOPE', 'Independent critic used an unexpected tool scope')
        if len(receipts) != 6 or len({r['directory'] for r in receipts}) != 6 or calls != 2:
            raise ContractError('RESOURCE_EXHAUSTED', 'Material completion does not have exactly six unique fine-call receipts')
        _unchanged(source_hashes)
        for receipt in receipts:
            _unchanged(receipt['files'])
        for name, digest in protected.items():
            if file_hash(work / name) != digest:
                raise ContractError('STALE_ARTIFACT', 'Protected material input changed')
        if _original_execution(host, confirmation) != original_execution:
            raise ContractError('STALE_ARTIFACT', 'Old failed command/job changed')
        await check_base()
        artifact = backend._artifact(output / 'asset.glb', scene_id, 'object_geometry', 'model/gltf-binary')
        update = {'instance_id': target_id, 'asset': {'asset_version': 'gpt-material-' + artifact['sha256'][:24],
                  'artifact': artifact, 'native_frame': 'asset_y_up', 'native_to_object': np.eye(4).tolist()},
                  'local_bounds_m': copy.deepcopy(target['local_bounds_m'])}
        draft = {key: copy.deepcopy(value) for key, value in frozen.items() if key not in {'revision', 'parent_revision'}}
        next(obj for obj in draft['objects'] if obj['instance_id'] == target_id)['asset'] = update['asset']
        accepted = dict(base, status='independently_accepted', candidate=candidate, critic=critic, geometry_proof=geometry_proof,
                        total_fine_calls=len(receipts), provider_call_receipts=[r['directory'] for r in receipts],
                        receipt_records=receipts, raw_source_hashes=source_hashes, previous_fine_calls=4, new_author_calls=1, new_critic_calls=1)
        write(root / 'accepted_material.json', accepted)
        status('validating', '材质已通过独立检查，正在构建、验证和提交候选。', stage='building_candidate')
        backend.timeout_s = remaining(300)
        build = await backend.build({'candidate': candidate, 'scene': draft}, context)
        build['asset_updates'] = [update]
        build['artifact_refs'].append(artifact)
        refs = [(work / 'object.py', 'refinement_source', 'text/x-python'),
                (work / 'asset.blend', 'authored_native_object', 'application/x-blender'),
                (work / 'input.blend', 'material_source_native', 'application/x-blender'),
                (work / 'execution_proof.json', 'author_execution', 'application/json'),
                (work / 'contract.json', 'author_contract', 'application/json'),
                (work / 'color_measurements.json', 'material_measurements', 'application/json'),
                (work / 'previous_critic.json', 'previous_visual_review', 'application/json'),
                (root / 'accepted_material.json', 'refinement_recovery', 'application/json'),
                (root / 'geometry_unchanged.json', 'material_geometry_proof', 'application/json'),
                (root / 'geometry_before/geometry.json', 'geometry_before_material', 'application/json'),
                (root / 'geometry_after/geometry.json', 'geometry_after_material', 'application/json'),
                (root / 'geometry_after/geometry_worker.py', 'geometry_verifier_source', 'text/x-python'),
                (root / 'geometry_after/geometry.log.execution.json', 'geometry_verifier_execution', 'application/json'),
                (root / 'raw_source_hashes.json', 'raw_source_hashes', 'application/json'),
                (output / 'asset.blend', 'object_blend', 'application/x-blender'),
                (output / 'result.json', 'refinement_manifest', 'application/json'),
                (output / 'reopen.json', 'asset_reopen', 'application/json'),
                (output / 'direct_validate.py', 'independent_export_source', 'text/x-python'),
                (output / 'validation.log.execution.json', 'independent_export_execution', 'application/json'),
                (output / 'turntable.jpg', 'object_turntable', 'image/jpeg'),
                (critic_dir / 'critic.json', 'object_visual_review', 'application/json')]
        refs += [(Path(name), 'fine_provider_receipt', 'application/json') for receipt in receipts for name in receipt['files']]
        refs += [(path, 'previous_material_render', 'image/jpeg') for path in work.glob('previous_*.jpg')]
        refs += [(output / f'comparison_{i:02d}.jpg', 'object_comparison', 'image/jpeg') for i in range(len(contract['frames']))]
        build['artifact_refs'].extend(backend._artifact(path, scene_id, role, mime) for path, role, mime in refs)
        validate(build, 'BuildOutput')
        remaining()
        await check_base()
        built = await host.repo.attach_build(candidate['candidate_id'], candidate['candidate_hash'], build, context)
        persisted = await host.repo.read_candidate(built['candidate_id'], built['candidate_hash'], context)
        report = await host.validator.validate(persisted, evidence, 'tabletop_design_v1', context)
        if not report['technical_pass'] or not report['committable']:
            raise ContractError('CONSTRAINT_VIOLATION', 'Material candidate failed technical validation')
        _unchanged(source_hashes)
        remaining()
        await check_base()
        commit = request('commit_candidate', {'candidate_id': built['candidate_id'], 'candidate_hash': built['candidate_hash'],
                         'validation_id': report['validation_id'], 'decision_ref': decision}, 'commit', True)
        write(root / 'commit_request.json', commit)
        committed = await host.repo.commit(commit, context)
        committed_head = {'revision': committed['revision'], 'head_seq': committed['head_seq']}
        final = await host.repo.inspect(request('inspect_scene', {'revision': 'head'}, 'final'), context)
        if final['head'] != committed_head:
            raise ContractError('STALE_REVISION', 'Scene changed after material commit')
        before_objects = {obj['instance_id']: obj for obj in frozen['objects']}
        for obj in final['snapshot']['objects']:
            previous = before_objects[obj['instance_id']]
            if obj['instance_id'] != target_id and obj != previous:
                raise ContractError('FORBIDDEN_SCOPE', 'Non-target changed during material commit')
            if obj['instance_id'] == target_id and (obj['asset'] != update['asset'] or any(obj[k] != previous[k] for k in ('transform', 'local_bounds_m', 'support_id'))):
                raise ContractError('FORBIDDEN_SCOPE', 'Material commit changed the accepted object constraints')
        completion = dict(accepted, status='complete', head=committed_head, candidate=built, validation_id=report['validation_id'],
                          artifact_refs=build['artifact_refs'], repaired_asset_sha256=artifact['sha256'],
                          original_command_state='failed', original_job_state='failed', completed_at=utc_now())
        _atomic_json(root / 'completion.json', completion)
        _atomic_json(run_root / 'recovery_completion.json', completion)
        ledger['status'] = 'complete'
        status('complete', '茶几材质通过独立复核并已保存，几何和已确认布局保持不变。', head=committed_head, stage='committed')
        await host.sync_outbox(scene_id)
        return completion
    except BaseException as error:
        ledger.update(status='failed', automatic_retry=False)
        details = error.as_dict() if isinstance(error, ContractError) else {'code': type(error).__name__, 'message': str(error)}
        message = ('材质已提交，但提交后的状态核对未完成；请检查记录，不自动重放。' if committed_head
                   else '材质修正未通过，原失败记录和本轮记录均已保留；没有提交或自动重放。')
        status('failed', message,
               error=details, committed_head=committed_head, automatic_retry=False)
        raise
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--color-measurements', type=Path, required=True)
    parser.add_argument('--timeout-s', type=int, default=1200)
    args = parser.parse_args()
    try:
        result = asyncio.run(refine_material(args.run_root, args.source_run, args.blender, args.color_measurements, args.timeout_s))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error), 'automatic_retry': False}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    print(json.dumps({'status': result['status'], 'head': result['head'], 'total_fine_calls': result['total_fine_calls']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
