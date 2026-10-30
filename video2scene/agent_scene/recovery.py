"""One-shot recovery of the known empty-world author failure.

The failed author response/job stay failed. Only a mechanical world initialization
is added to a copy of its script; one remaining independent critic may approve a
new candidate through the normal repository validation and commit boundary.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import difflib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time

import numpy as np

from video2scene.api_v1.__main__ import create_host
from video2scene.capture_evidence import transform_to_matrix
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, utc_now, validate
from .direct_asset import CRITIC_SCHEMA, DirectAssetBackend, critic_prompt, read, write
from .evidence import MaskEvidenceService


WORLD_LINE = b'bpy.context.scene.world.color = (.3,.3,.3)'
RECOVERY_NAME = 'recovery_world_init_v1'
FIELDS = frozenset({'asset.geometry', 'asset.materials'})


def patch_world_initialization(source):
    """Permit exactly one insertion before the known failing line, byte for byte."""
    lines = source.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.rstrip(b'\r\n') == WORLD_LINE]
    if len(matches) != 1 or b'Recovered World' in source:
        raise ContractError('BUILD_FAILED', 'Script is not the supported unrepaired empty-world failure')
    index = matches[0]
    newline = b'\r\n' if lines[index].endswith(b'\r\n') else b'\n'
    insertion = (b'if bpy.context.scene.world is None:' + newline
                 + b"    bpy.context.scene.world = bpy.data.worlds.new('Recovered World')" + newline)
    patched = b''.join(lines[:index]) + insertion + b''.join(lines[index:])
    ast.parse(patched.decode('utf-8'))
    return patched


def require_critic_accept(critic):
    from jsonschema import Draft202012Validator
    Draft202012Validator(CRITIC_SCHEMA).validate(critic)
    if critic['status'] != 'accept' or any(issue['severity'] == 'major' for issue in critic['issues']):
        raise ContractError('BUILD_FAILED', 'Final independent critic did not accept the repaired object', {'critic': critic})


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _hashes(paths):
    return {str(path.resolve()): file_hash(path) for path in paths}


def _unchanged(hashes):
    for name, digest in hashes.items():
        if not Path(name).is_file() or file_hash(name) != digest:
            raise ContractError('STALE_ARTIFACT', 'Recovery input changed', {'path': name})


def inspect_source(source_run):
    """Read only: prove the three completed calls and the known failed execution."""
    source_run = Path(source_run).resolve()
    ledger = read(source_run / 'run.json')
    if (ledger.get('origin') != 'codex_image' or ledger.get('status') != 'failed'
            or ledger.get('model_calls') != 3 or ledger.get('provider_calls') != 3):
        raise ContractError('FORBIDDEN_SCOPE', 'Recovery requires the failed three-call direct-model run')
    author = source_run / 'author_01'
    response, request = read(author / 'response.json'), read(author / 'request.json')
    if response.get('status') != 'failed' or response.get('instance_id') != ledger['instance_id']:
        raise ContractError('FORBIDDEN_SCOPE', 'The source author must retain its explicit failed response')
    events = [json.loads(line) for line in (author / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    failures = [event['item'] for event in events if event.get('type') == 'item.completed'
                and event.get('item', {}).get('type') == 'command_execution'
                and event['item'].get('exit_code') not in (None, 0)
                and "'NoneType' object has no attribute 'color'" in event['item'].get('aggregated_output', '')]
    if not failures or 'object.py' not in failures[-1].get('command', ''):
        raise ContractError('BUILD_FAILED', 'Source events do not establish the supported empty-world failure')
    paths = [source_run / name for name in ('run.json', 'frozen_scene.json', 'frozen_evidence.json', 'confirmation.json')]
    for call in ('author_00', 'critic_00', 'author_01'):
        folder = source_run / call
        record = read(folder / 'request.json')
        outcome = read(folder / 'invocation_outcome.json')
        if (record.get('status') != 'complete' or outcome.get('status') != 'complete'
                or record.get('requested_model') != request['requested_model']):
            raise ContractError('RESOURCE_EXHAUSTED', 'An original provider outcome is uncertain; no automatic recovery')
        for name in ('request.json', 'response.json', 'events.jsonl', 'invocation_started.json', 'invocation_outcome.json'):
            paths.append(folder / name)
    paths += [author / name for name in ('object.py', 'contract.json', 'dark_walnut.png')]
    paths += sorted(author.glob('ref_*.png'))
    paths.append(source_run / 'critic_00/critic.json')
    for image in request.get('images', []):
        path = Path(image['path']).resolve()
        if not path.is_relative_to(author) or not path.name.startswith('ref_') or file_hash(path) != image['sha256']:
            raise ContractError('STALE_ARTIFACT', 'Original author reference image changed')
    # Validate the exact permitted edit before any persistent recovery state exists.
    patch_world_initialization((author / 'object.py').read_bytes())
    if (source_run / 'critic_01/invocation_started.json').exists():
        raise ContractError('IDEMPOTENCY_CONFLICT', 'The fourth model call has already started')
    return ledger, request, _hashes(paths)


def _original_execution(host, confirmation):
    command = host.store.command(confirmation['scene_id'], confirmation['command_id'])
    request_id = 'req_' + canonical_hash([confirmation['scene_id'], confirmation['command_id'], 'refine'])[:40]
    database = host.root / 'job.sqlite'
    if not database.is_file():
        raise ContractError('NOT_FOUND', 'Original job record is absent')
    with sqlite3.connect(database) as connection:
        row = connection.execute('SELECT record_json,request_json FROM jobs WHERE scene_id=? AND request_id=?',
                                 (confirmation['scene_id'], request_id)).fetchone()
    if row is None:
        raise ContractError('NOT_FOUND', 'Original refinement job is absent')
    job, request = json.loads(row[0]), json.loads(row[1])
    if command['state'] != 'failed' or job['state'] != 'failed':
        raise ContractError('FORBIDDEN_SCOPE', 'Recovery requires a terminal failed command and job')
    return {'command': command, 'job': job, 'request': request}


async def recover(run_root, source_run, blender, timeout_s=600):
    run_root, source_run = Path(run_root).resolve(), Path(source_run).resolve()
    if timeout_s <= 0 or timeout_s > 600:
        raise ValueError('Recovery timeout must be in (0, 600] seconds')
    confirmation = read(run_root / 'confirmation.json')
    confirmation_hash = file_hash(run_root / 'confirmation.json')
    scene_id, target_id = confirmation['scene_id'], confirmation['target_id']
    runtime = run_root / 'runtime'
    if not source_run.is_relative_to((runtime / scene_id / 'assets').resolve()):
        raise ContractError('FORBIDDEN_SCOPE', 'Source run must belong to the confirmed scene assets')
    ledger, provider, source_hashes = inspect_source(source_run)
    frozen_scene, evidence = read(source_run / 'frozen_scene.json'), read(source_run / 'frozen_evidence.json')
    validate(frozen_scene, 'SceneSnapshot')
    validate(evidence, 'EvidenceBundle')
    confirmed_base = {'revision': confirmation['revision'], 'head_seq': confirmation['head_seq']}
    if (confirmation.get('confirmed') is not True or ledger['confirmation_sha256'] != confirmation_hash
            or read(source_run / 'confirmation.json') != confirmation
            or ledger['scene_id'] != scene_id or ledger['instance_id'] != target_id
            or ledger['candidate']['base_revision'] != confirmed_base['revision']
            or ledger['candidate']['base_head_seq'] != confirmed_base['head_seq']
            or canonical_hash(frozen_scene) != confirmation['snapshot_hash']):
        raise ContractError('STALE_REVISION', 'Original failed run does not match the confirmed whitebox')
    host = create_host(runtime)  # No model provider or job worker is started.
    host.evidence = MaskEvidenceService(runtime, host.repo)
    host.validator.evidence_service = host.evidence
    host.repo.evidence_getter = host.evidence.get
    decision = 'decision_' + canonical_hash([scene_id, confirmation['command_id']])[:32]
    context = host.context(actor='user', session_id=confirmation['session_id'], decision_ref=decision,
                           allowed_object_ids=frozenset({target_id}), allowed_fields=FIELDS)
    token = canonical_hash([str(source_run), source_hashes, confirmation_hash])[:28]

    def request(tool, params, suffix, mutate=False):
        value = {'contract_version': CONTRACT_VERSION, 'request_id': 'recovery_' + token + '_' + suffix,
                 'scene_id': scene_id, 'tool': tool, 'params': params}
        if mutate:
            value.update(base_revision=confirmed_base['revision'], base_head_seq=confirmed_base['head_seq'])
        return validate(value, 'ToolRequest')

    async def check_base():
        view = await host.repo.inspect(request('inspect_scene', {'revision': 'head'}, 'inspect'), context)
        if view['head'] != confirmed_base or view['snapshot'] != frozen_scene:
            raise ContractError('STALE_REVISION', 'Confirmed whitebox changed; this recovery cannot be rebased')
        if file_hash(run_root / 'confirmation.json') != confirmation_hash:
            raise ContractError('STALE_REVISION', 'User confirmation changed')
        return view

    await check_base()
    stored_evidence = await host.evidence.get(evidence['evidence_id'], context)
    if stored_evidence != evidence:
        raise ContractError('EVIDENCE_MISMATCH', 'Recovery evidence differs from original registered capture')
    original = _original_execution(host, confirmation)
    if original['request']['params']['evidence_id'] != evidence['evidence_id']:
        raise ContractError('EVIDENCE_MISMATCH', 'Original job used different evidence')
    original_metadata = await host.repo.candidate_metadata(ledger['candidate']['candidate_id'], context)
    if original_metadata['decision_ref'] != decision or original_metadata['base_snapshot'] != frozen_scene:
        raise ContractError('FORBIDDEN_SCOPE', 'Original candidate does not belong to the confirmed decision')
    backend = DirectAssetBackend(runtime, blender_path=blender, confirmation_path=run_root / 'confirmation.json',
                                 model=provider['requested_model'], effort=provider['reasoning_effort'])
    if not backend.blender_path.is_file():
        raise ContractError('TOOL_UNAVAILABLE', 'Configured Blender executable is absent')
    backend._scene_spec(frozen_scene)
    recovery_root = source_run / RECOVERY_NAME
    # This exclusive directory claims the last model call across all restarts.
    # Any interrupted recovery requires reconciliation, never automatic replay.
    try:
        recovery_root.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ContractError('IDEMPOTENCY_CONFLICT', 'Recovery already started; inspect its existing outcome without replay') from exc
    deadline = time.monotonic() + timeout_s
    timer = threading.Timer(timeout_s, context.cancel_event.set)
    timer.daemon = True
    timer.start()
    total_calls, committed_head = 3, None
    status_base = {'confirmed_base': confirmed_base, 'target_id': target_id, 'confirmation_sha256': confirmation_hash,
                   'source_run': str(source_run), 'recovery_root': str(recovery_root), 'scene_id': scene_id}

    def status(phase, message, **details):
        value = dict(status_base, phase=phase, message=message, time=utc_now(), total_fine_calls=total_calls, **details)
        _atomic_json(recovery_root / 'status.json', value)
        _atomic_json(run_root / 'recovery_status.json', value)

    def remaining(cap=600):
        context.check_cancelled()
        value = min(cap, deadline - time.monotonic())
        if value <= 0:
            raise ContractError('TIMEOUT', 'Recovery total timeout exhausted')
        return value

    try:
        status('refining', '正在保留原失败记录，并执行一次空场景 world 初始化修复。', stage='host_repair')
        write(recovery_root / 'original_execution_records.json', original)
        write(recovery_root / 'source_hashes.json', source_hashes)
        source_copy = recovery_root / 'original'
        for name, digest in source_hashes.items():
            path = Path(name)
            destination = source_copy / path.relative_to(source_run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if file_hash(destination) != digest:
                raise ContractError('STALE_ARTIFACT', 'Original failure copy differs from source')
        prepare = request('refine_asset', {'target_id': target_id, 'intent': 'design_edit',
                          'allowed_fields': sorted(FIELDS), 'evidence_id': evidence['evidence_id'],
                          'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                          'budget': {'max_iterations': 1, 'max_model_calls': 1, 'timeout_s': int(timeout_s)},
                          'decision_ref': decision}, 'prepare', True)
        candidate = await host.repo.prepare_refinement(prepare, context)
        if candidate['candidate_id'] == ledger['candidate']['candidate_id']:
            raise ContractError('STALE_ARTIFACT', 'Recovery requires its own candidate')
        write(recovery_root / 'prepare_request.json', prepare)
        write(recovery_root / 'candidate.json', candidate)
        author = source_run / 'author_01'
        work = recovery_root / 'repaired'
        work.mkdir()
        contract = read(author / 'contract.json')
        target = next(obj for obj in frozen_scene['objects'] if obj['instance_id'] == target_id)
        if (contract['candidate'] != ledger['candidate'] or contract['evidence_id'] != evidence['evidence_id']
                or contract['evidence_hash'] != evidence['evidence_hash'] or contract['instance_id'] != target_id
                or contract['accepted_transform'] != target['transform']
                or contract['local_bounds_m'] != target['local_bounds_m'] or contract['support_id'] != target['support_id']):
            raise ContractError('EVIDENCE_MISMATCH', 'Original author contract differs from confirmed target')
        initial = read(run_root / 'snapshot.json')
        observation = next(obj for obj in initial['objects'] if obj['instance_id'] == target_id)
        if (contract['observation_transform'] != transform_to_matrix(observation['transform']).tolist()
                or len(contract['frames']) != len(evidence['frames'])):
            raise ContractError('EVIDENCE_MISMATCH', 'Original observation pose/frame count differs from initial capture')
        for frame, observed in zip(contract['frames'], evidence['frames']):
            if any(frame[key] != value for key, value in observed.items()):
                raise ContractError('FRAME_MISMATCH', 'Author source camera/image differs from evidence')
        contract['candidate'] = candidate  # Only this host-owned binding changes.
        write(work / 'contract.json', contract)
        for path in sorted(author.glob('ref_*.png')) + [author / 'dark_walnut.png']:
            shutil.copyfile(path, work / path.name)
        raw = (author / 'object.py').read_bytes()
        patched = patch_world_initialization(raw)
        (work / 'object.py').write_bytes(patched)
        patch_path = recovery_root / 'world_initialization.patch'
        patch_path.write_text(''.join(difflib.unified_diff(raw.decode('utf-8').splitlines(keepends=True),
                                                        patched.decode('utf-8').splitlines(keepends=True),
                                                        fromfile='original/author_01/object.py', tofile='repaired/object.py')), encoding='utf-8')
        protected = _hashes([path for path in work.iterdir() if path.is_file()])
        _unchanged(source_hashes)
        await asyncio.to_thread(backend._process, [str(backend.blender_path), '--background', '--factory-startup',
                                '--threads', '4', '--python-exit-code', '1', '--python', str(work / 'object.py')],
                                work, context, remaining(180), log_name='host_repair.log')
        _unchanged(protected)  # The deterministic walnut texture must remain identical too.
        _unchanged(source_hashes)
        if not (work / 'asset.blend').is_file() or (work / 'asset.blend').stat().st_size < 10000:
            raise ContractError('BUILD_FAILED', 'Repaired script did not produce a usable asset')
        execution = read(work / 'host_repair.log.execution.json')
        if execution['returncode'] != 0:
            raise ContractError('BUILD_FAILED', 'Host repair execution did not finish successfully')
        proof = {'origin': 'gpt_authored_host_technical_repair', 'original_author_status': 'failed',
                 'original_model': provider['requested_model'], 'original_author_calls_replayed': 0,
                 'actual_author_execution': False, 'host_executed_repaired_script': True,
                 'host_generated_geometry': False, 'geometry_and_material_code_unchanged': True,
                 'source_script_sha256': file_hash(author / 'object.py'), 'script_sha256': file_hash(work / 'object.py'),
                 'patch_sha256': file_hash(patch_path), 'source_response_sha256': file_hash(author / 'response.json'),
                 'authored_blend_sha256': file_hash(work / 'asset.blend'), 'protected_inputs': protected,
                 'source_hashes': source_hashes, 'execution': execution,
                 'description': 'Host inserted only missing world initialization and executed the GPT-authored failed script copy.'}
        write(recovery_root / 'repair_proof.json', proof)
        status('validating', '正在独立检查修复后的 GLB、边界和原始相机渲染。', stage='validating_geometry')
        output = recovery_root / 'validated'
        result = await asyncio.to_thread(backend._validate_authored, work, output, contract, proof, context, remaining(300))
        if result['local_bounds_m'] != target['local_bounds_m']:
            raise ContractError('FORBIDDEN_SCOPE', 'Recovery changed locked local bounds')
        images = backend._montages(work, output, contract)
        await check_base()
        _unchanged(source_hashes)
        critic_dir = recovery_root / 'critic'
        if (critic_dir / 'invocation_started.json').exists():
            raise ContractError('IDEMPOTENCY_CONFLICT', 'The remaining critic call has already started')
        total_calls = 4
        status('validating', '正在进行最后一次独立视觉检查；精修模型调用总数为 4。', stage='independent_review')
        critic = await asyncio.to_thread(backend._invoke, critic_prompt(contract), images, critic_dir, context,
                                        timeout=remaining(180), schema=CRITIC_SCHEMA, sandbox='read-only', max_tool_calls=0)
        write(critic_dir / 'critic.json', critic)
        require_critic_accept(critic)
        critic_request = read(critic_dir / 'request.json')
        if (critic_request.get('status') != 'complete' or critic_request.get('requested_model') != provider['requested_model']
                or critic_request.get('sandbox') != 'read-only' or critic_request.get('max_tool_calls') != 0):
            raise ContractError('BUILD_FAILED', 'Independent critic provider record is incomplete or outside its allowed scope')
        _unchanged(protected)
        _unchanged(source_hashes)
        await check_base()
        if _original_execution(host, confirmation) != original:
            raise ContractError('STALE_ARTIFACT', 'Original failed command/job was modified')
        artifact = backend._artifact(output / 'asset.glb', scene_id, 'object_geometry', 'model/gltf-binary')
        update = {'instance_id': target_id, 'asset': {'asset_version': 'gpt-direct-recovery-' + artifact['sha256'][:24],
                  'artifact': artifact, 'native_frame': 'asset_y_up', 'native_to_object': np.eye(4).tolist()},
                  'local_bounds_m': copy.deepcopy(target['local_bounds_m'])}
        draft = {key: copy.deepcopy(value) for key, value in frozen_scene.items() if key not in {'revision', 'parent_revision'}}
        next(obj for obj in draft['objects'] if obj['instance_id'] == target_id)['asset'] = update['asset']
        manifest = dict(status_base, status='independently_accepted', candidate=candidate, critic=critic,
                        total_fine_calls=4, new_author_calls=0, new_critic_calls=1, raw_source_hashes=source_hashes,
                        original_execution_records_hash=canonical_hash(original), repaired_asset_sha256=artifact['sha256'])
        write(recovery_root / 'accepted_recovery.json', manifest)
        status('validating', '独立检查已接受，正在构建并验证新的单对象候选。', stage='building_candidate')
        backend.timeout_s = remaining(300)
        build = await backend.build({'candidate': candidate, 'scene': draft}, context)
        build['asset_updates'] = [update]
        build['artifact_refs'].append(artifact)
        refs = [(work / 'object.py', 'refinement_source', 'text/x-python'),
                (work / 'asset.blend', 'host_repaired_native_object', 'application/x-blender'),
                (work / 'contract.json', 'author_contract', 'application/json'),
                (work / 'host_repair.log', 'host_repair_log', 'text/plain'),
                (work / 'host_repair.log.execution.json', 'host_repair_execution', 'application/json'),
                (recovery_root / 'repair_proof.json', 'host_repair_proof', 'application/json'),
                (patch_path, 'host_repair_patch', 'text/plain'),
                (recovery_root / 'accepted_recovery.json', 'refinement_recovery', 'application/json'),
                (recovery_root / 'original_execution_records.json', 'original_failed_execution', 'application/json'),
                (recovery_root / 'source_hashes.json', 'raw_source_hashes', 'application/json'),
                (output / 'asset.blend', 'object_blend', 'application/x-blender'),
                (output / 'result.json', 'refinement_manifest', 'application/json'),
                (output / 'reopen.json', 'asset_reopen', 'application/json'),
                (output / 'direct_validate.py', 'independent_export_source', 'text/x-python'),
                (output / 'validation.log.execution.json', 'independent_export_execution', 'application/json'),
                (output / 'turntable.jpg', 'object_turntable', 'image/jpeg'),
                (critic_dir / 'critic.json', 'object_visual_review', 'application/json'),
                (critic_dir / 'request.json', 'recovery_critic_request', 'application/json'),
                (critic_dir / 'response.json', 'recovery_critic_response', 'application/json'),
                (critic_dir / 'events.jsonl', 'recovery_critic_events', 'application/json'),
                (critic_dir / 'invocation_outcome.json', 'recovery_critic_outcome', 'application/json'),
                (critic_dir / 'invocation_started.json', 'recovery_critic_invocation', 'application/json')]
        refs += [(source_copy / Path(name).relative_to(source_run), 'original_failed_run_source',
                  'image/png' if name.endswith('.png') else 'text/x-python' if name.endswith('.py') else 'application/json')
                 for name in source_hashes]
        refs += [(output / f'comparison_{i:02d}.jpg', 'object_comparison', 'image/jpeg') for i in range(len(contract['frames']))]
        build['artifact_refs'].extend(backend._artifact(path, scene_id, role, mime) for path, role, mime in refs)
        validate(build, 'BuildOutput')
        remaining()
        await check_base()
        built = await host.repo.attach_build(candidate['candidate_id'], candidate['candidate_hash'], build, context)
        persisted = await host.repo.read_candidate(built['candidate_id'], built['candidate_hash'], context)
        report = await host.validator.validate(persisted, evidence, 'tabletop_design_v1', context)
        if not report['technical_pass'] or not report['committable']:
            raise ContractError('CONSTRAINT_VIOLATION', 'Recovery candidate failed repository validation', {'validation_id': report['validation_id']})
        require_critic_accept(critic)
        _unchanged(source_hashes)
        if _original_execution(host, confirmation) != original:
            raise ContractError('STALE_ARTIFACT', 'Original failed command/job changed before recovery commit')
        remaining()
        await check_base()
        commit_request = request('commit_candidate', {'candidate_id': built['candidate_id'], 'candidate_hash': built['candidate_hash'],
                                 'validation_id': report['validation_id'], 'decision_ref': decision}, 'commit', True)
        write(recovery_root / 'commit_request.json', commit_request)
        committed = await host.repo.commit(commit_request, context)
        committed_head = {'revision': committed['revision'], 'head_seq': committed['head_seq']}
        final = await host.repo.inspect(request('inspect_scene', {'revision': 'head'}, 'final'), context)
        if final['head'] != committed_head:
            raise ContractError('STALE_REVISION', 'Scene changed immediately after recovery commit')
        before = {obj['instance_id']: obj for obj in frozen_scene['objects']}
        for obj in final['snapshot']['objects']:
            old = before[obj['instance_id']]
            if obj['instance_id'] != target_id and obj != old:
                raise ContractError('FORBIDDEN_SCOPE', 'Recovery unexpectedly changed a non-target object')
            if obj['instance_id'] == target_id:
                if any(obj[key] != old[key] for key in ('transform', 'support_id', 'local_bounds_m')) or obj['asset'] != update['asset']:
                    raise ContractError('FORBIDDEN_SCOPE', 'Recovery committed an unexpected target update')
        completion = dict(status_base, status='complete', head=committed_head, candidate=built,
                          validation_id=report['validation_id'], artifact_refs=build['artifact_refs'],
                          raw_source_hashes=source_hashes, total_fine_calls=4, critic=critic,
                          new_author_calls=0, new_critic_calls=1, original_command_state='failed', original_job_state='failed',
                          repaired_asset_sha256=artifact['sha256'], completed_at=utc_now())
        _atomic_json(recovery_root / 'completion.json', completion)
        _atomic_json(run_root / 'recovery_completion.json', completion)
        status('complete', '已通过独立检查并提交精修对象；原始失败记录已保留。', stage='committed', head=committed_head)
        await host.sync_outbox(scene_id)
        return completion
    except BaseException as error:
        details = error.as_dict() if isinstance(error, ContractError) else {'code': type(error).__name__, 'message': str(error)}
        status('failed', '恢复未完成；保留执行记录，不自动重放模型调用。', error=details,
               committed_head=committed_head, automatic_retry=False)
        raise
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--timeout-s', type=int, default=600)
    args = parser.parse_args()
    try:
        result = asyncio.run(recover(args.run_root, args.source_run, args.blender, args.timeout_s))
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error), 'automatic_retry': False}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    print(json.dumps({'status': result['status'], 'head': result['head'], 'total_fine_calls': result['total_fine_calls']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
