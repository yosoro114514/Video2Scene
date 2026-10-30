"""Sequential, independently reviewed refinement of an explicitly authorized batch.

Each target gets its own frozen base, confirmation, candidate and immutable
completion. Resume only skips verified commits; an interrupted/failed started
target must be reconciled outside this runner and is never replayed.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import re
import threading

from video2scene.api_v1.__main__ import create_host
from video2scene.capture_evidence import transform_to_matrix
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, camera_hash, canonical_hash, file_hash, utc_now, validate
from .direct_asset import DirectAssetBackend, read
from .evidence import MaskEvidenceService
from .material_refinement import provider_receipt
from .recovery import FIELDS, _atomic_json, _unchanged, require_critic_accept


IDENTITY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
LOCKED_FIELDS = ('instance_id', 'kind', 'category', 'transform', 'local_bounds_m', 'support_id')


def immutable_json(path, value):
    """Exclusive creation; never overwrite a previous decision or execution record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def locked_layout(snapshot, target_ids):
    """Lock all scene data except planned assets and their two provenance fields.

    Non-target objects remain complete, including already refined assets. Capture
    and geometry revision IDs remain locked; baseline.capture_record separately
    binds the actual camera/RGB/mask record bytes.
    """
    result = {key: copy.deepcopy(value) for key, value in snapshot.items() if key not in ('revision', 'parent_revision', 'objects')}
    targets = set(target_ids)
    result['objects'] = []
    for original in sorted(snapshot['objects'], key=lambda obj: obj['instance_id']):
        obj = copy.deepcopy(original)
        if obj['instance_id'] in targets:
            obj.pop('asset')
            for field in FIELDS:
                obj['field_state'].pop(field, None)
        result['objects'].append(obj)
    return result


def audit_transition(before, after, target_id):
    old = {obj['instance_id']: obj for obj in before['objects']}
    new = {obj['instance_id']: obj for obj in after['objects']}
    if old.keys() != new.keys() or target_id not in old:
        raise ContractError('FORBIDDEN_SCOPE', 'Object set changed during single-object refinement')
    for identity, previous in old.items():
        current = new[identity]
        if identity != target_id and current != previous:
            raise ContractError('FORBIDDEN_SCOPE', 'Non-target object changed', {'instance_id': identity})
        if identity == target_id and any(current[key] != previous[key] for key in LOCKED_FIELDS):
            raise ContractError('FORBIDDEN_SCOPE', 'Target layout/category/kind/support/bounds changed')
    if locked_layout(before, [target_id]) != locked_layout(after, [target_id]):
        raise ContractError('FORBIDDEN_SCOPE', 'Scene data outside the single-object asset scope changed')
    if new[target_id]['asset'] == old[target_id]['asset']:
        raise ContractError('STALE_ARTIFACT', 'Refinement reused the previous asset')
    if after.get('parent_revision') != before['revision']:
        raise ContractError('STALE_REVISION', 'Committed revision does not descend from this frozen base')
    return {'locked_fields_unchanged': True, 'non_targets_unchanged': True}


def authorized_targets(authorization, snapshot, requested=None):
    allowed = authorization.get('target_ids', [])
    if (authorization.get('authorized') is not True or authorization.get('authorization_source') != 'current_user_chat'
            or not authorization.get('user_instruction') or not authorization.get('authorized_at')
            or not allowed or len(set(allowed)) != len(allowed)
            or any(not isinstance(value, str) or not IDENTITY.fullmatch(value) for value in allowed)):
        raise ContractError('FORBIDDEN_SCOPE', 'Missing explicit batch authorization and target identities')
    selected = list(requested) if requested else list(allowed)
    if len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
        raise ContractError('FORBIDDEN_SCOPE', 'Requested targets exceed the authorized batch')
    objects = {obj['instance_id']: obj for obj in snapshot['objects']}
    for identity in selected:
        obj = objects.get(identity)
        if obj is None or obj['kind'] not in {'object', 'support'} or not obj['asset']['asset_version'].startswith('whitebox-'):
            raise ContractError('FORBIDDEN_SCOPE', 'Batch target must be a remaining non-room whitebox object', {'target_id': identity})
    return selected


def check_accepted_frames(evidence, target_id, capture, proposals):
    accepted = next((set(row.get('accepted_frame_indices', [])) for row in proposals['objects'] if row['object_id'] == target_id), set())
    targets = capture['metadata'].get('frame_targets', {})
    indices = capture['metadata'].get('original_source_indices', {})
    if not 1 <= len(evidence['frames']) <= 3:
        raise ContractError('EVIDENCE_MISMATCH', 'Expected one to three actual accepted source frames')
    for frame in evidence['frames']:
        identity = frame['source_frame_id']
        if targets.get(identity) != target_id or indices.get(identity) not in accepted or frame['mask'] is None:
            raise ContractError('EVIDENCE_MISMATCH', 'Refinement frame is not an accepted spatial observation', {'source_frame_id': identity})


def collect_run_proof(backend, output, candidate, evidence, snapshot, observation, policy):
    """Read canonical provider directories from this backend run, never archives."""
    scene_id = snapshot['scene_id']
    refs = {ref['role']: ref for ref in output['artifact_refs']}
    resolve = lambda role: backend.resolver.resolve(refs[role], scene_id=scene_id)
    run_path = resolve('refinement_run')
    ledger = read(run_path)
    root = run_path.parent
    if (ledger.get('origin') != 'codex_image' or ledger.get('status') != 'accepted_asset'
            or ledger['candidate'] != candidate or ledger['instance_id'] not in evidence['target_ids']):
        raise ContractError('BUILD_FAILED', 'Backend did not produce an accepted model-authored run for this candidate')
    receipts = []
    for row in ledger['iterations']:
        for name in (f"author_{row['iteration']:02d}", f"critic_{row['iteration']:02d}"):
            folder = root / name
            if (folder / 'invocation_started.json').is_file():
                receipts.append(provider_receipt(folder, backend.model))
    if len(receipts) != ledger['model_calls'] or len(receipts) != ledger['provider_calls'] or not 2 <= len(receipts) <= 4:
        raise ContractError('RESOURCE_EXHAUSTED', 'Provider receipt count differs from the bounded backend ledger')
    critic = read(resolve('object_visual_review'))
    require_critic_accept(critic)
    proof, contract, result = (read(resolve(role)) for role in ('author_execution', 'author_contract', 'refinement_manifest'))
    target = next(obj for obj in snapshot['objects'] if obj['instance_id'] == ledger['instance_id'])
    if (proof.get('origin') != 'codex_image' or proof.get('actual_author_execution') is not True
            or proof.get('host_generated_geometry') is not False):
        raise ContractError('BUILD_FAILED', 'Actual successful model-authored Blender execution is missing')
    if (contract['candidate'] != candidate or contract['evidence_id'] != evidence['evidence_id']
            or contract['evidence_hash'] != evidence['evidence_hash'] or contract['accepted_transform'] != target['transform']
            or contract['local_bounds_m'] != target['local_bounds_m'] or contract['support_id'] != target['support_id']
            or contract['observation_transform'] != transform_to_matrix(observation['transform']).tolist()
            or contract.get('visual_policy') != policy or len(contract['frames']) != len(evidence['frames'])):
        raise ContractError('EVIDENCE_MISMATCH', 'Author contract changed the frozen layout, policy or observation pose')
    for frame, source in zip(contract['frames'], evidence['frames']):
        if any(frame[key] != value for key, value in source.items()):
            raise ContractError('FRAME_MISMATCH', 'Author frame/camera differs from accepted evidence')
    update = output['asset_updates']
    if len(update) != 1 or update[0]['instance_id'] != target['instance_id'] or update[0]['local_bounds_m'] != target['local_bounds_m']:
        raise ContractError('FORBIDDEN_SCOPE', 'Backend asset update exceeds the single frozen target')
    if (result['input_fingerprint'] != candidate['candidate_hash'] or result.get('render_source') != 'reopened_canonical_glb'
            or result.get('rendered_asset_sha256') != update[0]['asset']['artifact']['sha256']
            or not result.get('native_reopened') or not result.get('glb_reopened')):
        raise ContractError('STALE_ARTIFACT', 'Independent critic did not inspect the final canonical asset')
    return {'backend_run': str(root), 'critic': critic, 'receipt_records': receipts,
            'provider_call_receipts': [row['directory'] for row in receipts], 'model_calls': len(receipts)}


async def run_batch(run_root, batch_root, blender, *, targets=None, limit=None, resume=False,
                    timeout_s=1200, max_iterations=2, max_model_calls=4, model='gpt-6-astra', effort='high'):
    run_root, batch_root = Path(run_root).resolve(), Path(batch_root).resolve()
    if (limit is not None and limit < 1) or not 1 <= max_iterations <= 2 or not 2 <= max_model_calls <= 4 or not 1 <= timeout_s <= 1200:
        raise ValueError('Invalid positive limit or bounded per-object budget')
    authorization_path = batch_root / 'authorization.json'
    authorization = read(authorization_path)
    authorization_hash = file_hash(authorization_path)
    scene_id = authorization['scene_id']
    if authorization.get('batch_id') != batch_root.name:
        raise ContractError('FORBIDDEN_SCOPE', 'Authorization belongs to another batch directory')
    runtime = run_root / 'runtime'
    host = create_host(runtime)  # Repository/validator only: no job worker/model starts.
    host.evidence = MaskEvidenceService(runtime, host.repo)
    host.validator.evidence_service = host.evidence
    host.repo.evidence_getter = host.evidence.get

    async def view(revision='head'):
        return await host.repo.inspect({'contract_version': CONTRACT_VERSION, 'request_id': 'batch_inspect',
                                       'scene_id': scene_id, 'tool': 'inspect_scene', 'params': {'revision': revision}}, host.context())

    lock = batch_root / '.runner.lock'
    try:
        immutable_json(lock, {'pid': os.getpid(), 'started_at': utc_now(), 'authorization_sha256': authorization_hash})
    except FileExistsError as exc:
        raise ContractError('RESOURCE_EXHAUSTED', 'Batch runner already active or interrupted; reconcile its lock before restarting') from exc
    baseline, completed, active = None, [], None
    current = {'head': authorization['baseline_head']}
    status_lock = threading.RLock()

    def status(phase, message, stage='', **details):
        if baseline is None:
            return
        receipts = [folder for row in completed for folder in row['provider_call_receipts']]
        value = {'scene_id': scene_id, 'batch_id': batch_root.name, 'authorization_sha256': authorization_hash,
                 'baseline_head': baseline['head'], 'baseline_snapshot_hash': baseline['snapshot_hash'],
                 'baseline_layout_hash': baseline['layout_hash'], 'target_ids': baseline['target_ids'],
                 'phase': phase, 'stage': stage, 'active_target_id': active, 'message': message,
                 'progress': {'completed': len(completed), 'total': len(baseline['target_ids'])},
                 'completed': copy.deepcopy(completed), 'current_head': current['head'],
                 'provider_call_receipts': receipts, 'updated_at': utc_now(), **details}
        with status_lock:
            _atomic_json(batch_root / 'status.json', value)

    def static_inputs():
        _unchanged({str(authorization_path): authorization_hash,
                    **{row['path']: row['sha256'] for row in (baseline['capture_record'], baseline['initial_snapshot'], baseline['geometry_proposals'])}})

    def target_folder(index, identity):
        return batch_root / 'targets' / f'{index + 1:03d}_{identity}'

    try:
        current = await view()
        baseline_path = batch_root / 'baseline.json'
        if baseline_path.exists():
            if not resume:
                raise ContractError('IDEMPOTENCY_CONFLICT', 'Batch already initialized; use --resume for verified committed targets only')
            baseline = read(baseline_path)
            frozen = read(batch_root / 'baseline_snapshot.json')
            if (baseline['authorization_sha256'] != authorization_hash or baseline['scene_id'] != scene_id
                    or canonical_hash(frozen) != baseline['snapshot_hash']
                    or canonical_hash(locked_layout(frozen, baseline['target_ids'])) != baseline['layout_hash']
                    or baseline['head'] != authorization['baseline_head']
                    or baseline['snapshot_hash'] != authorization['baseline_snapshot_hash']
                    or targets is not None and list(targets) != baseline['target_ids']):
                raise ContractError('STALE_ARTIFACT', 'Batch baseline/authorization/target plan changed')
            authorized_targets(authorization, frozen, baseline['target_ids'])
        else:
            if resume:
                raise ContractError('NOT_FOUND', 'No batch baseline exists to resume')
            frozen = current['snapshot']
            if current['head'] != authorization.get('baseline_head') or canonical_hash(frozen) != authorization.get('baseline_snapshot_hash'):
                raise ContractError('STALE_REVISION', 'Current head does not match the authorized batch baseline')
            selected = authorized_targets(authorization, frozen, targets)
            paths = {'capture_record': runtime / scene_id / 'captures' / f"{frozen['capture_revision']}.json",
                     'initial_snapshot': run_root / 'snapshot.json', 'geometry_proposals': run_root / 'geometry_proposals.json'}
            baseline = {'scene_id': scene_id, 'batch_id': batch_root.name, 'head': current['head'],
                        'snapshot_hash': canonical_hash(frozen), 'layout_hash': canonical_hash(locked_layout(frozen, selected)),
                        'authorization_sha256': authorization_hash, 'target_ids': selected, 'created_at': utc_now(),
                        **{key: {'path': str(path.resolve()), 'sha256': file_hash(path)} for key, path in paths.items()}}
            immutable_json(batch_root / 'baseline_snapshot.json', frozen)
            immutable_json(baseline_path, baseline)
        static_inputs()
        observations = {obj['instance_id']: obj for obj in read(Path(baseline['initial_snapshot']['path']))['objects']}
        capture = read(Path(baseline['capture_record']['path']))
        proposals = read(Path(baseline['geometry_proposals']['path']))
        expected_snapshot, expected_head = frozen, baseline['head']
        # Verify the complete contiguous chain before skipping any completed target.
        for index, identity in enumerate(baseline['target_ids']):
            folder = target_folder(index, identity)
            if not folder.exists():
                if any(target_folder(j, value).exists() for j, value in enumerate(baseline['target_ids']) if j > index):
                    raise ContractError('STALE_ARTIFACT', 'Batch has a non-contiguous execution history')
                break
            completion_path = folder / 'completion.json'
            if not completion_path.is_file():
                active = identity
                raise ContractError('RESOURCE_EXHAUSTED', 'An existing target attempt needs reconciliation; it will not be replayed', {'target_id': identity})
            completion = read(completion_path)
            before, after = read(folder / 'before.json'), read(folder / 'after.json')
            if (completion.get('status') != 'complete' or completion['target_id'] != identity
                    or completion['authorization_sha256'] != authorization_hash or completion['before_head'] != expected_head
                    or before != expected_snapshot or canonical_hash(after) != completion['after_snapshot_hash']
                    or completion['head']['head_seq'] != expected_head['head_seq'] + 1):
                raise ContractError('STALE_ARTIFACT', 'Saved target completion does not match the batch chain')
            audit_transition(before, after, identity)
            if (await view(completion['head']['revision']))['snapshot'] != after:
                raise ContractError('STALE_ARTIFACT', 'Saved after snapshot differs from the repository revision')
            actual = next(obj['asset'] for obj in after['objects'] if obj['instance_id'] == identity)
            if actual != completion['asset']:
                raise ContractError('STALE_ARTIFACT', 'Completion asset differs from its committed snapshot')
            host.resolver.resolve(actual['artifact'], scene_id=scene_id)
            require_critic_accept(completion['critic'])
            for receipt in completion['receipt_records']:
                if provider_receipt(receipt['directory'], completion['model']) != receipt:
                    raise ContractError('STALE_ARTIFACT', 'Completed provider receipt changed')
            saved_evidence = await host.evidence.get(completion['evidence_id'], host.context())
            if camera_hash(saved_evidence['frames']) != completion['camera_fingerprint']:
                raise ContractError('FRAME_MISMATCH', 'Completed target camera evidence changed')
            check_accepted_frames(saved_evidence, identity, capture, proposals)
            completed.append({'target_id': identity, 'before_head': completion['before_head'], 'head': completion['head'],
                              'asset': actual, 'completion_path': str(completion_path), 'completion_sha256': file_hash(completion_path),
                              'provider_call_receipts': completion['provider_call_receipts']})
            expected_snapshot, expected_head = after, completion['head']
        current = await view()
        if current['head'] != expected_head or current['snapshot'] != expected_snapshot:
            raise ContractError('STALE_REVISION', 'Current scene differs from the verified batch commit chain')
        started_this_run = 0
        for index in range(len(completed), len(baseline['target_ids'])):
            if limit is not None and started_this_run >= limit:
                active = None
                status('ready', '本次启动数量已达到内部限制；可用 --resume 继续剩余对象。', 'limit_reached')
                return read(batch_root / 'status.json')
            identity = baseline['target_ids'][index]
            active = identity
            static_inputs()
            current = await view()
            if current['head'] != expected_head or current['snapshot'] != expected_snapshot:
                raise ContractError('STALE_REVISION', 'Scene changed before the next target')
            if canonical_hash(locked_layout(current['snapshot'], baseline['target_ids'])) != baseline['layout_hash']:
                raise ContractError('FORBIDDEN_SCOPE', 'Batch layout or non-target asset changed')
            folder = target_folder(index, identity)
            folder.mkdir(parents=True, exist_ok=False)
            before, before_head = copy.deepcopy(current['snapshot']), copy.deepcopy(current['head'])
            immutable_json(folder / 'before.json', before)
            confirmation = {'confirmed': True, 'scene_id': scene_id, 'revision': before_head['revision'],
                            'head_seq': before_head['head_seq'], 'target_id': identity, 'confirmed_at': utc_now(),
                            'snapshot_hash': canonical_hash(before), 'user_instruction': authorization['user_instruction'],
                            'visual_policy': authorization.get('visual_policy', 'structure_first_relaxed_appearance'),
                            'authorization_ref': {'path': str(authorization_path), 'sha256': authorization_hash},
                            'authorization_source': authorization['authorization_source'], 'browser_confirmation_click_claimed': False}
            confirmation_path = folder / 'confirmation.json'
            immutable_json(confirmation_path, confirmation)
            confirmation_hash = file_hash(confirmation_path)
            decision = 'batch_' + canonical_hash([authorization_hash, identity, before_head])[:40]
            context = host.context(actor='user', session_id=batch_root.name, decision_ref=decision,
                                   allowed_object_ids=frozenset({identity}), allowed_fields=FIELDS)
            timer = threading.Timer(timeout_s, context.cancel_event.set)
            timer.daemon = True

            def request(tool, params, suffix, mutate=False):
                value = {'contract_version': CONTRACT_VERSION, 'request_id': decision + '_' + suffix, 'scene_id': scene_id,
                         'tool': tool, 'params': params}
                if mutate:
                    value.update(base_revision=before_head['revision'], base_head_seq=before_head['head_seq'])
                return validate(value, 'ToolRequest')

            def update(event):
                stage = event.get('stage', 'refining')
                status('running', f'正在精修 {identity}：{stage}', stage, backend_progress=event)

            backend = DirectAssetBackend(runtime, blender_path=blender, confirmation_path=confirmation_path,
                                         observation_objects=observations, model=model, effort=effort, status_callback=update)
            status('running', f'正在为 {identity} 选择已接受的原始观测。', 'selecting_evidence')
            try:
                evidence = await host.evidence.select(request('select_evidence', {'revision': before_head['revision'],
                                      'target_ids': [identity], 'purpose': 'refine', 'n': 3}, 'evidence'), context)
                check_accepted_frames(evidence, identity, capture, proposals)
                immutable_json(folder / 'evidence.json', evidence)
                prepare = request('refine_asset', {'target_id': identity, 'intent': 'design_edit', 'allowed_fields': sorted(FIELDS),
                                  'evidence_id': evidence['evidence_id'], 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                                  'budget': {'max_iterations': max_iterations, 'max_model_calls': max_model_calls, 'timeout_s': timeout_s},
                                  'decision_ref': decision}, 'prepare', True)
                candidate = await host.repo.prepare_refinement(prepare, context)
                immutable_json(folder / 'prepare_request.json', prepare)
                immutable_json(folder / 'candidate.json', candidate)
                immutable_json(folder / 'started.json', {'started_at': utc_now(), 'target_id': identity, 'candidate': candidate,
                               'before_head': before_head, 'authorization_sha256': authorization_hash, 'confirmation_sha256': confirmation_hash,
                               'budget': prepare['params']['budget'], 'automatic_replay': False})
                started_this_run += 1
                timer.start()
                spec = dict(prepare['params'], candidate=candidate, input_fingerprint=candidate['candidate_hash'])
                build = await backend.refine(spec, before, evidence, context)
                proof = collect_run_proof(backend, build, candidate, evidence, before, observations[identity], confirmation['visual_policy'])
                immutable_json(folder / 'model_proof.json', proof)
                immutable_json(folder / 'build_output.json', build)
                context.check_cancelled()
                static_inputs()
                if file_hash(confirmation_path) != confirmation_hash:
                    raise ContractError('STALE_ARTIFACT', 'Target confirmation changed while authoring')
                current = await view()
                if current['head'] != before_head or current['snapshot'] != before:
                    raise ContractError('STALE_REVISION', 'Scene changed during target refinement')
                built = await host.repo.attach_build(candidate['candidate_id'], candidate['candidate_hash'], build, context)
                persisted = await host.repo.read_candidate(built['candidate_id'], built['candidate_hash'], context)
                report = await host.validator.validate(persisted, evidence, 'tabletop_design_v1', context)
                immutable_json(folder / 'validation.json', report)
                if not report['technical_pass'] or not report['committable']:
                    raise ContractError('CONSTRAINT_VIOLATION', 'Target candidate failed technical validation')
                context.check_cancelled()
                unchanged_evidence = await host.evidence.get(evidence['evidence_id'], context)
                if unchanged_evidence != evidence:
                    raise ContractError('EVIDENCE_MISMATCH', 'Target evidence changed before commit')
                static_inputs()
                status('running', f'{identity} 已通过独立复核，正在保存。', 'committing')
                commit_request = request('commit_candidate', {'candidate_id': built['candidate_id'], 'candidate_hash': built['candidate_hash'],
                                         'validation_id': report['validation_id'], 'decision_ref': decision}, 'commit', True)
                immutable_json(folder / 'commit_request.json', commit_request)
                committed = await host.repo.commit(commit_request, context)
                immutable_json(folder / 'commit_result.json', committed)
                current = await view()
                head = {'revision': committed['revision'], 'head_seq': committed['head_seq']}
                if current['head'] != head:
                    raise ContractError('STALE_REVISION', 'Scene changed immediately after target commit')
                after = current['snapshot']
                audit = audit_transition(before, after, identity)
                if canonical_hash(locked_layout(after, baseline['target_ids'])) != baseline['layout_hash']:
                    raise ContractError('FORBIDDEN_SCOPE', 'Committed target changed the batch layout')
                immutable_json(folder / 'after.json', after)
                asset = next(obj['asset'] for obj in after['objects'] if obj['instance_id'] == identity)
                if asset != build['asset_updates'][0]['asset']:
                    raise ContractError('STALE_ARTIFACT', 'Committed target differs from the reviewed asset')
                host.resolver.resolve(asset['artifact'], scene_id=scene_id)
                completion = {'status': 'complete', 'scene_id': scene_id, 'batch_id': batch_root.name, 'target_id': identity,
                              'authorization_sha256': authorization_hash, 'confirmation_sha256': confirmation_hash,
                              'baseline_snapshot_hash': baseline['snapshot_hash'], 'baseline_layout_hash': baseline['layout_hash'],
                              'before_head': before_head, 'head': head, 'before_snapshot_hash': canonical_hash(before),
                              'after_snapshot_hash': canonical_hash(after), 'candidate': built, 'validation_id': report['validation_id'],
                              'asset': asset, 'artifact_refs': build['artifact_refs'], 'evidence_id': evidence['evidence_id'],
                              'camera_fingerprint': camera_hash(evidence['frames']), 'camera_evidence_unchanged': True,
                              'visual_policy': confirmation['visual_policy'], 'model': model, 'completed_at': utc_now(), **audit, **proof}
                completion_path = folder / 'completion.json'
                immutable_json(completion_path, completion)
                completed.append({'target_id': identity, 'before_head': before_head, 'head': head, 'asset': asset,
                                  'completion_path': str(completion_path), 'completion_sha256': file_hash(completion_path),
                                  'provider_call_receipts': proof['provider_call_receipts']})
                expected_snapshot, expected_head = after, head
                active = None
                status('running', f'{identity} 已复核并保存。', 'target_completed')
                await host.sync_outbox(scene_id)
            except BaseException as error:
                failure = {'status': 'failed', 'target_id': identity, 'before_head': before_head, 'automatic_replay': False,
                           'error': error.as_dict() if isinstance(error, ContractError) else {'type': type(error).__name__, 'message': str(error)},
                           'failed_at': utc_now()}
                if not (folder / 'failure.json').exists():
                    immutable_json(folder / 'failure.json', failure)
                raise
            finally:
                timer.cancel()
        current = await view()
        static_inputs()
        if current['head'] != expected_head or current['snapshot'] != expected_snapshot:
            raise ContractError('STALE_REVISION', 'Final scene differs from the completed batch chain')
        for row in completed:
            asset = next(obj['asset'] for obj in current['snapshot']['objects'] if obj['instance_id'] == row['target_id'])
            if asset != row['asset']:
                raise ContractError('STALE_ARTIFACT', 'A completed batch asset changed')
            host.resolver.resolve(asset['artifact'], scene_id=scene_id)
        active = None
        status('complete', '授权清单中的对象均已逐个复核并保存，墙体与布局保持不变。', 'complete', layout_preserved=True)
        return read(batch_root / 'status.json')
    except BaseException as error:
        try:
            current = await view()
        except Exception:
            pass
        uncertain = isinstance(error, ContractError) and error.code in ('RESOURCE_EXHAUSTED', 'STALE_REVISION', 'STALE_ARTIFACT')
        status('needs_reconciliation' if uncertain else 'failed', '批量精修已停止；请核对当前对象记录，不会自动重放已启动调用。',
               'stopped', error=error.as_dict() if isinstance(error, ContractError) else {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--targets', nargs='+')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timeout-s', type=int, default=1200)
    parser.add_argument('--max-iterations', type=int, default=2)
    parser.add_argument('--max-model-calls', type=int, default=4)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--effort', default='high')
    args = parser.parse_args()
    try:
        result = asyncio.run(run_batch(args.run_root, args.batch_root, args.blender, targets=args.targets, limit=args.limit,
                                      resume=args.resume, timeout_s=args.timeout_s, max_iterations=args.max_iterations,
                                      max_model_calls=args.max_model_calls, model=args.model, effort=args.effort))
    except Exception as error:
        print(json.dumps({'phase': 'failed', 'error': str(error), 'automatic_replay': False}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
    print(json.dumps({'phase': result['phase'], 'current_head': result['current_head'], 'progress': result['progress']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
