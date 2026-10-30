"""Independent Codex author/critic workers, immutable packages, serial integration.

Generation provenance stays bound to its original scene. Integration creates a
fresh candidate and real scene build; it never rewrites an old proof or token.
"""
import asyncio
import copy
from contextlib import contextmanager
import re
import threading
from types import SimpleNamespace

from video2scene.agent_scene.batch_refinement import immutable_json
from video2scene.agent_scene.planning import read, save
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash, utc_now
from .process_owner import owner_record


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', value):
        raise ContractError('INVALID_ARGUMENT', 'Invalid worker identity')
    return value


class ParallelRefinementMixin:
    @contextmanager
    def _package_lock(self, name):
        path = self.root / (identity(name) + '.lock')
        try:
            immutable_json(path, {'started_at': utc_now(),**owner_record()})
        except FileExistsError as exc:
            raise ContractError('RESOURCE_EXHAUSTED', 'Object workers are active or need reconciliation') from exc
        try:
            yield
        finally:
            path.unlink(missing_ok=True)

    def _worker_progress(self, target, event):
        with self._state_mutex:
            path = self.root / 'workers' / (identity(target) + '.json')
            state = read(path) if path.exists() else {}
            state.update(event, target_id=target, updated_at=utc_now())
            save(path, state)
        callback = getattr(self, 'worker_callback', None)
        if callback:
            callback(target, state)

    def cancel_workers(self, target=None):
        with self._state_mutex:
            for key, event in self._worker_cancels.items():
                if target is None or key == target:
                    event.set()

    async def _prepare_package(self, item, acceptance):
        """Short serial preparation only; author calls run after releasing lock."""
        target, op = identity(item['target_id']), identity(item['operation_id'])
        folder = self.root / 'packages' / op
        if (folder / 'package.json').exists():
            record = read(folder / 'package.json')
            if record['allowed_object_ids'] != [target] or record['acceptance_id'] != acceptance['acceptance_id']:
                raise ContractError('IDEMPOTENCY_CONFLICT', 'Package identity reused for another task')
            return {'reuse': await self._read_package(op, canonical_hash(record))}
        if folder.exists():
            raise ContractError('RESOURCE_EXHAUSTED', 'Existing author attempt needs reconciliation; no automatic replay')
        event, before, completed = await self._accepted_view(acceptance)
        if target not in acceptance['target_ids'] or target in {r['target_id'] for r in completed}:
            raise ContractError('FORBIDDEN_SCOPE', 'Worker must target an unfinished authorized object')
        folder.mkdir(parents=True, exist_ok=False)
        confirmation = {'confirmed': True, 'scene_id': self.scene_id, **before['head'], 'target_id': target,
            'confirmed_at': acceptance['confirmed_at'], 'user_instruction': event['payload']['user_instruction'],
            'visual_policy': event['payload'].get('visual_policy', 'source_matched'),
            'include_structures': event['payload'].get('include_structures') is True,
            'acceptance_id': acceptance['acceptance_id'], 'human_event_id': event['event_id'],
            'snapshot_hash': canonical_hash(before['snapshot']), 'authorization_source': 'browser'}
        immutable_json(folder / 'confirmation.json', confirmation)
        decision = 'pi_pkg_' + canonical_hash([op, confirmation])[:32]
        record = {'operation_id': op, 'kind': 'refine', 'decision_ref': decision,
            'allowed_object_ids': [target], 'allowed_fields': ['asset.geometry', 'asset.materials'],
            'human_event': event, 'human_event_token': acceptance['human_event_token'],
            'acceptance_id': acceptance['acceptance_id'], 'before': before, 'prepared_at': utc_now()}
        immutable_json(folder / 'started.json', record)
        context = self._context(record)
        observation = await self._seed_snapshot()
        observations = {obj['instance_id']: obj for obj in observation['objects']}
        capture = read(self.runtime / self.scene_id / 'captures' / (observation['capture_revision'] + '.json'))
        evidence = await self.host.evidence.select(self._request('select_evidence',
            {'revision': before['head']['revision'], 'target_ids': [target], 'purpose': 'refine', 'n': 3},
            decision + '_evidence'), context)
        self._refinement_evidence(evidence, observations[target], capture, self._measurements())
        params = {'target_id': target, 'intent': 'design_edit', 'allowed_fields': record['allowed_fields'],
            'evidence_id': evidence['evidence_id'], 'decision_ref': decision,
            'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
            'budget': {'max_iterations': 2, 'max_model_calls': 4, 'timeout_s': 1200}}
        ref = await self.host.repo.prepare_refinement(self._request('refine_asset', params,
            decision + '_prepare', before['head']), context)
        record.update(source_candidate=ref, evidence_id=evidence['evidence_id'])
        immutable_json(folder / 'prepared.json', record)
        return dict(record=record, folder=folder, context=context, params=params,
                    evidence=evidence, observations=observations)

    async def _generate_package(self, prepared):
        if 'reuse' in prepared:
            record = prepared['reuse']
            return self._package_result(record)
        record, folder, context = (prepared[k] for k in ('record', 'folder', 'context'))
        target = record['allowed_object_ids'][0]
        with self._state_mutex:
            self._worker_cancels[target] = context.cancel_event
        timer = threading.Timer(1200, context.cancel_event.set)
        timer.daemon = True
        try:
            backend = self.backend_factory(self.runtime, blender_path=self.blender,
                confirmation_path=folder / 'confirmation.json', observation_objects=prepared['observations'],
                model=self.model, effort=self.effort, status_callback=lambda event: self._worker_progress(target, event))
            backend.resource_budget=getattr(self,'resource_budget',None)
            backend.package_only=True
            protected = {p: file_hash(p) for p in (folder / 'confirmation.json', self.run / 'snapshot.json',
                self.run / 'geometry_proposals.json', self.root / 'acceptances' / (record['acceptance_id'] + '.json'))}
            if getattr(self, 'project_cancel_event', threading.Event()).is_set():
                context.cancel_event.set()
            context.check_cancelled()
            self._worker_progress(target, {'state': 'generating', 'operation_id': record['operation_id'], 'started_at': utc_now()})
            timer.start()
            build = await backend.refine(dict(prepared['params'], candidate=record['source_candidate'],
                input_fingerprint=record['source_candidate']['candidate_hash']), record['before']['snapshot'],
                prepared['evidence'], context)
            context.check_cancelled()
            if any(file_hash(p) != h for p, h in protected.items()):
                raise ContractError('STALE_ARTIFACT', 'Frozen author inputs changed')
            proof = self._collect_model_proof(backend, build, record['source_candidate'], prepared['evidence'],
                record['before']['snapshot'], prepared['observations'][target],
                record['human_event']['payload'].get('visual_policy', 'source_matched'))
            record.update(build_output=build, model_proof=proof, generated_at=utc_now())
            immutable_json(folder / 'package.json', record)
            self._worker_progress(target, {'state': 'package_ready', 'finished_at': utc_now()})
            return self._package_result(record)
        except Exception as error:
            failure = {'error': str(error), 'error_code':getattr(error,'code',None), 'automatic_replay': False, 'finished_at': utc_now()}
            immutable_json(folder / 'failure.json', failure)
            self._worker_progress(target, dict(failure, state='failed'))
            return {'target_id': target, 'operation_id': record['operation_id'], 'state': 'failed', **failure}
        finally:
            timer.cancel()
            with self._state_mutex:
                self._worker_cancels.pop(target, None)

    @staticmethod
    def _package_result(record):
        return {'target_id': record['allowed_object_ids'][0], 'operation_id': record['operation_id'],
            'state': 'package_ready', 'package_id': record['operation_id'], 'package_hash': canonical_hash(record),
            'acceptance_id':record['acceptance_id'],
            'base_head': record['before']['head'], 'critic_status': record['model_proof']['critic']['status']}

    async def refine_objects_parallel(self, acceptance_id, objects):
        if not 1 <= len(objects) <= 2 or len({o['target_id'] for o in objects}) != len(objects):
            raise ContractError('INVALID_ARGUMENT', 'One or two distinct object workers are required')
        acceptance = read(self.root / 'acceptances' / (identity(acceptance_id) + '.json'))
        with self._package_lock('parallel_workers'):
            prepared, failures = [], []
            with self._lock('prepare_parallel_workers'):
                for item in objects:
                    try:
                        prepared.append(await self._prepare_package(item, acceptance))
                    except Exception as error:
                        failures.append({**item, 'state': 'failed', 'error': str(error)})
            # Each backend has its own Codex sessions, directory, timeout and cancel event.
            results = await asyncio.gather(*(self._generate_package(item) for item in prepared))
            return {'parallelism': 2, 'objects': results + failures, 'saved': False}

    async def _read_package(self, package_id, package_hash):
        record = read(self.root / 'packages' / identity(package_id) / 'package.json')
        if canonical_hash(record) != package_hash or record['operation_id'] != package_id:
            raise ContractError('STALE_ARTIFACT', 'Immutable object package changed')
        context = self._context(record)
        source = record['source_candidate']
        persisted = await self.host.repo.read_candidate(source['candidate_id'], source['candidate_hash'], context)
        if persisted['candidate'] != source:
            raise ContractError('STALE_ARTIFACT', 'Object package source candidate changed')
        original = await self.inspect_scene(record['before']['head']['revision'])
        if (original['snapshot'] != record['before']['snapshot']
                or source['base_revision'] != record['before']['head']['revision']
                or source['base_head_seq'] != record['before']['head']['head_seq']):
            raise ContractError('STALE_ARTIFACT', 'Object package source snapshot changed')
        evidence = await self.host.evidence.get(record['evidence_id'], context)
        target = record['allowed_object_ids'][0]
        observation = next(obj for obj in (await self._seed_snapshot())['objects'] if obj['instance_id'] == target)
        proof = self._collect_model_proof(SimpleNamespace(resolver=self.host.resolver, model=self.model),
            record['build_output'], source, evidence, record['before']['snapshot'], observation,
            record['human_event']['payload'].get('visual_policy', 'source_matched'))
        if proof != record['model_proof']:
            raise ContractError('STALE_ARTIFACT', 'Object package model receipts changed')
        for ref in record['build_output']['artifact_refs']:
            self.host.resolver.resolve(ref, scene_id=self.scene_id)
        return record

    @staticmethod
    def _compatible_package(package, current):
        old, new = package['before']['snapshot'], current['snapshot']
        if any(old[k] != new[k] for k in ('capture_revision', 'geometry_revision')):
            raise ContractError('STALE_REVISION', 'Observation geometry changed during object generation')
        a = {o['instance_id']: o for o in old['objects']}
        b = {o['instance_id']: o for o in new['objects']}
        target, seen = package['allowed_object_ids'][0], set()
        # Support geometry affects contact: require it unchanged, along with target.
        while target in a and target not in seen:
            seen.add(target)
            if a[target] != b.get(target):
                raise ContractError('STALE_REVISION', 'Target or support changed since object generation')
            target = a[target].get('support_id')

    async def stage_refinement(self, package_id, package_hash):
        """Materialize a reviewed object on the latest authorized head, no model call."""
        with self._lock('stage_' + identity(package_id)):
            package = await self._read_package(package_id, package_hash)
            acceptance = read(self.root / 'acceptances' / (package['acceptance_id'] + '.json'))
            event, before, completed = await self._accepted_view(acceptance)
            target = package['allowed_object_ids'][0]
            if event != package['human_event'] or target in {r['target_id'] for r in completed}:
                raise ContractError('STALE_REVISION', 'Package is already saved or its grant changed')
            self._compatible_package(package, before)
            op = 'merge_' + canonical_hash([package_id, package_hash, before['head']])[:36]
            folder = self.root / 'operations' / op
            if (folder / 'result.json').exists():
                return read(folder / 'result.json')
            folder.mkdir(parents=True, exist_ok=True)
            decision = 'pi_' + op
            record = {k: copy.deepcopy(package[k]) for k in ('kind', 'allowed_object_ids', 'allowed_fields',
                'human_event', 'human_event_token', 'acceptance_id', 'model_proof')}
            record.update(operation_id=op, decision_ref=decision, before=before,
                          package={'package_id': package_id, 'package_hash': package_hash})
            if not (folder / 'started.json').exists():
                immutable_json(folder / 'started.json', record)
            elif read(folder / 'started.json') != record:
                raise ContractError('STALE_ARTIFACT', 'Integration input changed')
            context = self._context(record)
            evidence = await self.host.evidence.select(self._request('select_evidence',
                {'revision': before['head']['revision'], 'target_ids': [target], 'purpose': 'refine', 'n': 3},
                decision + '_evidence'), context)
            original_evidence = await self.host.evidence.get(package['evidence_id'], self._context(package))
            if evidence['frames'] != original_evidence['frames']:
                raise ContractError('EVIDENCE_MISMATCH', 'Integration must use the author-reviewed source frames')
            params = {'target_id': target, 'intent': 'design_edit', 'allowed_fields': record['allowed_fields'],
                'evidence_id': evidence['evidence_id'], 'decision_ref': decision,
                'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                'budget': {'max_iterations': 2, 'max_model_calls': 4, 'timeout_s': 1200}}
            ref = await self.host.repo.prepare_refinement(self._request('refine_asset', params,
                decision + '_prepare', before['head']), context)
            staged = await self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], context)
            update = package['build_output']['asset_updates']
            if len(update) != 1 or update[0]['instance_id'] != target:
                raise ContractError('FORBIDDEN_SCOPE', 'Package must replace exactly its authorized object')
            for obj in staged['scene']['objects']:
                if obj['instance_id'] == target:
                    obj['asset'] = copy.deepcopy(update[0]['asset'])
            self._worker_progress(target, {'state': 'staging'})
            build = await self.host.backend.build(staged, context)
            build['asset_updates'] = copy.deepcopy(update)
            # Object receipts remain original. Scene manifests are freshly built above.
            scene_roles = {r['role'] for r in build['artifact_refs']}
            build['artifact_refs'].extend(r for r in package['build_output']['artifact_refs'] if r['role'] not in scene_roles)
            built = await self.host.repo.attach_build(ref['candidate_id'], ref['candidate_hash'], build, context)
            record.update(candidate=built, source_candidate=ref, build_output=build,
                          evidence_id=evidence['evidence_id'], artifact_refs=build['artifact_refs'])
            result = self._register_candidate(record)
            immutable_json(folder / 'result.json', result)
            self._worker_progress(target, {'state': 'candidate_ready'})
            return result

    async def save_refinement(self, package_id, package_hash):
        """One Pi call, one scene lane, all original merge/validation/commit gates.

        Persist checkpoints before each next step. If the HTTP response is lost,
        the same package resumes the exact candidate/commit, never authors again.
        """
        identity(package_id)
        folder=self.root/'integrations'/package_id;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'transaction.json'
        with self._package_lock('integration_lane'):
            record=read(path) if path.exists() else {'package_id':package_id,'package_hash':package_hash,'started_at':utc_now()}
            if record['package_hash']!=package_hash:raise ContractError('IDEMPOTENCY_CONFLICT','Integration package hash changed')
            if record.get('candidate'):
                ref={k:record['candidate'][k] for k in ('candidate_id','candidate_hash')}
                previous,context,candidate=await self._candidate_record(**ref)
                if previous.get('package')!={'package_id':package_id,'package_hash':package_hash} or previous['operation_id']!=record.get('operation_id'):
                    raise ContractError('STALE_ARTIFACT','Integration checkpoint belongs to another package')
                committed=await self.host.repo.get_request_result(self.scene_id,previous['decision_ref']+'_commit')
                current=await self.inspect_scene()
                if not committed and previous['before']['head']!=current['head']:
                    package=await self._read_package(package_id,package_hash)
                    self._compatible_package(package,current)
                    save(folder/('previous_'+canonical_hash(record)+'.json'),record)
                    record={'package_id':package_id,'package_hash':package_hash,'started_at':utc_now()}
            if not record.get('candidate'):
                built=await self.stage_refinement(package_id,package_hash)
                record.update(candidate=built['candidate'],operation_id=built['operation_id']);save(path,record)
            ref={k:record['candidate'][k] for k in ('candidate_id','candidate_hash')}
            if not record.get('validation_id'):
                report=await self.validate_candidate(**ref)
                if not report['committable']:raise ContractError('BUILD_FAILED','Reviewed package failed scene validation',report)
                record['validation_id']=report['validation_id'];save(path,record)
            result=await self.commit_candidate(**ref,validation_id=record['validation_id'])
            record.update(head=result['head'],finished_at=utc_now());save(path,record)
            return result

    async def _verify_integrated_refinement(self, record):
        package = await self._read_package(**record['package'])
        if (record['model_proof'] != package['model_proof'] or record['human_event'] != package['human_event']
                or record['allowed_object_ids'] != package['allowed_object_ids']
                or record['build_output']['asset_updates'] != package['build_output']['asset_updates']):
            raise ContractError('STALE_ARTIFACT', 'Integrated asset differs from original reviewed package')
        self._compatible_package(package, record['before'])
        context = self._context(record)
        ref = record['candidate']
        persisted = await self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], context)
        metadata = await self.host.repo.candidate_metadata(ref['candidate_id'])
        if (metadata.get('source_candidate_id') != record['source_candidate']['candidate_id']
                or metadata['artifact_refs'] != record['build_output']['artifact_refs']
                or record['build_output']['input_fingerprint'] != record['source_candidate']['candidate_hash']):
            raise ContractError('STALE_ARTIFACT', 'Integration candidate or build binding changed')
        target = record['allowed_object_ids'][0]
        actual = next(obj for obj in persisted['scene']['objects'] if obj['instance_id'] == target)
        if actual['asset'] != package['build_output']['asset_updates'][0]['asset']:
            raise ContractError('STALE_ARTIFACT', 'Integrated asset was replaced after review')
        for ref in record['artifact_refs']:
            self.host.resolver.resolve(ref, scene_id=self.scene_id)
        return package['model_proof']
