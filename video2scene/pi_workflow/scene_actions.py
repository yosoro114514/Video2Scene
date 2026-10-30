"""Pi-directed scene tools; human grants and repository transactions own saves.

No planning model is invoked here. A bounded object modeller is the sole model
worker, and requires a verified browser acceptance of the exact current scene.
"""
from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
import inspect as inspect_module
from pathlib import Path
import re
import threading
from types import SimpleNamespace

from PIL import Image, ImageDraw

from video2scene.api_v1.__main__ import create_host
from video2scene.asset_backend.backend import DEFAULT_BLENDER
from video2scene.config import blender_path, load_settings
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, utc_now, validate
from .process_owner import owner_record
from video2scene.agent_scene import whitebox
from video2scene.agent_scene.planning import PRIOR_SCHEMA, read, save, validate_prior
from video2scene.agent_scene.run import LAYOUT_SCHEMA, validate_layout
from video2scene.agent_scene.evidence import MaskEvidenceService
from video2scene.agent_scene.direct_asset import DirectAssetBackend
from .parallel_refinement import ParallelRefinementMixin
from video2scene.agent_scene.batch_refinement import (
    immutable_json, locked_layout, audit_transition, check_accepted_frames, collect_run_proof,
)


IDENTITY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$')
ASSET_FIELDS = frozenset({'asset.geometry', 'asset.materials'})
PATCH_FIELDS = frozenset({'category', 'transform.translation_m', 'transform.rotation_xyzw', 'transform.scale', 'support_id'})
ROOM_KINDS = frozenset({'floor', 'wall', 'opening', 'ceiling'})


def _object(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required,
            'additionalProperties': False}


S = {'type': 'string'}
TOKEN = {'anyOf': [{'type': 'string'}, {'type': 'object'}]}
CANDIDATE = {'candidate_id': S, 'candidate_hash': S}
TOOL_SCHEMAS = {
    'record_prior': {'description': 'Save the structured prior after the Pi model actually inspected keyframes; no model call is hidden here.',
                     'parameters': _object({'prior': PRIOR_SCHEMA, 'observation_receipt': TOKEN})},
    'measure': {'description': 'Measure SAM masks against the selected ViPE cameras/depth and return a provisional top-view preview.',
                'parameters': _object({'sam_root': S})},
    'build_whitebox': {'description': 'Validate the Pi-authored complete layout, export a whitebox and import a NEW scene. Never overwrites a saved scene.',
                       'parameters': _object({'layout': LAYOUT_SCHEMA, 'sam_root': S})},
    'inspect_scene': {'description': 'Read the current or named scene revision.',
                      'parameters': _object({'revision': S}, [])},
    'apply_scene_patch': {'description': 'Build a candidate for explicitly user-authorized object/field edits. A verified browser event is required; this does not save.',
                          'parameters': _object({'operation_id': S, 'human_event': TOKEN, 'edits': {'type': 'array', 'minItems': 1,
                              'items': _object({'instance_id': S, 'field': {'enum': sorted(PATCH_FIELDS)}, 'value': {}})}})},
    'refine_object': {'description': 'Run a bounded Codex object author and independent critic, preserving the actual user-accepted layout. Returns a candidate, not a saved head.',
                      'parameters': _object({'operation_id': S, 'target_id': S, 'acceptance_id': S})},
    'refine_objects_parallel': {'description': 'Run up to two independent Codex author/critic workers concurrently. Returns immutable reviewed object packages and per-object errors; does not save or change the scene. Use only the host-assigned batch targets.',
        'parameters': _object({'acceptance_id': S, 'objects': {'type': 'array', 'minItems': 1, 'maxItems': 2,
            'items': _object({'target_id': S, 'operation_id': S})}})},
    'stage_refinement': {'description': 'Integrate one reviewed object package into the latest authorized scene. Creates a fresh candidate and build without another author call. Then validate_candidate and commit_candidate before staging the next package.',
        'parameters': _object({'package_id': S, 'package_hash': S})},
    'save_refinement': {'description': 'Merge one independently reviewed package onto the latest head, validate it, and commit serially in ONE host transaction. Reuses exact persisted checkpoints after a lost response. No author call; all original validation and authorization gates apply.',
        'parameters': _object({'package_id': S, 'package_hash': S})},
    'validate_candidate': {'description': 'Run technical validation on a candidate made by these tools and its frozen evidence.',
                           'parameters': _object(CANDIDATE)},
    'commit_candidate': {'description': 'Save a validated candidate using its original verified human grant and base version.',
                         'parameters': _object({**CANDIDATE, 'validation_id': S})},
    'export_scene': {'description': 'Export the exact current scene as Blender and GLB without changing the head.',
                     'parameters': _object({})},
}
ACTION_NAMES = tuple(TOOL_SCHEMAS)


def _identity(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ContractError('INVALID_ARGUMENT', 'Invalid project, operation or acceptance identity')
    return value


async def _call(function, argument):
    result = function(argument)
    return await result if inspect_module.isawaitable(result) else result


class SceneActions(ParallelRefinementMixin):
    def __init__(self, project_id, run_root, runtime_root, *, blender=None,
                 model=None, effort=None,
                 human_event_verifier=None, prior_observation_verifier=None,
                 host=None, backend_factory=DirectAssetBackend):
        self.scene_id = _identity(project_id)
        if self.scene_id == 'redwood-agent':
            raise ContractError('FORBIDDEN_SCOPE', 'The historical Redwood scene is outside Pi project scope')
        self.run = Path(run_root).resolve()
        self.runtime = Path(runtime_root).resolve()
        self.root = self.run / 'scene_actions'
        self.root.mkdir(parents=True, exist_ok=True)
        self.blender = blender_path(blender)
        settings = load_settings()
        self.model, self.effort = model or settings.model, effort or settings.effort
        self.host = host or create_host(self.runtime, blender=self.blender)
        self.human_event_verifier = human_event_verifier
        self.prior_observation_verifier = prior_observation_verifier
        self.backend_factory = backend_factory
        self._state_mutex = threading.RLock()
        self.preparation_gate=threading.RLock()
        self._worker_cancels = {}
        evidence = MaskEvidenceService(self.runtime, self.host.repo)
        self.host.evidence = evidence
        self.host.validator.evidence_service = evidence
        self.host.repo.evidence_getter = evidence.get
        if self.host.jobs:
            self.host.jobs.evidence_service = evidence
        self.host.dispatcher.handlers['select_evidence'] = evidence.select

    async def dispatch(self, name, arguments):
        import jsonschema
        if name not in TOOL_SCHEMAS:
            raise ContractError('FORBIDDEN_SCOPE', 'Unknown Pi scene tool; human acceptance is host-only')
        try:
            jsonschema.validate(arguments, TOOL_SCHEMAS[name]['parameters'])
            canonical_hash(arguments)
        except (ValueError, TypeError, jsonschema.ValidationError) as exc:
            raise ContractError('INVALID_ARGUMENT', 'Invalid scene tool arguments') from exc
        return await getattr(self, name)(**arguments)

    def _state(self, phase, **details):
        with self._state_mutex:
            path = self.root / 'state.json'
            old = read(path) if path.exists() else {}
            value = {**old, 'scene_id': self.scene_id, 'phase': phase, 'updated_at': utc_now(), **details}
            save(path, value)
            return value

    @staticmethod
    def _collect_model_proof(*args):
        return collect_run_proof(*args)

    @contextmanager
    def _lock(self, operation):
        path = self.root / 'active.lock'
        try:
            immutable_json(path, {'operation': operation, 'started_at': utc_now(),**owner_record()})
        except FileExistsError as exc:
            raise ContractError('RESOURCE_EXHAUSTED', 'A scene operation is active or interrupted; reconcile before resuming') from exc
        try:
            yield
        except BaseException as exc:
            self._state('failed', active_operation=operation, error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            path.unlink(missing_ok=True)

    def _path(self, value):
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (self.run / path).resolve()
        if not path.is_relative_to(self.run):
            raise ContractError('FORBIDDEN_SCOPE', 'Tool artifacts must belong to this project')
        return path

    def _request(self, tool, params, request_id, head=None):
        value = {'contract_version': CONTRACT_VERSION, 'scene_id': self.scene_id,
                 'tool': tool, 'request_id': request_id, 'params': params}
        if head is not None:
            value.update(base_revision=head['revision'], base_head_seq=head['head_seq'])
        return validate(value, 'ToolRequest')

    async def inspect_scene(self, revision='head'):
        return await self.host.repo.inspect(self._request('inspect_scene', {'revision': revision}, 'pi_inspect'), self.host.context())

    inspect = inspect_scene

    async def _new_scene(self):
        try:
            await self.inspect_scene()
        except ContractError as exc:
            if exc.code == 'NOT_FOUND':
                return
            raise
        raise ContractError('IDEMPOTENCY_CONFLICT', 'Scene already exists; use scoped candidate edits rather than reimport')

    async def record_prior(self, prior, observation_receipt):
        with self._lock('record_prior'):
            await self._new_scene()
            if self.prior_observation_verifier is None:
                raise ContractError('FORBIDDEN_SCOPE', 'A host-verified Pi image observation receipt is required')
            receipt = await _call(self.prior_observation_verifier, observation_receipt)
            if (receipt.get('scene_id') != self.scene_id or receipt.get('provider') != 'openai-codex'
                    or receipt.get('images_delivered') is not True or not receipt.get('run_id')
                    or not receipt.get('turn_id') or not receipt.get('model')):
                raise ContractError('EVIDENCE_MISMATCH', 'Pi image observation receipt is not bound to this project and turn')
            rows = receipt.get('frames', [])
            if not rows or len({r['source_frame_index'] for r in rows}) != len(rows):
                raise ContractError('FRAME_MISMATCH', 'Observation frames must be distinct and nonempty')
            for row in rows:
                if file_hash(self._path(row['path'])) != row['sha256']:
                    raise ContractError('STALE_ARTIFACT', 'Observed keyframe changed')
            validate_prior(prior, frame_indices=[r['source_frame_index'] for r in rows])
            binding = {'prior': prior, 'observation': receipt}
            identity = canonical_hash(binding)
            path = self.root / 'priors' / (identity + '.json')
            if not path.exists():
                immutable_json(path, binding)
            save(self.run / 'prior.json', prior)
            self._state('prior_recorded', prior_record={'path': str(path), 'sha256': file_hash(path)}, active_operation=None)
            return {'prior_sha256': file_hash(self.run / 'prior.json'), 'prior_path': str(self.run / 'prior.json'),
                    'observation': receipt, 'objects': len(prior['objects']), 'tool_plan': prior['tool_plan']}

    def _prior(self):
        state = read(self.root / 'state.json')
        record = state['prior_record']
        path = self._path(record['path'])
        if file_hash(path) != record['sha256']:
            raise ContractError('STALE_ARTIFACT', 'Prior observation binding changed')
        original = read(path)
        prior = read(self.run / 'prior.json')
        if prior != original['prior']:
            raise ContractError('STALE_ARTIFACT', 'Saved prior differs from the Pi-submitted prior')
        for row in original['observation']['frames']:
            if file_hash(self._path(row['path'])) != row['sha256']:
                raise ContractError('STALE_ARTIFACT', 'Original model-observed frame changed')
        return prior

    def _measurements(self):
        path = self.run / 'geometry_proposals.json'
        state = read(self.root / 'state.json')
        if state.get('measurement_sha256') != file_hash(path):
            raise ContractError('STALE_ARTIFACT', 'Measured geometry changed after the tool completed')
        if state.get('measurement_vipe_selection_hash') != self._vipe_selection_hash():
            raise ContractError('STALE_ARTIFACT', 'Selected ViPE output changed; measure again before building')
        return read(path)

    def _vipe_selection_hash(self):
        pointer = self.run / 'tools/vipe/latest.json'
        return canonical_hash(read(pointer)) if pointer.exists() else None

    async def _seed_snapshot(self):
        path = self.run / 'snapshot.json'
        provenance = read(self.run / 'snapshot_provenance.json')
        if file_hash(path) != provenance['snapshot_sha256']:
            raise ContractError('STALE_ARTIFACT', 'Original observed seed snapshot changed')
        snapshot = read(path)
        historical = await self.inspect_scene(snapshot['revision'])
        if historical['snapshot'] != snapshot:
            raise ContractError('STALE_ARTIFACT', 'Observation transform differs from the repository seed revision')
        return snapshot

    async def measure(self, sam_root):
        with self._lock('measure'):
            await self._new_scene()
            self._prior()
            sam = self._path(sam_root)
            if read(sam / 'result.json').get('status') != 'complete':
                raise ContractError('BUILD_FAILED', 'SAM operation is not complete')
            selection_hash = self._vipe_selection_hash()
            previous = whitebox.archive_geometry_proposals(self.run)
            self._state('measuring', active_operation='measure')
            result = await asyncio.to_thread(whitebox.propose, self.run, sam)
            self._prior()
            preview = self.root / 'measurement_preview.png'
            self._measurement_preview(result, preview)
            if self._vipe_selection_hash() != selection_hash:
                raise ContractError('STALE_ARTIFACT', 'Selected ViPE output changed during measurement; measure again')
            archived = whitebox.archive_geometry_proposals(self.run)
            self._state('measured', active_operation=None, sam_root=str(sam),
                        measurement_sha256=file_hash(self.run / 'geometry_proposals.json'), measurement_record=archived,
                        measurement_vipe_selection_hash=selection_hash)
            return {'measurements': result, 'measurements_path': str(self.run / 'geometry_proposals.json'),
                    'preview': str(preview), 'preview_sha256': file_hash(preview), 'status': result['status'],
                    'previous_measurement': previous, 'measurement_record': archived}

    @staticmethod
    def _measurement_preview(result, path):
        rows = [r for r in result.get('objects', []) if r.get('status') == 'observed_proposal']
        image = Image.new('RGB', (900, 700), '#eef2ee')
        draw = ImageDraw.Draw(image)
        draw.text((20, 15), 'MEASURED X/Z PROPOSALS - approximate, before Pi layout and human correction', fill='#20352d')
        if rows:
            import numpy as np
            centers = np.array([[r['center_m'][0], r['center_m'][2]] for r in rows])
            low, high = centers.min(axis=0) - 1, centers.max(axis=0) + 1
            scale = min(800 / (high[0] - low[0]), 580 / (high[1] - low[1]))
            for row, center in zip(rows, centers):
                x, y = 50 + (center[0] - low[0]) * scale, 70 + (center[1] - low[1]) * scale
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill='#187566')
                draw.text((x + 9, y - 8), row['object_id'], fill='#20352d')
        image.save(path)

    async def _export_snapshot(self, snapshot, label):
        """Read-only native/GLB export with a snapshot fingerprint, not a repo candidate."""
        fingerprint = canonical_hash(snapshot)
        payload = {'scene': self.host.backend._scene_spec(snapshot), 'input_fingerprint': fingerprint,
                   'candidate': {'candidate_id': label, 'candidate_hash': fingerprint,
                                 'scene_id': self.scene_id, 'base_revision': snapshot['revision'], 'base_head_seq': 0}}
        folder, result = await self.host.backend._run('build', self.scene_id, payload, self.host.context())
        refs = self.host.backend._build_refs(folder, self.scene_id)
        # Pi invokes native scene tools directly, bypassing the HTTP dispatcher's
        # normal response publication. Downloads resolve only published refs.
        self.host.store.register_artifacts(refs)
        return {'snapshot_hash': fingerprint, 'artifact_refs': refs, 'manifest': result,
                'read_only_snapshot_export': True, 'path': str(folder)}

    async def build_whitebox(self, layout, sam_root):
        with self._lock('build_whitebox'):
            await self._new_scene()
            prior = self._prior()
            proposals = self._measurements()
            sam = self._path(sam_root)
            if (proposals.get('prior_sha256') != file_hash(self.run / 'prior.json')
                    or proposals.get('sam_result_sha256') != file_hash(sam / 'result.json')):
                raise ContractError('STALE_ARTIFACT', 'Layout measurements differ from current prior or segmentation')
            validate_layout(layout, prior, proposals)
            if (self.run / 'snapshot.json').exists():
                if read(self.run / 'layout.json') != layout:
                    raise ContractError('IDEMPOTENCY_CONFLICT', 'An exported seed already binds a different layout')
                snapshot = read(self.run / 'snapshot.json')
                provenance = read(self.run / 'snapshot_provenance.json')
                if (provenance['snapshot_sha256'] != file_hash(self.run / 'snapshot.json')
                        or provenance['layout_sha256'] != file_hash(self.run / 'layout.json')):
                    raise ContractError('STALE_ARTIFACT', 'Exported seed or layout was changed before import')
                if snapshot['scene_id'] != self.scene_id:
                    raise ContractError('FORBIDDEN_SCOPE', 'Seed belongs to another project')
            else:
                save(self.run / 'layout.json', layout)
                snapshot = await asyncio.to_thread(whitebox.export_snapshot, self.run, layout, sam,
                                                  scene_id=self.scene_id, runtime_root=self.runtime)
            validate(snapshot, 'SceneSnapshot')
            self.host.repo._semantics(snapshot)
            for obj in snapshot['objects']:
                self.host.resolver.resolve(obj['asset']['artifact'], scene_id=self.scene_id)
            self._state('building_whitebox', active_operation='build_whitebox')
            exported = await self._export_snapshot(snapshot, 'whitebox_seed_export')
            save(self.root / 'whitebox_export.json', exported)
            self._prior()
            imported = await self.host.repo.import_scene(snapshot, self.host.context(), source_hash=canonical_hash(layout))
            self._state('needs_user', current_head=imported['head'], active_operation=None,
                        whitebox_export=str(self.root / 'whitebox_export.json'), accepted=None)
            return {**imported, 'export': exported, 'recommended_refinement_target': layout['recommended_refinement_target'],
                    'uncertainties': layout['uncertainties'], 'human_acceptance_required': True}

    async def _human(self, token, kind, *, require_current=True):
        if self.human_event_verifier is None:
            raise ContractError('FORBIDDEN_SCOPE', 'Verified browser feedback is required')
        event = await _call(self.human_event_verifier, token)
        if (event.get('scene_id') != self.scene_id or event.get('actor') != 'user'
                or event.get('source') != 'browser' or event.get('kind') != kind or not event.get('event_id')):
            raise ContractError('FORBIDDEN_SCOPE', 'Human event does not authorize this scene action')
        view = await self.inspect_scene()
        if require_current and (event.get('head') != view['head'] or event.get('snapshot_hash') != canonical_hash(view['snapshot'])):
            raise ContractError('STALE_REVISION', 'Human feedback refers to another scene version')
        return event, view

    async def _verify_refinement(self, record):
        """Recheck the canonical author/critic receipts and final asset, without inference."""
        if record.get('package'):
            return await self._verify_integrated_refinement(record)
        ref = record['candidate']
        persisted = await self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], self._context(record))
        metadata = await self.host.repo.candidate_metadata(ref['candidate_id'])
        if (metadata['operation'] != 'refine' or metadata.get('source_candidate_id') != record['source_candidate']['candidate_id']
                or metadata['artifact_refs'] != record['build_output']['artifact_refs']):
            raise ContractError('STALE_ARTIFACT', 'Refinement proof differs from the repository candidate')
        source = await self.host.repo.read_candidate(record['source_candidate']['candidate_id'],
                                                      record['source_candidate']['candidate_hash'], self._context(record))
        if source['candidate'] != record['source_candidate']:
            raise ContractError('STALE_ARTIFACT', 'Author proof is not bound to its real source candidate')
        evidence = await self.host.evidence.get(record['evidence_id'], self._context(record))
        observation = await self._seed_snapshot()
        observations = {obj['instance_id']: obj for obj in observation['objects']}
        target = record['allowed_object_ids'][0]
        actual = next(obj for obj in persisted['scene']['objects'] if obj['instance_id'] == target)
        update = record['build_output']['asset_updates']
        if (len(update) != 1 or update[0]['instance_id'] != target or update[0]['asset'] != actual['asset']
                or update[0]['local_bounds_m'] != actual['local_bounds_m']):
            raise ContractError('STALE_ARTIFACT', 'Refinement final asset differs from the bound candidate')
        proof = collect_run_proof(SimpleNamespace(resolver=self.host.resolver, model=self.model),
            record['build_output'], record['source_candidate'], evidence, record['before']['snapshot'], observations[target],
            record['human_event']['payload'].get('visual_policy', 'source_matched'))
        if proof != record['model_proof']:
            raise ContractError('STALE_ARTIFACT', 'Canonical model proof changed after candidate construction')
        for artifact in record['artifact_refs']:
            self.host.resolver.resolve(artifact, scene_id=self.scene_id)
        return proof

    async def _accepted_view(self, acceptance):
        """Allow only verified asset commits from this acceptance after its base."""
        event, current = await self._human(acceptance['human_event_token'], 'accept_whitebox', require_current=False)
        if (acceptance['acceptance_id'] != 'accept_' + canonical_hash(event)[:32]
                or event != acceptance['human_event'] or acceptance['scene_id'] != self.scene_id
                or event['head'] != acceptance['head'] or event['snapshot_hash'] != acceptance['snapshot_hash']
                or acceptance['target_ids'] != event['payload']['target_ids']
                or canonical_hash(acceptance['snapshot']) != acceptance['snapshot_hash']
                or canonical_hash(locked_layout(acceptance['snapshot'], acceptance['target_ids'])) != acceptance['layout_hash']):
            raise ContractError('STALE_ARTIFACT', 'Acceptance differs from the actual browser event and frozen layout')
        historical = await self.inspect_scene(acceptance['head']['revision'])
        if historical['snapshot'] != acceptance['snapshot']:
            raise ContractError('STALE_ARTIFACT', 'Accepted snapshot differs from its repository revision')
        expected = {'head': acceptance['head'], 'snapshot': acceptance['snapshot']}
        completed, seen = [], set()
        rows = []
        for path in (self.root / 'operations').glob('*/completion.json'):
            value = read(path)
            if value.get('acceptance_id') == acceptance['acceptance_id']:
                rows.append((path, value))
        for path, completion in sorted(rows, key=lambda row: row[1]['head']['head_seq']):
            ref = completion['candidate']
            record, context, candidate = await self._candidate_record(ref['candidate_id'], ref['candidate_hash'])
            if (completion.get('status') != 'complete' or completion.get('kind') != 'refine'
                    or record['kind'] != 'refine' or record.get('acceptance_id') != acceptance['acceptance_id']
                    or record['before'] != expected or record['human_event'] != event
                    or path.parent.name != record['operation_id'] or record['allowed_fields'] != sorted(ASSET_FIELDS)
                    or len(record['allowed_object_ids']) != 1 or candidate['candidate']['state'] != 'committed'):
                raise ContractError('STALE_ARTIFACT', 'Saved refinement does not form the authorized committed chain')
            target = record['allowed_object_ids'][0]
            if target not in acceptance['target_ids'] or target in seen:
                raise ContractError('FORBIDDEN_SCOPE', 'Committed target exceeds or repeats this acceptance')
            committed = await self.host.repo.get_request_result(self.scene_id, record['decision_ref'] + '_commit')
            if (not committed or committed != completion['commit_result'] or completion['before_head'] != expected['head']
                    or completion['head'] != {'revision': committed['revision'], 'head_seq': committed['head_seq']}
                    or committed['head_seq'] != expected['head']['head_seq'] + 1):
                raise ContractError('STALE_ARTIFACT', 'Completion lacks its exact repository commit receipt')
            report = await self.host.validator.get_report(completion['validation_id'], context)
            if (report['candidate_id'] != ref['candidate_id'] or report['candidate_hash'] != ref['candidate_hash']
                    or not report['technical_pass'] or not report['committable']):
                raise ContractError('CONSTRAINT_VIOLATION', 'Saved target has no matching passing validation')
            proof = await self._verify_refinement(record)
            if completion.get('model_proof') != proof:
                raise ContractError('STALE_ARTIFACT', 'Completion changed its canonical model receipts')
            after = (await self.inspect_scene(committed['revision']))['snapshot']
            if canonical_hash(after) != completion['snapshot_hash']:
                raise ContractError('STALE_ARTIFACT', 'Completed snapshot hash changed')
            audit_transition(expected['snapshot'], after, target)
            if canonical_hash(locked_layout(after, acceptance['target_ids'])) != acceptance['layout_hash']:
                raise ContractError('FORBIDDEN_SCOPE', 'Accepted layout changed during the refinement chain')
            expected = {'head': completion['head'], 'snapshot': after}
            seen.add(target)
            completed.append({'target_id': target, 'head': completion['head'], 'candidate': ref,
                              'validation_id': completion['validation_id'], 'model_proof': proof,
                              'completion_path': str(path), 'completion_sha256': file_hash(path)})
        if current != expected:
            raise ContractError('STALE_REVISION', 'Scene has edits outside the accepted refinement chain; obtain current human feedback')
        return event, current, completed

    async def accept_whitebox(self, payload, human_event):
        """Host-only: the browser POST is independently checked by its event store."""
        with self._lock('accept_whitebox'):
            event, view = await self._human(human_event, 'accept_whitebox')
            if payload != event.get('payload'):
                raise ContractError('FORBIDDEN_SCOPE', 'Acceptance payload differs from actual browser feedback')
            targets = payload.get('target_ids', [])
            objects = {obj['instance_id']: obj for obj in view['snapshot']['objects']}
            kinds = {'object', 'support'} | (ROOM_KINDS if payload.get('include_structures') is True else set())
            if (not targets or len(set(targets)) != len(targets)
                    or any(i not in objects or objects[i]['kind'] not in kinds for i in targets)):
                raise ContractError('INVALID_ARGUMENT', 'Accept existing targets; room structures require explicit inclusion')
            if not payload.get('user_instruction'):
                raise ContractError('INVALID_ARGUMENT', 'Acceptance must preserve the actual user instruction')
            identity = 'accept_' + canonical_hash(event)[:32]
            value = {'acceptance_id': identity, 'scene_id': self.scene_id, 'head': view['head'],
                     'snapshot_hash': canonical_hash(view['snapshot']), 'snapshot': view['snapshot'],
                     'layout_hash': canonical_hash(locked_layout(view['snapshot'], targets)), 'target_ids': targets,
                     'human_event': event, 'human_event_token': human_event, 'confirmed_at': utc_now()}
            path = self.root / 'acceptances' / (identity + '.json')
            if not path.exists():
                immutable_json(path, value)
            else:
                value = read(path)
            self._state('accepted', current_head=view['head'], accepted={k: value[k] for k in (
                'acceptance_id', 'head', 'snapshot_hash', 'target_ids')}, active_operation=None)
            return {k: value[k] for k in ('acceptance_id', 'head', 'snapshot_hash', 'target_ids', 'confirmed_at')}

    def _context(self, record):
        return self.host.context(actor='user', session_id=self.scene_id, decision_ref=record['decision_ref'],
                                 allowed_object_ids=frozenset(record['allowed_object_ids']),
                                 allowed_fields=frozenset(record['allowed_fields']))

    def _register_candidate(self, record):
        path = self.root / 'candidates' / (record['candidate']['candidate_id'] + '.json')
        immutable_json(path, record)
        return {'candidate': record['candidate'], 'evidence_id': record['evidence_id'],
                'operation_id': record['operation_id'], 'saved': False}

    async def _candidate_record(self, candidate_id, candidate_hash):
        path = self.root / 'candidates' / (_identity(candidate_id) + '.json')
        record = read(path)
        if record['candidate']['candidate_hash'] != candidate_hash:
            raise ContractError('STALE_ARTIFACT', 'Candidate hash differs from the authorized operation')
        context = self._context(record)
        persisted = await self.host.repo.read_candidate(candidate_id, candidate_hash, context)
        return record, context, persisted

    async def apply_scene_patch(self, operation_id, human_event, edits):
        operation_id = _identity(operation_id)
        with self._lock(operation_id):
            event, before = await self._human(human_event, 'scene_patch')
            grant = event['payload']
            targets, fields = {e['instance_id'] for e in edits}, {e['field'] for e in edits}
            if (not edits or not targets <= set(grant.get('allowed_object_ids', []))
                    or not fields <= set(grant.get('allowed_fields', [])) or not fields <= PATCH_FIELDS
                    or 'edits' in grant and grant['edits'] != edits):
                raise ContractError('FORBIDDEN_SCOPE', 'Patch exceeds the actual user object/field grant')
            folder = self.root / 'operations' / operation_id
            folder.mkdir(parents=True, exist_ok=False)
            decision = 'pi_' + canonical_hash([event, operation_id, edits])[:40]
            record = {'operation_id': operation_id, 'kind': 'patch', 'decision_ref': decision,
                      'allowed_object_ids': sorted(targets), 'allowed_fields': sorted(fields),
                      'human_event': event, 'human_event_token': human_event, 'before': before}
            immutable_json(folder / 'started.json', record)
            context = self._context(record)
            request = self._request('apply_scene_patch', {'intent': 'design_edit', 'edits': edits,
                                    'decision_ref': decision}, decision + '_prepare', before['head'])
            ref = await self.host.repo.prepare_patch(request, context)
            scene = await self.host.repo.read_candidate(ref['candidate_id'], ref['candidate_hash'], context)
            build = await self.host.backend.build(scene, context)
            built = await self.host.repo.attach_build(ref['candidate_id'], ref['candidate_hash'], build, context)
            evidence = await self.host.evidence.select(self._request('select_evidence', {'revision': before['head']['revision'],
                'target_ids': sorted(targets), 'purpose': 'validation', 'n': 3}, decision + '_evidence'), context)
            record.update(candidate=built, evidence_id=evidence['evidence_id'], artifact_refs=build['artifact_refs'])
            result = self._register_candidate(record)
            immutable_json(folder / 'result.json', result)
            self._state('candidate_ready', active_operation=None, current_head=before['head'], candidate=built)
            return result

    apply_patch = apply_scene_patch

    def _refinement_evidence(self, evidence, target, capture, proposals):
        if target['kind'] not in ROOM_KINDS:
            check_accepted_frames(evidence, target['instance_id'], capture, proposals)
            return
        # Doors/windows can be actual SAM instances in a new input video. Accept
        # their own spatially filtered masks, never an arbitrary neighbor mask.
        if target['kind']=='opening' and any(frame.get('mask') for frame in evidence['frames']):
            check_accepted_frames(evidence,target['instance_id'],capture,proposals)
            originals={frame['source_frame_id']:frame for frame in capture['frames']}
            for frame in evidence['frames']:
                source=originals.get(frame['source_frame_id'])
                if source is None or any(frame.get(key)!=source.get(key) for key in ('rgb','mask','c2w','K','timestamp_s')):
                    raise ContractError('EVIDENCE_MISMATCH','Opening evidence differs from its original target mask or camera')
            return
        # Unsegmented structures keep the actual unmasked RGB/camera binding.
        originals = {frame['source_frame_id']: frame for frame in capture['frames'] if frame['mask'] is None}
        if not 1 <= len(evidence['frames']) <= 3:
            raise ContractError('EVIDENCE_MISMATCH', 'Structure needs one to three original RGB observations')
        for frame in evidence['frames']:
            source = originals.get(frame['source_frame_id'])
            if source is None or any(frame.get(key) != source.get(key) for key in ('rgb', 'mask', 'c2w', 'K', 'timestamp_s')):
                raise ContractError('EVIDENCE_MISMATCH', 'Structure evidence must use original unmasked RGB and cameras')

    async def refine_object(self, target_id, acceptance_id, operation_id):
        operation_id, acceptance_id = _identity(operation_id), _identity(acceptance_id)
        with self._lock(operation_id):
            acceptance_path = self.root / 'acceptances' / (acceptance_id + '.json')
            acceptance = read(acceptance_path)
            event, before, completed = await self._accepted_view(acceptance)
            if target_id not in acceptance['target_ids'] or target_id in {row['target_id'] for row in completed}:
                raise ContractError('STALE_REVISION', 'Refinement needs the exact actually accepted whitebox')
            folder = self.root / 'operations' / operation_id
            folder.mkdir(parents=True, exist_ok=False)
            payload = event['payload']
            confirmation = {'confirmed': True, 'scene_id': self.scene_id, **before['head'], 'target_id': target_id,
                            'confirmed_at': acceptance['confirmed_at'], 'user_instruction': payload['user_instruction'],
                            'visual_policy': payload.get('visual_policy', 'source_matched'),
                            'acceptance_id': acceptance_id, 'human_event_id': event['event_id'],
                            'snapshot_hash': canonical_hash(before['snapshot']), 'authorization_source': 'browser'}
            confirmation['include_structures'] = payload.get('include_structures') is True
            confirmation_path = folder / 'confirmation.json'
            immutable_json(confirmation_path, confirmation)
            decision = 'pi_' + canonical_hash([operation_id, confirmation])[:40]
            record = {'operation_id': operation_id, 'kind': 'refine', 'decision_ref': decision,
                      'allowed_object_ids': [target_id], 'allowed_fields': sorted(ASSET_FIELDS),
                      'human_event': event, 'human_event_token': acceptance['human_event_token'],
                      'acceptance_id': acceptance_id, 'before': before}
            immutable_json(folder / 'started.json', record)
            context = self._context(record)
            observation = await self._seed_snapshot()
            observations = {obj['instance_id']: obj for obj in observation['objects']}
            capture = read(self.runtime / self.scene_id / 'captures' / (observation['capture_revision'] + '.json'))
            proposals = self._measurements()
            evidence = await self.host.evidence.select(self._request('select_evidence', {'revision': before['head']['revision'],
                'target_ids': [target_id], 'purpose': 'refine', 'n': 3}, decision + '_evidence'), context)
            self._refinement_evidence(evidence, observations[target_id], capture, proposals)
            params = {'target_id': target_id, 'intent': 'design_edit', 'allowed_fields': sorted(ASSET_FIELDS),
                      'evidence_id': evidence['evidence_id'], 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                      'budget': {'max_iterations': 2, 'max_model_calls': 4, 'timeout_s': 1200}, 'decision_ref': decision}
            ref = await self.host.repo.prepare_refinement(self._request('refine_asset', params, decision + '_prepare', before['head']), context)
            backend = self.backend_factory(self.runtime, blender_path=self.blender, confirmation_path=confirmation_path,
                observation_objects=observations, model=self.model, effort=self.effort,
                status_callback=lambda e: self._state('refining', active_operation=operation_id, target_id=target_id, backend_progress=e))
            backend.resource_budget=getattr(self,'resource_budget',None)
            timer = threading.Timer(1200, context.cancel_event.set)
            timer.daemon = True
            protected = {p: file_hash(p) for p in (acceptance_path, confirmation_path, self.run / 'snapshot.json', self.run / 'geometry_proposals.json')}
            timer.start()
            try:
                build = await backend.refine(dict(params, candidate=ref, input_fingerprint=ref['candidate_hash']), before['snapshot'], evidence, context)
                proof = collect_run_proof(backend, build, ref, evidence, before['snapshot'], observations[target_id], confirmation['visual_policy'])
                context.check_cancelled()
                if any(file_hash(path) != digest for path, digest in protected.items()):
                    raise ContractError('STALE_ARTIFACT', 'Frozen refinement input changed')
                if await self.inspect_scene() != before:
                    raise ContractError('STALE_REVISION', 'Scene changed during bounded object authoring')
                built = await self.host.repo.attach_build(ref['candidate_id'], ref['candidate_hash'], build, context)
                record.update(candidate=built, source_candidate=ref, build_output=build,
                              evidence_id=evidence['evidence_id'], artifact_refs=build['artifact_refs'], model_proof=proof)
                result = self._register_candidate(record)
                immutable_json(folder / 'result.json', {**result, 'model_proof': proof})
                self._state('candidate_ready', current_head=before['head'], candidate=built, active_operation=None)
                return {**result, 'model_proof': proof}
            except BaseException as exc:
                immutable_json(folder / 'failure.json', {'message': str(exc), 'automatic_replay': False, 'failed_at': utc_now()})
                raise
            finally:
                timer.cancel()

    async def validate_candidate(self, candidate_id, candidate_hash):
        record, context, candidate = await self._candidate_record(candidate_id, candidate_hash)
        evidence = await self.host.evidence.get(record['evidence_id'], context)
        report = await self.host.validator.validate(candidate, evidence, 'tabletop_design_v1', context)
        self._state('candidate_validated' if report['committable'] else 'candidate_rejected', validation=report)
        return report

    async def commit_candidate(self, candidate_id, candidate_hash, validation_id):
        with self.preparation_gate:
            return await self._commit_candidate(candidate_id,candidate_hash,validation_id)

    async def _commit_candidate(self, candidate_id, candidate_hash, validation_id):
        with self._lock('commit_' + _identity(candidate_id)):
            record, context, candidate = await self._candidate_record(candidate_id, candidate_hash)
            kind = 'accept_whitebox' if record['kind'] == 'refine' else 'scene_patch'
            before = record['before']
            request = self._request('commit_candidate', {'candidate_id': candidate_id, 'candidate_hash': candidate_hash,
                'validation_id': validation_id, 'decision_ref': record['decision_ref']}, record['decision_ref'] + '_commit', before['head'])
            previous = await self.host.repo.get_request_result(self.scene_id, request['request_id'])
            if previous is None:
                if record['kind'] == 'refine':
                    acceptance = read(self.root / 'acceptances' / (record['acceptance_id'] + '.json'))
                    event, current, _ = await self._accepted_view(acceptance)
                    await self._verify_refinement(record)
                else:
                    event, current = await self._human(record['human_event_token'], kind)
                if event != record['human_event'] or current != before:
                    raise ContractError('STALE_REVISION', 'Candidate no longer matches its real human feedback and base')
            else:
                # A completed database transaction may outlive the HTTP response.
                # Verify its original human event and ask the repository to replay
                # the exact request, never generate another candidate or model call.
                if self.human_event_verifier is None:
                    raise ContractError('FORBIDDEN_SCOPE', 'Original human event verifier is required for replay')
                event = await _call(self.human_event_verifier, record['human_event_token'])
                if event != record['human_event']:
                    raise ContractError('FORBIDDEN_SCOPE', 'Original human event changed before commit replay')
            if record['kind'] == 'refine' and previous is None:
                draft = dict(candidate['scene'], revision='pending', parent_revision=before['head']['revision'])
                audit_transition(before['snapshot'], draft, record['allowed_object_ids'][0])
            committed = await self.host.repo.commit(request, context)
            await self.host.sync_outbox(self.scene_id)
            committed_head = {'revision': committed['revision'], 'head_seq': committed['head_seq']}
            after = await self.inspect_scene(committed['revision'])
            completion = {'status': 'complete', 'kind': record['kind'], 'before_head': before['head'],
                          'head': committed_head, 'snapshot_hash': canonical_hash(after['snapshot']), 'commit_result': committed,
                          'candidate': record['candidate'], 'validation_id': validation_id, 'artifact_refs': record['artifact_refs'],
                          'acceptance_id': record.get('acceptance_id'),
                          'human_event_id': event['event_id'], 'model_proof': record.get('model_proof'), 'completed_at': utc_now()}
            path = self.root / 'operations' / record['operation_id'] / 'completion.json'
            if path.exists():
                old = read(path)
                if any(old[key] != value for key, value in completion.items() if key != 'completed_at'):
                    raise ContractError('STALE_ARTIFACT', 'Saved completion differs from the actual repository commit')
                completion = old
            else:
                immutable_json(path, completion)
            self._state('saved' if after['head'] == committed_head else 'scene_changed', current_head=after['head'],
                        active_operation=None, candidate=None, last_completion=completion)
            return {**completion, 'snapshot': after['snapshot']}

    async def verify_delivery(self, acceptance_id, export_record):
        """Host-only completion gate, checked against repository and provider receipts."""
        acceptance = read(self.root / 'acceptances' / (_identity(acceptance_id) + '.json'))
        _, current, completed = await self._accepted_view(acceptance)
        if not completed or {row['target_id'] for row in completed} != set(acceptance['target_ids']):
            raise ContractError('CONSTRAINT_VIOLATION', 'Delivery requires every approved target to be independently reviewed and committed')
        saved_export = read(self.root / 'export.json')
        if (export_record != saved_export or export_record.get('scene_id') != self.scene_id
                or export_record.get('head') != current['head']
                or export_record.get('snapshot_hash') != canonical_hash(current['snapshot'])):
            raise ContractError('STALE_ARTIFACT', 'Delivery export does not match the exact current accepted scene')
        roles = {ref['role'] for ref in export_record.get('artifact_refs', [])}
        if not {'scene_glb', 'scene_blend', 'reopen_report', 'build_input', 'build_manifest'} <= roles:
            raise ContractError('BUILD_FAILED', 'Delivery export lacks the native/GLB/reopen build evidence')
        resolved = {ref['role']: self.host.resolver.resolve(ref, scene_id=self.scene_id) for ref in export_record['artifact_refs']}
        reopen, manifest, inputs = (read(resolved[role]) for role in ('reopen_report', 'build_manifest', 'build_input'))
        if (not reopen.get('passed') or manifest.get('input_fingerprint') != export_record['snapshot_hash']
                or inputs.get('input_fingerprint') != export_record['snapshot_hash']):
            raise ContractError('BUILD_FAILED', 'Export reopen proof or snapshot fingerprint is invalid')
        if await self.inspect_scene() != current:
            raise ContractError('STALE_REVISION', 'Scene changed during delivery verification')
        return {'status': 'complete', 'scene_id': self.scene_id, 'head': current['head'],
                'snapshot_hash': canonical_hash(current['snapshot']), 'acceptance_id': acceptance_id,
                'refined_objects': len(completed), 'targets': completed, 'export': export_record,
                'layout_preserved': True, 'provider_call_receipts': [path for row in completed
                    for path in row['model_proof']['provider_call_receipts']]}

    async def export_scene(self):
        with self._lock('export_scene'):
            view = await self.inspect_scene()
            exported = await self._export_snapshot(view['snapshot'], 'saved_scene_export')
            if await self.inspect_scene() != view:
                raise ContractError('STALE_REVISION', 'Scene changed during export')
            value = {**exported, 'head': view['head'], 'scene_id': self.scene_id, 'exported_at': utc_now()}
            save(self.root / 'export.json', value)
            self._state('exported', current_head=view['head'], active_operation=None, export=str(self.root / 'export.json'))
            return value
