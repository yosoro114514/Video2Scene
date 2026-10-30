"""Frozen evidence bundles, bound to a repository snapshot and an immutable capture."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

import cv2
import numpy as np

from video2scene.contracts_v1 import ContractError, canonical_hash, validate

from .coordinates import (
    bounds_corners,
    intrinsics_matrix,
    project,
    rigid,
    transform_points,
    transform_to_matrix,
)
from .storage import checked_record, immutable_json, resolve_artifact


def check_frame(frame, storage_root, *, camera_frame_id=None, mask_frame_id=None):
    validate(frame, 'EvidenceFrame')
    source_id = frame['source_frame_id']
    if camera_frame_id is not None and camera_frame_id != source_id:
        raise ContractError('FRAME_MISMATCH', 'Camera and RGB source frame IDs differ')
    if frame['mask'] is not None and mask_frame_id is not None and mask_frame_id != source_id:
        raise ContractError('FRAME_MISMATCH', 'Mask and RGB source frame IDs differ')
    rigid(frame['c2w'])
    intrinsics_matrix(frame['K'])
    rgb = cv2.imread(str(resolve_artifact(storage_root, frame['rgb'])), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2] != (frame['height'], frame['width']):
        raise ContractError('FRAME_MISMATCH', 'RGB dimensions do not match the delivered camera')
    if frame['mask'] is not None:
        mask = cv2.imread(str(resolve_artifact(storage_root, frame['mask'])), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape[:2] != rgb.shape[:2]:
            raise ContractError('FRAME_MISMATCH', 'Mask dimensions differ from RGB')
    return rgb


def register_capture(storage_root, scene_id, capture_revision, geometry_revision, frames, *, metadata=None):
    """Persist validated, aligned frames. Metadata is internal provenance, not a wire DTO."""
    if not frames or len({frame['source_frame_id'] for frame in frames}) != len(frames):
        raise ContractError('FRAME_MISMATCH', 'Capture requires distinct source frame IDs')
    for frame in frames:
        check_frame(frame, storage_root)
        if not frame['rgb']['uri'].startswith(f'artifact://{scene_id}/'):
            raise ContractError('FORBIDDEN_SCOPE', 'Capture RGB belongs to another scene')
    payload = {'scene_id': scene_id, 'capture_revision': capture_revision, 'geometry_revision': geometry_revision,
               'frames': frames, 'metadata': metadata or {}}
    payload['capture_hash'] = canonical_hash(payload)
    immutable_json(Path(storage_root) / scene_id / 'captures' / f'{capture_revision}.json', payload)
    return payload


class EvidenceService:
    def __init__(self, storage_root, repository):
        self.storage_root = Path(storage_root).resolve()
        self.repository = repository

    async def select(self, request, context):
        validate(request, 'ToolRequest')
        params = request['params']
        inspect_request = {key: request[key] for key in ('contract_version', 'request_id', 'scene_id')}
        inspect_request.update(tool='inspect_scene', params={'revision': params['revision']})
        result = self.repository.inspect(inspect_request, context)
        view = await result if inspect.isawaitable(result) else result
        snapshot = view['snapshot']
        objects = {obj['instance_id']: obj for obj in snapshot['objects']}
        if not set(params['target_ids']) <= objects.keys():
            raise ContractError('NOT_FOUND', 'Evidence target is absent from this revision')
        capture = self._capture(snapshot['scene_id'], snapshot['capture_revision'])
        if capture['geometry_revision'] != snapshot['geometry_revision']:
            raise ContractError('EVIDENCE_MISMATCH', 'Capture geometry revision differs from snapshot')
        ranked = []
        for frame in capture['frames']:
            rgb = check_frame(frame, self.storage_root)
            visible = []
            total_area = 0.
            for target in params['target_ids']:
                obj = objects[target]
                corners = transform_points(transform_to_matrix(obj['transform']), bounds_corners(obj['local_bounds_m']))
                pixels, depths = project(corners, frame['K'], frame['c2w'])
                pixels = pixels[(depths > 1e-6) & np.isfinite(pixels).all(axis=1)]
                if not len(pixels):
                    continue
                lo = np.maximum(pixels.min(axis=0), [0, 0])
                hi = np.minimum(pixels.max(axis=0), [frame['width'], frame['height']])
                area = float(np.prod(np.maximum(hi - lo, 0)))
                if area > 1:
                    visible.append(target)
                    total_area += area / (frame['width'] * frame['height'])
            if not visible:
                continue
            sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            ranked.append({'frame': frame, 'area': total_area, 'visible': visible, 'sharpness': sharpness,
                           'center': np.asarray(frame['c2w'])[:3, 3]})
        if not ranked:
            raise ContractError('FRAME_MISMATCH', 'No delivered RGB frame projects the requested target bounds')
        selected = []
        while ranked and len(selected) < params['n']:
            def rank(row):
                diversity = min((float(np.linalg.norm(row['center'] - other['center'])) for other in selected), default=0.)
                return (len(row['visible']), row['area'] + .05 * np.log1p(row['sharpness']) + diversity)
            chosen = max(range(len(ranked)), key=lambda index: rank(ranked[index]))
            selected.append(ranked.pop(chosen))
        frames = []
        for row in selected:
            frame = copy.deepcopy(row['frame'])
            frame['selection_reason'] = ('Projected bounds cover ' + ', '.join(row['visible']) +
                '; RGB sharpness and camera-center diversity ranking. Occlusion is unknown without observed masks.')
            frames.append(frame)
        payload = {'scene_id': snapshot['scene_id'], 'scene_revision': snapshot['revision'],
                   'capture_revision': snapshot['capture_revision'], 'geometry_revision': snapshot['geometry_revision'],
                   'target_ids': params['target_ids'], 'purpose': params['purpose'], 'frames': frames}
        payload['evidence_id'] = 'ev-' + canonical_hash(payload)[:32]
        payload['evidence_hash'] = canonical_hash(payload)
        validate(payload, 'EvidenceBundle')
        immutable_json(self.storage_root / snapshot['scene_id'] / 'evidence' / f"{payload['evidence_id']}.json", payload)
        return copy.deepcopy(payload)

    async def select_for_extension(self, request, candidate_id, candidate_hash, context, *, frame_ids=None):
        """Bind new identities to unchanged, unmasked frames of the frozen capture.

        This host-only entry requires an existing authorized extension candidate.
        Added objects are design proposals; it never invents observed masks or
        changes the registered capture. Optional frame_ids must name real original
        unmasked frames, and otherwise projected candidate bounds rank the views.
        """
        validate(request, 'ToolRequest')
        if request['tool'] != 'select_evidence':
            raise ContractError('INVALID_ARGUMENT', 'Extension evidence expects a select_evidence request')
        candidate = await self.repository.read_candidate(candidate_id, candidate_hash, context)
        metadata = await self.repository.candidate_metadata(candidate_id, context)
        ref, scene, params = candidate['candidate'], candidate['scene'], request['params']
        if metadata.get('operation') != 'extension' or metadata['decision_ref'] != context.decision_ref:
            raise ContractError('FORBIDDEN_SCOPE', 'Evidence is not bound to this authorized scene extension')
        for identity, fields in metadata['target_fields'].items():
            for field in fields:
                context.authorize(identity, field)
        if (request['scene_id'] != ref['scene_id'] or params['revision'] != ref['base_revision']
                or set(params['target_ids']) != set(metadata['allowed_object_ids'])):
            raise ContractError('EVIDENCE_MISMATCH', 'Extension evidence must cover the exact frozen target scope/base')
        query = {key: request[key] for key in ('contract_version', 'request_id', 'scene_id')}
        query.update(tool='inspect_scene', params={'revision': 'head'})
        view = await self.repository.inspect(query, context)
        if (view['head'] != {'revision': ref['base_revision'], 'head_seq': ref['base_head_seq']}
                or view['snapshot'] != metadata['base_snapshot']):
            raise ContractError('STALE_REVISION', 'Scene changed before extension evidence selection')
        capture = self._capture(ref['scene_id'], scene['capture_revision'])
        if capture['geometry_revision'] != scene['geometry_revision']:
            raise ContractError('EVIDENCE_MISMATCH', 'Extension capture geometry differs from the frozen scene')
        originals = {frame['source_frame_id']: frame for frame in capture['frames'] if frame['mask'] is None}
        objects = {obj['instance_id']: obj for obj in scene['objects']}
        if frame_ids is not None:
            if (not isinstance(frame_ids, (list, tuple)) or not 1 <= len(frame_ids) <= params['n']
                    or len(set(frame_ids)) != len(frame_ids) or not set(frame_ids) <= set(originals)):
                raise ContractError('FRAME_MISMATCH', 'Extension frames must be distinct original unmasked capture frames')
            selected = [originals[identity] for identity in frame_ids]
            reason = 'Host-selected original capture RGB/camera for authorized scene extension; additions are design proposals, not observed segmentations.'
        else:
            rows = []
            for frame in originals.values():
                check_frame(frame, self.storage_root)
                area = 0.
                for identity in params['target_ids']:
                    obj = objects[identity]
                    corners = transform_points(transform_to_matrix(obj['transform']), bounds_corners(obj['local_bounds_m']))
                    pixels, depths = project(corners, frame['K'], frame['c2w'])
                    pixels = pixels[(depths > 1e-6) & np.isfinite(pixels).all(axis=1)]
                    if len(pixels):
                        lo = np.maximum(pixels.min(axis=0), [0, 0])
                        hi = np.minimum(pixels.max(axis=0), [frame['width'], frame['height']])
                        area += float(np.prod(np.maximum(hi - lo, 0))) / (frame['width'] * frame['height'])
                if area > 0:
                    rows.append((frame, area))
            selected = []
            while rows and len(selected) < params['n']:
                def score(row):
                    position = np.asarray(row[0]['c2w'])[:3, 3]
                    diversity = min((np.linalg.norm(position - np.asarray(old['c2w'])[:3, 3]) for old in selected), default=0.)
                    return row[1] + min(float(diversity), 2.) * .12
                selected.append(rows.pop(max(range(len(rows)), key=lambda i: score(rows[i])))[0])
            reason = 'Projected extension bounds and camera diversity selected this original RGB frame; added identities have no claimed observed mask.'
        if not selected:
            raise ContractError('FRAME_MISMATCH', 'No original unmasked frame covers the proposed extension')
        frames = []
        for frame in selected:
            check_frame(frame, self.storage_root)
            value = copy.deepcopy(frame)
            value['selection_reason'] = reason
            frames.append(value)
        payload = {'scene_id': ref['scene_id'], 'scene_revision': ref['base_revision'],
                   'capture_revision': scene['capture_revision'], 'geometry_revision': scene['geometry_revision'],
                   'target_ids': params['target_ids'], 'purpose': params['purpose'], 'frames': frames}
        payload['evidence_id'] = 'ev-' + canonical_hash(payload)[:32]
        payload['evidence_hash'] = canonical_hash(payload)
        validate(payload, 'EvidenceBundle')
        folder = self.storage_root / ref['scene_id'] / 'evidence'
        immutable_json(folder / f"{payload['evidence_id']}.json", payload)
        provenance = {'evidence_id': payload['evidence_id'], 'evidence_hash': payload['evidence_hash'],
                      'candidate_id': candidate_id, 'candidate_hash': candidate_hash, 'decision_ref': context.decision_ref,
                      'new_design_identities': sorted(metadata['exact_additions']), 'source_capture_hash': capture['capture_hash'],
                      'source_frames_unchanged': True, 'observed_masks_for_additions_claimed': False}
        # Identical evidence may be reused by another candidate, so bind the
        # provenance filename to the candidate while leaving EvidenceBundle v1 intact.
        immutable_json(folder / f"{payload['evidence_id']}.{candidate_id}.extension.json", provenance)
        return copy.deepcopy(payload)

    def _capture(self, scene_id, capture_revision):
        path = self.storage_root / scene_id / 'captures' / f'{capture_revision}.json'
        if not path.is_file():
            raise ContractError('NOT_FOUND', 'Capture is not registered')
        return checked_record(path, 'capture_hash')

    async def get(self, evidence_id, context=None, scene_id=None):
        if not evidence_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in evidence_id):
            raise ContractError('INVALID_ARGUMENT', 'Invalid evidence ID')
        if scene_id:
            paths = [self.storage_root / scene_id / 'evidence' / f'{evidence_id}.json']
        else:
            paths = list(self.storage_root.glob(f'*/evidence/{evidence_id}.json'))
        paths = [p for p in paths if p.is_file()]
        if len(paths) != 1:
            raise ContractError('NOT_FOUND', 'Evidence was not found uniquely')
        value = checked_record(paths[0], 'evidence_hash')
        validate(value, 'EvidenceBundle')
        capture = self._capture(value['scene_id'], value['capture_revision'])
        if capture['geometry_revision'] != value['geometry_revision']:
            raise ContractError('EVIDENCE_MISMATCH', 'Evidence geometry no longer matches capture')
        by_id = {frame['source_frame_id']: frame for frame in capture['frames']}
        for frame in value['frames']:
            original = by_id.get(frame['source_frame_id'])
            if not original or any(frame[key] != original[key] for key in frame if key != 'selection_reason'):
                raise ContractError('FRAME_MISMATCH', 'Evidence frame differs from registered capture')
            check_frame(frame, self.storage_root)
        return copy.deepcopy(value)
