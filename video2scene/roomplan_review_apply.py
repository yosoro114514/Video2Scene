"""Validate data-only review proposals and preserve unresolved candidates."""
import copy
import itertools

import numpy as np
from scipy.spatial import cKDTree

from .roomplan_minimal import fuse_track


def validate_decisions(decisions, expected_ids, allowed_frames):
    ids = [d['id'] for d in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Review must cover each input candidate exactly once')
    for d in decisions:
        if d['action'] not in ['keep', 'relabel', 'reject', 'uncertain']:
            raise ValueError('Unknown review action')
        if not 0 <= d['confidence'] <= 1:
            raise ValueError('Confidence outside [0,1]')
        if not set(d['evidence_frames']) <= set(allowed_frames):
            raise ValueError('Review cites a heldout or nonexistent frame')
        if not isinstance(d['label'], str) or not 0 < len(d['label']) <= 80:
            raise ValueError('Invalid label')


def apply_review(tracks, surfaces, response, protocol):
    """No model code or coordinates are executed. Uncertainty is retained."""
    validate_decisions(response['objects'], [t['id'] for t in tracks], protocol['fit_frames'])
    validate_decisions(response['surfaces'], [s['id'] for s in surfaces], protocol['fit_frames'])
    threshold = protocol['minimum_confidence']
    result = {}; audit = []; remap = {}; decisions = {}
    for original in tracks:
        t = copy.deepcopy(original)
        d = next(x for x in response['objects'] if x['id'] == t['id'])
        confident = d['confidence'] >= threshold and bool(d['evidence_frames'])
        state = 'reviewed_candidate' if confident and d['action'] != 'uncertain' else 'pending'
        applied = 'keep'
        if confident and d['action'] == 'reject':
            remap[t['id']] = None; applied = 'reject'
        else:
            if confident and d['action'] == 'relabel':
                t['label'] = d['label']; applied = 'relabel'
            t['review_status'] = state
            result[t['id']] = t; remap[t['id']] = t['id']
        audit.append(dict(kind='object', id=t['id'], requested=d['action'], applied=applied,
                          status=state, original_label=original['label'], **{k:d[k] for k in ['label','confidence','evidence_frames','reason']}))
        decisions[t['id']] = d
    used = set()
    for merge in response['merges']:
        ids = merge['ids']; reason = None; checks = []
        if len(ids) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(remap):
            raise ValueError('Invalid merge IDs')
        if used.intersection(ids):
            raise ValueError('Overlapping merge groups')
        used.update(ids)
        if not 0 <= merge['confidence'] <= 1 or not set(merge['evidence_frames']) <= set(protocol['fit_frames']):
            raise ValueError('Invalid merge confidence/evidence')
        if merge['confidence'] < threshold or not merge['evidence_frames']:
            reason = 'low confidence or no cited evidence'
        elif any(i not in result for i in ids):
            reason = 'group contains rejected object'
        else:
            for a,b in itertools.combinations(ids,2):
                pa,pb = result[a]['points'],result[b]['points']
                distance = float(np.linalg.norm(np.median(pa,axis=0)-np.median(pb,axis=0)))
                nearest = float(min(np.median(cKDTree(pa).query(pb)[0]),np.median(cKDTree(pb).query(pa)[0])))
                passed = distance <= protocol['merge_max_centroid_distance_m'] and nearest <= protocol['merge_max_nearest_median_m']
                checks.append(dict(first=a,second=b,centroid_distance_m=distance,nearest_median_m=nearest,passed=passed))
            if not all(c['passed'] for c in checks):
                reason = 'geometry guard failed; preserve separate candidates'
        if reason:
            for tid in ids:
                if tid in result: result[tid]['review_status'] = 'pending'
            audit.append(dict(kind='merge',applied=False,blocked_reason=reason,checks=checks,**merge))
            continue
        target = sorted(ids)[0]; combined = result[target]
        for tid in ids:
            if tid != target:
                fuse_track(combined,result.pop(tid))
            remap[tid] = target
        combined['label'] = merge['label']
        combined['review_status'] = 'pending' if any(decisions[i]['action']=='uncertain' for i in ids) else 'reviewed_candidate'
        audit.append(dict(kind='merge',applied=True,target=target,checks=checks,**merge))
    surface_result = []
    for original in surfaces:
        s = copy.deepcopy(original)
        d = next(x for x in response['surfaces'] if x['id']==s['id'])
        confident = d['confidence'] >= threshold and bool(d['evidence_frames'])
        # Surface relabels are not supported by this fixed-role geometry experiment.
        applied = 'reject' if confident and d['action']=='reject' else 'keep'
        s['review_status'] = 'reviewed_candidate' if confident and d['action']=='keep' else 'pending'
        if applied != 'reject': surface_result.append(s)
        audit.append(dict(kind='surface',applied=applied,**d))
    return list(result.values()),surface_result,remap,audit
