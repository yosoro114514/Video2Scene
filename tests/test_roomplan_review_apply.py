import numpy as np
import pytest

from video2scene.roomplan_review_apply import apply_review


def track(tid, x=0.):
    return dict(id=tid,label='monitor',points=np.array([[x,0.,0.],[x+.03,.03,.03],[x,.04,.04]]),
                feature=np.array([1.,0.]),frames=[0,12],detection_ids=[tid+'a',tid+'b'],raw_ids=[tid])


def decision(tid,action='keep',confidence=.99):
    return dict(id=tid,action=action,label='laptop',confidence=confidence,evidence_frames=[0],reason='visible evidence')


def protocol():
    return dict(fit_frames=[0,12],minimum_confidence=.85,merge_max_centroid_distance_m=.65,merge_max_nearest_median_m=.18)


def response(decisions,merges=None):
    return dict(objects=decisions,surfaces=[],merges=merges or [])


def merge(ids):
    return dict(ids=ids,label='laptop',confidence=.99,evidence_frames=[0],reason='same physical object')


def test_low_confidence_reject_and_uncertainty_preserve_observations():
    tracks=[track('a'),track('b')]
    out,_,remap,_=apply_review(tracks,[],response([decision('a','reject',.8),decision('b','uncertain')]),protocol())
    assert len(out)==2 and all(t['review_status']=='pending' for t in out)
    assert remap=={'a':'a','b':'b'}
    assert all(t['label']=='monitor' for t in tracks)


def test_high_confidence_relabel_does_not_change_points():
    original=track('a')
    out,_,_,audit=apply_review([original],[],response([decision('a','relabel')]),protocol())
    assert out[0]['label']=='laptop'
    np.testing.assert_array_equal(out[0]['points'],original['points'])
    assert audit[0]['applied']=='relabel'


def test_reject_retains_explicit_remap_and_audit():
    out,_,remap,audit=apply_review([track('a')],[],response([decision('a','reject')]),protocol())
    assert not out and remap=={'a':None} and audit[0]['applied']=='reject'


def test_distant_model_merge_blocked_but_near_merge_combines_source_ids():
    resp=response([decision('a'),decision('b')],[merge(['a','b'])])
    out,_,_,audit=apply_review([track('a'),track('b',2)],[],resp,protocol())
    assert len(out)==2 and not audit[-1]['applied']
    out,_,remap,audit=apply_review([track('a'),track('b',.02)],[],resp,protocol())
    assert len(out)==1 and remap['a']==remap['b'] and audit[-1]['applied']
    assert set(out[0]['detection_ids'])=={'aa','ab','ba','bb'}


def test_invalid_coverage_and_heldout_evidence_cannot_apply():
    with pytest.raises(ValueError):
        apply_review([track('a')],[],response([]),protocol())
    d=decision('a');d['evidence_frames']=[6]
    with pytest.raises(ValueError):
        apply_review([track('a')],[],response([d]),protocol())
