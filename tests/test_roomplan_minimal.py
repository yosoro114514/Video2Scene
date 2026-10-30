import numpy as np

from video2scene.roomplan_minimal import (
    association_score, map_objects, merge_allowed, plane_mask, project,vote_point_semantics,
)


def observation(name, frame, offset=0, feature=(1.,0.), role='fit'):
    points=np.random.default_rng(7).uniform(0,.2,(150,3))+offset
    return dict(id=name,frame=frame,role=role,label='chair',points=points,
                feature=np.array(feature),frames=[frame],detection_ids=[name])


def test_appearance_cannot_override_distant_geometry():
    first=observation('a',0)
    far=observation('b',6,offset=4)
    assert association_score(first,far,True) is None


def test_appearance_veto_and_same_frame_exclusion():
    a=observation('a',0)
    b=observation('b',6,feature=(0.,1.))
    assert association_score(a,b,False) is not None
    assert association_score(a,b,True) is None
    b=observation('b',0)
    assert not merge_allowed(a,b,False)
    assert not merge_allowed(a,b,True)


def test_heldout_observations_never_update_map_and_assignment_is_one_to_one():
    observations=[observation('a',0),observation('b',0),observation('c',6),
                  observation('heldout',12,offset=4,role='heldout')]
    tracks,log=map_objects(observations,True)
    assert len(tracks)==2
    assert log['assignments']['a']!=log['assignments']['b']
    assert 'heldout' not in log['assignments']
    assert all(12 not in t['frames'] for t in tracks)


def test_planar_projection_uses_depth_and_respects_occlusion():
    rec=dict(c2w=np.eye(4).tolist(),intrinsics=[4,4,5,5])
    surface=dict(equation=[0,0,1,-2],polygon=[[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]])
    uv,z=project(np.array([[0,0,2],[1,1,2]]),rec)
    np.testing.assert_allclose(uv,[[5,5],[7,7]])
    visible,t=plane_mask(surface,rec,np.full((10,10),2.))
    assert visible[5,5]
    np.testing.assert_allclose(t[visible],2)
    hidden,_=plane_mask(surface,rec,np.ones((10,10)))
    assert not hidden.any()


def test_semantic_voting_excludes_occluded_and_heldout_votes():
    points=np.array([[0,0,2.],[0,0,3.]])
    rec=dict(c2w=np.eye(4).tolist(),intrinsics=[4,4,5,5],role='fit')
    depth=np.full((10,10),2.)
    label=np.full((10,10),7,np.uint8)
    frames=[(rec,None,depth,label),(rec,None,depth,label),
            (dict(rec,role='heldout'),None,depth,np.zeros((10,10),np.uint8))]
    voted,details=vote_point_semantics(points,frames,{'wall':[7]})
    assert voted.tolist()==[1,-1]
    assert details['observation_count'].tolist()==[2,0]
