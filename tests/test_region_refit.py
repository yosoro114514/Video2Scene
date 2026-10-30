import numpy as np
import pytest

from video2scene.region_refit import cross_view_filter,local_depth_mask,validate_annotation,fit_region_plane


def protocol():
    return dict(fit_frames=[0,1,2],depth_consistency_abs=.04,depth_consistency_relative=.02,
        mask_margin_pixels=3,minimum_other_view_support=1,minimum_consistency_ratio=.6,
        depth_median_window=5,local_depth_tolerance=.04,object_min_depth_band=.06,object_mad_scale=3.5)


def record(frame,role='fit'):
    return dict(frame=frame,role=role,c2w=np.eye(4).tolist(),intrinsics=[10,10,10,10])


def test_local_spikes_rejected_without_modifying_depth_values():
    depth=np.ones((21,21),np.float32);depth[10,10]=3;depth[5,5]=np.nan
    original=depth.copy();mask,counts=local_depth_mask(depth,np.ones_like(depth,bool),True,protocol())
    assert not mask[10,10] and not mask[5,5] and mask[12,12]
    np.testing.assert_array_equal(depth,original)
    assert counts['rejected']>=1


def test_heldout_cannot_supply_missing_support():
    xyz=np.array([[0,0,1.]])
    masks={(1,'x'):np.ones((21,21),bool)}
    keep,stats=cross_view_filter(xyz,0,'x',[(record(1,'heldout'),np.ones((21,21)))],masks,protocol())
    assert not keep[0] and stats['support'][0]==0


def test_occlusion_is_not_a_conflict_and_valid_other_view_supports():
    xyz=np.array([[0,0,1.]])
    masks={(i,'x'):np.ones((21,21),bool) for i in [1,2]}
    keep,stats=cross_view_filter(xyz,0,'x',[(record(1),np.ones((21,21))),
        (record(2),np.full((21,21),.5))],masks,protocol())
    assert keep[0] and stats['support'][0]==1 and stats['occluded'][0]==1 and stats['conflict'][0]==0


def test_points_in_free_space_are_rejected():
    xyz=np.array([[0,0,1.]])
    masks={(i,'x'):np.ones((21,21),bool) for i in [1,2]}
    keep,stats=cross_view_filter(xyz,0,'x',[(record(i),np.full((21,21),2.)) for i in [1,2]],masks,protocol())
    assert not keep[0] and stats['conflict'][0]==2


def test_bad_coordinates_and_holdout_annotations_rejected():
    a=dict(frame=0,target_id='t008',confidence=.9,status='usable',box_xyxy=[10,20,100,120],
        positive_points=[[50,60]],negative_points=[],keep_polygons=[],exclude_polygons=[])
    validate_annotation(a,[0],['t008'])
    a['positive_points']=[[900,600]]
    with pytest.raises(ValueError):validate_annotation(a,[0],['t008'])
    a['frame']=6
    with pytest.raises(ValueError):validate_annotation(a,[0],['t008'])


def test_disconnected_regions_can_support_one_plane_without_hidden_height_guess():
    x,z=np.meshgrid(np.linspace(0,.3,15),np.linspace(0,.3,15))
    patch=np.column_stack([x.ravel(),np.ones(x.size),z.ravel()])
    sources={f:patch+np.array([f*.5,0,0]) for f in range(3)}
    candidates,_=fit_region_plane(sources,'table')
    assert candidates and len(candidates[0]['frame_counts'])==3
    assert abs(candidates[0]['normal'][1])>.99
    assert candidates[0]['fit_residual_median_m']<1e-6
