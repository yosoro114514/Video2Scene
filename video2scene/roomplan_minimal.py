"""Evidence-based geometry for the frozen-input RoomPlan replacement experiment.

This is a small local implementation inspired by point-label lifting and
geometry/appearance association, not a claim to run LabelMaker or ConceptGraphs.
"""
from collections import Counter
import copy

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import point_cloud, fit_box
from .segmentation import spatial_overlap


CONFIG = dict(plane_distance=.035, plane_min_area=.06, plane_min_points=100,
              plane_min_views=3, voxel=.025, holdout_depth_tolerance=.08,
              association_overlap=.22, appearance_min=.70,
              merge_overlap=.50, merge_appearance=.85, min_object_views=2,
              max_depth=8.0, seed=20261027)
SURFACE_COLORS = {'table':[234,156,45], 'wall':[57,157,229],
                  'floor':[88,177,124], 'ceiling':[173,117,219]}


def voxel_points(points, size=.025):
    if not len(points):
        return np.empty((0,3))
    return np.asarray(point_cloud(points).voxel_down_sample(size).points).copy()


def project(points, rec):
    pose=np.asarray(rec['c2w'])
    cam=(np.asarray(points)-pose[:3,3])@pose[:3,:3]
    fx,fy,cx,cy=rec['intrinsics']
    z=cam[:,2]
    uv=cam[:,:2]/np.maximum(z[:,None],1.e-6)*[fx,fy]+[cx,cy]
    return uv,z


def semantic_groups(id2label):
    groups={k:[] for k in SURFACE_COLORS}
    for key,label in id2label.items():
        base=label.lower().strip()
        if base in ('desk','table','coffee table'):
            groups['table'].append(int(key))
        elif base in ('wall','floor','ceiling'):
            groups[base].append(int(key))
    return groups


def vote_point_semantics(points,frames,groups):
    """LabelMaker-style depth-visible reprojection voting, fitting views only.

    Each frame contributes one vote including OTHER; unseen is not a class.
    Scores are empirical agreement, not calibrated confidence.
    """
    names=['other']+list(groups)
    votes=np.zeros((len(points),len(names)),np.uint16)
    lookup=np.zeros(256,np.uint8)
    for idx,name in enumerate(names[1:],1):
        lookup[groups[name]]=idx
    for rec,rgb,depth,sem in frames:
        if rec['role']!='fit':continue
        uv,z=project(points,rec);uv=np.rint(uv).astype(int)
        h,w=depth.shape
        valid=(z>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ids=np.flatnonzero(valid);x,y=uv[ids].T
        d=depth[y,x]
        good=np.isfinite(d)&(d>.05)&(np.abs(d-z[ids])<=.08)
        ids=ids[good];x,y=uv[ids].T
        np.add.at(votes,(ids,lookup[sem[y,x]]),1)
    count=votes.sum(axis=1)
    winner=votes.argmax(axis=1)
    agreement=votes.max(axis=1)/np.maximum(count,1)
    accepted=(count>=2)&(agreement>=.60)
    labels=np.where(accepted,winner,-1)
    return labels,dict(names=names,votes=votes,observation_count=count,agreement=agreement,
                       accepted=accepted)


def fit_surfaces(points_by_label):
    """Find observed connected plane patches, retaining source frame support."""
    o3d.utility.random.seed(CONFIG['seed'])
    surfaces,rejected=[],[]
    for label,sources in points_by_label.items():
        if not sources:
            continue
        pooled=voxel_points(np.concatenate(list(sources.values())),CONFIG['voxel'])
        remaining=pooled
        # A wall can consist of several disconnected observed patches.
        for attempt in range(8):
            if len(remaining)<CONFIG['plane_min_points']:
                break
            equation,ids=point_cloud(remaining).segment_plane(
                distance_threshold=CONFIG['plane_distance'],ransac_n=3,num_iterations=1000)
            ids=np.asarray(ids)
            if len(ids)<CONFIG['plane_min_points']:
                break
            mask=np.ones(len(remaining),bool);mask[ids]=False
            inliers=remaining[ids];remaining=remaining[mask]
            equation=np.asarray(equation);equation/=np.linalg.norm(equation[:3])
            normal=equation[:3]
            alignment=abs(normal[1])
            if (label in ('table','floor','ceiling') and alignment<.85) or (label=='wall' and alignment>.35):
                rejected.append(dict(label=label,reason='semantic/normal conflict',points=len(inliers),normal=normal.tolist()))
                continue
            labels=np.asarray(point_cloud(inliers).cluster_dbscan(eps=.11,min_points=5))
            for component in sorted(set(labels)-{-1}):
                patch=inliers[labels==component]
                if len(patch)<CONFIG['plane_min_points']:
                    continue
                center=np.median(patch,axis=0)
                center-=normal*(center@normal+equation[3])
                axis=np.eye(3)[np.argmin(np.abs(normal))]
                u=np.cross(normal,axis);u/=np.linalg.norm(u)
                v=np.cross(normal,u)
                basis=np.column_stack([u,v])
                local=(patch-center)@basis
                hull=cv2.convexHull(local.astype(np.float32)).reshape(-1,2)
                area=float(cv2.contourArea(hull))
                if area<CONFIG['plane_min_area']:
                    continue
                source_counts={}
                # Count support near the observed patch, not arbitrary coplanar pixels.
                tree=cKDTree(patch)
                for frame,pts in sources.items():
                    near=np.abs(pts@normal+equation[3])<CONFIG['plane_distance']
                    count=int((tree.query(pts[near])[0]<.08).sum()) if near.any() else 0
                    if count>=20:
                        source_counts[str(frame)]=count
                if len(source_counts)<CONFIG['plane_min_views']:
                    rejected.append(dict(label=label,reason='less than three fitting views',area=area))
                    continue
                polygon=center+hull@basis.T
                surfaces.append(dict(id=f'{label}_{len(surfaces):02d}',label=label,
                    equation=equation.tolist(),normal=normal.tolist(),center=center.tolist(),
                    polygon=polygon.tolist(),area=area,color=SURFACE_COLORS[label],
                    point_count=len(patch),frame_counts=source_counts,
                    fit_residual_median_m=float(np.median(np.abs(patch@normal+equation[3]))),
                    extent_kind='convex footprint interpolated inside observed inliers; not complete room boundary',
                    status='automatic candidate; no human approval',
                    sample_points=patch[::max(1,len(patch)//1200)].tolist()))
    return surfaces,rejected


def plane_mask(surface,rec,depth):
    """Render a planar polygon with foreground occlusion from frozen depth."""
    h,w=depth.shape
    polygon=np.asarray(surface['polygon'])
    uv,z=project(polygon,rec)
    valid=z>.05
    if not valid.all() or len(polygon)<3:
        return np.zeros((h,w),bool),np.full((h,w),np.nan)
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.clip(uv,-10000,10000).astype(np.int32)],1)
    fx,fy,cx,cy=rec['intrinsics'];yy,xx=np.mgrid[:h,:w]
    rays=np.stack([(xx-cx)/fx,(yy-cy)/fy,np.ones_like(xx)],axis=-1)
    pose=np.asarray(rec['c2w']);n=np.asarray(surface['equation'][:3]);d=surface['equation'][3]
    denom=rays@(pose[:3,:3].T@n)
    t=np.divide(-(pose[:3,3]@n+d),denom,out=np.full((h,w),np.nan),where=np.abs(denom)>1.e-6)
    mask=mask.astype(bool)&(t>.05)&(t<CONFIG['max_depth'])
    # Keep unknown depth visible, but exclude these pixels from numerical evaluation.
    visible=mask&((depth<=0)|(~np.isfinite(depth))|(t<=depth+CONFIG['holdout_depth_tolerance']))
    return visible,t


def evaluate_surfaces(surfaces,frames,groups):
    rows=[]
    for rec,rgb,depth,sem in frames:
        if rec['role']!='heldout':
            continue
        for label,ids in groups.items():
            target=np.isin(sem,ids)&np.isfinite(depth)&(depth>.05)&(depth<CONFIG['max_depth'])
            pred=np.zeros(depth.shape,bool)
            errors=[]
            for surface in [s for s in surfaces if s['label']==label]:
                mask,t=plane_mask(surface,rec,depth)
                pred|=mask
                valid=mask&target
                errors.extend(np.abs(t[valid]-depth[valid]).tolist())
            if target.sum()<100:
                continue
            intersection=int((pred&target).sum())
            rows.append(dict(frame=rec['frame'],label=label,semantic_pixels=int(target.sum()),
                             projected_pixels=int(pred.sum()),intersection_pixels=intersection,
                             semantic_coverage=intersection/int(target.sum()),
                             semantic_precision=intersection/max(1,int(pred.sum())),
                             depth_residual_median_m=float(np.median(errors)) if errors else None,
                             note='heldout model-label agreement and same-ViPE depth consistency; not accuracy'))
    return rows


def association_score(a,b,appearance=False):
    if a['label']!=b['label']:
        return None
    distance=np.linalg.norm(np.median(a['points'],axis=0)-np.median(b['points'],axis=0))
    extent=np.linalg.norm(np.ptp(np.quantile(b['points'],[.05,.95],axis=0),axis=0))
    if distance>max(.20,extent*1.25):
        return None
    overlap=float(spatial_overlap(a['points'],b['points']))
    if overlap<CONFIG['association_overlap']:
        return None
    cosine=float(np.dot(a['feature'],b['feature']))
    if appearance and cosine<CONFIG['appearance_min']:
        return None
    score=(.6*min(overlap,1)+.4*cosine) if appearance else min(overlap,1)
    return score,overlap,cosine


def merge_allowed(a,b,appearance):
    if set(a['frames']) & set(b['frames']):
        return False
    score=association_score(a,b,appearance)
    if score is None or score[1]<CONFIG['merge_overlap']:
        return False
    if appearance and score[2]<CONFIG['merge_appearance']:
        return False
    # AABB overlap can be misleading with a thin monitor beside another monitor.
    distance=cKDTree(a['points']).query(b['points'])[0]
    return float(np.mean(distance<.07))>=.25


def fuse_track(a,b):
    na,nb=len(a['detection_ids']),len(b['detection_ids'])
    a['points']=voxel_points(np.vstack([a['points'],b['points']]),.02)
    feature=a['feature']*na+b['feature']*nb
    a['feature']=feature/max(np.linalg.norm(feature),1.e-9)
    a['frames']+=b['frames'];a['detection_ids']+=b['detection_ids']
    a['raw_ids']=sorted(set(a.get('raw_ids',[a['id']])+b.get('raw_ids',[b['id']])))


def map_objects(observations,appearance=False):
    tracks=[];assignments={};events=[]
    for frame in sorted({o['frame'] for o in observations if o['role']=='fit'}):
        batch=[o for o in observations if o['frame']==frame and o['role']=='fit']
        scores=np.full((len(batch),len(tracks)),-1.)
        details={}
        for i,obs in enumerate(batch):
            for j,track in enumerate(tracks):
                result=association_score(obs,track,appearance)
                if result is not None:
                    scores[i,j]=result[0];details[i,j]=result
        matches={}
        if scores.size:
            row,col=linear_sum_assignment(-scores)
            matches={int(i):int(j) for i,j in zip(row,col) if scores[i,j]>=0}
        for i,obs in enumerate(batch):
            if i in matches:
                track=tracks[matches[i]]
                sample=dict(id=obs['id'],points=obs['points'],feature=obs['feature'],frames=[frame],detection_ids=[obs['id']])
                fuse_track(track,sample)
                events.append(dict(detection=obs['id'],track=track['id'],score=details[i,matches[i]][0],
                                   overlap=details[i,matches[i]][1],cosine=details[i,matches[i]][2]))
            else:
                track=dict(id=f"t{len(tracks):03d}",label=obs['label'],points=obs['points'].copy(),
                           feature=obs['feature'].copy(),frames=[frame],detection_ids=[obs['id']],raw_ids=[f't{len(tracks):03d}'])
                tracks.append(track)
            assignments[obs['id']]=track['id']
    online_count=len(tracks)
    merge_events=[]
    changed=True
    while changed:
        changed=False
        for i in range(len(tracks)):
            for j in range(i+1,len(tracks)):
                if merge_allowed(tracks[i],tracks[j],appearance):
                    a,b=tracks[i],tracks[j]
                    merge_events.append(dict(into=a['id'],removed=b['id'],cosine=float(a['feature']@b['feature']),
                                             overlap=float(spatial_overlap(a['points'],b['points']))))
                    fuse_track(a,b)
                    for det in b['detection_ids']:
                        assignments[det]=a['id']
                    tracks.pop(j);changed=True;break
            if changed:
                break
    return tracks,dict(assignments=assignments,online_tracks=online_count,final_tracks=len(tracks),
                       online_matches=events,consolidation_merges=merge_events,
                       method='geometry+CLIP with appearance veto' if appearance else 'geometry-only; identical geometric gates and consolidation')


def track_specs(tracks):
    specs=[]
    for track in tracks:
        center,dims,rotation=fit_box(track['points'])
        feature=track['feature'].tolist()
        specs.append(dict(id=track['id'],label=track['label'],center=center.tolist(),dimensions=dims.tolist(),
                          rotation=rotation.tolist(),frame_indices=sorted(track['frames']),
                          detection_ids=track['detection_ids'],raw_ids=track['raw_ids'],feature=feature,
                          point_count=len(track['points']),
                          status='multi_view_candidate' if len(set(track['frames']))>=CONFIG['min_object_views'] else 'single_view_uncertain',
                          extent_kind='observed point extent; no hidden geometry completion'))
    return specs


def box_corners(obj):
    corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    return (corners*np.asarray(obj['dimensions']))@np.asarray(obj['rotation']).T+np.asarray(obj['center'])


BOX_EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]


def association_differences(a,b,observations):
    """Compare co-assignment pairs; track numbers have no cross-method meaning."""
    ids=sorted(set(a['assignments'])&set(b['assignments']))
    changed=[]
    for i,first in enumerate(ids):
        for second in ids[i+1:]:
            sa=a['assignments'][first]==a['assignments'][second]
            sb=b['assignments'][first]==b['assignments'][second]
            if sa!=sb:
                changed.append(dict(first=first,second=second,geometry_same=sa,appearance_same=sb))
    return dict(pairs_compared=len(ids)*(len(ids)-1)//2,changed_pairs=len(changed),examples=changed[:40],
                interpretation='different association is not necessarily correct; inspect source views')
