"""Pixel-region validation and conservative multi-view depth rejection."""
import cv2
import numpy as np

from .roomplan_minimal import project


def validate_annotation(a,fit_frames,targets,width=480,height=640):
    if a['frame'] not in fit_frames or a['target_id'] not in targets:
        raise ValueError('Unknown target or heldout frame in annotation')
    if not 0 <= a['confidence'] <= 1:raise ValueError('Invalid confidence')
    for key in ['positive_points','negative_points']:
        for point in a[key]:
            if len(point)!=2 or not 0<=point[0]<width or not 0<=point[1]<height:
                raise ValueError('Point outside original image coordinates')
    for key in ['keep_polygons','exclude_polygons']:
        for poly in a[key]:
            p=np.asarray(poly,float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
                raise ValueError('Invalid polygon')
            if (p<0).any() or (p[:,0]>=width).any() or (p[:,1]>=height).any():
                raise ValueError('Polygon outside original image coordinates')
            if abs(cv2.contourArea(p.astype(np.float32)))<4:raise ValueError('Degenerate polygon')
    if a['status']=='usable':
        x0,y0,x1,y1=a['box_xyxy']
        if not (0<=x0<x1<width and 0<=y0<y1<height):raise ValueError('Invalid image box')
        if a['target_id'].startswith('t0'):
            if not a['positive_points']:raise ValueError('SAM2 needs a positive point')
            if any(not(x0<=x<=x1 and y0<=y<=y1) for x,y in a['positive_points']):
                raise ValueError('Positive point outside target box')
        elif not a['keep_polygons']:raise ValueError('Surface needs visible-region polygons')


def polygon_mask(polygons,shape=(640,480)):
    mask=np.zeros(shape,np.uint8)
    for poly in polygons:cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],1)
    return mask.astype(bool)


def local_depth_mask(depth,mask,object_target,p):
    valid=mask & np.isfinite(depth) & (depth>.05) & (depth<8.)
    safe=np.where(np.isfinite(depth),depth,0).astype(np.float32)
    median=cv2.medianBlur(safe,p['depth_median_window'])
    good=valid & (median>.05) & (np.abs(safe-median)<=p['local_depth_tolerance'])
    if object_target and good.any():
        values=depth[good];center=np.median(values);mad=np.median(np.abs(values-center))
        band=max(p['object_min_depth_band'],p['object_mad_scale']*1.4826*mad)
        good &= np.abs(depth-center)<=band
    return good,dict(input_valid=int(valid.sum()),retained=int(good.sum()),rejected=int(valid.sum()-good.sum()))


def cross_view_filter(points,source_frame,target,views,masks,p):
    support=np.zeros(len(points),np.uint8);conflict=np.zeros(len(points),np.uint8);occluded=np.zeros(len(points),np.uint8)
    for rec,depth in views:
        if rec['role']!='fit' or rec['frame']==source_frame or rec['frame'] not in p['fit_frames']:continue
        mask=masks.get((rec['frame'],target))
        if mask is None or not mask.any():continue
        uv,z=project(points,rec);xy=np.rint(uv).astype(int);h,w=depth.shape
        valid=(z>.05)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        ids=np.flatnonzero(valid);x,y=xy[ids].T;d=depth[y,x]
        valid_depth=np.isfinite(d)&(d>.05)&(d<8)
        ids=ids[valid_depth];x,y=xy[ids].T;d=depth[y,x]
        tolerance=p['depth_consistency_abs']+p['depth_consistency_relative']*d
        residual=z[ids]-d
        expanded=cv2.dilate(mask.astype(np.uint8),np.ones((p['mask_margin_pixels'],)*2,np.uint8)).astype(bool)
        labelled=expanded[y,x];consistent=np.abs(residual)<=tolerance
        support[ids]+=consistent&labelled
        conflict[ids]+=(residual < -tolerance)|(consistent & ~labelled)
        occluded[ids]+=residual>tolerance
    ratio=(support+1)/(support+conflict+1)
    keep=(support>=p['minimum_other_view_support'])&(ratio>=p['minimum_consistency_ratio'])
    return keep,dict(support=support,conflict=conflict,occluded=occluded,consistency_ratio=ratio)


def fit_region_plane(sources,label):
    """Fit a common observed plane across disconnected trusted image regions.

    Applied identically to original/corrected/depth variants. The convex footprint
    interpolates gaps; it is explicitly not a full semantic silhouette.
    """
    import open3d as o3d
    from .geometry import point_cloud
    from .roomplan_minimal import CONFIG,SURFACE_COLORS,voxel_points
    o3d.utility.random.seed(CONFIG['seed'])
    if not sources:return [],[dict(reason='no samples')]
    remaining=voxel_points(np.concatenate(list(sources.values())),.015)
    candidates=[];rejected=[]
    for attempt in range(8):
        if len(remaining)<100:break
        equation,indices=point_cloud(remaining).segment_plane(distance_threshold=.035,ransac_n=3,num_iterations=1000)
        if len(indices)<100:break
        inliers=remaining[indices];keep=np.ones(len(remaining),bool);keep[indices]=False;remaining=remaining[keep]
        equation=np.array(equation);equation/=np.linalg.norm(equation[:3]);normal=equation[:3]
        if (label=='table' and abs(normal[1])<.85) or (label=='wall' and abs(normal[1])>.35):
            rejected.append(dict(reason='normal conflict',points=len(inliers)));continue
        support={str(f):int((np.abs(x@normal+equation[3])<.035).sum()) for f,x in sources.items()}
        support={f:n for f,n in support.items() if n>=20}
        if len(support)<3:
            rejected.append(dict(reason='fewer than three supporting views',support=support));continue
        center=np.median(inliers,axis=0);center-=normal*(center@normal+equation[3])
        axis=np.eye(3)[np.argmin(np.abs(normal))];u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
        basis=np.column_stack([u,v]);local=(inliers-center)@basis
        hull=cv2.convexHull(local.astype(np.float32)).reshape(-1,2);area=float(cv2.contourArea(hull))
        if area<.06:
            rejected.append(dict(reason='observed convex footprint below minimum area',area=area));continue
        candidates.append(dict(id=label+'_region',label=label,equation=equation.tolist(),normal=normal.tolist(),
            center=center.tolist(),polygon=(center+hull@basis.T).tolist(),area=area,color=SURFACE_COLORS[label],
            point_count=len(inliers),frame_counts=support,fit_residual_median_m=float(np.median(np.abs(inliers@normal+equation[3]))),
            extent_kind='convex footprint between disconnected observed patches; interpolated gaps, not complete boundary',
            status='experimental observed-plane candidate',sample_points=inliers[::max(1,len(inliers)//1200)].tolist()))
    return candidates,rejected
