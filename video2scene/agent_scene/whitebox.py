"""Numerical proposals from GPT-selected masks and measured camera/depth outputs.

The model reviews these proposals before they become the editable scene seed.
No previous scene, YOLO class, reference mesh or reference pose is read here.
"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from video2scene.artifacts import ViPEArtifacts, clean_depth, unproject
from video2scene.capture_evidence import register_capture
from video2scene.capture_evidence.storage import copy_artifact
from video2scene.contracts_v1 import canonical_hash, validate, utc_now
from video2scene.codex_review import digest
from .planning import read, save

FIELDS = ('category','transform.translation_m','transform.rotation_xyzw','transform.scale','support_id','asset.geometry','asset.materials')


def fit_plane(points):
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 40:
        raise ValueError('Not enough plane samples')
    active = points
    for _ in range(4):
        center = np.median(active, axis=0)
        _, singular, vectors = np.linalg.svd(active-center, full_matrices=False)
        if singular[1] < 1e-5:
            raise ValueError('Degenerate structural observations')
        normal = vectors[-1]
        distances = np.abs((points-center)@normal)
        cutoff = max(.025, float(np.quantile(distances,.75))*1.5)
        active = points[distances <= cutoff]
    return normal, center, active, float(np.quantile(np.abs((active-center)@normal), .9))


def polygon_points(frame, polygon):
    height,width = frame.depth.shape
    points = np.rint(np.asarray(polygon)*[width-1,height-1]).astype(np.int32)
    mask = np.zeros((height,width),np.uint8)
    cv2.fillPoly(mask,[points],1)
    depth = clean_depth(frame.depth,max_depth=12)
    return unproject(depth,frame.intrinsics,frame.c2w,mask=mask,step=3)


def _indexed_rows(rows, label):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label} has no frame mapping')
    indexed, sources = {}, set()
    for row in rows:
        index, source = row.get('frame_index'), row.get('source_frame_index')
        if (type(index) is not int or type(source) is not int or min(index, source) < 0
                or index in indexed or source in sources):
            raise ValueError(f'{label} has invalid or duplicate frame/source indices')
        indexed[index] = row
        sources.add(source)
    return indexed


def _vipe_mapping(run):
    run = Path(run).resolve()
    observation_path = run/'frames/manifest.json'
    manifest = read(observation_path)
    observed = _indexed_rows(manifest['frames'], 'Observation manifest')
    source_to_observation = {row['source_frame_index']: index for index, row in observed.items()}
    pointer_path = run/'tools/vipe/latest.json'
    pointer = read(pointer_path) if pointer_path.is_file() else {}
    vipe = run/'vipe'
    dense = bool(pointer.get('vipe_manifest_path'))
    if dense or not vipe.is_dir() and pointer:
        vipe = Path(pointer['vipe_dir']).resolve()
        if not vipe.is_relative_to(run):
            raise ValueError('ViPE output must belong to this project')
    mapping = {index: row['source_frame_index'] for index, row in observed.items()}
    all_indices = sorted(mapping)
    provenance = {'vipe_dir': str(vipe), 'sampling_version': 'legacy_observation_frames',
                  'observation_manifest_path': str(observation_path), 'observation_manifest_sha256': digest(observation_path)}
    if dense:
        path = Path(pointer['vipe_manifest_path']).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError('Dense ViPE manifest must belong to this project')
        if pointer.get('vipe_manifest_sha256') != digest(path):
            raise ValueError('Dense ViPE manifest hash mismatch')
        value = read(path)
        if (value.get('sampling_version') != 'vipe_dense_v1' or pointer.get('sampling_version') != 'vipe_dense_v1'
                or value.get('observation_manifest_sha256') != digest(observation_path)
                or pointer.get('observation_manifest_sha256') != digest(observation_path)
                or Path(value.get('observation_manifest_path', '')).resolve() != observation_path
                or not manifest.get('source_sha256') or value.get('source_sha256') != manifest['source_sha256']):
            raise ValueError('Dense ViPE mapping is not bound to the current original observation manifest/video')
        indexed = _indexed_rows(value['frames'], 'Dense ViPE manifest')
        all_indices = sorted(indexed)
        if all_indices != list(range(len(indexed))):
            raise ValueError('Dense ViPE frame indices must be contiguous from zero')
        mapping = {}
        for index, row in indexed.items():
            source = row['source_frame_index']
            if row.get('dense_frame_index') != index:
                raise ValueError('Dense frame index mismatch')
            expected = source_to_observation.get(source)
            if (row.get('observation_frame_index') != expected
                    or row.get('is_observation_frame') is not (expected is not None)):
                raise ValueError('Dense/observation frame mapping mismatch')
            if expected is not None:
                original = observed[expected]
                if any(row.get(key) != original[key] for key in ('source_width', 'source_height', 'source_fps') if key in original):
                    raise ValueError('Dense frame source dimensions/rate mismatch')
                if 'source_time_s' in original and (row.get('source_time_s') is None
                        or abs(row['source_time_s'] - original['source_time_s']) > 1e-5):
                    raise ValueError('Dense frame source timestamp mismatch')
                if row.get('observation_rgb_sha256') != original.get('sha256'):
                    raise ValueError('Dense frame observation RGB hash mismatch')
                mapping[index] = source
        if set(mapping.values()) != set(source_to_observation):
            raise ValueError('Dense ViPE mapping is missing required observation source frames')
        provenance.update(sampling_version=value['sampling_version'], vipe_manifest_path=str(path),
                          vipe_manifest_sha256=digest(path), source_sha256=value['source_sha256'],
                          prepared_fps=value.get('prepared_fps'))
    provenance['selected_source_mapping'] = [{'vipe_frame_index': index, 'source_frame_index': source}
                                               for index, source in sorted(mapping.items())]
    return vipe, mapping, all_indices, dense, provenance


def frames_by_source(run):
    vipe, mapping, all_indices, dense, _ = _vipe_mapping(run)
    artifacts = ViPEArtifacts(vipe)
    if dense and sorted(artifacts.indices) != all_indices:
        raise ValueError('ViPE artifact frame indices differ from the dense input manifest')
    iterator = artifacts.frames(indices=sorted(mapping)) if dense else artifacts.frames()
    result = {}
    for frame in iterator:
        if frame.index not in mapping or mapping[frame.index] in result:
            raise ValueError('ViPE yielded an unexpected or duplicate mapped frame')
        result[mapping[frame.index]] = frame
    if set(result) != set(mapping.values()):
        raise ValueError('ViPE artifacts are missing required observation source frames')
    return result


def archive_geometry_proposals(run_root):
    """Preserve exact proposal bytes and point clouds before a new measurement."""
    run = Path(run_root).resolve()
    source = run/'geometry_proposals.json'
    if not source.exists():
        return None
    checksum = digest(source)
    folder = run/'geometry/proposal_history'/checksum
    folder.mkdir(parents=True, exist_ok=True)
    files = [source]
    files += sorted((run/'geometry/points').glob('*.npy'))
    preview = run/'scene_actions/measurement_preview.png'
    if preview.exists():
        files.append(preview)
    archived = []
    for path in files:
        destination = folder/path.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != digest(path):
                # Identical JSON with different auxiliary output is a different
                # provenance state; preserve it by its own content hash too.
                destination = destination.with_name(destination.stem+'-'+digest(path)[:16]+destination.suffix)
        if not destination.exists():
            with destination.open('xb') as stream:
                stream.write(path.read_bytes())
        archived.append({'path': str(destination), 'sha256': digest(destination)})
    value = read(source)
    record = {'proposal_sha256': checksum, 'files': archived,
              'vipe_evidence': value.get('vipe_evidence'), 'legacy_origin_unrecorded': 'vipe_evidence' not in value}
    record_path = folder/('archive-'+canonical_hash(record)[:24]+'.json')
    if not record_path.exists():
        save(record_path, {**record, 'archived_at': utc_now()})
    return {'path': str(folder/'geometry_proposals.json'), 'sha256': checksum, 'record_path': str(record_path)}


def propose(run_root, sam_root):
    run=Path(run_root).resolve(); sam=Path(sam_root).resolve()
    archive_geometry_proposals(run)
    vipe_evidence = _vipe_mapping(run)[-1]
    prior=read(run/'prior.json'); frames=frames_by_source(run)
    if _vipe_mapping(run)[-1] != vipe_evidence:
        raise ValueError('ViPE mapping changed while reading measurement frames')
    camera_up=np.mean([-f.c2w[:3,1] for f in frames.values()],axis=0)
    camera_up/=np.linalg.norm(camera_up)
    floor_fits=[]
    for observation in prior['floor_observations']:
        source=observation['source_frame_index']
        points=polygon_points(frames[source],observation['polygon_normalized'])
        try:n,c,inliers,residual=fit_plane(points)
        except ValueError:continue
        if n@camera_up<0:n=-n
        floor_fits.append({'source_frame_index':source,'normal':n.tolist(),'center':c.tolist(),
                           'p90_residual_m':residual,'points':len(inliers)})
    usable=[f for f in floor_fits if np.dot(f['normal'],camera_up)>.65 and f['p90_residual_m']<.15]
    if not usable:
        save(run/'geometry_proposals.json',{'status':'needs_more_floor_evidence','floor_fits':floor_fits,
                                           'vipe_evidence':vipe_evidence,'prior_sha256':digest(run/'prior.json')})
        raise ValueError('No sufficiently planar GPT-selected floor region; ask the model for new evidence')
    best=min(usable,key=lambda f:f['p90_residual_m']+.1/np.sqrt(f['points']))
    up=np.array(best['normal']); origin=np.array(best['center'])
    first=frames[min(frames)]
    right=first.c2w[:3,0].copy();right-=right@up*up;right/=np.linalg.norm(right)
    rotation=np.stack([right,up,np.cross(right,up)])
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=-rotation@origin
    for fit in floor_fits:
        fit['normal_difference_deg']=float(np.degrees(np.arccos(np.clip(np.dot(fit['normal'],up),-1,1))))
        fit['height_difference_m']=float((np.array(fit['center'])-origin)@up)
    warnings=[]
    if any(f['normal_difference_deg']>15 or abs(f['height_difference_m'])>.25 for f in floor_fits):
        warnings.append('Floor observations disagree across views; alignment uses the declared best local anchor and remains provisional for human review.')
    proposals=[]; point_folder=run/'geometry/points';point_folder.mkdir(parents=True,exist_ok=True)
    for target in prior['objects']:
        identity=target['object_id']; clouds=[]; observations=[]
        for source,frame in frames.items():
            path=sam/identity/'target_masks'/f'{source:05d}.npz'
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as archive:
                mask=archive['mask'].astype(np.uint8)
                if int(archive['source_frame_index'])!=source:raise ValueError('SAM source frame mismatch')
            area=int(mask.sum())
            if area<100:continue
            if mask.shape!=frame.depth.shape:
                mask=cv2.resize(mask,(frame.depth.shape[1],frame.depth.shape[0]),interpolation=cv2.INTER_NEAREST)
            mask=cv2.erode(mask,np.ones((3,3),np.uint8))
            points=unproject(clean_depth(frame.depth,max_depth=12),frame.intrinsics,frame.c2w,mask=mask,step=4)
            if len(points)<25:continue
            points=points@rotation.T+transform[:3,3]
            center=np.median(points,axis=0)
            dist=np.linalg.norm(points-center,axis=1)
            points=points[dist<=np.quantile(dist,.92)]
            clouds.append((source,points,area))
            observations.append({'source_frame_index':source,'mask_pixels':area,'points':len(points),'visible_center_m':center.tolist()})
        if not clouds:
            proposals.append({'object_id':identity,'status':'no_valid_mask_depth','prior':target})
            continue
        anchor=next((c for c in clouds if c[0]==target['seed_source_frame_index']),max(clouds,key=lambda c:c[2]))
        anchor_center=np.median(anchor[1],axis=0)
        limit=max(.45,float(max(target['approx_dimensions_m']))*.55)
        accepted=[c for c in clouds if np.linalg.norm(np.median(c[1],axis=0)-anchor_center)<=limit]
        points=np.concatenate([c[1][::max(1,len(c[1])//1500)] for c in accepted])
        np.save(point_folder/f'{identity}.npy',points,allow_pickle=False)
        horizontal=points[:,[0,2]]
        covariance=np.cov(horizontal.T)
        eigen,axes=np.linalg.eigh(covariance)
        long=axes[:,-1]
        yaw=float(np.arctan2(-long[1],long[0]))
        R=Rotation.from_euler('y',yaw).as_matrix()
        local=points@R
        lo,hi=np.quantile(local,[.03,.97],axis=0)
        dims=hi-lo; center=(lo+hi)/2@R.T
        # These are proposal values; GPT must adjudicate visible extents vs hidden-side priors.
        proposals.append({'object_id':identity,'status':'observed_proposal','center_m':center.tolist(),
                          'visible_dimensions_m':dims.tolist(),'yaw_deg':float(np.degrees(yaw)),
                          'anchor_source_frame_index':anchor[0], 'accepted_frame_indices':[c[0] for c in accepted],
                          'rejected_frame_indices':[c[0] for c in clouds if c[0] not in {a[0] for a in accepted}],
                          'per_view':observations,'prior':target})
    floor_points=np.concatenate([polygon_points(frames[o['source_frame_index']],o['polygon_normalized']) for o in prior['floor_observations']])
    floor_points=floor_points@rotation.T+transform[:3,3]
    floor_points=floor_points[np.abs(floor_points[:,1])<.2]
    bounds=np.quantile(floor_points[:,[0,2]],[.01,.99],axis=0).tolist() if len(floor_points)>20 else [[-2,-2],[2,2]]
    wall_fits=[]
    for observation in prior['wall_observations']:
        source=observation['source_frame_index']
        points=polygon_points(frames[source],observation['polygon_normalized'])
        try:n,c,inliers,residual=fit_plane(points)
        except ValueError:continue
        world_points=inliers@rotation.T+transform[:3,3]
        wall_fits.append({'wall_id':observation['wall_id'],'source_frame_index':source,
                         'normal':(rotation@n).tolist(),'center_m':(rotation@c+transform[:3,3]).tolist(),
                         'observed_bounds_m':np.quantile(world_points,[.03,.97],axis=0).tolist(),
                         'p90_residual_m':residual})
    value={'status':'proposed','vipe_evidence':vipe_evidence,'scene_from_vipe':transform.tolist(),'floor_anchor':best,'floor_fits':floor_fits,
           'floor_observed_xz_bounds':bounds,'wall_fits':wall_fits,'objects':proposals,
           'warnings':warnings,'scale_status':'estimated','cameras_frozen':True,
           'sam_result_sha256':digest(sam/'result.json'),'prior_sha256':digest(run/'prior.json')}
    if _vipe_mapping(run)[-1] != vipe_evidence:
        raise ValueError('ViPE mapping changed during geometry measurement')
    save(run/'geometry_proposals.json',value)
    return value


def export_snapshot(run_root, layout, sam_root, *, scene_id='redwood-agent', runtime_root=None):
    run=Path(run_root).resolve();sam=Path(sam_root).resolve()
    runtime=Path(runtime_root).resolve() if runtime_root is not None else run/'runtime'
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', scene_id):
        raise ValueError('Invalid scene identity')
    proposals=read(run/'geometry_proposals.json')
    if 'vipe_evidence' in proposals and proposals['vipe_evidence'] != _vipe_mapping(run)[-1]:
        raise ValueError('Geometry proposals belong to a different ViPE input mapping; measure again before export')
    transform=np.array(proposals['scene_from_vipe'])
    geometry_revision='geom-'+canonical_hash({'transform':transform.tolist(),'layout':layout})[:24]
    frames=frames_by_source(run)
    manifest=read(run/'frames/manifest.json')
    source_rows={row['source_frame_index']:row for row in manifest.get('frames',[])}
    preview_fps=float(proposals.get('vipe_evidence',{}).get('prepared_fps') or
                      manifest.get('preview_fps', manifest.get('prepared_fps', 10.)))
    if not np.isfinite(preview_fps) or preview_fps<=0:raise ValueError('Invalid preview frame rate')
    capture_frames=[]; frame_targets={}; original_map={}; folder=run/'geometry/evidence';folder.mkdir(parents=True,exist_ok=True)
    needed={o['source_frame_index'] for o in read(run/'prior.json')['floor_observations']}
    target_indices={}
    accepted_sources={row['object_id']:set(row.get('accepted_frame_indices',[])) for row in proposals['objects']}
    for target in layout['objects']:
        identity=target['object_id'];visible=[]
        for path in sorted((sam/identity/'target_masks').glob('*.npz')):
            with np.load(path,allow_pickle=False) as data:
                area=int(data['mask'].sum());source=int(data['source_frame_index'])
            # SAM IDs may jump to another similar instance after an occlusion.
            # Only the same spatial cluster used for this object's proposal may
            # become downstream modeling evidence.
            if source in accepted_sources.get(identity,set()) and area>100:visible.append((source,area))
        visible.sort(key=lambda row:row[1],reverse=True)
        chosen=[]
        for source,area in visible:
            if not chosen or all(abs(source-other)>80 for other in chosen):chosen.append(source)
            if len(chosen)==6:break
        if not chosen:raise ValueError('Cannot export an object without observed segmentation evidence: '+identity)
        target_indices[identity]=chosen;needed.update(chosen)
    for source in sorted(needed):
        frame=frames[source];height,width=frame.depth.shape
        rgb_path=folder/f'rgb-{source:05d}.png'
        cv2.imwrite(str(rgb_path),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
        rgb=copy_artifact(rgb_path,runtime,scene_id,'source_rgb')
        fx,fy,cx,cy=frame.intrinsics
        source_row=source_rows.get(source,{})
        source_width=source_row.get('source_width',640);source_height=source_row.get('source_height',480)
        common={'source_time_s':source_row.get('source_time_s'),'preview_time_s':frame.index/preview_fps,
                'width':width,'height':height,'rgb':rgb,
                'K':[[float(fx),0.,float(cx)],[0.,float(fy),float(cy)],[0.,0.,1.]],
                'c2w':(transform@frame.c2w).tolist(),'source_to_image':np.diag([width/source_width,height/source_height,1.]).tolist(),
                'selection_reason':'GPT-planned object, actual SAM observed mask and exactly joined ViPE source-frame camera.'}
        for identity,chosen in target_indices.items():
            if source not in chosen:continue
            with np.load(sam/identity/'target_masks'/f'{source:05d}.npz',allow_pickle=False) as data:mask=data['mask'].astype(np.uint8)
            if mask.shape!=(height,width):mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
            mask_path=folder/f'mask-{identity}-{source:05d}.png';cv2.imwrite(str(mask_path),mask*255)
            source_id=f'rgb-{source:05d}-{identity}'
            capture_frames.append(dict(common,source_frame_id=source_id,mask=copy_artifact(mask_path,runtime,scene_id,'observed_mask')))
            frame_targets[source_id]=identity;original_map[source_id]=source
        source_id=f'rgb-{source:05d}'
        capture_frames.append(dict(common,source_frame_id=source_id,mask=None))
        original_map[source_id]=source
    capture_revision='cap-'+canonical_hash(capture_frames)[:24]
    register_capture(runtime,scene_id,capture_revision,geometry_revision,capture_frames,
        metadata={'frame_targets':frame_targets,'original_source_indices':original_map,'scene_from_vipe':transform.tolist(),
                  'source':'fresh RGB-only ViPE and GPT-requested SAM3.1','ground_truth_inputs':[]})
    objects=[];assets=runtime/scene_id/'seed_assets';assets.mkdir(parents=True,exist_ok=True)
    entries=layout['objects']+layout['structures']
    known_ids={o['object_id'] for o in entries}
    for item in entries:
        identity=item['object_id'];dims=np.array(item['dimensions_m'],float)
        if dims.shape!=(3,) or not np.isfinite(dims).all() or min(dims)<.01 or max(dims)>30:raise ValueError('Invalid layout dimensions')
        center=np.array(item['center_m'],float)
        if center.shape!=(3,) or not np.isfinite(center).all() or np.abs(center).max()>40:raise ValueError('Invalid layout center')
        mesh=trimesh.creation.box(extents=dims)
        mesh.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name='Whitebox clay',baseColorFactor=[225,229,222,255],metallicFactor=0.0,roughnessFactor=.9))
        path=assets/f'{identity}.glb';path.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
        support=item.get('support_object_id','unknown')
        support=support if support in known_ids and support!=identity else None
        state={field:{'source':'model' if field in ['category','support_id'] else 'derived','locked':False,'decision_ref':None} for field in FIELDS}
        value={'instance_id':identity,'kind':item.get('kind','object'),'category':item['category'],
            'transform':{'translation_m':center.tolist(),'rotation_xyzw':Rotation.from_euler('y',item['yaw_deg'],degrees=True).as_quat().tolist(),'scale':[1.,1.,1.]},
            'local_bounds_m':{'min':(-dims/2).tolist(),'max':(dims/2).tolist()},'support_id':support,
            'asset':{'asset_version':'whitebox-'+digest(path)[:24],'artifact':{'uri':f'artifact://{scene_id}/seed_assets/{path.name}',
                'sha256':digest(path),'media_type':'model/gltf-binary','role':'agent_whitebox'},'native_frame':'asset_y_up','native_to_object':np.eye(4).tolist()},
            'field_state':state,'evidence_ids':[],'geometry_source':'mixed'}
        validate(value,'SceneObject');objects.append(value)
    snapshot={'contract_version':'video2scene/1.0','scene_id':scene_id,'capture_revision':capture_revision,
              'geometry_revision':geometry_revision,'world_frame':'scene_y_up','object_frame':'object_y_up','unit':'m',
              'scale_status':'estimated','scope':'room','objects':objects,
              'revision':'agent-whitebox-'+canonical_hash(objects)[:20],'parent_revision':None}
    validate(snapshot,'SceneSnapshot');save(run/'snapshot.json',snapshot)
    save(run/'snapshot_provenance.json',{'layout_sha256':digest(run/'layout.json'),
        'snapshot_sha256':digest(run/'snapshot.json'),'sam_result_sha256':digest(sam/'result.json')})
    return snapshot
