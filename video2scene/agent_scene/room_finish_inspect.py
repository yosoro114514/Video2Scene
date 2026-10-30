"""Host-owned Blender inspection of the assembled room and actual door aperture."""
from pathlib import Path
import hashlib
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')


def main(path):
    spec=json.loads(Path(path).read_text(encoding='utf-8'))
    if digest(spec['scene_blend'])!=spec['scene_blend_sha256'] or digest(__file__)!=spec['worker_sha256']:
        raise ValueError('Inspection inputs changed')
    output=Path(spec['output_dir'])
    bpy.ops.wm.open_mainfile(filepath=spec['scene_blend'])
    scene=bpy.context.scene
    baseline=spec['baseline_snapshot']; snapshot=spec['candidate_scene']
    by_id={o['instance_id']:o for o in snapshot['objects']}
    roots={o['instance_id']:o for o in scene.objects if o.type=='EMPTY' and o.get('instance_id')}
    if set(roots)!=set(by_id): raise ValueError('Scene roots differ from candidate')
    # B converts Blender world coordinates to the immutable scene Y-up frame.
    B=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,-1.,0.,0.],[0.,0.,0.,1.]])
    C=B.T
    old={o['instance_id']:o for o in baseline['objects']}
    for identity,original in old.items():
        now=by_id[identity]
        for key in ['transform','support_id','local_bounds_m','category','kind']:
            if now[key]!=original[key]: raise ValueError('Existing layout changed: '+identity)
        if identity not in spec['replace_ids'] and now!=original:
            raise ValueError('Existing furnishing changed: '+identity)
    # This check reads the final imported mesh. Opening-kind OBB exemptions are
    # insufficient: nine rays through the specified cutout must be unobstructed.
    root=roots['wall_sofa_color']; inv=np.linalg.inv(np.asarray(root.matrix_world))
    vertices=[]; polygons=[]
    for mesh in scene.objects:
        if mesh.type!='MESH' or mesh.get('instance_id')!='wall_sofa_color': continue
        offset=len(vertices)
        for v in mesh.data.vertices:
            p=inv@np.r_[np.asarray(mesh.matrix_world@v.co),1.]
            vertices.append(tuple(p[:3]))
        polygons.extend(tuple(offset+i for i in p.vertices) for p in mesh.data.polygons)
    bvh=BVHTree.FromPolygons(vertices,polygons)
    aperture=spec['aperture']
    hits=[]
    for fx in [.2,.5,.8]:
        for fy in [.2,.5,.8]:
            x=aperture['x'][0]+fx*(aperture['x'][1]-aperture['x'][0])
            y=aperture['y'][0]+fy*(aperture['y'][1]-aperture['y'][0])
            hit=bvh.ray_cast(Vector((x,y,-.3)),Vector((0,0,1)),.6)
            hits.append({'point_local_y_up':[x,y],'clear':hit[0] is None})
    if not all(h['clear'] for h in hits): raise ValueError('The final wall mesh blocks the specified door opening')
    bounds=by_id['wall_sofa_color']['local_bounds_m']
    solid_tests=[((bounds['min'][0]+aperture['x'][0])/2,0.),
                 ((aperture['x'][1]+bounds['max'][0])/2,0.),
                 (sum(aperture['x'])/2,(aperture['y'][1]+bounds['max'][1])/2)]
    if not all(bvh.ray_cast(Vector((x,y,-.3)),Vector((0,0,1)),.6)[0] is not None for x,y in solid_tests):
        raise ValueError('Door opening removed a required wall pier/lintel')
    cabinet=by_id['tv_stand']; vase=by_id['vase_tv_stand']
    cabinet_top=cabinet['transform']['translation_m'][1]+cabinet['local_bounds_m']['max'][1]*cabinet['transform']['scale'][1]
    vase_bottom=vase['transform']['translation_m'][1]+vase['local_bounds_m']['min'][1]*vase['transform']['scale'][1]
    if abs(cabinet_top-vase_bottom)>1e-5 or vase['support_id']!='tv_stand': raise ValueError('Vase does not rest on the cabinet')
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24;scene.cycles.use_denoising=True
    scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='AgX'
    scene.world=bpy.data.worlds.new('Room inspection world');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.65,.7,.75,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6
    floor=roots['floor']; floor_pose=B@np.asarray(floor.matrix_world)
    center=floor_pose[:3,3].copy();center[1]=.7
    for name,offset,energy,size in [('Room key',[0,4,0],1100,5),('Window fill',[-4,2,0],650,4),('Room fill',[2,3,2],700,3)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.size=size
        ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob)
        ob.location=Vector((C@np.r_[center+offset,1])[:3]);target=Vector((C@np.r_[center,1])[:3])
        ob.rotation_euler=(target-ob.location).to_track_quat('-Z','Y').to_euler()
    room_axes=floor_pose[:3,:3];room_axes=room_axes/np.linalg.norm(room_axes,axis=0)
    def room_point(offset):return center+room_axes@np.asarray(offset)
    door_center=np.asarray(by_id['interior_door']['transform']['translation_m'])
    vase_center=np.asarray(vase['transform']['translation_m'])
    wall_inside=(B@np.asarray(roots['wall_sofa_color'].matrix_world))[:3,2]
    wall_inside/=np.linalg.norm(wall_inside)
    cabinet_front=center-vase_center
    cabinet_front[1]=0.
    cabinet_front/=np.linalg.norm(cabinet_front)
    views=[('overview',room_point([-8,6,-6]),center,9.0),
           ('top',room_point([-.01,10,.01]),center,6.8),
           ('door',door_center+wall_inside*3+np.array([.15,.35,.1]),door_center,2.7),
           ('vase',vase_center+cabinet_front*2+np.array([.3,.5,.15]),vase_center,1.25)]
    renders=[]
    for name,eye,aim,scale in views:
        data=bpy.data.cameras.new(name);ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob)
        ob.location=Vector((C@np.r_[eye,1])[:3]);target=Vector((C@np.r_[aim,1])[:3])
        ob.rotation_euler=(target-ob.location).to_track_quat('-Z','Y').to_euler()
        data.type='ORTHO';data.ortho_scale=scale;data.clip_start=.01;data.clip_end=100
        scene.camera=ob;scene.render.resolution_x=1000;scene.render.resolution_y=800;scene.render.resolution_percentage=100
        scene.render.filepath=str(output/(name+'.png'));bpy.ops.render.render(write_still=True)
        renders.append(name+'.png')
    result={'passed':True,'existing_transforms_bounds_support_unchanged':True,
        'non_target_furniture_unchanged':True,'door_aperture_rays':hits,'door_piers_and_lintel_present':True,
        'vase_support_gap_m':vase_bottom-cabinet_top,'source_scene_sha256':spec['scene_blend_sha256'],
        'worker_sha256':spec['worker_sha256'],'renders':{name:digest(output/name) for name in renders}}
    write(output/'inspection.json',result)


if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1])
