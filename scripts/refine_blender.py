"""Refine an existing scene, render source cameras, export a separate native file."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from video2scene.blender_assets import Asset, C, material


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def camera(entry):
    name=f"Evidence {entry['index']:03d}"
    o=bpy.data.objects.get(name)
    if o is None:
        o=bpy.data.objects.new(name,bpy.data.cameras.new(name));bpy.context.scene.collection.objects.link(o)
    cv=np.array(entry['c2w']);m=np.eye(4);m[:3,:3]=C@cv[:3,:3]@np.diag([1,-1,-1]);m[:3,3]=C@cv[:3,3];o.matrix_world=Matrix(m.tolist())
    w,h=entry.get('resolution',[640,480]);fx,fy,cx,cy=entry['intrinsics'];d=o.data
    d.type='PERSP';d.sensor_fit='HORIZONTAL';d.sensor_width=36;d.lens=fx*36/w
    d.shift_x=(w/2-cx)/w;d.shift_y=(cy-h/2)*(fx/fy)/w;d.clip_start=.01;d.clip_end=100
    s=bpy.context.scene;s.render.resolution_x=w;s.render.resolution_y=h;s.render.resolution_percentage=100;s.render.pixel_aspect_x=1;s.render.pixel_aspect_y=fx/fy
    return o


def render_views(config,out,tag,samples,frames):
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=samples;s.cycles.use_denoising=True
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
        for d in prefs.devices:d.use=d.type=='OPTIX'
        s.cycles.device='GPU'
    except Exception:s.cycles.device='CPU'
    s.render.image_settings.file_format='PNG'
    for entry in config['cameras']:
        if entry['index'] not in frames:continue
        s.camera=camera(entry)
        s.render.filepath=str(out/'renders'/f"{tag}_{entry['index']:03d}.png")
        bpy.ops.render.render(write_still=True)


def render_labels(config,out,tag,frames):
    """Occlusion-aware instance silhouettes, without lighting or textures."""
    s=bpy.context.scene;restore=[];colors={}
    assets=[e['id'] for e in config.get('assets',[])+config.get('additions',[])]
    for o in s.objects:
        if o.type not in {'MESH','CURVE','FONT'}:continue
        identity=o.get('source_instance',o.get('instance',o.name))
        if identity.endswith('_label'):identity=identity[:-6]
        rgb=[0,0,0]
        if identity in assets:
            raw=hashlib.sha256(identity.encode()).digest();rgb=[40+v%190 for v in raw[:3]];colors[identity]=rgb
        key='MASK '+identity
        m=bpy.data.materials.get(key) or bpy.data.materials.new(key);m.use_nodes=True;m.node_tree.nodes.clear()
        emission=m.node_tree.nodes.new('ShaderNodeEmission');value=np.array(rgb)/255;value=np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)
        emission.inputs['Color'].default_value=(*value,1)
        target=m.node_tree.nodes.new('ShaderNodeOutputMaterial');m.node_tree.links.new(emission.outputs[0],target.inputs['Surface'])
        restore.append((o,list(o.data.materials),[p.material_index for p in o.data.polygons] if o.type=='MESH' else None))
        for slot in o.material_slots:slot.material=m
        if not len(o.material_slots):o.data.materials.append(m)
    view=(s.view_settings.view_transform,s.view_settings.look,s.view_settings.exposure,s.view_settings.gamma)
    s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=0;s.view_settings.gamma=1
    denoise=s.cycles.use_denoising;s.cycles.use_denoising=False;s.cycles.samples=8
    for e in config['cameras']:
        if e['index'] not in frames:continue
        s.camera=camera(e);s.render.filepath=str(out/'renders'/f"{tag}_labels_{e['index']:03d}.png");bpy.ops.render.render(write_still=True)
    for o,mats,indices in restore:
        o.data.materials.clear()
        for m in mats:o.data.materials.append(m)
        if indices is not None:
            for p,index in zip(o.data.polygons,indices):p.material_index=index
    s.view_settings.view_transform=view[0];s.view_settings.look=view[1];s.view_settings.exposure=view[2];s.view_settings.gamma=view[3];s.cycles.use_denoising=denoise
    (out/'label_colors.json').write_text(json.dumps(colors,indent=2),encoding='utf8')


def belongs(o,name):
    return o.get('source_instance')==name or o.get('instance')==name or o.name.startswith(name+'__')


def delete_asset(name,keep_screen=False):
    for o in list(bpy.data.objects):
        if o.type not in {'MESH','CURVE','FONT'} or not belongs(o,name):continue
        if keep_screen and ('screen' in o.name or 'display' in o.name):continue
        bpy.data.objects.remove(o,do_unlink=True)


def add_can(e,out):
    a=Asset(e,out);w,h,l=a.d
    a.cylinder('body',[0,0,0],[w,h,l],material('V2 red can',[151,12,27],.38))
    for side in [-1,1]:a.cylinder('rim_'+str(side),[0,side*h*.49,0],[w*.94,.003,l*.94],a.steel)
    a.ellipsoid('tab',[0,h*.51,0],[.010,.002,.018],a.steel)
    return a


def observed_can_label(e,out,texture,camera_entry):
    a=Asset(dict(e,id=e['id']+'_label'),out);w,h,l=a.d
    pos=np.array(camera_entry['c2w'])[:3,3]
    view=a.r.T@(pos-a.c);theta=math.atan2(view[0],view[2])
    vertices=[]
    for y in [-h*.40,h*.39]:
        for j in range(33):
            angle=theta+(j/32-.5)*math.pi*.70
            vertices.append([(w*.5+.0003)*math.sin(angle),y,(l*.5+.0003)*math.cos(angle)])
    faces=[(j,j+1,34+j,33+j) for j in range(32)]
    m=material('V2 observed '+e['id'],[255,255,255],.45,texture=texture)
    o=a.mesh('observed_label_sector',vertices,faces,m)
    uv=o.data.uv_layers.new()
    for p in o.data.polygons:
        for li in p.loop_indices:
            vi=o.data.loops[li].vertex_index;uv.data[li].uv=(vi%33/32,vi//33)
    o['appearance_provenance']='Visible source patch projected to front sector; rear remains plain, no invented text'


def appearance(config,out):
    xyz=config['sample']=='fr2_xyz'
    # Source-derived laminate patch. It carries baked lighting and is marked as such.
    table=material('V2 observed laminate',[255,255,255],.70,texture=out/'textures/table.png')
    for o in list(bpy.data.objects):
        if o.type!='MESH':continue
        is_top=(o.name.startswith('desk_') and '__top' in o.name) if xyz else (o.name.startswith('surface_table') and '_top' in o.name)
        if is_top:
            o.data.materials.clear();o.data.materials.append(table)
            uv=o.data.uv_layers.active or o.data.uv_layers.new()
            for p in o.data.polygons:
                for li in p.loop_indices:
                    v=o.matrix_world@o.data.vertices[o.data.loops[li].vertex_index].co
                    uv.data[li].uv=(v.x/.65,v.y/.65)
            o['appearance_provenance']='Unoccluded tabletop crop; repeated outside observed patch'
        if xyz and o.name.startswith('plant__leaf'):
            green=material('V2 observed leaf green',[49,76,23],.72)
            o.data.materials.clear();o.data.materials.append(green)
            # The elongated-leaf experiment regressed all checked silhouettes.
            # Keep the original geometry unless explicitly running that experiment.
            if config.get('leaf_shape_experiment',False):
                for v in o.data.vertices:
                    radius=math.hypot(v.co.x,v.co.z)
                    v.co.y-=.35*radius*radius
                    v.co.y=.82+(v.co.y-.82)*1.40
                    v.co.x*=.88;v.co.z*=.88
                o.location.x-=.025
        if xyz and o.name.startswith('teddy__') and any(x in o.name for x in ['head','body','arm','ear','foot']):
            fur=material('V2 observed teddy brown',[67,30,12],.95)
            if not fur.node_tree.nodes.get('Fur microvariation'):
                tex=fur.node_tree.nodes.new('ShaderNodeTexNoise');tex.name='Fur microvariation';tex.inputs['Scale'].default_value=180
                bump=fur.node_tree.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.18;bump.inputs['Distance'].default_value=.001
                fur.node_tree.links.new(tex.outputs['Fac'],bump.inputs['Height']);fur.node_tree.links.new(bump.outputs['Normal'],fur.node_tree.nodes.get('Principled BSDF').inputs['Normal'])
            o.data.materials.clear();o.data.materials.append(fur)
        if xyz and o.name.startswith('teddy__ear'):
            o.scale*=.77
            o.location.z+=.025
        if xyz and o.name=='teddy__head':
            o.scale.x*=1.15;o.scale.y*=1.08;o.scale.z*=1.15;o.location.z+=.025
        if xyz and o.name=='teddy__body':
            o.scale.z*=1.35;o.location.z+=.075
        if xyz and o.name.startswith('teddy__arm'):
            o.location.z+=.065
        if xyz and o.name=='teddy__chair_back':
            # Prior support protruded above the observed bear; it has no video evidence.
            o.hide_render=True;o.hide_set(True)
        if xyz and o.name.startswith('teddy__') and any(x in o.name for x in ['muzzle','nose','eye']):
            o.location.x+=.025
        if xyz and o.name=='floor__slab':
            floor=material('V2 observed concrete',[103,99,96],.94)
            o.data.materials.clear();o.data.materials.append(floor)
            for v in o.data.vertices:
                v.co.x*=3;v.co.z*=3
            uv=o.data.uv_layers.active or o.data.uv_layers.new()
            for p in o.data.polygons:
                for li in p.loop_indices:
                    v=o.matrix_world@o.data.vertices[o.data.loops[li].vertex_index].co
                    uv.data[li].uv=(v.x/2,v.y/2)
    if xyz:
        cams={c['index']:c for c in config['cameras']}
        for e in config['assets']+config['additions']:
            if e['id'] in ['can_red','can_red_back','can_rear']:
                for o in list(bpy.data.objects):
                    if o.name.startswith(e['id']+'__label_'):bpy.data.objects.remove(o,do_unlink=True)
                key='can_rear' if e['id']=='can_rear' else 'can_red'
                observed_can_label(e,out,out/f'textures/{key}.png',cams[80 if key=='can_rear' else 0])
        e=next(e for e in config['assets'] if e['id']=='book_upright')
        a=Asset(dict(e,id='book_upright_observed'),out)
        # Same measured plane as the old volume, with a source-derived face.
        a.decal('printed_spine',[a.d[0]*.33,0,a.d[2]/2+.0005],[a.d[0]*.34,a.d[1]],out/'textures/upright.png',axis='front')


def project_surface_textures(config,out):
    """Project source pixels only onto visible top control surfaces, not whole assets."""
    cams={c['index']:c for c in config['cameras']}
    frame_by_id=({'keyboard':159,'telephone':159} if config['sample']=='fr2_xyz' else {'laptop':80,'keyboard_front':40,'keyboard_reverse':80,'keyboard_rear':60})
    for identity,index in frame_by_id.items():
        c=cams[index];t=np.array(c['c2w']);fx,fy,cx,cy=c['intrinsics']
        mat=material('V2 video key legends '+identity,[255,255,255],.8,texture=out/f'evidence/frame_{index:03d}.png')
        for o in bpy.data.objects:
            if o.type!='MESH' or not belongs(o,identity):continue
            if not ('__key_' in o.name or '__dial_' in o.name or '__directory_panel' in o.name):continue
            o.data.materials.append(mat);slot=len(o.data.materials)-1
            uv=o.data.uv_layers.active or o.data.uv_layers.new()
            for p in o.data.polygons:
                if p.normal.y<.8:continue
                p.material_index=slot
                for li in p.loop_indices:
                    v=np.array(o.matrix_world@o.data.vertices[o.data.loops[li].vertex_index].co)@C
                    q=(v-t[:3,3])@t[:3,:3];u=fx*q[0]/q[2]+cx;v=fy*q[1]/q[2]+cy
                    uv.data[li].uv=(u/640,1-v/480)
            o['appearance_provenance']=f'Key top source projection from frame {index}; source shading baked; side faces remain material'


def add_clutter(config,out):
    changes=[]
    for e in config.get('clutter',[]):
        a=Asset(e,out);w,h,l=a.d;kind=e['template']
        if kind=='pen_cup':
            a.cylinder('cup',[0,0,0],[w,h,l])
            a.cylinder('dark_interior',[0,h*.50,0],[w*.85,.001,l*.85],a.dark)
            for j,(x,z,length) in enumerate([(-.018,.01,.16),(.012,-.012,.19),(.005,.018,.14),(-.013,-.017,.18),(.018,.013,.17)]):
                col=material('V2 pen '+str(j),[[171,22,20],[32,63,128],[194,165,33],[37,83,41],[23,25,26]][j],.5)
                a.curve('pen_'+str(j),[[x,-h*.3,z],[x*1.3,-h*.3+length,z*1.3]],.003,col)
        elif kind in ['carton','photo_card']:
            a.box('body',[0,0,0],[w,h,l],bevel=.0008)
            a.decal('observed_print',[0,0,l/2+.0003],[w*.98,h*.98],out/f"textures/{e['id']}.png",axis='front')
        elif kind=='accessory':
            a.loft('red_body',[(-h/2,w*.9,l*.9,0),(0,w,l,0),(h/2,w*.83,l*.86,0)],exponent=3)
            a.cylinder('blue_ring',[0,h*.52,0],[w*.62,.003,l*.62],material('V2 accessory blue',[25,96,151],.5))
            a.cylinder('center',[0,h*.60,0],[w*.20,.002,l*.20],a.dark)
        elif kind=='power_strip':
            a.box('body',[0,0,0],[w,h,l],bevel=.009)
            for j in range(5):
                x=(j-2)*w*.18
                a.cylinder('socket_'+str(j),[x,h*.5,0],[.038,.001,.038],material('V2 socket recess',[155,151,143],.8))
                for side in [-1,1]:a.cylinder(f'hole_{j}_{side}',[x+side*.007,h*.53,0],[.004,.001,.004],a.dark)
        changes.append(e['id'])
    for e in config.get('paper_quads',[]):
        vertices=np.array(e['vertices']);center=vertices.mean(0)
        a=Asset(dict(id=e['id'],center=center.tolist(),dimensions=[1,.001,1],rotation=np.eye(3).tolist(),color=[225,222,213],source=e['source'],evidence_frames=[e['frame']]),out)
        m=material('V2 printed '+e['id'],[255,255,255],.9,texture=out/e['texture'])
        o=a.mesh('sheet',(vertices-center).tolist(),[(0,3,2,1)],m,smooth=False)
        uv=o.data.uv_layers.new();coords=[(0,1),(1,1),(1,0),(0,0)]
        for li in o.data.polygons[0].loop_indices:uv.data[li].uv=coords[o.data.loops[li].vertex_index]
        s=o.modifiers.new('Paper thickness','SOLIDIFY');s.thickness=.00025
        changes.append(e['id'])
    for e in config.get('cables',[]):
        a=Asset(dict(id=e['id'],center=[0,0,0],dimensions=[1,1,1],rotation=np.eye(3).tolist(),color=[15,15,15],source=e['source']),out)
        a.curve('trace',e['points'],radius=e['radius'],world=True)
        changes.append(e['id'])
    return changes


def refine(config,out):
    changes=[]
    for name,f in config.get('panel_updates',{}).items():
        if name!='laptop_screen':continue
        old=f['original'];r0=np.array(old['rotation']);r1=np.array(f['rotation'])
        c0=np.array(old['center']);c1=np.array(f['center']);scale=np.array(f['dimensions'])/old['dimensions']
        a=np.eye(4);a[:3,:3]=C@r1@np.diag(scale)@r0.T@C.T;a[:3,3]=C@c1-a[:3,:3]@C@c0
        for o in bpy.data.objects:
            if belongs(o,'laptop') and ('screen' in o.name or 'display' in o.name):o.matrix_world=Matrix(a.tolist())@o.matrix_world
    for e0 in config['assets']:
        e=dict(e0,table_height=config['table_height'])
        if e['template'] not in {'keyboard','laptop','telephone','mouse','monitor','gamepad','apparatus'}:continue
        delete_asset(e['id'],keep_screen=e['template']=='laptop')
        a=Asset(e,out);a.build();changes.append(e['id'])
    for e in config['additions']:
        if e['template']=='can':add_can(e,out);changes.append(e['id'])
    appearance(config,out)
    project_surface_textures(config,out)
    changes.extend(add_clutter(config,out))
    # Exposure is calibrated as a scene-wide appearance parameter, not per object.
    bpy.context.scene.view_settings.exposure=-.65
    return changes


def export(config,out,changes):
    s=bpy.context.scene
    meshes=[o for o in s.objects if o.type in {'MESH','CURVE','FONT'} and not o.hide_render and not any(c.hide_render for c in o.users_collection)]
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.hide_set(False);o.select_set(True)
        if o.parent:o.parent.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True,export_cameras=False,export_lights=False)
    overview=bpy.data.objects.get('Overview') or next((o for o in s.objects if o.type=='CAMERA' and 'Overview' in o.name),None)
    if overview:s.camera=overview
    s.render.pixel_aspect_x=1;s.render.pixel_aspect_y=1
    s.render.resolution_x=1200;s.render.resolution_y=900;s.render.filepath=str(out/'overview.png')
    bpy.ops.render.render(write_still=True)
    bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
    native=out/(config['sample']+'_refined.blend');bpy.ops.wm.save_as_mainfile(filepath=str(native),compress=True)
    meta=[]
    for o in meshes:
        points=np.array([o.matrix_world@Vector(p) for p in o.bound_box])@C
        meta.append(dict(id=o.name,parent_id=o.get('source_instance',o.get('instance',o.parent.name if o.parent else o.name)),center=points.mean(0).tolist(),dimensions=np.ptp(points,axis=0).tolist()))
    (out/'scene.json').write_text(json.dumps(dict(objects=meta,cameras=config['all_cameras'],stats=dict(model_parts=len(meshes)),editable_source=native.name),indent=2),encoding='utf8')
    report=dict(native=str(native),modified_instances=changes,parts=len(meshes),modifiers=sum(len(o.modifiers) for o in meshes),images=[im.name for im in bpy.data.images if im.packed_file],source_unchanged=digest(Path(config['source_blend']))==config['source_blend_sha256'])
    (out/'build_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('REFINEMENT_COMPLETE',json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--samples',type=int,default=24);p.add_argument('--frames',default='0,80,159');p.add_argument('--baseline',action='store_true');p.add_argument('--tag',default='after');p.add_argument('--render-only',action='store_true');p.add_argument('--masks',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);config=json.loads(a.config.read_text(encoding='utf8'));out=Path(config['output'])
    source=Path(config['source_blend'])
    if digest(source)!=config['source_blend_sha256']:raise ValueError('Baseline has changed')
    bpy.ops.wm.open_mainfile(filepath=str(source))
    frames={int(x) for x in a.frames.split(',')}
    changes=[] if a.baseline else refine(config,out)
    render_views(config,out,'before' if a.baseline else a.tag,a.samples,frames)
    if a.masks:
        render_labels(config,out,'before' if a.baseline else a.tag,frames)
        bpy.context.scene.cycles.samples=a.samples
    if not a.baseline and not a.render_only:export(config,out,changes)


if __name__=='__main__':main()
