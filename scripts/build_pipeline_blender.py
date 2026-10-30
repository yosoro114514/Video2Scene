"""Build a first editable scene from source-bound automatic or reviewed assets."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from video2scene.blender_assets import Asset,C,material
from video2scene.automatic_assets import AutomaticAsset
from refine_blender import render_views,render_labels

p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--samples',type=int,default=24)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);config=json.loads(a.recipe.read_text(encoding='utf8'));out=Path(config['output'])
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.unit_settings.system='METRIC'
for e in config['assets']:
    asset=AutomaticAsset(e,out)
    if not asset.build():
        shape=e.get('primitive','box')
        if shape=='prism':
            footprint=e['footprint'];n=len(footprint)
            vertices=[[x,y,z] for y in [-asset.d[1]/2,asset.d[1]/2] for x,z in footprint]
            faces=[tuple(range(n)),tuple(reversed(range(n,2*n)))]+[(i,i+n,(i+1)%n+n,(i+1)%n) for i in range(n)]
            asset.mesh('observed_plane_footprint',vertices,faces,smooth=False)
        elif shape=='cylinder':asset.cylinder('observed_proxy',[0,0,0],asset.d)
        elif shape=='ellipsoid':asset.ellipsoid('observed_proxy',[0,0,0],asset.d)
        else:asset.box('observed_proxy',[0,0,0],asset.d,bevel=min(.003,min(asset.d)*.05))
model=[o for o in scene.objects if o.type in {'MESH','CURVE'}]
points=np.vstack([np.array([o.matrix_world@Vector(v) for v in o.bound_box]) for o in model]) if model else None
if points is None:raise ValueError('No supported geometry in scene')
center=np.median(points,axis=0);size=max(np.ptp(points,axis=0).max(),1)
for name,offset,power in [('Key',[-1,-1,2],500),('Fill',[1,0,1.5],260)]:
    light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
    obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=center+np.array(offset)*size
    obj.rotation_euler=(Vector(center)-obj.location).to_track_quat('-Z','Y').to_euler()
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
overview=bpy.data.objects.new('Overview',bpy.data.cameras.new('Overview'));scene.collection.objects.link(overview)
overview.location=center+np.array([1,-1,.85])*size;overview.rotation_euler=(Vector(center)-overview.location).to_track_quat('-Z','Y').to_euler();overview.data.type='ORTHO';overview.data.ortho_scale=size*1.4
render_views(config,out,'after',a.samples,set(config['review_frames']))
render_labels(config,out,'after',set(config['review_frames']))
scene.cycles.samples=a.samples
scene.camera=overview;scene.render.resolution_x=1200;scene.render.resolution_y=900;scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1;scene.render.filepath=str(out/'overview.png');bpy.ops.render.render(write_still=True)
bpy.ops.object.select_all(action='DESELECT')
for o in model:o.select_set(True);o.parent.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True)
if Path(config['scan']).exists():
    before=set(scene.objects);bpy.ops.import_scene.gltf(filepath=config['scan'])
    col=bpy.data.collections.new('REFERENCE scan');scene.collection.children.link(col)
    for o in set(scene.objects)-before:
        for c in list(o.users_collection):c.objects.unlink(o)
        col.objects.link(o);o.hide_render=True;o.hide_set(True)
    col.hide_render=True;col.hide_viewport=True
bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out/'video_scene.blend'),compress=True)
(out/'scene.json').write_text(json.dumps(dict(objects=config['assets'],cameras=config['all_cameras'],editable_source='video_scene.blend',stats=dict(model_parts=len(model))),indent=2),encoding='utf8')
(out/'build_manifest.json').write_text(json.dumps(dict(parts=len(model),assets=len(config['assets']),mode='Codex reviewed assets with measured parts' if config.get('review_backend') else 'automatic proxies and category assets',limitations=config['limitations']),indent=2),encoding='utf8')
