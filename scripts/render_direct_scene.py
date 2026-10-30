"""Render/export GPT-authored Blender geometry under frozen source cameras."""
import argparse
import json
import re
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from refine_blender import render_views,render_labels

p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=32)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);config=json.loads(a.recipe.read_text(encoding='utf8'));out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
config['baseline_dir']=config['output']
(out/'renders').mkdir(exist_ok=True)
config['output']=str(out);config['sample']='video_scene_codex';config['review_backend']='codex-direct'
config['limitations']=['Geometry was directly authored and executed through the Codex ChatGPT subscription.',
                      'Fixed ViPE cameras and monocular scale are estimates; native projection checks do not measure reconstruction accuracy.',
                      'Visible shapes and materials are interpreted from video; unseen geometry remains inferred. See actual independent visual reviews.']
model=[o for o in bpy.context.scene.objects if o.type in {'MESH','CURVE'} and not o.hide_render and not any(c.hide_render for c in o.users_collection)]
if not model:raise ValueError('GPT candidate has no visible geometry')
groups={};C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
for o in model:
    identity=re.sub(r'[^A-Za-z0-9_-]','_',str(o.get('source_instance',o.get('instance',o.name.split('__')[0]))))[:90] or 'unnamed_object'
    o['source_instance']=identity;o['instance']=identity
    points=np.array([o.matrix_world@Vector(v) for v in o.bound_box])@C
    groups.setdefault(identity,[]).append(points)
assets=[]
for identity,boxes in groups.items():
    points=np.vstack(boxes);lo=points.min(0);hi=points.max(0)
    assets.append(dict(id=identity,template='direct_mesh',center=((lo+hi)/2).tolist(),dimensions=np.maximum(hi-lo,.001).tolist(),rotation=np.eye(3).tolist(),color=[128,128,128],source='GPT-authored bpy geometry; evaluated bounds for inspection only',evidence_frames=config['review_frames']))
config['assets']=assets;config['additions']=[]
render_views(config,out,'after',a.samples,set(config['review_frames']));render_labels(config,out,'after',set(config['review_frames']))
scene=bpy.context.scene;scene.cycles.samples=a.samples
overview=bpy.data.objects.get('Overview')
if overview:
    scene.camera=overview;scene.render.resolution_x=1200;scene.render.resolution_y=900;scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1;scene.render.filepath=str(out/'overview.png');bpy.ops.render.render(write_still=True)
bpy.ops.object.select_all(action='DESELECT')
for o in model:
    o.hide_set(False);o.select_set(True)
    parent=o.parent
    while parent:parent.select_set(True);parent=parent.parent
bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True)
bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(out/'video_scene.blend'),compress=True)
(out/'recipe.json').write_text(json.dumps(config,indent=2),encoding='utf8')
(out/'scene.json').write_text(json.dumps(dict(objects=assets,cameras=config['all_cameras'],editable_source='video_scene.blend',stats=dict(model_parts=len(model))),indent=2),encoding='utf8')
(out/'build_manifest.json').write_text(json.dumps(dict(mode='GPT directly authored and executed bpy; host only renders and exports',parts=len(model),instances=len(assets)),indent=2),encoding='utf8')
