"""Geometry-neutral validation, four-view renders and original-camera renders.

Copied into each fresh modelling workspace. Contains no object geometry.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_tools import setup,camera


def main(contract_path,output):
    contract=json.loads(contract_path.read_text(encoding='utf8'))
    output.mkdir(parents=True,exist_ok=True)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene
    original=[o for o in scene.objects if o.type in {'MESH','CURVE','SURFACE','FONT'} and not o.hide_render]
    if not original:raise ValueError('No authored geometry')
    deps=bpy.context.evaluated_depsgraph_get();baked=[]
    for ob in original:
        ev=ob.evaluated_get(deps)
        data=bpy.data.meshes.new_from_object(ev,preserve_all_data_layers=True,depsgraph=deps)
        data.transform(ev.matrix_world)
        new=bpy.data.objects.new('Fresh '+ob.name,data);scene.collection.objects.link(new)
        new['fresh_instance']=contract['instance_id'];new['part_name']=ob.name
        baked.append(new)
    for ob in list(scene.objects):
        if ob not in baked:bpy.data.objects.remove(ob,do_unlink=True)
    points=np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in baked])
    if not np.isfinite(points).all():raise ValueError('Non-finite geometry')
    lo,hi=points.min(0),points.max(0);target=np.array(contract['local_bounds']);span=hi-lo;expected=target[1]-target[0]
    if (span<1e-7).any():raise ValueError('Degenerate asset bounds')
    error=np.abs(span/expected-1)
    report=dict(instance_id=contract['instance_id'],raw_bounds=[lo.tolist(),hi.tolist()],target_bounds=target.tolist(),
                dimension_relative_error=error.tolist(),parts=[o['part_name'] for o in baked],vertices=len(points))
    (output/'raw_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    # Small fitting adjustment only. A wrong-sized object must be corrected by its author.
    if error.max()>.12:raise ValueError('Asset size differs by more than 12%; fix script to contract bounds: '+str(error.tolist()))
    factor=expected/span
    for ob in baked:
        for v in ob.data.vertices:v.co=Vector(((np.array(v.co)-lo)*factor+target[0]).tolist())
    report['fit_scale']=factor.tolist()
    report['fit_translation']=(target[0]-lo*factor).tolist()
    report['normalization']='Small explicit local bounds fit; accepted scene transform is never adjusted.'
    bpy.context.view_layer.update();bpy.ops.file.pack_all()
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'asset_validated.blend'),compress=True)
    report['packed_images']=[im.name for im in bpy.data.images if im.packed_file]
    report['unpacked_images']=[im.name for im in bpy.data.images if im.source=='FILE' and not im.packed_file]
    if report['unpacked_images']:raise ValueError('Unpacked image dependencies')
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    cameras=contract['cameras'];setup(cameras);scene.cycles.samples=16
    center=target.mean(0);size=max(expected.max(),.05)
    for index,angle in enumerate([45,135,225,315]):
        a=math.radians(angle);cam=bpy.data.objects.new('Turntable '+str(index),bpy.data.cameras.new('Turntable '+str(index)))
        scene.collection.objects.link(cam);cam.location=center+np.array([math.cos(a),math.sin(a),.85])*size*2
        cam.rotation_euler=(Vector(center)-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.type='ORTHO';cam.data.ortho_scale=size*1.5;scene.camera=cam
        scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
        scene.render.filepath=str(output/f'turn_{index}.png');bpy.ops.render.render(write_still=True)
    # Capture-pose previews assess observed shape without undoing the user's design edits.
    for ob in baked:ob.matrix_world=Matrix(contract['observation_matrix'])
    for entry in cameras:
        camera(entry);scene.render.filepath=str(output/f'source_{entry["frame"]:03d}.png');bpy.ops.render.render(write_still=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,default=Path('contract.json'));p.add_argument('--output',type=Path,default=Path('preview'))
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);main(a.contract.resolve(),a.output.resolve())
