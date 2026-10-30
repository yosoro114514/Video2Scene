"""Make an input package containing white geometry ONLY. Executed by Blender."""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from simple_direct_blender_tools import setup
from whitebox_loop_blender import render_view


def main(out,job):
    state=json.loads((job/'accepted_scene.json').read_text(encoding='utf8'))
    bpy.ops.wm.open_mainfile(filepath=str(out/'assets/assets.blend'))
    bpy.context.preferences.filepaths.save_version=0
    for ob in list(bpy.context.scene.objects):
        if ob.type in {'MESH','CURVE'} and ob.get('variant')!='white':
            bpy.data.objects.remove(ob,do_unlink=True)
    # Remove archived textures, embedded scripts, and orphaned fine geometry.
    for text in list(bpy.data.texts):bpy.data.texts.remove(text)
    for image in list(bpy.data.images):bpy.data.images.remove(image)
    bpy.data.orphans_purge(do_recursive=True)
    table=Matrix(state['objects']['table_00']['base_matrix']).to_3x3()
    all_points=[]
    proof={'fine_objects':[],'images':[],'objects':{}}
    import math
    for root in [o for o in bpy.context.scene.objects if o.get('instance_id')]:
        key=root['instance_id'];obj=state['objects'][key]
        base=Matrix(obj['base_matrix']);base.translation+=table@Vector((*obj['offset_m'],0))
        root.matrix_world=base@Matrix.Rotation(math.radians(obj['yaw_deg']),4,'Z')@Matrix.Diagonal((*obj['scale'],1))
        root['asset_id']='white/'+key+'/v1'
        parts=[];local=[]
        for ob in root.children:
            ob.hide_render=False;ob.hide_set(False)
            parts.append(ob.name)
            local.extend([list(v.co) for v in ob.data.vertices])
        proof['objects'][key]={'parts':parts,'vertices':len(local),'accepted_matrix':[list(r) for r in root.matrix_world]}
        object_dir=job/'inputs'/key;object_dir.mkdir(parents=True,exist_ok=True)
        (object_dir/'white_vertices.json').write_text(json.dumps(local),encoding='utf8')
        all_points.extend([list(root.matrix_world@Vector(v)) for v in local])
    bpy.context.view_layer.update()
    cameras=json.loads((out/'cameras.json').read_text())
    setup(cameras)
    proof['fine_objects']=[o.name for o in bpy.context.scene.objects if o.get('variant')=='fine']
    proof['images']=[im.name for im in bpy.data.images if im.source=='FILE']
    assert not proof['fine_objects'] and not proof['images']
    (job/'white_input_proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8')
    bpy.ops.wm.save_as_mainfile(filepath=str(job/'white_reference.blend'),compress=True)
    p=np.array(all_points);center=(p.min(0)+p.max(0))/2;size=np.ptp(p,axis=0).max()
    render_view(job,'white_overview.png',center+np.array([.9,-1.25,1.45])*size,center,size*1.4)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);main(a.output,a.job)
