"""Reopen native deliverable, verify finite evaluated geometry and packed textures."""
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

root=Path(bpy.data.filepath).parent
objects=[o for o in bpy.context.scene.objects if o.type in {'MESH','CURVE'} and not o.hide_render and not any(c.hide_render for c in o.users_collection)]
deps=bpy.context.evaluated_depsgraph_get();bad=[];vertices=0;triangles=0
for o in objects:
    evaluated=o.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
    xyz=np.array([v.co for v in mesh.vertices]);vertices+=len(xyz);triangles+=len(mesh.loop_triangles)
    if not len(xyz) or not np.isfinite(xyz).all():bad.append(o.name)
    evaluated.to_mesh_clear()
missing=[im.name for im in bpy.data.images if im.source=='FILE' and not im.packed_file]
assert objects and not bad and not missing,(bad,missing)
report=dict(opened_file=bpy.data.filepath,parts=len(objects),evaluated_vertices=vertices,evaluated_triangles=triangles,invalid_meshes=bad,unpacked_images=missing,editable_modifiers=sum(len(o.modifiers) for o in objects),editable_curves=sum(o.type=='CURVE' for o in objects),cameras=[o.name for o in bpy.context.scene.objects if o.type=='CAMERA'],scope='File/geometry validation only. Visual fidelity is reviewed separately.')
config_path=root/'refinement.json' if (root/'refinement.json').exists() else root/'recipe.json'
if config_path.exists():
    config=json.loads(config_path.read_text(encoding='utf8'));errors=[];C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    for c in config['cameras']:
        cam=bpy.data.objects.get(f"Evidence {c['index']:03d}")
        if cam is None:continue
        w,h=c.get('resolution',[640,480]);fx,fy,cx,cy=c['intrinsics'];scene=bpy.context.scene
        scene.render.resolution_x=w;scene.render.resolution_y=h;scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=fx/fy
        t=np.array(c['c2w'])
        for q in [np.array([.12,.08,1.3]),np.array([-.25,-.14,2.1])]:
            world=C@(t[:3,:3]@q+t[:3,3]);ndc=world_to_camera_view(scene,cam,Vector(world))
            actual=np.array([ndc.x*w,(1-ndc.y)*h]);expected=q[:2]/q[2]*[fx,fy]+[cx,cy]
            errors.append(float(np.linalg.norm(actual-expected)))
    report['camera_projection_max_px']=max(errors,default=0)
    assert report['camera_projection_max_px']<.01,report['camera_projection_max_px']
    contact={}
    for e in config.get('assets',[]):
        if e['template'] not in ['keyboard','laptop','laptop_assembly','telephone','mouse']:continue
        suffix={'keyboard':'tapered_lower_case','laptop':'tapered_lower_case','laptop_assembly':'tapered_lower_case','telephone':'wedge_case','mouse':'flat_base'}[e['template']]
        ob=bpy.data.objects.get(e['id']+'__'+suffix)
        if ob is None:continue
        r=np.array(e['rotation']);dims=np.array(e['dimensions']);expected=e['center'][1]-float(abs(r[1])@dims/2)
        fitted=e.get('fitted_parts',{}).get('screen')
        if e['template']=='laptop_assembly' and fitted:
            panel_r=np.array(fitted['rotation']);panel_c=np.array(fitted['center']);panel_h=fitted['dimensions'][1]
            thickness=e.get('parameters',{}).get('base_thickness',min(.022,dims[1]*.13))
            expected=(panel_c-panel_r[:,1]*panel_h/2)[1]-thickness*1.05
        evaluated=ob.evaluated_get(deps);mesh=evaluated.to_mesh()
        bottom=min((ob.matrix_world@v.co).z for v in mesh.vertices);evaluated.to_mesh_clear()
        contact[e['id']]=dict(expected_support_y=expected,actual_bottom_y=bottom,error_m=abs(bottom-expected))
    report['asset_bottom_checks']=contact
    assert all(x['error_m']<.006 for x in contact.values()),contact
    missing_parts=[]
    expected={'laptop_assembly':['tapered_lower_case','display_back','display_glass'],
              'chair':['seat_cushion','back_frame','arm_-1','arm_1'],
              'bottle':['bottle_body','cap'],'cup':['cup_body','cup_rim'],
              'backpack':['fabric_body','strap_-1','strap_1']}
    for e in config.get('assets',[]):
        for part in expected.get(e['template'],[]):
            if bpy.data.objects.get(e['id']+'__'+part) is None:missing_parts.append(e['id']+'__'+part)
    report['missing_required_parts']=missing_parts
    assert not missing_parts,missing_parts
(root/'native_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
