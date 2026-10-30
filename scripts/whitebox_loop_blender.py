"""Compile versioned scene data into native Blender/GLB. Run inside Blender."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from simple_direct_blender_tools import setup, camera

spec = importlib.util.spec_from_file_location('aha_semantics', ROOT / 'third_party/aha3d_semantics/semantics.py')
semantics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantics)
LABELS = {'t000': '白碗', 't001': '红色马克杯', 't002': '易拉罐', 't003': '麦片盒', 'table_00': '圆桌面'}
CLASSES = {'t000': 'prop/tableware/bowl', 't001': 'prop/tableware/mug', 't002': 'prop/container/can', 't003': 'prop/container/carton', 'table_00': 'furniture/table'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def mat(name, color, rough=.5):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bs = m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, 1)
    bs.inputs['Roughness'].default_value = rough
    return m


def mesh(name, vertices, faces, parent, material):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    ob = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.data.materials.append(material)
    return ob


def lathe(name, profile, parent, material, n=20):
    vs = [(r*math.cos(i*2*math.pi/n), r*math.sin(i*2*math.pi/n), z) for r,z in profile for i in range(n)]
    fs = [(j*n+i, j*n+(i+1)%n, (j+1)*n+(i+1)%n, (j+1)*n+i) for j in range(len(profile)-1) for i in range(n)]
    return mesh(name, vs, fs, parent, material)


def initialize(out, source):
    bpy.ops.wm.open_mainfile(filepath=str(source / 'geometry_guided/result/scene.blend'))
    bpy.context.preferences.filepaths.save_version = 0
    params = read(source / 'geometry_guided/author/fit_parameters.json')
    white = mat('Whitebox • matte porcelain', (.75,.78,.77), .72)
    table_mat = mat('Whitebox • warm neutral support', (.54,.57,.55), .8)
    roots = {o.get('source_instance'):o for o in bpy.context.scene.objects if o.type == 'EMPTY' and o.get('source_instance') in LABELS}
    assert set(roots) == set(LABELS), roots
    scene_data = {'schema_version': 'whitebox-loop/1', 'coordinate_system': 'Blender Z-up; offsets in table-local XY',
                  'units': 'estimated_monocular_metres', 'objects': {}, 'evidence_frames': [0,40,79],
                  'source': 'Existing GPT-authored geometry-guided UW tabletop; no new geometry inference',
                  'asset_note': 'Fine assets are archived from the previous GPT-generated scene; this run verifies replacement, not a new fine reconstruction.'}
    for identity, root in roots.items():
        fine = []
        old = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.get('source_instance') == identity]
        deps = bpy.context.evaluated_depsgraph_get()
        for ob in old:
            ev = ob.evaluated_get(deps)
            data = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=deps)
            data.transform(root.matrix_world.inverted() @ ev.matrix_world)
            dup = bpy.data.objects.new('Fine ' + identity + ' ' + ob.name, data)
            bpy.context.scene.collection.objects.link(dup)
            dup.parent = root
            dup['variant'] = 'fine'
            dup['source_instance'] = identity
            fine.append(dup)
        for ob in old:
            bpy.data.objects.remove(ob, do_unlink=True)
        points = np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in fine])
        # Explicitly normalize each foreground asset's contact origin to local z=0.
        bottom = float(points[:,2].min()) if identity != 'table_00' else 0
        for ob in fine:
            for v in ob.data.vertices:
                v.co.z -= bottom
        points[:,2] -= bottom
        lo, hi = points.min(0), points.max(0)
        coarse = []
        p = params[identity]
        if identity == 't000':
            r,h = p['radius'],p['height']
            coarse.append(lathe('White bowl shell', [(0,0),(r*.53,0),(r*.75,h*.3),(r,h),(r-.004,h),(r*.7,h*.3),(r*.45,.012),(0,.012)], root, white, 20))
        elif identity == 't001':
            r,h = p['radius'],p['height']
            coarse.append(lathe('White mug shell', [(0,0),(r*.89,0),(r,h),(r-.004,h),(r*.78,.010),(0,.010)], root, white, 16))
            direction = Vector((-.97,-.243,0)).normalized()
            tangent = Vector((-direction.y,direction.x,0))
            vs,fs = [],[]
            for i in range(16):
                a = i*2*math.pi/16
                normal = direction*math.cos(a)+Vector((0,0,math.sin(a)))
                center = direction*(.045+.023*math.cos(a))+Vector((0,0,h*.51+.031*math.sin(a)))
                for j in range(6):
                    b = j*2*math.pi/6
                    vs.append(tuple(center+.0043*(normal*math.cos(b)+tangent*math.sin(b))))
            for i in range(16):
                for j in range(6):
                    fs.append((i*6+j,((i+1)%16)*6+j,((i+1)%16)*6+(j+1)%6,i*6+(j+1)%6))
            coarse.append(mesh('White mug handle',vs,fs,root,white))
        elif identity == 't002':
            r,h = p['radius'],p['height']
            coarse.append(lathe('White can',[(0,0),(r,0),(r,h),(0,h)],root,white,24))
        else:
            for original in fine:
                ob = bpy.data.objects.new('White '+identity+' '+original.name, original.data.copy())
                bpy.context.scene.collection.objects.link(ob)
                ob.parent = root
                ob.data.materials.clear()
                ob.data.materials.append(table_mat if identity == 'table_00' else white)
                for poly in ob.data.polygons:
                    poly.material_index = 0
                coarse.append(ob)
        cp = np.concatenate([np.array([v.co[:] for v in o.data.vertices]) for o in coarse])
        cmin,cmax = cp.min(0),cp.max(0)
        factor = (hi-lo)/np.maximum(cmax-cmin,1e-9)
        for ob in coarse:
            for v in ob.data.vertices:
                v.co = Vector(((np.array(v.co)-cmin)*factor+lo).tolist())
            ob['variant'] = 'white'
            ob['source_instance'] = identity
        semantics.tag_root(root, CLASSES[identity], identity, 'white/'+identity+'/v1', None if identity=='table_00' else 'table_00')
        root['asset_origin'] = 'support_contact'
        if identity == 'table_00':
            semantics.tag_support(root, float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]),0.0)
        scene_data['objects'][identity] = dict(label=LABELS[identity], semantic_class=CLASSES[identity], support_id=None if identity=='table_00' else 'table_00',
            base_matrix=[list(row) for row in root.matrix_world], offset_m=[0.0,0.0], yaw_deg=0.0, scale=[1.0,1.0,1.0],
            asset_id='white/'+identity+'/v1', variant='white', locks=[], local_bounds=[lo.tolist(),hi.tolist()],
            contact_normalization_m=bottom, evidence_frames=[0,20,40,60,79], editable=identity!='table_00')
    write(out/'initial_scene.json',scene_data)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'assets/assets.blend'),compress=True)


def visible_geometry():
    deps = bpy.context.evaluated_depsgraph_get()
    items = []
    for root in semantics.roots():
        for ob in semantics.descendants(root):
            if ob.type != 'MESH' or ob.hide_render:
                continue
            ev = ob.evaluated_get(deps)
            m = ev.to_mesh()
            m.calc_loop_triangles()
            v = np.array([ev.matrix_world @ vert.co for vert in m.vertices],np.float32)
            f = np.array([t.vertices[:] for t in m.loop_triangles],np.int32)
            items.append((root['instance_id'], ob.name, v, f))
            ev.to_mesh_clear()
    return items


def render_view(output, name, position, target, scale, size=(900,680)):
    scene = bpy.context.scene
    cam = bpy.data.objects.new('Review '+name,bpy.data.cameras.new('Review '+name))
    scene.collection.objects.link(cam)
    cam.location = position
    cam.rotation_euler = (Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale
    scene.camera = cam
    scene.render.resolution_x,scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.filepath = str(output/name)
    bpy.ops.render.render(write_still=True)


def compile_scene(out, revision):
    directory = out/'revisions'/revision
    state = read(directory/'scene.json')
    bpy.ops.wm.open_mainfile(filepath=str(out/'assets/assets.blend'))
    bpy.context.preferences.filepaths.save_version = 0
    cameras = read(out/'cameras.json')
    setup(cameras)
    scene = bpy.context.scene
    scene.cycles.samples = 12
    table_rot = Matrix(state['objects']['table_00']['base_matrix']).to_3x3()
    summaries = {}
    # Generated assets are independently authored from clean references. Remove all
    # legacy children for those IDs, including hidden archived fine meshes.
    for root in semantics.roots():
        identity=root['instance_id'];obj=state['objects'][identity]
        if obj['variant']!='generated':
            continue
        asset=(out/obj['asset_file']).resolve()
        if not asset.is_relative_to((out/'jobs').resolve()) or not asset.is_file():
            raise ValueError('Generated asset must be a validated local job artifact')
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=obj['asset_sha256']:
            raise ValueError('Generated asset fingerprint mismatch')
        for child in reversed(semantics.descendants(root)):
            bpy.data.objects.remove(child,do_unlink=True)
        with bpy.data.libraries.load(str(asset),link=False) as (available,loaded):
            loaded.objects=list(available.objects)
        for child in loaded.objects:
            if child is None:continue
            if child.type!='MESH' or child.get('fresh_instance')!=identity:
                raise ValueError('Unexpected object in validated asset')
            scene.collection.objects.link(child)
            child.parent=root;child.matrix_parent_inverse=Matrix.Identity(4)
            child.matrix_basis=Matrix.Identity(4)
            child['variant']='generated';child['source_instance']=identity
        root['asset_provenance']=obj['asset_file']
    bpy.data.orphans_purge(do_recursive=True)
    for root in semantics.roots():
        identity = root['instance_id']
        obj = state['objects'][identity]
        base = Matrix(obj['base_matrix'])
        delta = table_rot @ Vector((*obj['offset_m'],0))
        base.translation += delta
        root.matrix_world = base @ Matrix.Rotation(math.radians(obj['yaw_deg']),4,'Z') @ Matrix.Diagonal((*obj['scale'],1))
        root['asset_id'] = obj['asset_id']
        root['revision'] = revision
        root['locked_fields_json'] = json.dumps(obj['locks'])
        for child in semantics.descendants(root):
            child.hide_render = child.get('variant') != obj['variant']
            child.hide_set(child.hide_render)
        summaries[identity] = {'instance_id':identity,'asset_id':root['asset_id'],'support_id':root.get('support_id'),
            'matrix_world':[list(row) for row in root.matrix_world], 'locked_fields':obj['locks']}
    bpy.context.view_layer.update()
    items = visible_geometry()
    arrays = {}
    part_rows = []
    for i,(identity,name,v,f) in enumerate(items):
        arrays['v'+str(i)], arrays['f'+str(i)] = v,f
        part_rows.append(dict(index=i,identity=identity,name=name,vertices=len(v),triangles=len(f)))
    np.savez_compressed(directory/'geometry.npz',**arrays)
    write(directory/'geometry_parts.json',part_rows)
    for identity,record in summaries.items():
        p = np.concatenate([v for key,_,v,_ in items if key==identity])
        local = (np.c_[p,np.ones(len(p))] @ np.linalg.inv(np.array(record['matrix_world'])).T)[:,:3]
        record.update(world_bounds=[p.min(0).tolist(),p.max(0).tolist()],local_bounds=[local.min(0).tolist(),local.max(0).tolist()],
                      vertices=len(p),triangles=sum(len(f) for key,_,_,f in items if key==identity))
        record['geometry_sha256'] = hashlib.sha256(b''.join(v.tobytes()+f.tobytes() for key,_,v,f in items if key==identity)).hexdigest()
    scene['scene_revision'] = revision
    scene['authoritative_scene_json'] = str(directory/'scene.json')
    scene['source_camera_sha256'] = hashlib.sha256((out/'cameras.json').read_bytes()).hexdigest()
    write(directory/'compiled.json',dict(revision=revision,objects=summaries,camera_sha256=scene['source_camera_sha256']))
    bpy.ops.object.select_all(action='DESELECT')
    for root in semantics.roots():
        root.select_set(True)
        for ob in semantics.descendants(root):
            if not ob.hide_render:
                ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(directory/'model.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True)
    if any(o['variant']=='generated' for o in state['objects'].values()):
        from whitebox_web_export import export_web
        export_web(directory)
    points = np.concatenate([v for _,_,v,_ in items])
    center = (points.min(0)+points.max(0))/2
    size = max(np.ptp(points,axis=0).max(),.5)
    render_view(directory,'overview.png',center+np.array([.9,-1.25,1.45])*size,center,size*1.4)
    # Freeze the close-up camera to r000, so a moved object visibly moves in the comparison.
    initial = read(out/'revisions/r000/compiled.json')
    mug_bounds = np.array(initial['objects']['t001']['world_bounds'])
    center = mug_bounds.mean(0)
    render_view(directory,'mug.png',center+np.array([-.22,-.30,.23]),center,.23,(600,600))
    if any(o['variant']=='generated' for o in state['objects'].values()):
        for entry in cameras:
            if entry['frame'] in (0,40,79):
                camera(entry)
                scene.render.filepath=str(directory/f'capture_{entry["frame"]:03d}.png')
                bpy.ops.render.render(write_still=True)
    # Restore a reproducible source camera in the saved native file.
    camera(next(c for c in cameras if c['frame']==40))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(directory/'scene.blend'),compress=True)


def verify_saved(out, revision):
    directory = out/'revisions'/revision
    bpy.ops.wm.open_mainfile(filepath=str(directory/'scene.blend'))
    state,compiled = read(directory/'scene.json'),read(directory/'compiled.json')
    checks = {'revision_matches':bpy.context.scene['scene_revision']==revision,'objects':{},'missing_packed_images':[]}
    for root in semantics.roots():
        key = root['instance_id']
        expected = compiled['objects'][key]
        checks['objects'][key] = dict(asset_matches=root['asset_id']==expected['asset_id'],support_matches=root.get('support_id')==expected['support_id'],
            matrix_max_error=float(np.abs(np.array(root.matrix_world)-expected['matrix_world']).max()),
            locks_match=json.loads(root['locked_fields_json'])==state['objects'][key]['locks'])
    checks['missing_packed_images'] = [im.name for im in bpy.data.images if im.source=='FILE' and not im.packed_file]
    checks['generated_asset_checks']={}
    for root in semantics.roots():
        key=root['instance_id']
        if state['objects'][key]['variant']=='generated':
            children=semantics.descendants(root)
            checks['generated_asset_checks'][key]=dict(
                legacy_parts=[o.name for o in children if o.get('variant')!='generated'],
                foreign_parts=[o.name for o in children if o.type=='MESH' and o.get('fresh_instance')!=key],
                authored_mesh_parts=len([o for o in children if o.type=='MESH']))
    checks['passed'] = checks['revision_matches'] and set(checks['objects'])==set(state['objects']) and not checks['missing_packed_images'] and all(
        r['asset_matches'] and r['support_matches'] and r['locks_match'] and r['matrix_max_error']<1e-6 for r in checks['objects'].values())
    checks['passed'] &= all(not r['legacy_parts'] and not r['foreign_parts'] and r['authored_mesh_parts']>0 for r in checks['generated_asset_checks'].values())
    write(directory/'reopen_validation.json',checks)
    if not checks['passed']:
        raise RuntimeError(checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['initialize','compile','verify'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--revision')
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.action=='initialize':
        (a.output/'assets').mkdir(parents=True,exist_ok=True)
        initialize(a.output,a.source)
    elif a.action=='compile':
        compile_scene(a.output,a.revision)
    else:
        verify_saved(a.output,a.revision)
