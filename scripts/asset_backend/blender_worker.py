"""Pinned per-run Blender CPU worker; invoked with a host-authored manifest only.

Supports explicit procedural fallback or a data-only image-authored recipe.
Model text is never executed. This compiler is not LiteReality.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

B = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
C = B.T
CAMERA_FLIP = np.diag([1., -1., -1., 1.])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0


def object_matrix(obj):
    transform = obj['transform']
    x, y, z, w = transform['rotation_xyzw']
    rotation = Quaternion((w, x, y, z)).to_matrix().to_4x4()
    scale = Matrix.Diagonal((*transform['scale'], 1.))
    translation = Matrix.Translation(Vector(transform['translation_m']))
    return np.array(translation @ rotation @ scale)


def assemble(scene):
    reset()
    for obj in scene['objects']:
        identity = obj['instance_id']
        asset = Path(obj['_asset_path'])
        if digest(asset) != obj['asset']['artifact']['sha256']:
            raise ValueError('Asset hash changed before Blender import: ' + identity)
        root = bpy.data.objects.new(identity, None)
        bpy.context.scene.collection.objects.link(root)
        root['instance_id'] = identity
        root['category'] = obj['category']
        root['support_id'] = obj['support_id'] or ''
        root['asset_version'] = obj['asset']['asset_version']
        root['field_state_json'] = json.dumps(obj['field_state'], sort_keys=True)
        root['declared_local_bounds_json'] = json.dumps(obj['local_bounds_m'], sort_keys=True)
        root.matrix_world = Matrix((C @ object_matrix(obj)).tolist())
        native = np.asarray(obj['asset']['native_to_object'])
        if asset.suffix.lower() == '.blend':
            with bpy.data.libraries.load(str(asset), link=False) as (available, loaded):
                loaded.objects = list(available.objects)
            meshes = [mesh for mesh in loaded.objects if mesh and mesh.type == 'MESH']
            if not meshes:
                raise ValueError('Asset has no mesh: ' + identity)
            for mesh in meshes:
                tagged = mesh.get('fresh_instance') or mesh.get('instance_id') or mesh.get('source_instance')
                if tagged and tagged != identity:
                    raise ValueError('Asset contains a foreign instance: ' + str(tagged))
                original = mesh.matrix_world.copy()
                bpy.context.scene.collection.objects.link(mesh)
                mesh.parent = root
                mesh.matrix_parent_inverse = Matrix.Identity(4)
                mesh.matrix_basis = Matrix(native.tolist()) @ original
                mesh['instance_id'] = identity
                mesh.hide_render = False
                mesh.hide_set(False)
            for other in loaded.objects:
                if other and other.type != 'MESH':
                    bpy.data.objects.remove(other, do_unlink=True)
        elif asset.suffix.lower() == '.glb':
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=str(asset))
            imported = set(bpy.data.objects) - before
            meshes = [mesh for mesh in imported if mesh.type == 'MESH']
            # Blender's GLB importer already maps native glTF Y-up into Blender Z-up.
            for mesh in meshes:
                original = mesh.matrix_world.copy()
                mesh.parent = root
                mesh.matrix_parent_inverse = Matrix.Identity(4)
                mesh.matrix_basis = Matrix((native @ B).tolist()) @ original
                mesh['instance_id'] = identity
            for other in imported:
                if other.type != 'MESH':
                    bpy.data.objects.remove(other, do_unlink=True)
        else:
            raise ValueError('Unsupported asset format: ' + asset.suffix)
    bpy.context.view_layer.update()
    return inspect_scene()


def inspect_scene():
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    records = {}
    for root in scene.objects:
        if root.type != 'EMPTY' or not root.get('instance_id'):
            continue
        identity = root['instance_id']
        world, local, parts, partial_records = [], [], [], []
        inv = root.matrix_world.inverted()
        for child in scene.objects:
            if child.type != 'MESH' or child.get('instance_id') != identity:
                continue
            evaluated = child.evaluated_get(deps)
            mesh = evaluated.to_mesh()
            vertices = np.array([evaluated.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=float)
            if not len(vertices) or not np.isfinite(vertices).all():
                raise ValueError('Invalid mesh vertices: ' + identity)
            world.append(vertices)
            local.append(np.array([inv @ Vector(point) for point in vertices]))
            parts.append({'name': child.name, 'vertices': len(vertices), 'polygons': len(mesh.polygons),
                          'geometry_sha256': hashlib.sha256(vertices.tobytes()).hexdigest()})
            if child.get('observation_kind') == 'rgb_inferred_surface_crop':
                partial_records.append({'source_scan_sha256': child.get('source_scan_sha256'),
                    'observed_record_hash': child.get('observed_record_hash'),
                    'declared_local_bounds_json': child.get('declared_local_bounds_json')})
            evaluated.to_mesh_clear()
        if not world:
            raise ValueError('No visible meshes for root ' + identity)
        world, local = np.concatenate(world), np.concatenate(local)
        standard_world = world @ B[:3, :3].T
        records[identity] = {'matrix_world_blender': [list(row) for row in root.matrix_world],
                             'transform_scene': (B @ np.array(root.matrix_world)).tolist(),
                             'support_id': root['support_id'] or None,
                             'local_bounds_m': {'min': local.min(0).tolist(), 'max': local.max(0).tolist()},
                             'world_bounds_m': {'min': standard_world.min(0).tolist(), 'max': standard_world.max(0).tolist()},
                             'vertices': len(world), 'parts': parts,
                             'partial_observed': len(partial_records) == len(parts), 'partial_records': partial_records,
                             'declared_local_bounds_m': json.loads(root['declared_local_bounds_json'])}
    return records


def export_scene(output, scene_spec, input_fingerprint):
    scene = bpy.context.scene
    scene['candidate_hash'] = input_fingerprint
    scene['world_frame'] = 'Blender Z-up; conversion from scene_y_up recorded in input.json'
    scene['scene_id'] = scene_spec['scene_id']
    scene.unit_settings.system = 'METRIC'
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(output / 'scene.glb'), export_format='GLB', use_selection=True,
                             export_apply=True, export_extras=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'scene.blend'), compress=True)
    before = inspect_scene()
    bpy.ops.wm.open_mainfile(filepath=str(output / 'scene.blend'))
    after = inspect_scene()
    if set(after) != {obj['instance_id'] for obj in scene_spec['objects']}:
        raise ValueError('Reopened roots differ from the input scene')
    errors, bounds_errors = [], []
    for obj in scene_spec['objects']:
        identity = obj['instance_id']
        error = float(np.max(np.abs(np.asarray(after[identity]['transform_scene']) - object_matrix(obj))))
        errors.append(error)
        bounds_error = float(np.max(np.abs(np.asarray([after[identity]['local_bounds_m']['min'], after[identity]['local_bounds_m']['max']]) -
                                           np.asarray([obj['local_bounds_m']['min'], obj['local_bounds_m']['max']]))))
        bounds_errors.append(bounds_error)
        if error > 1e-5 or after[identity]['vertices'] != before[identity]['vertices']:
            raise ValueError('Reopened geometry or root transform changed: ' + identity)
        partial = after[identity]['partial_observed']
        if partial:
            actual = after[identity]['local_bounds_m']
            declared = obj['local_bounds_m']
            if (np.any(np.asarray(actual['min']) < np.asarray(declared['min']) - 1e-6) or
                    np.any(np.asarray(actual['max']) > np.asarray(declared['max']) + 1e-6)):
                raise ValueError('Partial observed geometry escapes declared bounds: ' + identity)
            for record in after[identity]['partial_records']:
                if (not record['source_scan_sha256'] or not record['observed_record_hash'] or
                        json.loads(record['declared_local_bounds_json']) != declared):
                    raise ValueError('Partial observed asset lacks matching source/bounds provenance')
        elif bounds_error > 1e-5:
            raise ValueError('Reopened geometry bounds differ from standard input: ' + identity + ' ' + str(bounds_error))
    write(output / 'reopen.json', {'passed': True, 'objects': after, 'max_root_transform_error': max(errors),
                                 'max_local_bounds_error_m': max(bounds_errors),
                                 'bounds_policy': 'Exact for full assets; containment within unchanged declared bounds for audited partial observed assets',
                                 'blender_version': bpy.app.version_string, 'glb_native_frame': 'asset_y_up'})


def mesh_object(name, vertices, faces, material):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def lathe(name, profile, material, segments=64):
    vertices = [(r * math.cos(i * 2 * math.pi / segments), r * math.sin(i * 2 * math.pi / segments), z)
                for r, z in profile for i in range(segments)]
    faces = [(j * segments + i, j * segments + (i + 1) % segments,
              (j + 1) * segments + (i + 1) % segments, (j + 1) * segments + i)
             for j in range(len(profile) - 1) for i in range(segments)]
    obj = mesh_object(name, vertices, faces, material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def material(name, color, metallic=0., roughness=.32):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1.)
    result.use_nodes = True
    shader = result.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1.)
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Roughness'].default_value = roughness
    return result


def refine(spec, output):
    reset()
    target, recipe = spec['target'], spec['recipe']
    color = {'mug': (.60, .025, .035), 'cup': (.60, .025, .035), 'bowl': (.87, .88, .84),
             'can': (.12, .45, .18), 'carton': (.77, .56, .12), 'table': (.34, .19, .09)}[recipe]
    mat = material('Procedural ' + recipe, color, .7 if recipe == 'can' else 0.)
    if recipe in {'mug', 'cup'}:
        profile = [(0., 0.), (.30, 0.), (.34, .03), (.35, .96), (.34, 1.),
                   (.30, 1.), (.29, .96), (.28, .09), (0., .09)]
        parts = [lathe('new_cup_body', profile, mat)]
        vertices, faces = [], []
        segments, ring = 40, 12
        for i in range(segments + 1):
            theta = -math.pi / 2 + i * math.pi / segments
            center = np.array([.33 + .29 * math.cos(theta), 0., .52 + .34 * math.sin(theta)])
            normal = np.array([math.cos(theta), 0., math.sin(theta)])
            for j in range(ring):
                phi = j * 2 * math.pi / ring
                point = center + .045 * (math.cos(phi) * normal + math.sin(phi) * np.array([0., 1., 0.]))
                vertices.append(point.tolist())
        for i in range(segments):
            for j in range(ring):
                faces.append((i * ring + j, i * ring + (j + 1) % ring,
                              (i + 1) * ring + (j + 1) % ring, (i + 1) * ring + j))
        parts.append(mesh_object('new_cup_handle', vertices, faces, mat))
    elif recipe == 'bowl':
        parts = [lathe('new_bowl', [(0, 0), (.25, 0), (.30, .03), (.70, .65), (.72, .72),
                                    (.67, .74), (.63, .66), (.25, .10), (0, .10)], mat)]
    elif recipe == 'can':
        parts = [lathe('new_can', [(0, 0), (.30, 0), (.32, .025), (.30, .07), (.34, .12),
                                  (.34, .90), (.30, .95), (.32, .98), (.30, 1.), (0, 1.)], mat)]
    elif recipe == 'table':
        parts = [lathe('new_tabletop', [(0, 0), (.95, 0), (1., .12), (1., .88), (.95, 1.), (0, 1.)], mat)]
    else:
        bpy.ops.mesh.primitive_cube_add(size=1)
        cube = bpy.context.object
        cube.name = 'new_carton'
        cube.data.materials.append(mat)
        bevel = cube.modifiers.new('Small formed edges', 'BEVEL')
        bevel.width = .015
        bevel.segments = 2
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        parts = [cube]
    lo = np.asarray(target['local_bounds_m']['min'])
    hi = np.asarray(target['local_bounds_m']['max'])
    corners = np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
    native = corners @ C[:3, :3].T
    lower, upper = native.min(0), native.max(0)
    all_vertices = np.array([obj.matrix_world @ vertex.co for obj in parts for vertex in obj.data.vertices])
    current_lo, current_hi = all_vertices.min(0), all_vertices.max(0)
    for obj in parts:
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            vertex.co = Vector(((np.asarray(point) - current_lo) / (current_hi - current_lo) * (upper - lower) + lower).tolist())
        obj.matrix_world = Matrix.Identity(4)
        obj['fresh_instance'] = target['instance_id']
        obj['instance_id'] = target['instance_id']
        obj['generation_method'] = 'new parameterized geometry from category and bounds; no model call'
        obj['input_fingerprint'] = spec['input_fingerprint']
    bpy.context.view_layer.update()
    generated = np.array([vertex.co[:] for obj in parts for vertex in obj.data.vertices])
    standard = generated @ B[:3, :3].T
    measured = {'min': standard.min(0).tolist(), 'max': standard.max(0).tolist()}
    if not np.allclose([measured['min'], measured['max']], [lo, hi], atol=1e-6):
        raise ValueError('Generated asset does not preserve requested bounds')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(output / 'asset.glb'), export_format='GLB', use_selection=True,
                             export_apply=True, export_extras=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'asset.blend'), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(output / 'asset.blend'))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes or any(obj.get('fresh_instance') != target['instance_id'] for obj in meshes):
        raise ValueError('Refined asset reopen failed')
    return {'local_bounds_m': target['local_bounds_m'], 'measured_local_bounds_m': measured,
            'vertices': sum(len(obj.data.vertices) for obj in meshes), 'recipe': recipe,
            'geometry_sha256': hashlib.sha256(generated.tobytes()).hexdigest(), 'reopen_passed': True,
            'new_geometry': True, 'model_calls': 0,
            'quality_claim': 'Parametric asset generation verified; photo-consistency and quality improvement are not established.'}


def render_setup():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    if scene.world is None:
        scene.world = bpy.data.worlds.new('Evidence World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.6, .65, .7, 1.)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .5
    for name, location, energy in [('Key', (-2, -2, 4), 450), ('Fill', (2, 1, 3), 220)]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.size = energy, 4
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, .5)) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def refine_recipe(spec, output):
    reset()
    compiler=Path(__file__).with_name('recipe_geometry.py')
    if digest(compiler)!=spec['compiler_sha256']:
        raise ValueError('Recipe compiler differs from its pinned input hash')
    import importlib.util
    module_spec=importlib.util.spec_from_file_location('pinned_recipe_geometry',compiler)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    target=spec['target'];recipe=spec['authored_recipe'];record=spec['recipe_record']
    if target['instance_id']!=recipe['instance_id']:
        raise ValueError('Authored recipe belongs to another object')
    compiled,stats=module.compile_object(recipe,target['local_bounds_m'])
    generated=[]
    for part in compiled:
        material_spec=part['material']
        mat=material(material_spec['name'],material_spec['base_color_rgb'],material_spec['metallic'],material_spec['roughness'])
        # The compiler produces object Y-up; convert once to Blender-native Z-up.
        vertices=np.asarray(part['vertices'])@C[:3,:3].T
        mesh=mesh_object(part['name'],vertices.tolist(),part['faces'],mat)
        for polygon in mesh.data.polygons:polygon.use_smooth=part['smooth']
        mesh['fresh_instance']=target['instance_id'];mesh['instance_id']=target['instance_id']
        mesh['generation_method']='data-only recipe compiled by pinned host code'
        mesh['author_origin']=record['provenance']['origin'];mesh['recipe_hash']=record['recipe_hash']
        mesh['author_bundle_hash']=record['bundle_hash'];mesh['input_fingerprint']=spec['input_fingerprint']
        generated.extend(vertices.tolist())
    generated=np.asarray(generated);standard=generated@B[:3,:3].T
    measured={'min':standard.min(0).tolist(),'max':standard.max(0).tolist()}
    expected=[target['local_bounds_m']['min'],target['local_bounds_m']['max']]
    if not np.allclose([measured['min'],measured['max']],expected,atol=1e-6):
        raise ValueError('Recipe changed locked local bounds')
    bpy.context.view_layer.update();bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(output/'asset.glb'),export_format='GLB',use_selection=True,export_apply=True,export_extras=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'asset.blend'),compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(output/'asset.blend'))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if len(meshes)!=len(compiled) or any(o.get('recipe_hash')!=record['recipe_hash'] for o in meshes):
        raise ValueError('Authored geometry reopen validation failed')
    return {'local_bounds_m':target['local_bounds_m'],'measured_local_bounds_m':measured,
            'recipe_hash':record['recipe_hash'],'author_bundle_hash':record['bundle_hash'],
            'author_provenance':record['provenance'],'compiler_sha256':spec['compiler_sha256'],
            'geometry_sha256':hashlib.sha256(generated.tobytes()).hexdigest(),'new_geometry':True,
            'reopen_passed':True,'model_calls':0,'author_batch_model_calls':record['provenance']['model_calls'],
            'observed_features':recipe['observed_features'],'assumptions':recipe['assumptions'],**stats,
            'quality_claim':'New data-recipe geometry compiled and reopened. Photo-consistency requires a separate source/render critic.'}


def refine_observed(spec, output):
    reset()
    target, observed, record = spec['target'], spec['observed_mesh'], spec['observed_record']
    canonical = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if record['origin'] != 'rgb_inferred_surface_crop' or canonical(observed) != record['cropped_mesh_hash']:
        raise ValueError('Observed mesh provenance or content hash differs')
    if record['candidate'] != spec['candidate'] or record['evidence_hash'] != spec['evidence']['evidence_hash']:
        raise ValueError('Observed crop belongs to another candidate/evidence')
    for name, row in record['source_files'].items():
        if digest(output / name) != row['sha256']:
            raise ValueError('Frozen observed source differs from its recorded hash')
    vertices = np.asarray(observed['vertices'], dtype=float)
    faces = np.asarray(observed['faces'], dtype=int)
    colors = np.asarray(observed['colors_rgba'], dtype=float) / 255.
    lo, hi = np.asarray(target['local_bounds_m']['min']), np.asarray(target['local_bounds_m']['max'])
    if (not len(vertices) or not len(faces) or not np.isfinite(vertices).all() or
            np.any(vertices < lo - 1e-6) or np.any(vertices > hi + 1e-6)):
        raise ValueError('Partial observed geometry escapes declared bounds or has no surface')
    if colors.shape != (len(vertices), 4) or np.any(colors < 0) or np.any(colors > 1):
        raise ValueError('Observed vertex color dimensions or values are invalid')
    mat = material('Observed RGB vertex colors', [1., 1., 1.], 0., .85)
    node = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    node.layer_name = 'COLOR_0'
    mat.node_tree.links.new(node.outputs['Color'], mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    obj = mesh_object('partial_observed_' + target['instance_id'], (vertices @ C[:3, :3].T).tolist(), faces.tolist(), mat)
    attr = obj.data.color_attributes.new(name='COLOR_0', type='FLOAT_COLOR', domain='POINT')
    for idx, color in enumerate(colors):
        attr.data[idx].color = color.tolist()
    obj.data.color_attributes.active_color = attr
    obj['fresh_instance'] = target['instance_id']
    obj['instance_id'] = target['instance_id']
    obj['observation_kind'] = record['origin']
    obj['source_scan_sha256'] = record['source_files']['source_scan.glb']['sha256']
    obj['observed_record_hash'] = canonical(record)
    obj['declared_local_bounds_json'] = json.dumps(target['local_bounds_m'], sort_keys=True)
    obj['input_fingerprint'] = spec['input_fingerprint']
    obj['quality_claim'] = record['quality_claim']
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(output / 'asset.glb'), export_format='GLB', use_selection=True,
                             export_apply=True, export_extras=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'asset.blend'), compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(output / 'asset.blend'))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if len(meshes) != 1 or meshes[0].get('observed_record_hash') != canonical(record):
        raise ValueError('Partial observed asset failed reopen provenance validation')
    return {'local_bounds_m': target['local_bounds_m'], 'measured_local_bounds_m': observed['measured_local_bounds_m'],
            'declared_bounds_policy': 'unchanged enclosure, not a tight mesh AABB', 'partial': True,
            'observed_record': record, 'observed_record_hash': canonical(record), 'stats': observed['stats'],
            'geometry_sha256': hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest(),
            'model_calls': 0, 'reopen_passed': True, 'quality_claim': record['quality_claim']}


def camera(frame):
    scene = bpy.context.scene
    K = np.asarray(frame['K'])
    if abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Blender requires a rectified positive-focal-length K')
    data = bpy.data.cameras.new(frame['source_frame_id'])
    obj = bpy.data.objects.new(frame['source_frame_id'], data)
    scene.collection.objects.link(obj)
    obj.matrix_world = Matrix((C @ np.asarray(frame['c2w']) @ CAMERA_FLIP).tolist())
    width, height = frame['width'], frame['height']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    data.type, data.sensor_fit, data.sensor_width = 'PERSP', 'HORIZONTAL', 36
    data.lens = fx * 36 / width
    data.shift_x = (width / 2 - cx) / width
    data.shift_y = (cy - height / 2) * fx / fy / width
    data.clip_start, data.clip_end = .005, 1000
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1, fx / fy
    scene.camera = obj
    bpy.context.view_layer.update()
    errors = []
    for local in [np.array([.12, .08, 1.3]), np.array([-.25, -.14, 2.1])]:
        point = np.asarray(frame['c2w'])[:3, :3] @ local + np.asarray(frame['c2w'])[:3, 3]
        uv = world_to_camera_view(scene, obj, Vector((C[:3, :3] @ point).tolist()))
        actual = np.array([uv.x * width, (1 - uv.y) * height])
        projected = K @ local
        errors.append(float(np.linalg.norm(actual - projected[:2] / projected[2])))
    if max(errors) > .01:
        raise ValueError('Camera pixel projection differs from evidence: ' + str(errors))
    return max(errors)


def mask_material(identity):
    color = hashlib.sha256(identity.encode()).digest()[:3]
    mat = bpy.data.materials.new('Instance mask ' + identity)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emit = mat.node_tree.nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (*[(40 + item % 190) / 255 for item in color], 1.)
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emit.outputs[0], output.inputs['Surface'])
    return mat


def render(spec, output):
    assemble(spec['scene'])
    render_setup()
    scene = bpy.context.scene
    scene.cycles.samples = spec['samples']
    mode = spec['view_mode']
    targets = spec['evidence']['target_ids']
    if mode != 'normal':
        for mesh in [obj for obj in scene.objects if obj.type == 'MESH']:
            mesh.data.materials.clear()
            mesh.data.materials.append(mask_material(mesh['instance_id']))
            for polygon in mesh.data.polygons:
                polygon.material_index = 0
        scene.view_settings.view_transform = 'Standard'
        scene.cycles.use_denoising = False
        scene.cycles.samples = 1
    frames = {frame['source_frame_id']: frame for frame in spec['evidence']['frames']}
    rows = []
    for frame_id in spec['render_frame_ids']:
        frame = frames[frame_id]
        error = camera(frame)
        for target in targets if mode == 'xray' else [None]:
            for mesh in [obj for obj in scene.objects if obj.type == 'MESH']:
                mesh.hide_render = target is not None and mesh['instance_id'] != target
            filename = frame_id + ('--' + target if target else '') + '--' + mode + '.png'
            scene.render.filepath = str(output / filename)
            bpy.ops.render.render(write_still=True)
            rows.append({'source_frame_id': frame_id, 'target_id': target, 'filename': filename,
                         'camera_projection_max_error_px': error})
    return {'renders': rows, 'view_mode': mode, 'device': 'CPU', 'blender_version': bpy.app.version_string}


def main():
    manifest = Path(sys.argv[sys.argv.index('--') + 1])
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    output = Path(spec['output_dir'])
    if digest(Path(__file__)) != spec['worker_sha256']:
        raise ValueError('Worker script differs from pinned build manifest')
    if spec['action'] == 'build':
        assemble(spec['scene'])
        export_scene(output, spec['scene'], spec['input_fingerprint'])
        details = {'blender_version': bpy.app.version_string, 'native_scene_glb_frame': 'asset_y_up'}
    elif spec['action'] == 'refine':
        details = refine(spec, output)
    elif spec['action'] == 'refine_recipe':
        details = refine_recipe(spec, output)
    elif spec['action'] == 'refine_observed':
        details = refine_observed(spec, output)
    elif spec['action'] == 'render':
        details = render(spec, output)
    else:
        raise ValueError('Unknown action')
    files = {path.name: digest(path) for path in output.iterdir()
             if path.suffix in {'.blend', '.glb', '.png'} or path.name == 'reopen.json'}
    write(output / 'result.json', {'input_fingerprint': spec['input_fingerprint'],
                                 'worker_sha256': spec['worker_sha256'], 'files': files, **details})


if __name__ == '__main__':
    main()
