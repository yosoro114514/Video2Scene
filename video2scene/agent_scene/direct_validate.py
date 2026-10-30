"""Geometry-neutral CPU Blender verifier/exporter for a freshly authored object.

Invoked in a new Blender process with a host-generated JSON contract. This file
contains no category recipes and never modifies a scene repository.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
import numpy as np

B = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
              [0., -1., 0., 0.], [0., 0., 0., 1.]])
C = B.T


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def visible_meshes():
    return [ob for ob in bpy.context.scene.objects if ob.type == 'MESH' and not ob.hide_render]


def vertices(meshes):
    if not meshes:
        raise ValueError('No mesh parts were produced')
    chunks = []
    for ob in meshes:
        points = np.array([ob.matrix_world @ vertex.co for vertex in ob.data.vertices])
        if not len(points) or not np.isfinite(points).all() or not len(ob.data.polygons):
            raise ValueError('Empty, nonfinite or faceless part: ' + ob.name)
        chunks.append(points)
    return np.concatenate(chunks)


def setup_lighting(center, extent, samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.world = bpy.data.worlds.new('Inspection world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.6, .65, .7, 1.)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .6
    for name, direction, energy in [('Key', (-2, -2, 3), 500), ('Fill', (2, 1, 2), 250)]:
        lamp = bpy.data.lights.new(name, 'AREA')
        lamp.energy = energy * extent ** 2
        lamp.size = extent * 3
        ob = bpy.data.objects.new(name, lamp)
        scene.collection.objects.link(ob)
        ob.location = Vector(center + np.array(direction) * extent)
        ob.rotation_euler = (Vector(center) - ob.location).to_track_quat('-Z', 'Y').to_euler()


def fixed_camera(frame):
    scene = bpy.context.scene
    K = np.asarray(frame['K'], float)
    if abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Camera requires rectified positive-focal intrinsics')
    data = bpy.data.cameras.new(frame['source_frame_id'])
    camera = bpy.data.objects.new(frame['source_frame_id'], data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix((C @ np.asarray(frame['c2w']) @ np.diag([1., -1., -1., 1.])).tolist())
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
    scene.camera = camera
    bpy.context.view_layer.update()
    errors = []
    for local in [np.array([.12, .08, 1.3]), np.array([-.25, -.14, 2.1])]:
        world = np.asarray(frame['c2w'])[:3, :3] @ local + np.asarray(frame['c2w'])[:3, 3]
        uv = world_to_camera_view(scene, camera, Vector(C[:3, :3] @ world))
        actual = np.array([uv.x * width, (1 - uv.y) * height])
        expected = K @ local
        errors.append(float(np.linalg.norm(actual - expected[:2] / expected[2])))
    if max(errors) > .01:
        raise ValueError('Fixed camera projection mismatch: ' + str(errors))
    return max(errors)


def main(manifest):
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    output = Path(spec['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if digest(Path(__file__)) != spec['worker_sha256']:
        raise ValueError('Verifier source hash mismatch')
    source = Path(spec['authored_blend'])
    if digest(source) != spec['authored_sha256']:
        raise ValueError('Author asset changed before validation')
    bpy.ops.wm.open_mainfile(filepath=str(source))
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    original = [ob for ob in scene.objects if ob.type in {'MESH', 'CURVE', 'SURFACE', 'FONT'} and not ob.hide_render]
    if not original:
        raise ValueError('Author asset has no visible geometry')
    baked = []
    for ob in original:
        evaluated = ob.evaluated_get(deps)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=deps)
        mesh.transform(evaluated.matrix_world)
        part = bpy.data.objects.new(ob.name + ' [validated]', mesh)
        scene.collection.objects.link(part)
        part['instance_id'] = spec['instance_id']
        part['authored_part'] = ob.name
        baked.append(part)
    for ob in list(scene.objects):
        if ob not in baked:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.context.view_layer.update()
    points = vertices(baked)
    if len(points) > spec.get('max_vertices', 500000):
        raise ValueError('Authored mesh exceeds vertex budget')
    lo, hi = points.min(axis=0), points.max(axis=0)
    target = np.asarray(spec['local_bounds_blender'], float)
    span, expected = hi - lo, target[1] - target[0]
    if (span < 1e-7).any() or (expected < 1e-7).any():
        raise ValueError('Degenerate source or target dimensions')
    dimension_error = np.abs(span / expected - 1)
    center_error = np.abs((lo + hi - target[0] - target[1]) / (2 * expected))
    report = {'instance_id': spec['instance_id'], 'raw_bounds_blender': [lo.tolist(), hi.tolist()],
              'target_bounds_blender': target.tolist(), 'dimension_relative_error': dimension_error.tolist(),
              'center_relative_error': center_error.tolist(), 'vertices': len(points),
              'parts': [ob['authored_part'] for ob in baked]}
    write(output / 'raw_validation.json', report)
    tolerance = float(spec.get('bounds_fit_tolerance', .12))
    if dimension_error.max() > tolerance or center_error.max() > tolerance:
        raise ValueError('Authored size/origin differs beyond permitted small fit: ' + str(report))
    factors = expected / span
    for ob in baked:
        for vertex in ob.data.vertices:
            vertex.co = Vector(((np.asarray(vertex.co) - lo) * factors + target[0]).tolist())
    report.update(fit_scale=factors.tolist(), fit_translation=(target[0] - lo * factors).tolist(),
                  normalization='Explicit small local bounds fit only; world root/support are untouched.')
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    unpacked = [im.name for im in bpy.data.images if im.source == 'FILE' and not im.packed_file and im.users]
    if unpacked:
        raise ValueError('Unpacked external texture dependencies: ' + str(unpacked))
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'asset.blend'), compress=True)
    bpy.ops.object.select_all(action='DESELECT')
    for ob in baked:
        ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(output / 'asset.glb'), export_format='GLB',
                             use_selection=True, export_yup=True, export_apply=True)

    # Reopen both native and glTF output in this independent process.
    bpy.ops.wm.open_mainfile(filepath=str(output / 'asset.blend'))
    native_points = vertices(visible_meshes())
    if not np.allclose([native_points.min(0), native_points.max(0)], target, atol=2e-5):
        raise ValueError('Reopened Blender bounds differ from locked contract')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(output / 'asset.glb'))
    glb_points = vertices(visible_meshes()) @ B[:3, :3].T
    expected_y_up = np.array([spec['local_bounds_m']['min'], spec['local_bounds_m']['max']])
    if not np.allclose([glb_points.min(0), glb_points.max(0)], expected_y_up, atol=2e-5):
        raise ValueError('Reopened glTF is not canonical object Y-up')
    report.update(native_reopened=True, glb_reopened=True, glb_bounds_y_up=[glb_points.min(0).tolist(), glb_points.max(0).tolist()],
                  local_bounds_m=spec['local_bounds_m'], native_frame='asset_y_up')
    write(output / 'reopen.json', report)

    # Inspect the exported asset that will actually be committed. Blender-only
    # procedural materials/modifiers can differ from glTF after export; rendering
    # the native file here would let the critic approve a different appearance.
    scene = bpy.context.scene
    meshes = visible_meshes()
    center, size = target.mean(0), float(expected.max())
    setup_lighting(center, size, int(spec.get('samples', 8)))
    renders = []
    for index, angle in enumerate([45, 135, 225, 315]):
        a = math.radians(angle)
        camera = bpy.data.objects.new('Turntable ' + str(index), bpy.data.cameras.new('Turntable ' + str(index)))
        scene.collection.objects.link(camera)
        camera.location = Vector(center + np.array([math.cos(a), math.sin(a), .85]) * size * 2)
        camera.rotation_euler = (Vector(center) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.type, camera.data.ortho_scale = 'ORTHO', size * 1.55
        camera.data.clip_start, camera.data.clip_end = .005, 1000
        scene.camera = camera
        scene.render.resolution_x = scene.render.resolution_y = int(spec.get('preview_resolution', 384))
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
        filename = f'turn_{index}.png'
        scene.render.filepath = str(output / filename)
        bpy.ops.render.render(write_still=True)
        renders.append({'filename': filename, 'kind': 'turntable'})

    # Source-pose shape inspection is separate from accepted user layout.
    pose = Matrix((C @ np.asarray(spec['observation_transform']) @ B).tolist())
    observed_matrices = [(ob, pose @ ob.matrix_world.copy()) for ob in meshes]
    for ob, matrix in observed_matrices:
        ob.matrix_world = matrix
    for ob in list(scene.objects):
        if ob.type == 'LIGHT':
            bpy.data.objects.remove(ob, do_unlink=True)
    world_center = np.asarray(pose)[:3, :3] @ center + np.asarray(pose)[:3, 3]
    setup_lighting(world_center, size, int(spec.get('samples', 8)))
    for index, frame in enumerate(spec['frames']):
        error = fixed_camera(frame)
        filename = f'source_{index:02d}.png'
        scene.render.filepath = str(output / filename)
        bpy.ops.render.render(write_still=True)
        renders.append({'filename': filename, 'kind': 'source_pose', 'source_frame_id': frame['source_frame_id'],
                        'camera_projection_max_error_px': error})
    report.update(input_fingerprint=spec['input_fingerprint'], worker_sha256=spec['worker_sha256'],
                  authored_sha256=spec['authored_sha256'], renders=renders, device='CPU',
                  render_source='reopened_canonical_glb',
                  rendered_asset_sha256=digest(output / 'asset.glb'),
                  blender_version=bpy.app.version_string)
    report['files'] = {p.name: digest(p) for p in output.iterdir()
                       if p.suffix in {'.glb', '.blend', '.png'} or p.name == 'reopen.json'}
    write(output / 'result.json', report)


if __name__ == '__main__':
    main(Path(sys.argv[sys.argv.index('--') + 1]).resolve())
