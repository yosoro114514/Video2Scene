"""Shared camera/render tools for the simple-table authoring experiment.

Contains no object geometry or scene measurements. Copied into each author
workspace so the image-only condition has the same rendering tools.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def camera(entry):
    name = f"Evidence {entry['frame']:03d}"
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        bpy.context.scene.collection.objects.link(ob)
    t = np.array(entry['c2w'])
    matrix = np.eye(4)
    matrix[:3, :3] = C @ t[:3, :3] @ np.diag([1, -1, -1])
    matrix[:3, 3] = C @ t[:3, 3]
    ob.matrix_world = Matrix(matrix.tolist())
    w, h = entry['resolution']
    fx, fy, cx, cy = entry['intrinsics']
    data = ob.data
    data.type = 'PERSP'
    data.sensor_fit = 'HORIZONTAL'
    data.sensor_width = 36
    data.lens = fx * 36 / w
    data.shift_x = (w / 2 - cx) / w
    data.shift_y = (cy - h / 2) * fx / fy / w
    data.clip_start = .01
    data.clip_end = 100
    scene = bpy.context.scene
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = fx / fy
    scene.camera = ob
    return ob


def setup(cameras):
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    for ob in list(scene.objects):
        if ob.type == 'LIGHT':
            bpy.data.objects.remove(ob, do_unlink=True)
    for name, location, energy, size in [('Key', (-2, -2, 4), 450, 4), ('Fill', (2, 1, 3), 220, 3)]:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy
        data.shape = 'DISK'
        data.size = size
        ob = bpy.data.objects.new(name, data)
        scene.collection.objects.link(ob)
        ob.location = location
        ob.rotation_euler = (Vector((0, 0, .5)) - ob.location).to_track_quat('-Z', 'Y').to_euler()
    if scene.world is None:
        scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.65, .70, .75, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .55
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'OPTIX'
        scene.cycles.device = 'GPU'
    except Exception:
        scene.cycles.device = 'CPU'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    for entry in cameras:
        camera(entry)


def visible_objects():
    return [o for o in bpy.context.scene.objects if o.type in {'MESH', 'CURVE', 'FONT'}
            and not o.hide_render and not any(c.hide_render for c in o.users_collection)]


def render(cameras, output, samples=16, labels=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    setup(cameras)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    for entry in cameras:
        camera(entry)
        scene.render.filepath = str(output / f"rgb_{entry['frame']:03d}.png")
        bpy.ops.render.render(write_still=True)
    if not labels:
        return
    restore = []
    colors = {}
    for ob in visible_objects():
        identity = str(ob.get('source_instance', ob.get('instance', 'unassigned')))
        raw = hashlib.sha256(identity.encode()).digest()
        rgb = [40 + v % 190 for v in raw[:3]]
        colors[identity] = rgb
        mat = bpy.data.materials.new('MASK ' + identity)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        emission = mat.node_tree.nodes.new('ShaderNodeEmission')
        value = np.array(rgb) / 255
        value = np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)
        emission.inputs['Color'].default_value = (*value, 1)
        target = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emission.outputs[0], target.inputs['Surface'])
        indices = [p.material_index for p in ob.data.polygons] if ob.type == 'MESH' else None
        restore.append((ob, list(ob.data.materials), indices))
        ob.data.materials.clear()
        ob.data.materials.append(mat)
        if ob.type == 'MESH':
            for p in ob.data.polygons:
                p.material_index = 0
    scene.view_settings.view_transform = 'Standard'
    scene.cycles.use_denoising = False
    scene.cycles.samples = 1
    scene.render.film_transparent = True
    for entry in cameras:
        camera(entry)
        scene.render.filepath = str(output / f"labels_{entry['frame']:03d}.png")
        bpy.ops.render.render(write_still=True)
    for ob, mats, indices in restore:
        ob.data.materials.clear()
        for mat in mats:
            ob.data.materials.append(mat)
        if indices is not None:
            for p, index in zip(ob.data.polygons, indices):
                p.material_index = index
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.cycles.use_denoising = True
    write(output / 'label_colors.json', colors)


def inspect(cameras):
    scene = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    groups = {}
    meshes = []
    for ob in visible_objects():
        ev = ob.evaluated_get(deps)
        mesh = ev.to_mesh()
        points = np.array([ev.matrix_world @ v.co for v in mesh.vertices])
        assert len(points) and np.isfinite(points).all(), ob.name
        identity = str(ob.get('source_instance', ob.get('instance', 'unassigned')))
        groups.setdefault(identity, []).append(points)
        meshes.append(dict(name=ob.name, identity=identity, vertices=len(points), polygons=len(mesh.polygons)))
        ev.to_mesh_clear()
    groups = {key: np.concatenate(arrays) for key, arrays in groups.items()}
    rows = []
    errors = []
    for entry in cameras:
        cam = camera(entry)
        t = np.array(entry['c2w'])
        w2c = np.linalg.inv(t)
        fx, fy, cx, cy = entry['intrinsics']
        w, h = entry['resolution']
        for q in [np.array([.12, .08, 1.3]), np.array([-.25, -.14, 2.1])]:
            world = C @ (t[:3, :3] @ q + t[:3, 3])
            uv = world_to_camera_view(scene, cam, Vector(world))
            errors.append(float(np.linalg.norm(np.array([uv.x*w, (1-uv.y)*h]) - (q[:2]/q[2]*[fx,fy]+[cx,cy]))))
        for identity, points in groups.items():
            p = points @ C
            p = p @ w2c[:3, :3].T + w2c[:3, 3]
            valid = p[:, 2] > .01
            if valid.all():
                uv = p[:, :2] / p[:, 2, None] * [fx, fy] + [cx, cy]
                lo = np.clip(uv.min(0), [0, 0], [w, h])
                hi = np.clip(uv.max(0), [0, 0], [w, h])
                box = [*lo.tolist(), *hi.tolist()]
            else:
                box = None
            rows.append(dict(frame=entry['frame'], target=identity, projected_box=box, vertices=len(points)))
    missing = [im.name for im in bpy.data.images if im.source == 'FILE' and not im.packed_file]
    assert not missing, missing
    assert max(errors, default=0) < .01, errors
    return dict(parts=meshes, groups={key:dict(min=p.min(0).tolist(),max=p.max(0).tolist()) for key,p in groups.items()},
                projections=rows, camera_projection_max_px=max(errors, default=0), unpacked_images=missing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cameras', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--samples', type=int, default=16)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    cameras = read(a.cameras)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.init:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        setup(cameras)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'input.blend'))
        return
    render(cameras, a.output/'renders', a.samples, a.final)
    if a.final:
        bpy.ops.file.pack_all()
        write(a.output/'geometry_validation.json', inspect(cameras))
        objs = visible_objects()
        bpy.ops.object.select_all(action='DESELECT')
        for ob in objs:
            ob.hide_set(False)
            ob.select_set(True)
            parent = ob.parent
            while parent:
                parent.select_set(True)
                parent = parent.parent
        bpy.ops.export_scene.gltf(filepath=str(a.output/'model.glb'), export_format='GLB', use_selection=True, export_apply=True, export_extras=True)
        points = []
        for ob in objs:
            points.extend([ob.matrix_world @ Vector(v) for v in ob.bound_box])
        points = np.array(points)
        center = (points.min(0)+points.max(0))/2
        size = max(np.ptp(points, axis=0).max(), .5)
        overview = bpy.data.objects.new('Overview', bpy.data.cameras.new('Overview'))
        bpy.context.scene.collection.objects.link(overview)
        overview.location = center + np.array([1.1, -1.4, 1.3])*size
        overview.rotation_euler = (Vector(center)-overview.location).to_track_quat('-Z', 'Y').to_euler()
        overview.data.type = 'ORTHO'
        overview.data.ortho_scale = size*1.45
        s = bpy.context.scene
        s.camera = overview
        s.render.resolution_x = 960
        s.render.resolution_y = 720
        s.render.pixel_aspect_x = s.render.pixel_aspect_y = 1
        s.cycles.samples = a.samples
        s.render.filepath = str(a.output/'overview.png')
        bpy.ops.render.render(write_still=True)
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'scene.blend'), compress=True)


if __name__ == '__main__':
    main()
