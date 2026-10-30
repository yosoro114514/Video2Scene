"""Scene semantics v1. Blender custom properties are authoritative; JSON is derived.

A semantic root is an Empty with a stable instance_id and replaceable descendants.
Classes use slash-delimited paths. Surface roles are a default surface_role string
and optional surface_roles JSON mapping of material-slot indices to role strings.
An optional support_plane JSON stores local xmin/xmax/ymin/ymax/z on a root.
"""
import json
import math
import re
from pathlib import Path

import bpy
from mathutils import Matrix

SCHEMA_VERSION = 1


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ' must be a nonempty string')
    return value


def descendants(root):
    result = []
    for child in root.children:
        result.append(child)
        result.extend(descendants(child))
    return result


def roots():
    result = [obj for obj in bpy.context.scene.objects if obj.get('instance_id')]
    ids = [obj['instance_id'] for obj in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate semantic instance_id; explicitly assign unique IDs after duplication')
    return sorted(result, key=lambda obj: obj['instance_id'])


def _new_id(name):
    stem = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'instance'
    used = {obj.get('instance_id') for obj in bpy.context.scene.objects}
    candidate, number = stem, 2
    while candidate in used:
        candidate = '{}-{}'.format(stem, number)
        number += 1
    return candidate


def tag_root(root, semantic_class, instance_id=None, asset_id=None, support_id=None):
    """Tag a complete object; wrap mesh/collection-instance roots without moving them.

    Callers must use the returned root: a legacy mesh becomes its geometry child.
    A complete Empty hierarchy is tagged directly. No heuristic grouping by name.
    """
    _text(semantic_class, 'semantic_class')
    if not semantic_class.strip('/'):
        raise ValueError('semantic_class must name a class')
    bpy.context.view_layer.update()
    identity = instance_id or root.get('instance_id') or _new_id(root.name)
    _text(identity, 'instance_id')
    if any(obj != root and obj.get('instance_id') == identity for obj in bpy.context.scene.objects):
        raise ValueError('Duplicate instance_id: ' + identity)
    if root.type != 'EMPTY' or root.instance_type != 'NONE':
        original = root
        world = original.matrix_world.copy()
        root = bpy.data.objects.new(original.name + ' semantic root', None)
        for collection in original.users_collection:
            collection.objects.link(root)
        if not root.users_collection:
            bpy.context.scene.collection.objects.link(root)
        root.parent = original.parent
        root.matrix_world = world
        original.parent = root
        original.matrix_world = world
        for key in ('instance_id', 'semantic_class', 'asset_id', 'support_id',
                    'asset_orientation_json', 'asset_front_axis', 'asset_up_axis', 'asset_origin', 'facing_target_json'):
            if key in original:
                root[key] = original[key]
                del original[key]
    root['semantics_schema_version'] = SCHEMA_VERSION
    root['instance_id'] = identity
    root['semantic_class'] = semantic_class.strip('/')
    for key, value in (('asset_id', asset_id), ('support_id', support_id)):
        if value is not None:
            root[key] = _text(value, key)
    bpy.context.scene['semantics_schema_version'] = SCHEMA_VERSION
    return root


def adopt_group(objects, semantic_class, instance_id, asset_id=None, support_id=None, placement_matrix=None):
    """Group explicit legacy parts; provide their reviewed floor-centred placement.

    The default identity placement keeps the world origin. It is suitable only
    for groups already authored in normalized asset coordinates.
    """
    objects = list(objects)
    if not objects:
        raise ValueError('adopt_group requires explicit objects')
    if any(obj.get('instance_id') for obj in objects):
        raise ValueError('Group contains existing semantic roots')
    root = bpy.data.objects.new(instance_id, None)
    bpy.context.scene.collection.objects.link(root)
    if placement_matrix is not None:
        root.matrix_world = Matrix(placement_matrix)
    bpy.context.view_layer.update()
    for obj in objects:
        if obj.parent not in objects:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    return tag_root(root, semantic_class, instance_id, asset_id, support_id)


def tag_surface(obj, surface_role, slot=None):
    _text(surface_role, 'surface_role')
    if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META'):
        raise ValueError('Surface tags require renderable geometry')
    if slot is None:
        obj['surface_role'] = surface_role
    else:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 or slot >= len(obj.material_slots):
            raise ValueError('Surface slot must refer to an existing material slot')
        roles = json.loads(obj.get('surface_roles', '{}'))
        roles[str(slot)] = surface_role
        obj['surface_roles'] = json.dumps(roles, sort_keys=True)
    obj['semantics_schema_version'] = SCHEMA_VERSION
    return obj


def surface_slots(obj):
    """Map slot index to role; explicit per-slot assignments override the default."""
    roles = {int(index): value for index, value in json.loads(obj.get('surface_roles', '{}')).items()}
    if obj.get('surface_role'):
        for index in range(max(1, len(obj.material_slots))):
            roles.setdefault(index, obj['surface_role'])
    return roles


def tag_support(root, xmin, xmax, ymin, ymax, z):
    """Describe a horizontal support plane in the semantic root's local metres."""
    values = (xmin, xmax, ymin, ymax, z)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Support coordinates must be finite numbers')
    if not root.get('instance_id') or xmin >= xmax or ymin >= ymax:
        raise ValueError('Support requires a semantic root and positive plane bounds')
    root['support_plane'] = json.dumps(dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, z=z))
    return root


def _json_value(value):
    """Convert Blender ID properties without serializing live RNA objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bpy.types.ID):
        return {'datablock': value.name, 'type': value.bl_rna.identifier}
    if hasattr(value, 'items'):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, 'to_list'):
        return [_json_value(item) for item in value.to_list()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError('Unsupported semantic metadata value type: ' + type(value).__name__)


def export_semantics(path=None):
    result = {'schema_version': SCHEMA_VERSION, 'source': bpy.data.filepath,
              'units': 'metres', 'world_up': 'Z', 'frame': bpy.context.scene.frame_current,
              'instances': [], 'surfaces': [], 'joints': {}}
    for root in roots():
        item = {key: root.get(key) for key in ('instance_id', 'semantic_class', 'asset_id', 'support_id')}
        item['capabilities'] = {key: root[key] for key in ('asset_articulated', 'asset_import_mode', 'variant_uniform_scale') if key in root}
        item.update(object=root.name, matrix_world=[list(row) for row in root.matrix_world],
                    geometry_objects=[obj.name for obj in descendants(root)])
        from .orientation import get_orientation, facing_report
        item['orientation'] = get_orientation(root)
        if root.get('facing_target_json'):
            item['facing_target'] = json.loads(root['facing_target_json'])
            item['facing_check'] = facing_report(root)
        if root.get('support_plane'):
            item['support_plane'] = json.loads(root['support_plane'])
        result['instances'].append(item)
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.get('joint_id'):
            owner = obj.parent
            while owner is not None and not owner.get('instance_id'):
                owner = owner.parent
            if owner is None:
                raise ValueError('Joint has no semantic owner: ' + obj.name)
            owner_id, joint_id = owner['instance_id'], str(obj['joint_id'])
            joints = result['joints'].setdefault(owner_id, {})
            if joint_id in joints:
                raise ValueError('Duplicate joint_id under {}: {}'.format(owner_id, joint_id))
            prop = obj.get('roomkit_open_property')
            if not prop:
                prop = 'open_amount' if 'open_amount' in obj else 'Open' if 'Open' in obj else None
            metadata = {key: _json_value(value) for key, value in obj.items()
                        if key.startswith('joint_') or key in ('axis', 'limit_min', 'limit_max', 'closed_position')}
            joints[joint_id] = {'instance_id': owner_id, 'joint_id': joint_id, 'object': obj.name,
                'property': prop, 'value': _json_value(obj.get(prop)) if prop else None,
                'type': obj.get('joint_type'), 'axis': obj.get('joint_axis', obj.get('axis')),
                'limit': _json_value(obj.get('joint_limit')), 'limit_unit': obj.get('joint_limit_unit'),
                'metadata': metadata, 'matrix_world': [list(row) for row in obj.matrix_world]}
        slots = surface_slots(obj)
        if slots:
            result['surfaces'].append({'object': obj.name, 'slots': {str(i): role for i, role in slots.items()}})
    if path is not None:
        Path(path).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result

