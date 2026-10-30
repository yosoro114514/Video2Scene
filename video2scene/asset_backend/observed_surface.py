"""Explicit partial surface extraction from a frozen RGB-inferred TSDF scan.

No template completion, shape rescaling, semantic relabeling, or model calls.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import trimesh

from video2scene.capture_evidence import transform_points, transform_to_matrix
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash, validate
from .backend import BlenderAssetBackend, DEFAULT_BLENDER


QUALITY = 'RGB-inferred observed surface crop, partial, not GPT authored or photo-quality pass'


def crop_observed_mesh(mesh, source_target, *, min_faces=24):
    """Keep original triangles wholly inside the source object's local ROI."""
    vertices = transform_points(np.linalg.inv(transform_to_matrix(source_target['transform'])),
                                np.asarray(mesh.vertices, dtype=float))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ContractError('BUILD_FAILED', 'Observed scan has no finite triangular surface')
    bounds = source_target['local_bounds_m']
    lo, hi = np.asarray(bounds['min']), np.asarray(bounds['max'])
    inside = np.all((vertices >= lo) & (vertices <= hi), axis=1)
    keep = np.flatnonzero(np.all(inside[faces], axis=1))
    if len(keep) < min_faces:
        raise ContractError('BUILD_FAILED', 'Insufficient observed triangles in the target ROI; no completion fallback',
                            {'retained_faces': int(len(keep)), 'required_faces': min_faces})
    kept_faces = faces[keep]
    used, inverse = np.unique(kept_faces.ravel(), return_inverse=True)
    local = vertices[used]
    remapped = inverse.reshape(-1, 3)
    tri = local[remapped]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or float(areas.sum()) < 1e-5:
        raise ContractError('BUILD_FAILED', 'Observed ROI contains no meaningful surface area')
    if len(local) > 100000 or len(remapped) > 200000:
        raise ContractError('BUDGET_EXCEEDED', 'Observed ROI exceeds the CPU mesh budget')
    if mesh.visual.kind != 'vertex':
        raise ContractError('TOOL_UNAVAILABLE', 'Observed scan must carry vertex colors; texture synthesis is not supported')
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[used]
    return {'vertices': local.tolist(), 'faces': remapped.tolist(), 'colors_rgba': colors.tolist(),
            'source_vertex_indices': used.tolist(), 'source_face_indices': keep.tolist(),
            'measured_local_bounds_m': {'min': local.min(0).tolist(), 'max': local.max(0).tolist()},
            'stats': {'source_faces': len(faces), 'retained_faces': len(remapped), 'retained_vertices': len(local),
                      'surface_area_m2': float(areas.sum()), 'selection': 'all triangle vertices inside original source object ROI',
                      'boundary_crossing_triangles': 'omitted; no new boundary or hidden-side geometry',
                      'geometry_rescaled': False}}


class ObservedSurfaceBackend(BlenderAssetBackend):
    def __init__(self, storage_root, scan_path, source_scene_path, source_snapshot,
                 blender_path=DEFAULT_BLENDER, *, timeout_s=300, samples=8, min_faces=24):
        super().__init__(storage_root, blender_path, timeout_s=timeout_s, samples=samples,
                         refinement_mode='observed_surface')
        self.scan_path = Path(scan_path).resolve()
        self.source_scene_path = Path(source_scene_path).resolve()
        self.snapshot_path = Path(source_snapshot).resolve()
        self.source_files = {name: {'path': str(path), 'sha256': file_hash(path)} for name, path in [
            ('source_scan.glb', self.scan_path), ('observed_source_scene.json', self.source_scene_path),
            ('observed_source_snapshot.json', self.snapshot_path)]}
        self.source_snapshot = json.loads(self.snapshot_path.read_text(encoding='utf-8'))
        validate(self.source_snapshot, 'SceneSnapshot')
        legacy = json.loads(self.source_scene_path.read_text(encoding='utf-8'))
        if not legacy.get('coordinate_system', '').startswith('right-handed Y-up'):
            raise ContractError('FRAME_MISMATCH', 'Observed scan source must declare the imported scene Y-up frame')
        self.source_ids = {obj['id'] for obj in legacy['objects']}
        self.min_faces = min_faces

    def _observed_payload(self, target, scene, evidence, candidate, context):
        context.check_cancelled()
        for row in self.source_files.values():
            if file_hash(row['path']) != row['sha256']:
                raise ContractError('STALE_ARTIFACT', 'Frozen observed scan/source metadata changed')
        for key in ('scene_id', 'capture_revision', 'geometry_revision'):
            if scene[key] != self.source_snapshot[key] or evidence[key] != scene[key]:
                raise ContractError('EVIDENCE_MISMATCH', 'Observed scan is bound to another scene/capture/geometry')
        source_target = next((obj for obj in self.source_snapshot['objects']
                              if obj['instance_id'] == target['instance_id']), None)
        if source_target is None or target['instance_id'] not in self.source_ids:
            raise ContractError('EVIDENCE_MISMATCH', 'Target is absent from the frozen source scene')
        for key in ('instance_id', 'kind', 'category', 'local_bounds_m'):
            if target[key] != source_target[key]:
                raise ContractError('STALE_ARTIFACT', 'Target identity or declared bounds differ from source observation')
        mesh = trimesh.load(self.scan_path, force='mesh', process=False)
        cropped = crop_observed_mesh(mesh, source_target, min_faces=self.min_faces)
        context.check_cancelled()
        for row in self.source_files.values():
            if file_hash(row['path']) != row['sha256']:
                raise ContractError('STALE_ARTIFACT', 'Observed source changed while extracting the ROI')
        record = {'origin': 'rgb_inferred_surface_crop', 'quality_claim': QUALITY, 'partial': True,
                  'model_calls': 0, 'source_files': copy.deepcopy(self.source_files),
                  'source_snapshot_hash': canonical_hash(self.source_snapshot),
                  'source_target_hash': canonical_hash(source_target),
                  'source_object_transform': source_target['transform'],
                  'current_user_transform': target['transform'],
                  'scene_id': scene['scene_id'], 'capture_revision': scene['capture_revision'],
                  'geometry_revision': scene['geometry_revision'], 'candidate': candidate,
                  'evidence_id': evidence['evidence_id'], 'evidence_hash': evidence['evidence_hash'],
                  'rgb_sha256': {f['source_frame_id']: f['rgb']['sha256'] for f in evidence['frames']},
                  'cropped_mesh_hash': canonical_hash(cropped),
                  'limitations': ['Partial surface only; no hidden-side completion or chair template.',
                                 'A box ROI is not an exact instance mask and can include neighboring surfaces.',
                                 'Monocular depth, thin geometry, reflections and source color remain uncertain.']}
        return {'observed_mesh': cropped, 'observed_record': record}
