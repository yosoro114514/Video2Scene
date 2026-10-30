"""Reproduce migration and evidence checks without a GPU or a model call."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

import numpy as np

from .coordinates import (
    BLENDER_TO_SCENE,
    SCENE_TO_BLENDER,
    bounds_corners,
    legacy_object_matrix,
    project,
    transform_points,
    transform_to_matrix,
    unproject,
)
from .prototype import migrate_prototype
from .service import EvidenceService
from .storage import file_hash, immutable_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--storage-root', required=True, type=Path)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--revision', default='r005')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    protected = args.source / 'revisions' / args.revision / 'scene.blend'
    before = file_hash(protected)
    draft = migrate_prototype(args.source, args.storage_root, args.scene_id, args.revision)
    snapshot = dict(draft, revision='import-' + args.revision, parent_revision=None)

    class ImportedSnapshot:
        async def inspect(self, request, context):
            return {'head': {'revision': snapshot['revision'], 'head_seq': 0}, 'snapshot': copy.deepcopy(snapshot)}

    service = EvidenceService(args.storage_root, ImportedSnapshot())
    old = read_json(args.source / 'revisions' / args.revision / 'scene.json')
    compiled = read_json(args.source / 'revisions' / args.revision / 'compiled.json')
    table_rotation = np.asarray(old['objects']['table_00']['base_matrix'])[:3, :3]
    object_results = []

    async def checks():
        for obj in draft['objects']:
            identity = obj['instance_id']
            legacy = legacy_object_matrix(old['objects'][identity], table_rotation)
            standard = transform_to_matrix(obj['transform'])
            points = bounds_corners(old['objects'][identity]['local_bounds'])
            expected = transform_points(BLENDER_TO_SCENE @ legacy, points)
            actual = transform_points(standard @ np.asarray(obj['asset']['native_to_object']), points)
            result = {'instance_id': identity,
                      'compiled_matrix_max_error': float(np.max(np.abs(legacy - compiled['objects'][identity]['matrix_world']))),
                      'world_point_max_error_m': float(np.max(np.abs(expected - actual))),
                      'roundtrip_matrix_max_error': float(np.max(np.abs(SCENE_TO_BLENDER @ standard @ BLENDER_TO_SCENE - legacy)))}
            req = {'contract_version': 'video2scene/1.0', 'request_id': 'migration-' + identity,
                   'scene_id': args.scene_id, 'tool': 'select_evidence',
                   'params': {'revision': 'head', 'target_ids': [identity], 'purpose': 'review', 'n': 3}}
            bundle = await service.select(req, None)
            assert await service.get(bundle['evidence_id']) == bundle
            errors = []
            for frame in bundle['frames']:
                pixels, depth = project(expected, frame['K'], frame['c2w'])
                recovered = unproject(pixels, depth, frame['K'], frame['c2w'])
                errors.append(float(np.max(np.abs(recovered - expected))))
            result.update(evidence_id=bundle['evidence_id'], evidence_hash=bundle['evidence_hash'],
                          selected_frame_ids=[f['source_frame_id'] for f in bundle['frames']],
                          projection_roundtrip_max_error_m=max(errors))
            object_results.append(result)

    asyncio.run(checks())
    report = {'task': 'T02', 'contract_version': 'video2scene/1.0', 'actual_prototype_migration': True,
              'transaction_repository_used': False, 'repository_note': 'Read-only imported snapshot for capture checks; transaction integration belongs to T00/T01.',
              'scene_id': args.scene_id, 'capture_revision': draft['capture_revision'],
              'geometry_revision': draft['geometry_revision'], 'objects': object_results,
              'source_blend_sha256_before': before, 'source_blend_sha256_after': file_hash(protected),
              'source_preserved': before == file_hash(protected),
              'not_run': ['new ViPE inference', 'Blender render/reopen', 'observed segmentation mask quality'],
              'limitations': ['Selection uses projected bounds and sharpness; occlusion is unknown without observed masks.',
                              'Source acquisition timestamps are unknown; preview clock uses 10 FPS.',
                              'Prototype geometry is estimated monocular scale.',
                              'Asset files are copied existing artifacts and are not reported as new refinement.']}
    report['passed'] = report['source_preserved'] and all(
        row['compiled_matrix_max_error'] < 1e-6 and row['world_point_max_error_m'] < 1e-6 and
        row['roundtrip_matrix_max_error'] < 1e-6 and row['projection_roundtrip_max_error_m'] < 1e-6
        for row in object_results)
    immutable_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
