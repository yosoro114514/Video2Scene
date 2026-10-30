"""Actually regenerate all five desktop objects, one asset at a time, using CPU Blender."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

import numpy as np

from video2scene.capture_evidence import EvidenceService, migrate_prototype
from video2scene.contracts_v1 import ExecutionContext, canonical_hash, file_hash
from video2scene.contracts_v1.artifacts import ArtifactResolver

from .backend import BlenderAssetBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--scene-id', default='t03-five-refined')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    original_path = args.source / 'revisions/r005/scene.blend'
    protected_hash = file_hash(original_path)
    draft = migrate_prototype(args.source, args.storage_root, args.scene_id)
    snapshot = dict(draft, revision='import-r005', parent_revision=None)

    class ImportedSnapshot:
        async def inspect(self, request, context):
            return {'head': {'revision': snapshot['revision'], 'head_seq': 0}, 'snapshot': copy.deepcopy(snapshot)}

    async def run():
        backend = BlenderAssetBackend(args.storage_root, samples=4)
        resolver = ArtifactResolver(args.storage_root)
        evidence_service = EvidenceService(args.storage_root, ImportedSnapshot())
        targets = ['t001', 't000', 't002', 't003', 'table_00']
        original_by_id = {obj['instance_id']: obj for obj in draft['objects']}
        final_draft = copy.deepcopy(draft)
        records = []
        for index, target in enumerate(targets):
            print(f'Actual new asset {index + 1}/5: {target}', flush=True)
            context = ExecutionContext(actor='user', session_id='t03-five', decision_ref='explicit-' + target,
                                       allowed_object_ids=frozenset({target}),
                                       allowed_fields=frozenset({'asset.geometry', 'asset.materials'}),
                                       storage_root=args.storage_root)
            candidate = {'candidate_id': 'five-staged-' + target, 'scene_id': args.scene_id,
                         'base_revision': snapshot['revision'], 'base_head_seq': 0,
                         'candidate_hash': canonical_hash({'scene': draft, 'target': target}), 'state': 'staged'}
            evidence = await evidence_service.select({'contract_version': 'video2scene/1.0',
                'request_id': 'five-evidence-' + target, 'scene_id': args.scene_id, 'tool': 'select_evidence',
                'params': {'revision': 'head', 'target_ids': [target], 'purpose': 'refine', 'n': 3}}, context)
            spec = {'target_id': target, 'intent': 'design_edit', 'allowed_fields': ['asset.geometry', 'asset.materials'],
                    'evidence_id': evidence['evidence_id'], 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                    'budget': {'max_iterations': 1, 'timeout_s': 300}, 'decision_ref': context.decision_ref,
                    'candidate': candidate, 'input_fingerprint': candidate['candidate_hash']}
            before_hash = canonical_hash(snapshot)
            output = await backend.refine(spec, snapshot, evidence, context)
            if canonical_hash(snapshot) != before_hash:
                raise AssertionError('Backend mutated input snapshot')
            assert len(output['asset_updates']) == 1
            update = output['asset_updates'][0]
            assert update['instance_id'] == target
            asset = update['asset']
            assert asset['artifact']['media_type'] == 'model/gltf-binary'
            assert asset['native_frame'] == 'asset_y_up'
            assert np.array_equal(asset['native_to_object'], np.eye(4))
            path = resolver.resolve(asset['artifact'])
            assert path.read_bytes()[:4] == b'glTF'
            assert asset['artifact']['sha256'] != original_by_id[target]['asset']['artifact']['sha256']
            reopen_ref = next(ref for ref in output['artifact_refs'] if ref['role'] == 'reopen_report')
            reopened = json.loads(resolver.resolve(reopen_ref).read_text(encoding='utf-8'))
            assert reopened['passed'] and reopened['max_local_bounds_error_m'] < 1e-5
            generation_ref = next(ref for ref in output['artifact_refs'] if ref['role'] == 'refinement_manifest')
            generation = json.loads(resolver.resolve(generation_ref).read_text(encoding='utf-8'))
            assert generation['new_geometry'] and generation['reopen_passed']
            for obj in final_draft['objects']:
                if obj['instance_id'] == target:
                    obj['asset'], obj['local_bounds_m'] = asset, update['local_bounds_m']
            records.append({'target': target, 'new_geometry': True, 'vertices': generation['vertices'],
                            'recipe': generation['recipe'], 'asset': asset,
                            'max_root_transform_error': reopened['max_root_transform_error'],
                            'max_imported_local_bounds_error_m': reopened['max_local_bounds_error_m'],
                            'input_snapshot_unchanged': True, 'output': output})
        context = ExecutionContext(storage_root=args.storage_root)
        candidate = {'candidate_id': 'five-all-built', 'scene_id': args.scene_id,
                     'base_revision': snapshot['revision'], 'base_head_seq': 0,
                     'candidate_hash': canonical_hash(final_draft), 'state': 'built'}
        final_snapshot = {'candidate': candidate, 'scene': final_draft}
        print('Assemble/reopen all five new GLB assets together', flush=True)
        combined = await backend.build(final_snapshot, context)
        evidence = await evidence_service.select({'contract_version': 'video2scene/1.0',
            'request_id': 'five-final-evidence', 'scene_id': args.scene_id, 'tool': 'select_evidence',
            'params': {'revision': 'head', 'target_ids': targets, 'purpose': 'review', 'n': 3}}, context)
        render = await backend.render(final_snapshot, evidence, 'normal', context,
                                      frame_ids=['frame-000000', 'frame-000040', 'frame-000079'])
        preserved = all(obj['transform'] == original_by_id[obj['instance_id']]['transform'] and
                        obj['support_id'] == original_by_id[obj['instance_id']]['support_id'] and
                        obj['local_bounds_m'] == original_by_id[obj['instance_id']]['local_bounds_m']
                        for obj in final_draft['objects'])
        report = {'task': 'T03', 'attempt': 'attempt_002', 'passed': preserved and file_hash(original_path) == protected_hash,
                  'actual_blender': True, 'new_assets_generated_count': len(records), 'objects': records,
                  'combined_build': combined, 'final_render': render, 'final_scene': final_draft,
                  'root_support_bounds_preserved': preserved, 'source_preserved': file_hash(original_path) == protected_hash,
                  'source_sha256': protected_hash, 'model_calls': 0, 'repository_commit_run': False,
                  'quality_verified': False, 'generation_method': 'project procedural Blender recipes; no LiteReality or image-conditioned model',
                  'scope_note': 'Five distinct actual per-object refine invocations followed by combined GLB build/reopen and three fixed-camera renders.'}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key: report[key] for key in ['passed', 'new_assets_generated_count', 'root_support_bounds_preserved', 'source_preserved']}), flush=True)
        return report['passed']

    return 0 if asyncio.run(run()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
