"""Run a real CPU Blender build/refine/render demonstration in an independent run root."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from video2scene.capture_evidence import EvidenceService, migrate_prototype
from video2scene.contracts_v1 import (
    ExecutionContext,
    canonical_hash,
    file_hash,
    validate,
)

from .backend import BlenderAssetBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--storage-root', required=True, type=Path)
    parser.add_argument('--scene-id', default='t03-desktop')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    protected = args.source / 'revisions/r005/scene.blend'
    original_hash = file_hash(protected)
    draft = migrate_prototype(args.source, args.storage_root, args.scene_id)
    snapshot = dict(draft, revision='import-r005', parent_revision=None)
    context = ExecutionContext(actor='user', session_id='t03-demo', decision_ref='t03-explicit-refine',
                               allowed_object_ids=frozenset({'t001'}),
                               allowed_fields=frozenset({'asset.geometry', 'asset.materials'}), storage_root=args.storage_root)
    candidate = {'candidate_id': 't03-before', 'scene_id': args.scene_id, 'base_revision': 'import-r005',
                 'base_head_seq': 0, 'candidate_hash': canonical_hash(draft), 'state': 'staged'}

    class ImportedSnapshot:
        async def inspect(self, request, ctx):
            return {'head': {'revision': 'import-r005', 'head_seq': 0}, 'snapshot': copy.deepcopy(snapshot)}

    async def run():
        evidence = await EvidenceService(args.storage_root, ImportedSnapshot()).select(
            {'contract_version': 'video2scene/1.0', 'request_id': 't03-evidence', 'scene_id': args.scene_id,
             'tool': 'select_evidence', 'params': {'revision': 'head', 'target_ids': ['t001'], 'purpose': 'refine', 'n': 3}}, context)
        backend = BlenderAssetBackend(args.storage_root, samples=4)
        before_candidate = {'candidate': candidate, 'scene': draft}
        print('Actual five-object build and reopen', flush=True)
        build = await backend.build(before_candidate, context)
        print('Actual before render at source frame 40', flush=True)
        before_render = await backend.render(before_candidate, evidence, 'normal', context, frame_ids=['frame-000040'])
        print('Generate new single-cup asset, reopen, then assemble five-object scene', flush=True)
        spec = {'target_id': 't001', 'intent': 'design_edit', 'allowed_fields': ['asset.geometry', 'asset.materials'],
                'evidence_id': evidence['evidence_id'], 'constraints': {'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True},
                'budget': {'max_iterations': 1, 'timeout_s': 300}, 'decision_ref': context.decision_ref,
                'candidate': candidate, 'input_fingerprint': candidate['candidate_hash']}
        refined = await backend.refine(spec, snapshot, evidence, context)
        update = refined['asset_updates'][0]
        after_draft = copy.deepcopy(draft)
        for obj in after_draft['objects']:
            if obj['instance_id'] == 't001':
                obj['asset'] = update['asset']
                obj['local_bounds_m'] = update['local_bounds_m']
        after_ref = dict(candidate, candidate_id='t03-after', candidate_hash=canonical_hash(after_draft), state='built')
        after_candidate = {'candidate': after_ref, 'scene': after_draft}
        print('Actual after render at the same source frame', flush=True)
        after_render = await backend.render(after_candidate, evidence, 'normal', context, frame_ids=['frame-000040'])
        collisions = await backend.collisions(after_candidate, 'box', context)
        before_by_id = {obj['instance_id']: obj for obj in draft['objects']}
        after_by_id = {obj['instance_id']: obj for obj in after_draft['objects']}
        non_target_equal = all(after_by_id[key] == obj for key, obj in before_by_id.items() if key != 't001')
        root_preserved = after_by_id['t001']['transform'] == before_by_id['t001']['transform']
        support_preserved = after_by_id['t001']['support_id'] == before_by_id['t001']['support_id']
        bounds_preserved = after_by_id['t001']['local_bounds_m'] == before_by_id['t001']['local_bounds_m']
        changed = after_by_id['t001']['asset']['artifact']['sha256'] != before_by_id['t001']['asset']['artifact']['sha256']
        report = {'task': 'T03', 'actual_blender': True, 'device': 'CPU', 'scene_objects': len(draft['objects']),
                  'refined_target': 't001', 'new_procedural_asset': changed, 'non_target_objects_unchanged': non_target_equal,
                  'root_preserved': root_preserved, 'support_preserved': support_preserved, 'bounds_preserved': bounds_preserved,
                  'source_preserved': file_hash(protected) == original_hash, 'protected_sha256': original_hash,
                  'build': build, 'refinement': refined, 'before_render': before_render, 'after_render': after_render,
                  'collisions': collisions, 'evidence': evidence,
                  'repository_commit_run': False, 'model_calls': 0, 'litereality_integrated': False,
                  'quality': 'not assessed; procedural model is newly built, not a claimed reconstruction improvement',
                  'limitations': ['No installed LiteReality checkout; uses local Blender worker.',
                                  'Only geometry+materials refinement is currently supported.',
                                  'Collision screening uses conservative world bounds, not mesh intersection.',
                                  'Rendering requires rectified positive-focal-length camera intrinsics.']}
        report['passed'] = all([changed, non_target_equal, root_preserved, support_preserved, bounds_preserved, report['source_preserved']])
        for result in [build, refined]:
            validate(result, 'BuildOutput')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key: report[key] for key in ['passed', 'actual_blender', 'new_procedural_asset', 'non_target_objects_unchanged', 'source_preserved']}), flush=True)
        return report['passed']

    return 0 if asyncio.run(run()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
