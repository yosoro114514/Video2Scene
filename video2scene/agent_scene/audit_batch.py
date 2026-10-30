"""Verify the saved batch against its authorization, original layout and receipts."""
import argparse
from pathlib import Path
import uuid

from video2scene.contracts_v1 import canonical_hash, file_hash, utc_now
from video2scene.contracts_v1.artifacts import ArtifactResolver
from .batch_refinement import audit_transition, locked_layout
from .material_refinement import provider_receipt
from .planning import read, save
from .structural_edit import api


def audit(run_root, batch_root, url):
    run, batch = Path(run_root).resolve(), Path(batch_root).resolve()
    status, baseline = read(batch/'status.json'), read(batch/'baseline.json')
    authorization = read(batch/'authorization.json')
    assert status['phase'] == 'complete'
    assert status['authorization_sha256'] == baseline['authorization_sha256'] == file_hash(batch/'authorization.json')
    assert status['target_ids'] == baseline['target_ids'] == authorization['target_ids']
    frozen = read(batch/'baseline_snapshot.json')
    assert canonical_hash(frozen) == baseline['snapshot_hash'] == authorization['baseline_snapshot_hash']
    for name in ('capture_record','initial_snapshot','geometry_proposals'):
        assert file_hash(Path(baseline[name]['path'])) == baseline[name]['sha256']
    response = api(url,'/api/v1/tools/execute',{'contract_version':'video2scene/1.0',
        'request_id':'batch-audit-'+uuid.uuid4().hex,'scene_id':status['scene_id'],
        'tool':'inspect_scene','params':{'revision':'head'}})
    assert response['status'] != 'error', response
    view=response['result']; scene=view['snapshot']; targets=status['target_ids']
    assert view['head'] == status['current_head']
    assert len(status['completed']) == len(targets) == status['progress']['completed']
    assert canonical_hash(locked_layout(scene,targets)) == baseline['layout_hash']
    assert locked_layout(scene,targets) == locked_layout(frozen,targets)
    resolver = ArtifactResolver(run/'runtime')
    previous, head = frozen, baseline['head']
    receipts, objects, last = [], [], None
    for identity,row in zip(targets,status['completed']):
        path=Path(row['completion_path'])
        assert row['target_id'] == identity and file_hash(path) == row['completion_sha256']
        record=read(path); before=read(path.parent/'before.json'); after=read(path.parent/'after.json')
        assert before == previous and record['before_head'] == head
        assert record['head']['head_seq'] == head['head_seq']+1
        audit_transition(before,after,identity)
        assert canonical_hash(after) == record['after_snapshot_hash']
        assert record['critic']['status'] == 'accept' and not any(i['severity']=='major' for i in record['critic']['issues'])
        assert record['visual_policy'] == authorization['visual_policy'] == 'structure_first_relaxed_appearance'
        current=next(obj for obj in scene['objects'] if obj['instance_id']==identity)
        assert current['asset'] == row['asset'] == record['asset']
        resolver.resolve(current['asset']['artifact'],scene_id=status['scene_id'])
        for saved in record['receipt_records']:
            actual=provider_receipt(saved['directory'],record['model'])
            assert actual == saved
            receipts.append(actual)
        assert record['model_calls'] == len(record['receipt_records'])
        objects.append({'target_id':identity,'head':record['head'],'asset_sha256':current['asset']['artifact']['sha256'],
            'critic':record['critic'],'model_calls':record['model_calls'],'completion_path':str(path)})
        previous,head,last=after,record['head'],record
    assert previous == scene and head == view['head']
    assert len(set(row['directory'] for row in receipts)) == len(receipts)
    refs={ref['role']:ref for ref in last['artifact_refs']}
    resolve=lambda role:resolver.resolve(refs[role],scene_id=status['scene_id'])
    build=read(resolve('build_manifest')); folder=resolve('scene_blend').parent
    assert all(file_hash(folder/name)==digest for name,digest in build['files'].items())
    reopened=read(resolve('reopen_report'))
    assert reopened['passed'] is True
    web=api(url,'/agent/status?scene_id='+status['scene_id'])
    assert web['phase']=='complete' and web['head']==view['head'] and web['batch']['layout_preserved'] is True
    assert web['batch']['editing_locked'] is False
    result={'status':'complete','audited_at':utc_now(),'baseline_head':baseline['head'],'head':view['head'],
        'target_count':len(targets),'objects':objects,'model_calls':len(receipts),
        'provider_call_receipts':[row['directory'] for row in receipts],
        'all_transforms_bounds_support_preserved':True,'camera_capture_unchanged':True,
        'fully_preserved_objects':[obj['instance_id'] for obj in frozen['objects'] if obj['instance_id'] not in targets],
        'scene_blend':str(resolve('scene_blend')),'scene_glb':str(resolve('scene_glb')),
        'scene_reopen':reopened,'visual_policy':authorization['visual_policy'],
        'scope':'Single Redwood room with estimated scale; hidden geometry completed conservatively.'}
    save(batch/'final_snapshot.json',scene);save(batch/'final_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--batch-root',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8793')
    args=parser.parse_args();value=audit(args.run_root,args.batch_root,args.url)
    print({key:value[key] for key in ('status','head','target_count','model_calls','scene_blend')})
