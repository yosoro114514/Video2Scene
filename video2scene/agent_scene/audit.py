"""Verify the real Redwood pilot's saved human correction and refined commit."""
from __future__ import annotations
import argparse
from pathlib import Path

from video2scene.contracts_v1 import canonical_hash, file_hash, utc_now
from .planning import read, save
from .structural_edit import api


def verified_calls(folders, asset_root):
    """Count completed real calls from unique original receipts, not copied logs."""
    folders=[Path(folder).resolve() for folder in folders]
    assert len(set(folders))==len(folders),'Duplicate provider receipts'
    records=[]
    for folder in folders:
        assert folder.is_relative_to(asset_root.resolve())
        request=read(folder/'request.json');outcome=read(folder/'invocation_outcome.json')
        response=read(folder/'response.json')
        assert request['backend']=='codex-cli' and request['status']=='complete'
        assert request['response_sha256']==file_hash(folder/'response.json')
        assert outcome['origin']=='codex_image' and outcome['status']=='complete'
        assert outcome['response_hash']==canonical_hash(response)
        assert (folder/'invocation_started.json').is_file() and (folder/'events.jsonl').is_file()
        records.append({'directory':str(folder),'model':request['requested_model'],
            'response_sha256':request['response_sha256'],'request_sha256':file_hash(folder/'request.json')})
    assert len({row['request_sha256'] for row in records})==len(records),'Copied provider receipts'
    return records


def audit(run_root,url):
    run=Path(run_root).resolve()
    status=api(url,'/agent/status?scene_id=redwood-agent')
    if status['phase']!='complete':raise RuntimeError('Real refinement is not complete: '+status['phase'])
    confirmation=read(run/'confirmation.json')
    base=read(run/'user_edits/walls_001/after.json')['snapshot']
    response=api(url,'/api/v1/tools/execute',{'contract_version':'video2scene/1.0',
        'request_id':'final-audit-'+confirmation['command_id'],'scene_id':'redwood-agent',
        'tool':'inspect_scene','params':{'revision':'head'}})
    if response['status']=='error':raise RuntimeError(response['error'])
    view=response['result'];scene=view['snapshot'];target=confirmation['target_id']
    if base['revision']!=confirmation['revision']:raise AssertionError('Confirmed scene is not the saved human correction')
    if view['head']['head_seq']!=confirmation['head_seq']+1:raise AssertionError('Unexpected intervening scene commit')
    old={o['instance_id']:o for o in base['objects']};new={o['instance_id']:o for o in scene['objects']}
    assert old.keys()==new.keys()
    unchanged=[key for key in old if key!=target and old[key]==new[key]]
    assert len(unchanged)==len(old)-1,'A non-target object changed'
    for field in ('transform','local_bounds_m','support_id','category','kind'):
        assert old[target][field]==new[target][field],field
    assert scene['capture_revision']==base['capture_revision']
    assert scene['geometry_revision']==base['geometry_revision']
    assert new[target]['asset']!=old[target]['asset']
    runs=[p for p in (run/'runtime/redwood-agent/assets').glob('gpt-direct-*/run.json')
        if read(p).get('candidate',{}).get('base_revision')==confirmation['revision']]
    assert len(runs)==1
    ledger=read(runs[0]);asset_root=runs[0].parent
    recovery_path=run/'recovery_completion.json'
    recovery=read(recovery_path) if recovery_path.exists() else None
    if recovery:
        assert recovery['status']=='complete' and recovery['head']==view['head']
        assert recovery['confirmed_base']=={'revision':confirmation['revision'],'head_seq':confirmation['head_seq']}
        assert recovery['confirmation_sha256']==file_hash(run/'confirmation.json')
        assert ledger['status']=='failed','Original failed author outcome must remain intact'
        assert Path(recovery['source_run']).resolve()==asset_root.resolve()
        from video2scene.contracts_v1.artifacts import ArtifactResolver
        validated=ArtifactResolver(run/'runtime').resolve(new[target]['asset']['artifact'],scene_id='redwood-agent').parent
        critic=recovery['critic'];fine_calls=recovery['total_fine_calls']
        folders=recovery.get('provider_call_receipts') or [
            asset_root/'author_00',asset_root/'critic_00',asset_root/'author_01',
            asset_root/'recovery_world_init_v1/critic']
        receipts=verified_calls(folders,asset_root)
        assert fine_calls==len(receipts)
    else:
        assert ledger['status']=='accepted_asset'
        iteration=ledger['accepted_iteration']
        validated=asset_root/f'validated_{iteration:02d}'
        critic=read(asset_root/f'critic_{iteration:02d}/critic.json');fine_calls=ledger['model_calls']
        receipts=verified_calls([asset_root/f'{role}_{i:02d}' for i in range(iteration+1)
            for role in ('author','critic')],asset_root)
        assert fine_calls==len(receipts)
    assert critic['status']=='accept' and all(i['severity']!='major' for i in critic['issues'])
    assert file_hash(validated/'asset.glb')==new[target]['asset']['artifact']['sha256']
    planning_calls=[]
    for name in ('prior','layout'):
        folder=run/'calls'/name;request=read(folder/'request.json')
        assert request['backend']=='codex-cli' and request['status']=='complete'
        assert request['response_sha256']==file_hash(folder/'response.json')
        assert all(file_hash(Path(image['path']))==image['sha256'] for image in request['images'])
        planning_calls.append({'stage':name,'model':request['requested_model'],
            'request_sha256':file_hash(folder/'request.json'),'images':len(request['images'])})
    builds=[]
    for path in (run/'runtime/redwood-agent/assets').glob('build-*/input.json'):
        value=read(path)
        if value.get('candidate',{}).get('base_revision')!=confirmation['revision']:continue
        built_target=next((o for o in value['scene']['objects'] if o['instance_id']==target),None)
        if built_target and built_target['asset']==new[target]['asset']:
            result=read(path.parent/'result.json')
            assert all(file_hash(path.parent/name)==sha for name,sha in result['files'].items())
            builds.append(path.parent)
    assert len(builds)==1,'Final assembled scene must be identified uniquely'
    result={'completed_at':utc_now(),'status':'complete','head':view['head'],'target_id':target,
        'human_input':'Actual chat feedback requesting connected, straight walls; saved through the normal versioned patch API.',
        'browser_confirmation_click_claimed':False,'human_revision':confirmation['revision'],
        'non_target_objects_unchanged':unchanged,'target_transform_bounds_support_preserved':True,
        'cameras_preserved':True,'whitebox_object_count':len(old),
        'model_calls':len(planning_calls)+fine_calls,'fine_model_calls':fine_calls,
        'planning_provider_receipts':planning_calls,
        'fine_author_calls':sum(Path(row['directory']).name.startswith('author') for row in receipts),
        'fine_provider_receipts':receipts,'independent_critic':critic,
        'scene_blend':str(builds[0]/'scene.blend'),'scene_glb':str(builds[0]/'scene.glb'),
        'object_blend':str(validated/'asset.blend'),'object_glb':str(validated/'asset.glb'),
        'object_glb_sha256':file_hash(validated/'asset.glb'),'fine_run':str(asset_root),
        'technical_recovery':str(recovery_path) if recovery else None,
        'scope':'One Redwood RGB-only pilot and one refined object; estimated scale, not room-wide reconstruction accuracy.'}
    save(run/'final_snapshot.json',scene);save(run/'final_status.json',status);save(run/'final_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8793')
    args=parser.parse_args();result=audit(args.run_root,args.url)
    print({key:result[key] for key in ('status','head','target_id','model_calls','scene_blend')})
