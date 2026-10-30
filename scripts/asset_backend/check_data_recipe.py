"""Isolated CPU check of all recipe primitives, GLB reimport and fixed cameras.

--fixture explicitly uses local test data and makes no model-quality claim.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/asset_backend'))
from video2scene.asset_backend import BlenderAssetBackend,FileRecipeProvider
from video2scene.asset_backend.recipe_author import install_local_fixture
from video2scene.contracts_v1 import ExecutionContext,canonical_hash,file_hash
from video2scene.contracts_v1.artifacts import ArtifactResolver


async def run(args):
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=False)
    proposal=Path(args.proposal).resolve();scene=json.loads((proposal/'scene_snapshot.json').read_text(encoding='utf8'))
    evidence=json.loads((proposal/'evidence.json').read_text(encoding='utf8'))
    if args.fixture:
        from test_recipes import fixture_recipe
        copied=output/'local_fixture_proposal';copied.mkdir()
        for name in ('upload_proposal.json','prompt.txt','output_schema.json','input.json','scene_snapshot.json','evidence.json'):
            shutil.copy2(proposal/name,copied/name)
        bundle=install_local_fixture(copied,fixture_recipe(scene))
    else:bundle=proposal/'recipe_bundle.json'
    source=ArtifactResolver(args.source_storage);resolver=ArtifactResolver(output)
    refs=[o['asset']['artifact'] for o in scene['objects']]+[f['rgb'] for f in evidence['frames']]
    for ref in refs:
        dst=resolver.path(ref['uri']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source.resolve(ref),dst)
        assert file_hash(dst)==ref['sha256']
    backend=BlenderAssetBackend(output,recipe_provider=FileRecipeProvider(bundle,allow_local_fixture=args.fixture),refinement_mode='image_recipe',samples=4)
    original=copy.deepcopy(scene);builds=[]
    for index,target in enumerate(original['objects']):
        identity=target['instance_id'];fingerprint=canonical_hash({'scene':scene,'target':identity,'bundle':file_hash(bundle)})
        candidate={'candidate_id':'recipecheck_'+str(index),'scene_id':scene['scene_id'],'base_revision':scene['revision'],
                   'base_head_seq':0,'candidate_hash':fingerprint,'state':'staged'}
        context=ExecutionContext(storage_root=output,actor='user',decision_ref='local_recipe_check',
            allowed_object_ids=frozenset([identity]),allowed_fields=frozenset(['asset.geometry','asset.materials']))
        spec={'candidate':candidate,'target_id':identity,'input_fingerprint':fingerprint,
              'allowed_fields':['asset.geometry','asset.materials'],'constraints':{'keep_root_transform':True,'keep_support':True,'keep_bounds':True},
              'budget':{'timeout_s':120,'max_model_calls':0,'max_iterations':1}}
        result=await backend.refine(spec,scene,evidence,context)
        for update in result['asset_updates']:
            obj=next(o for o in scene['objects'] if o['instance_id']==update['instance_id'])
            assert obj['transform']==target['transform'] and obj['support_id']==target['support_id']
            assert update['local_bounds_m']==obj['local_bounds_m'];obj['asset']=update['asset']
        for ref in result['artifact_refs']:resolver.resolve(ref)
        (output/f'build_{identity}.json').write_text(json.dumps(result,indent=2),encoding='utf8');builds.append({'instance_id':identity,'asset':result['asset_updates'][0]['asset']})
        print('CPU recipe compiled, assembled, and reopened: '+identity,flush=True)
    draft={k:v for k,v in scene.items() if k not in ('revision','parent_revision')}
    candidate['state']='built';candidate['candidate_hash']=canonical_hash(draft)
    renders=await backend.render({'candidate':candidate,'scene':draft},evidence,'normal',context,frame_ids=[f['source_frame_id'] for f in evidence['frames']])
    (output/'renders.json').write_text(json.dumps(renders,indent=2),encoding='utf8')
    (output/'scene_snapshot.json').write_text(json.dumps(scene,indent=2),encoding='utf8')
    report={'passed':True,'origin':'local_fixture' if args.fixture else 'codex_image_recipe','model_calls_this_check':0,
            'compiled_object_count':len(builds),'objects':builds,'render_frame_count':len(renders['frame_ids']),
            'all_roots_support_bounds_preserved':True,'device':'CPU','not_run':['independent visual quality assessment','repository commit','browser']}
    (output/'check_result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--proposal',required=True);parser.add_argument('--source-storage',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    asyncio.run(run(args))
