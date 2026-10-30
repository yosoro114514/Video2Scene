import copy
import json
from pathlib import Path

import pytest

from video2scene.asset_backend.recipe_author import FileRecipeProvider,install_local_fixture,invoke_author_once,write_new
from video2scene.asset_backend.recipe_geometry import compile_object
from video2scene.asset_backend.recipe_schema import validate_recipes
from video2scene.contracts_v1 import ContractError,canonical_hash,file_hash
from video2scene.contracts_v1.mocks import example


def material(color=(.4,.5,.6)):
    return {'name':'fixture_mat','base_color_rgb':list(color),'metallic':.1,'roughness':.4}


def part(kind,**properties):
    return {'kind':kind,'name':'fixture_'+kind,'center':[0,0,0],'rotation_xyzw':[0,0,0,1],
            'material':material(),'smooth':kind!='box',**properties}


def fixture_recipe(scene):
    """Locally authored exercise data; deliberately not claimed as source inference."""
    primitives=[part('box',dimensions=[.8,.9,.7]),
        part('lathe',profile=[[0,0],[.35,0],[.4,.9],[.35,1],[.3,.9],[.27,.1],[0,.1]],segments=24),
        part('tube',path=[[-.3,.2,0],[-.55,.3,0],[-.6,.65,0],[-.3,.85,0]],radius=.05,sides=12,caps=True),
        part('torus',major_radius=.4,tube_radius=.06,major_segments=32,minor_segments=8),
        part('mesh',vertices=[[-.5,0,-.5],[.5,0,-.5],[.4,0,.5],[-.5,0,.4],[-.5,.1,-.5],[.5,.1,-.5],[.4,.1,.5],[-.5,.1,.4]],
             faces=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])]
    objects=[]
    for index,target in enumerate(scene['objects']):
        p=copy.deepcopy(primitives[index%len(primitives)])
        objects.append({'instance_id':target['instance_id'],'observed_features':['Local compiler test only; no image model called'],
                        'assumptions':['Not an image reconstruction result'],'parts':[p]})
    return {'recipe_version':'video2scene.recipe/1','scene_id':scene['scene_id'],'coordinate_frame':'normalized_object_y_up',
            'summary':'Explicit local fixture exercising arbitrary data primitives','objects':objects}


@pytest.fixture
def prepared(tmp_path):
    scene=copy.deepcopy(example('model_SceneSnapshot'))
    prototype=copy.deepcopy(scene['objects'][1]);scene['objects']=[]
    for index in range(5):
        target=copy.deepcopy(prototype);target['instance_id']='obj'+str(index);target['support_id']=None;scene['objects'].append(target)
    evidence=copy.deepcopy(example('model_EvidenceBundle'));evidence.update(scene_id=scene['scene_id'],scene_revision=scene['revision'],
        target_ids=[o['instance_id'] for o in scene['objects']],capture_revision=scene['capture_revision'],geometry_revision=scene['geometry_revision'])
    image=tmp_path/'rgb.png';image.write_bytes(b'local fixture only');evidence['frames'][0]['rgb']['sha256']=file_hash(image)
    write_new(tmp_path/'scene_snapshot.json',scene);write_new(tmp_path/'evidence.json',evidence)
    (tmp_path/'prompt.txt').write_text('Local fixture. Never send to a model.',encoding='utf8')
    manifest={'scene_id':scene['scene_id'],'capture_revision':scene['capture_revision'],'geometry_revision':scene['geometry_revision'],
              'evidence_hash':evidence['evidence_hash'],'images':[{'path':str(image),'source_frame_id':evidence['frames'][0]['source_frame_id'],'sha256':file_hash(image)}],
              'files':{name:file_hash(tmp_path/name) for name in ('scene_snapshot.json','evidence.json','prompt.txt')}}
    write_new(tmp_path/'upload_proposal.json',manifest)
    return tmp_path,scene,evidence,fixture_recipe(scene)


def test_arbitrary_primitives_fit_locked_bounds(prepared):
    _,scene,_,recipe=prepared;validate_recipes(recipe,scene)
    for target,description in zip(scene['objects'],recipe['objects']):
        parts,stats=compile_object(description,target['local_bounds_m'])
        for axis in range(3):
            coords=[v[axis] for part in parts for v in part['vertices']]
            assert min(coords)==pytest.approx(target['local_bounds_m']['min'][axis])
            assert max(coords)==pytest.approx(target['local_bounds_m']['max'][axis])
        assert stats['part_count']==1


def test_recipe_controls_handle_direction_not_category_template(prepared):
    _,scene,_,recipe=prepared
    body=recipe['objects'][1]['parts'][0];handle=recipe['objects'][2]['parts'][0]
    cup={'instance_id':'cup','observed_features':['fixture'],'assumptions':[],'parts':[body,handle]}
    parts,_=compile_object(cup,scene['objects'][0]['local_bounds_m'])
    body_center=sum(v[0] for v in parts[0]['vertices'])/len(parts[0]['vertices'])
    handle_center=sum(v[0] for v in parts[1]['vertices'])/len(parts[1]['vertices'])
    assert handle_center<body_center
    mirrored=copy.deepcopy(cup);mirrored['parts'][1]['path']=[[-x,y,z] for x,y,z in handle['path']]
    parts2,_=compile_object(mirrored,scene['objects'][0]['local_bounds_m'])
    assert sum(v[0] for v in parts2[1]['vertices'])/len(parts2[1]['vertices'])>sum(v[0] for v in parts2[0]['vertices'])/len(parts2[0]['vertices'])


@pytest.mark.parametrize('mutation',[
    lambda r:r['objects'][0]['parts'][0].update(script='import os; os.remove(...)'),
    lambda r:r['objects'][0]['parts'][0].update(center=[float('nan'),0,0]),
    lambda r:r['objects'][0]['parts'][0].update(rotation_xyzw=[0,0,0,2]),
    lambda r:r['objects'][1]['parts'][0].update(segments=100000),
    lambda r:r['objects'][1]['parts'][0]['profile'].__setitem__(0,[-.1,0]),
    lambda r:r['objects'][4]['parts'][0]['faces'].__setitem__(0,[0,1,99]),
    lambda r:r['objects'][1].update(instance_id=r['objects'][0]['instance_id']),
])
def test_bad_recipe_rejected_before_blender(prepared,mutation):
    _,scene,_,recipe=prepared;mutation(recipe)
    with pytest.raises(ContractError):validate_recipes(recipe,scene)


def test_small_model_quaternion_rounding_normalized_and_recorded(prepared):
    _,scene,_,recipe=prepared;recipe['objects'][0]['parts'][0]['rotation_xyzw']=[0,.7071,0,.7071]
    validate_recipes(recipe,scene)
    _,stats=compile_object(recipe['objects'][0],scene['objects'][0]['local_bounds_m'])
    assert stats['part_rotation_numeric_normalization'][0]['input']==[0,.7071,0,.7071]
    assert stats['part_rotation_numeric_normalization'][0]['normalized'][1]==pytest.approx(2**-.5)


def test_fixture_cannot_claim_image_author_origin(prepared):
    folder,scene,evidence,recipe=prepared;bundle=install_local_fixture(folder,recipe)
    with pytest.raises(ContractError) as error:FileRecipeProvider(bundle).get_recipe(scene['objects'][0],scene,evidence,None)
    assert error.value.code=='FORBIDDEN_SCOPE'
    result=FileRecipeProvider(bundle,allow_local_fixture=True).get_recipe(scene['objects'][0],scene,evidence,None)
    assert result['provenance']['model_calls']==0 and result['provenance']['origin']=='local_fixture'


def test_recipe_binding_and_mutated_bundle_rejected(prepared):
    folder,scene,evidence,recipe=prepared;bundle=install_local_fixture(folder,recipe);provider=FileRecipeProvider(bundle,allow_local_fixture=True)
    changed=copy.deepcopy(scene['objects'][0]);changed['local_bounds_m']['max'][0]*=2
    with pytest.raises(ContractError) as error:provider.get_recipe(changed,scene,evidence,None)
    assert error.value.code=='STALE_ARTIFACT'
    bundle.write_text(bundle.read_text(encoding='utf8')+' ',encoding='utf8')
    with pytest.raises(ContractError) as error:provider.get_recipe(scene['objects'][0],scene,evidence,None)
    assert error.value.code=='STALE_ARTIFACT'


def test_missing_specific_upload_approval_never_invokes(prepared):
    folder,*_=prepared;calls=[]
    with pytest.raises(ContractError) as error:invoke_author_once(folder,invoker=lambda *a,**kw:calls.append(a))
    assert error.value.code=='FORBIDDEN_SCOPE' and calls==[]
    assert not (folder/'author_call_started.json').exists()


def test_uncertain_call_is_not_retried(prepared):
    folder,*_=prepared
    write_new(folder/'upload_authorization.json',{'status':'approved','proposal_sha256':file_hash(folder/'upload_proposal.json'),'approval_id':'offline_unit_test_only'})
    calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise TimeoutError('simulated lost response; no network called')
    with pytest.raises(TimeoutError):invoke_author_once(folder,invoker=fail)
    with pytest.raises(ContractError) as error:invoke_author_once(folder,invoker=fail)
    assert error.value.code=='IDEMPOTENCY_CONFLICT' and len(calls)==1
    assert calls[0]['timeout']==240 and calls[0]['max_tool_calls']==0
    assert json.loads((folder/'author_call_outcome.json').read_text())['retry_allowed'] is False
