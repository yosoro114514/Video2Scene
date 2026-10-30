"""Strict JSON schema and independent semantic budgets for model-authored recipes."""
import json
import math

from jsonschema import Draft202012Validator

from video2scene.contracts_v1 import ContractError
from .recipe_geometry import compile_object

VERSION='video2scene.recipe/1'
def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
def array(items,minimum=1,maximum=64):
    return {'type':'array','items':items,'minItems':minimum,'maxItems':maximum}
NUMBER={'type':'number','minimum':-4,'maximum':4}
POSITIVE={'type':'number','exclusiveMinimum':0,'maximum':4}
VEC3=array(NUMBER,3,3)
QUAT=array({'type':'number','minimum':-1,'maximum':1},4,4)
TEXT={'type':'string','minLength':1,'maxLength':1200}
NAME={'type':'string','pattern':'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'}
MATERIAL=obj({'name':NAME,'base_color_rgb':array({'type':'number','minimum':0,'maximum':1},3,3),
              'metallic':{'type':'number','minimum':0,'maximum':1},'roughness':{'type':'number','minimum':0.04,'maximum':1}})
COMMON={'name':NAME,'center':VEC3,'rotation_xyzw':QUAT,'material':MATERIAL,'smooth':{'type':'boolean'}}
def part(kind,properties):return obj({'kind':{'type':'string','const':kind},**COMMON,**properties})
PART={'anyOf':[
    part('box',{'dimensions':array(POSITIVE,3,3)}),
    part('lathe',{'profile':array(array(NUMBER,2,2),3,48),'segments':{'type':'integer','minimum':8,'maximum':128}}),
    part('tube',{'path':array(VEC3,2,96),'radius':POSITIVE,'sides':{'type':'integer','minimum':6,'maximum':32},'caps':{'type':'boolean'}}),
    part('torus',{'major_radius':POSITIVE,'tube_radius':POSITIVE,'major_segments':{'type':'integer','minimum':8,'maximum':128},'minor_segments':{'type':'integer','minimum':6,'maximum':32}}),
    part('mesh',{'vertices':array(VEC3,4,4096),'faces':array(array({'type':'integer','minimum':0,'maximum':4095},3,64),4,8192)}),
]}
OBJECT_RECIPE=obj({'instance_id':NAME,'observed_features':array(TEXT,1,20),'assumptions':array(TEXT,0,20),'parts':array(PART,1,32)})
RECIPE_SCHEMA=obj({'recipe_version':{'type':'string','const':VERSION},'scene_id':NAME,
                   'coordinate_frame':{'type':'string','const':'normalized_object_y_up'},
                   'summary':TEXT,'objects':array(OBJECT_RECIPE,1,5)})


def validate_recipes(value, scene_snapshot):
    try:body=json.dumps(value,allow_nan=False).encode('utf8')
    except (ValueError,TypeError) as exc:raise ContractError('INVALID_ARGUMENT','Recipe must be finite JSON') from exc
    if len(body)>2_000_000:raise ContractError('RESOURCE_EXHAUSTED','Recipe JSON exceeds 2 MB')
    errors=list(Draft202012Validator(RECIPE_SCHEMA).iter_errors(value))
    if errors:raise ContractError('INVALID_ARGUMENT','Recipe does not match the data-only schema',{'errors':[e.message for e in errors[:4]]})
    by_id={o['instance_id']:o for o in scene_snapshot['objects']}
    ids=[o['instance_id'] for o in value['objects']]
    if value['scene_id']!=scene_snapshot['scene_id'] or len(ids)!=len(set(ids)) or set(ids)!=set(by_id):
        raise ContractError('FORBIDDEN_SCOPE','Recipe must describe exactly the prepared scene object IDs')
    vertices=0
    for recipe in value['objects']:
        for p in recipe['parts']:
            if abs(sum(x*x for x in p['rotation_xyzw'])-1)>1e-3:raise ContractError('INVALID_ARGUMENT','Recipe quaternion is not close enough to unit length')
            if p['kind']=='lathe' and any(point[0]<0 for point in p['profile']):raise ContractError('INVALID_ARGUMENT','Lathe radius must not be negative')
            if p['kind']=='torus' and p['tube_radius']>=p['major_radius']:raise ContractError('INVALID_ARGUMENT','Torus tube must be smaller than its major radius')
        try:_,stats=compile_object(recipe,by_id[recipe['instance_id']]['local_bounds_m'])
        except (ValueError,TypeError,KeyError) as exc:raise ContractError('INVALID_ARGUMENT','Recipe geometry rejected',{'reason':str(exc)}) from exc
        vertices+=stats['vertex_count']
    if vertices>150000:raise ContractError('RESOURCE_EXHAUSTED','Five-object recipe exceeds total vertex budget')
    return value
