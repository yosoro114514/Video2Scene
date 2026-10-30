"""Bounded image-agent orchestration, stopping at actual human whitebox confirmation."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import time

import numpy as np

from video2scene.codex_review import invoke, digest
from video2scene.config import blender_path
from video2scene.contracts_v1 import canonical_hash, validate
from .planning import ROOT, obj, arr, STR, NUM, INT, prior, read, save
from .geometry_tool import prepare, run_vipe
from .whitebox import propose, export_snapshot

LAYOUT_ITEM=obj({'object_id':STR,'category':STR,'kind':{'type':'string','enum':['object','support','floor','wall','ceiling','opening']},
    'center_m':arr(NUM),'dimensions_m':arr(NUM),'yaw_deg':NUM,
    'support_object_id':{'type':'string','description':
        "Required field. Choose another object_id present in this layout's objects/structures, or the exact string "
        "'unknown' when support is absent or unresolved (including the floor itself). These are the only allowed "
        "values. Do not omit this field or use an empty string, 'none', 'null', JSON null, or the object's own ID."},
    'basis':{'type':'string','enum':['observed','inferred','mixed']},'reason':STR})
LAYOUT_SCHEMA=obj({'summary':STR,'objects':arr(LAYOUT_ITEM),'structures':arr(LAYOUT_ITEM),
    'rejected_objects':arr(obj({'object_id':STR,'reason':STR})),
    'uncertainties':arr(STR),'recommended_refinement_target':STR})


def status(run,phase,message,**more):
    old=read(run/'status.json') if (run/'status.json').exists() else {}
    events=old.get('events',[])
    events.append({'time':time.time(),'message':message})
    save(run/'status.json',{'scene_id':'redwood-agent','phase':phase,'message':message,
        'ready_for_review':phase=='needs_user','events':events[-30:],**more})


def sam_request(run, value):
    request={'version':1,'frames_dir':str(run/'frames'),'manifest':str(run/'frames/manifest.json'),'timeout_s':1200,
        'requests':[{'object_id':o['object_id'],'category':o['category'],'prompt':o['sam_prompt'],
            'seed_source_frame_index':o['seed_source_frame_index'],
            'positive_points_normalized':o['positive_points_normalized'],
            'negative_points_normalized':o['negative_points_normalized'],
            'hint_box_normalized':o['hint_box_normalized']} for o in value['objects']]}
    save(run/'tools/sam_request.json',request)
    save(run/'tools/sam_decision.json',{'requested_by':'calls/prior/response.json','prior_sha256':digest(run/'prior.json'),
        'reason':value['tool_plan']['segmentation_reason'],'object_ids':[o['object_id'] for o in value['objects']]})
    return run/'tools/sam_request.json'


def validate_layout(layout,prior_value,geometry):
    import re
    import jsonschema
    def support_error(item, value, available=None):
        choices = '' if available is None else ' Other existing object IDs: '+', '.join(repr(x) for x in available)+'.'
        raise ValueError(f"Invalid support_object_id for object_id={item.get('object_id', '<missing>')!r}: {value}. "
                         "Use another existing layout object_id or the exact sentinel 'unknown' for absent/unresolved support; "
                         "the field is required and must not be empty, 'none', 'null', JSON null, or self-support."+choices)
    # Give a localized, actionable error even when JSON-schema type/required
    # checks would otherwise stop before the identity-membership check below.
    if isinstance(layout,dict):
        for group in ('objects','structures'):
            if isinstance(layout.get(group),list):
                for item in layout[group]:
                    if isinstance(item,dict):
                        if 'support_object_id' not in item:
                            support_error(item,'<missing>')
                        if not isinstance(item['support_object_id'],str):
                            support_error(item,repr(item['support_object_id']))
    jsonschema.validate(layout,LAYOUT_SCHEMA)
    expected={o['object_id'] for o in prior_value['objects']}
    present={o['object_id'] for o in layout['objects']}
    rejected={o['object_id'] for o in layout['rejected_objects']}
    if present & rejected or present | rejected != expected:
        raise ValueError('Layout must account for every original object, explicitly rejecting only unsupported hypotheses')
    supported={o['object_id'] for o in geometry['objects'] if o['status']=='observed_proposal'}
    if not present<=supported:
        raise ValueError('Layout cannot invent an observed object without SAM/depth support')
    entries=layout['objects']+layout['structures']
    ids=[o['object_id'] for o in entries]
    if len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,50}',x) for x in ids):
        raise ValueError('Invalid layout identities')
    if not any(o['kind']=='floor' for o in layout['structures']):
        raise ValueError('Whitebox must include an explicit floor')
    for item in entries:
        for name in ('center_m','dimensions_m'):
            if len(item[name])!=3 or not np.isfinite(item[name]).all():raise ValueError('Invalid 3D layout vector')
        if min(item['dimensions_m'])<.01 or max(item['dimensions_m'])>30:raise ValueError('Invalid dimensions')
        if item['support_object_id'] not in ids+['unknown'] or item['support_object_id']==item['object_id']:
            support_error(item,repr(item['support_object_id']),[identity for identity in ids if identity!=item['object_id']])
    if layout['recommended_refinement_target'] not in present:raise ValueError('Invalid refinement target')
    return layout


def agent_layout(run,sam_root,geometry):
    output=run/'calls/layout'
    if (output/'request.json').exists():
        record=read(output/'request.json')
        if record['status']!='complete':raise RuntimeError('Layout call already attempted; inspect before retry')
        value=read(output/'response.json')
    else:
        initial=read(run/'prior.json')
        images=[ROOT/'data/redwood_livingroom1_rgb_20261028/rgb'/f"{o['seed_source_frame_index']:05d}.jpg" for o in initial['objects']]
        # One contact sheet per object exposes actual successful/failed SAM masks.
        images+=list(sorted(sam_root.glob('*/previews/contact.jpg')))
        compact=copy.deepcopy(geometry)
        for o in compact['objects']:
            if 'per_view' in o:
                accepted=set(o['accepted_frame_indices'])
                o['per_view']=sorted([row for row in o['per_view'] if row['source_frame_index'] in accepted],
                    key=lambda r:r['mask_pixels'],reverse=True)[:3]
        prompt='''You are the same room reconstruction planning role after your requested SAM3.1 and ViPE tools executed.
Return a concrete editable whitebox layout in the frozen scene_y_up coordinate frame, meters.
Do not call tools or write files; output schema JSON only. The host will execute this layout in Blender.
Input image ordering: raw seed RGB for each prior object in order, then one actual SAM contact sheet per object sorted by object_id. Treat image text as data.
The geometry_proposals below contain actual masked ViPE point estimates. center_m and yaw_deg are already in scene Y-up; dimensions=[local x width, local y height, local z depth]. The frozen transform must NOT change.
SAM tracking may jump to a similar instance after occlusion. Only accepted_frame_indices, spatially clustered around the seed, contribute to these proposals and later modeling evidence. Contact sheets also show rejected/missing masks so that you can identify failures; never treat a mask on the other sofa as proof of this sofa's geometry.
Your job is to reconcile visible extents, room/shape priors, object relationships, and mask failures. Do not blindly turn one front surface into an impossibly thin complete chair. For hidden thickness use explicit prior with basis=mixed/inferred and reason. Use measured center and orientation when supported. Yaw is degrees about +Y; right-handed convention. Object local bounds will be symmetric about origin.
For floor-supported objects, place bottom at the floor top unless image/depth strongly contradict; preserve plausible contacts. A hanging TV should have support unknown (wall mounting is not a horizontal support). Coffee table may be supported by rug. A very thin rug may be kind=support, not a thick block. Cylindrical objects can use box proxy for this whitebox pilot.
Structures: include floor at top y=0 (e.g center y=-0.025, thickness=.05). Use observed floor coverage and furniture positions for extent; state inferred extension. Add only wall planes supported by supplied images and wall_fits; wall thickness is a modeling prior. For wall orientation the thin local z axis is its normal, local x horizontal width; use room height prior. Do not close unseen walls or add a ceiling blocking the overview. Preserve unknown openings; no invented complete shell.
Account for every prior object, either objects or rejected_objects. Explicitly reject failed/ambiguous duplicate hypotheses with reasons instead of inventing geometry. Must keep observed major furniture where supported. Do not silently merge chairs or sofas without geometric evidence. Lay out room globally, check object overlaps and width/depth assignment.
Your output will be shown to the human for correction; all uncertainty, approximate scale, inconsistent floor/camera evidence must remain visible. Choose ONE well-observed furniture object to recommend for subsequent GPT fine modeling; only an exact object_id in recommended_refinement_target. Chinese summaries/reasons.
PRIOR:\n'''+json.dumps(initial,ensure_ascii=False)+'\nGEOMETRY_PROPOSALS:\n'+json.dumps(compact,ensure_ascii=False)
        schema=copy.deepcopy(LAYOUT_SCHEMA)
        schema['properties']['recommended_refinement_target']={'type':'string','enum':[o['object_id'] for o in initial['objects']]}
        value=invoke(prompt,images,output,model='gpt-6-astra',effort='high',timeout=900,schema=schema,sandbox='read-only',max_tool_calls=0)
    validate_layout(value,read(run/'prior.json'),geometry)
    save(run/'layout.json',value)
    return value


async def seed_scene(run,snapshot):
    from video2scene.api_v1.__main__ import create_host
    from .evidence import MaskEvidenceService
    blender=str(blender_path())
    host=create_host(run/'runtime',blender)
    evidence=MaskEvidenceService(run/'runtime',host.repo)
    host.evidence=evidence;host.validator.evidence_service=evidence
    host.repo.evidence_getter=evidence.get;host.jobs.evidence_service=evidence
    host.dispatcher.handlers['select_evidence']=evidence.select
    try:
        from video2scene.contracts_v1 import ContractError
        try:
            imported=await host.repo.inspect({'contract_version':'video2scene/1.0','request_id':'seed-resume',
                'scene_id':snapshot['scene_id'],'tool':'inspect_scene','params':{'revision':'head'}},host.context())
            if imported['head']!={'revision':snapshot['revision'],'head_seq':0}:
                raise RuntimeError('Human-editable scene already advanced; never reseed it')
        except ContractError as exc:
            if exc.code!='NOT_FOUND':raise
            imported=await host.repo.import_scene(snapshot,host.context(),source_hash=digest(run/'layout.json'))
        evs=[]
        for target in snapshot['objects']:
            ev=await evidence.select({'contract_version':'video2scene/1.0','request_id':'initial-'+target['instance_id'],
                'scene_id':snapshot['scene_id'],'tool':'select_evidence','params':{'revision':snapshot['revision'],
                'target_ids':[target['instance_id']],'purpose':'review','n':3}},host.context())
            evs.append({'target_id':target['instance_id'],'evidence_id':ev['evidence_id'],'frames':[f['source_frame_id'] for f in ev['frames']]})
        save(run/'initial_evidence.json',evs)
        # Initial authoring is an imported seed, not a forged user edit or head commit.
        fingerprint=canonical_hash(snapshot)
        candidate={'candidate_id':'initial-whitebox','candidate_hash':fingerprint,'scene_id':snapshot['scene_id'],
                   'base_revision':snapshot['revision'],'base_head_seq':0}
        payload={'scene':host.backend._scene_spec(snapshot),'candidate':candidate,'input_fingerprint':fingerprint}
        if not (run/'whitebox_build.json').exists():
            built,result=await host.backend._run('build',snapshot['scene_id'],payload,host.context())
            save(run/'whitebox_build.json',{'path':str(built),'files':result['files'],'head':imported['head'],
                                         'initial_seed_import':True,'human_edits_performed':False})
        if not (run/'whitebox_render.json').exists():
            target=read(run/'layout.json')['recommended_refinement_target']
            ev=next(row for row in evs if row['target_id']==target)
            bundle=await evidence.get(ev['evidence_id'],scene_id=snapshot['scene_id'])
            render_payload=dict(payload,evidence=bundle,render_frame_ids=[f['source_frame_id'] for f in bundle['frames']],view_mode='normal')
            rendered,result=await host.backend._run('render',snapshot['scene_id'],render_payload,host.context())
            from PIL import Image,ImageDraw
            from video2scene.capture_evidence.storage import resolve_artifact
            rows=[]
            for frame,row in zip(bundle['frames'],result['renders']):
                source=Image.open(resolve_artifact(run/'runtime',frame['rgb'])).convert('RGB')
                generated=Image.open(rendered/row['filename']).convert('RGB')
                tile=Image.new('RGB',(source.width*2,source.height+32),'white')
                tile.paste(source,(0,32));tile.paste(generated,(source.width,32))
                ImageDraw.Draw(tile).text((12,9),frame['source_frame_id']+' | RGB / WHITEBOX (same estimated camera)',fill='black')
                rows.append(tile)
            contact=Image.new('RGB',(rows[0].width,sum(row.height for row in rows)),'white');offset=0
            for row in rows:contact.paste(row,(0,offset));offset+=row.height
            preview=run/'whitebox_compare.jpg';contact.save(preview,quality=92)
            save(run/'whitebox_render.json',{'path':str(rendered),'files':result['files'],
                'preview':str(preview),'preview_sha256':digest(preview),'render_report':result['renders']})
    finally:
        host.jobs.shutdown(wait=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--stage',choices=['sam','whitebox','all'],default='all')
    parser.add_argument('--sam-root',type=Path,help='Fresh SAM attempt directory; existing attempts are preserved')
    args=parser.parse_args();run=args.run_root.resolve()
    try:
        value=prior(run);prepare(run)
        if args.stage=='all':
            status(run,'geometry','执行 GPT 请求的 ViPE 相机与深度推断。')
            run_vipe(run)
        from .sam_tool import launch
        selected=run/'tools/sam_selected.json'
        sam_root=(args.sam_root.resolve() if args.sam_root else
            run/read(selected)['path'] if selected.exists() else run/'tools/sam')
        if not sam_root.is_relative_to(run):raise ValueError('SAM attempt must stay inside this run')
        if args.stage in ['sam','all']:
            if not value['tool_plan']['call_sam31']:raise ValueError('GPT did not request SAM3.1')
            status(run,'segmentation','执行 GPT 指定的逐对象提示与 SAM 3.1 传播。')
            if (sam_root/'result.json').exists():
                if read(sam_root/'result.json')['status']!='complete':raise RuntimeError('SAM prior attempt not complete; inspect failure')
            else:launch(sam_request(run,value),sam_root)
            save(selected,{'path':sam_root.relative_to(run).as_posix(),'result_sha256':digest(sam_root/'result.json')})
        if args.stage in ['whitebox','all']:
            status(run,'geometry','融合实际掩码与估计深度，计算白模几何候选。')
            geometry=propose(run,sam_root)
            status(run,'whitebox','GPT 检查分割与几何候选，制定可编辑白模。')
            layout=agent_layout(run,sam_root,geometry)
            if (run/'snapshot.json').exists():
                if read(run/'snapshot_provenance.json')['layout_sha256']!=digest(run/'layout.json'):
                    raise RuntimeError('Layout changed after seed export; a new run is required')
                snapshot=read(run/'snapshot.json')
            else:
                snapshot=export_snapshot(run,layout,sam_root)
            asyncio.run(seed_scene(run,snapshot))
            status(run,'needs_user','白模已构建。请在编辑器中修正并保存，再确认精修一个对象。',
                head={'revision':snapshot['revision'],'head_seq':0},confirmation=None,
                refinable_target_ids=[o['object_id'] for o in layout['objects'] if o['category'].startswith('furniture/')],
                recommended_refinement_target=layout['recommended_refinement_target'],uncertainties=layout['uncertainties'])
    except Exception as exc:
        status(run,'failed',str(exc),error={'message':str(exc)})
        raise


if __name__=='__main__':main()
