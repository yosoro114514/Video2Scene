"""Authorized Redwood structural finishes and additions, with actual author/review receipts.

The existing furniture and all accepted transforms remain frozen. Authoring and
independent GLB inspection precede the normal extension candidate transaction.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from video2scene.api_v1.__main__ import create_host
from video2scene.config import blender_path
from video2scene.capture_evidence import transform_to_matrix
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, utc_now
from video2scene.codex_review import obj, STR
from .direct_asset import DirectAssetBackend, local_blender_bounds, read, write
from .batch_refinement import immutable_json
from .material_refinement import provider_receipt
from .recovery import _atomic_json

SCENE = 'redwood-agent'
RUN_ID = 'room_finish_001'
REPLACE = ['floor', 'wall_tv', 'wall_sofa_color', 'wall_sofa_green']
ADD = ['interior_door', 'balcony_door', 'vase_tv_stand']
TARGETS = REPLACE + ADD
INSTRUCTION = '墙壁 地板 还有 门都应该上色/建模 另外tv stand 上面应该有个花瓶'
FIELDS = frozenset({'asset.geometry', 'asset.materials', 'scene.objects.add'})
AUTHOR_SCHEMA = obj({'status': {'type': 'string', 'enum': ['complete', 'failed']},
    'script_path': STR, 'assets': {'type': 'array', 'items': obj({'instance_id': STR,
    'asset_blend': STR, 'parts': {'type': 'array', 'items': STR},
    'observed_features': {'type': 'array', 'items': STR}, 'assumptions': {'type': 'array', 'items': STR}})}})
CRITIC_SCHEMA = obj({'status': {'type': 'string', 'enum': ['accept', 'revise', 'unclear']},
    'summary': STR, 'objects': {'type': 'array', 'items': obj({'instance_id': STR,
    'status': {'type': 'string', 'enum': ['accept', 'revise', 'unclear']}, 'summary': STR,
    'issues': {'type': 'array', 'items': obj({'severity': {'type': 'string', 'enum': ['minor','major']},
        'evidence': STR, 'requested_fix': STR})}})}, 'limitations': {'type': 'array', 'items': STR}})


class RoomFinish:
    def __init__(self, run_root):
        self.run = Path(run_root).resolve()
        self.root = self.run / RUN_ID
        self.runtime = self.run / 'runtime'
        self.assets = self.runtime / SCENE / 'assets' / 'gpt-room-finish-001'
        self.host = create_host(self.runtime, blender=blender_path())
        self.context = self.host.context(actor='user', session_id=RUN_ID, decision_ref=RUN_ID,
            allowed_object_ids=frozenset(TARGETS), allowed_fields=FIELDS)
        self.backend = DirectAssetBackend(self.runtime, confirmation_path=self.root/'authorization.json', samples=8)

    async def view(self):
        return await self.host.repo.inspect({'contract_version': CONTRACT_VERSION,
            'request_id': RUN_ID+'_inspect', 'scene_id': SCENE, 'tool':'inspect_scene',
            'params': {'revision':'head'}}, self.context)

    def status(self, phase, stage, message, **extra):
        auth = read(self.root/'authorization.json')
        value = {'scene_id': SCENE, 'run_id': RUN_ID, 'authorization_sha256': file_hash(self.root/'authorization.json'),
            'baseline_head': auth['baseline_head'], 'current_head': auth['baseline_head'],
            'phase': phase, 'stage': stage, 'message': message, 'targets': TARGETS,
            'completed': [], 'active_target': None, 'updated_at': utc_now(), **extra}
        _atomic_json(self.root/'status.json', value)
        print(json.dumps({'phase': phase, 'stage': stage, 'message': message}, ensure_ascii=False), flush=True)

    async def initialize(self):
        view = await self.view()
        if (self.root/'authorization.json').exists():
            auth = read(self.root/'authorization.json')
            if view['head'] != auth['baseline_head']:
                raise RuntimeError('Scene head differs from this finish authorization')
            return
        self.root.mkdir(parents=True, exist_ok=True)
        immutable_json(self.root/'baseline_snapshot.json', view['snapshot'])
        immutable_json(self.root/'authorization.json', {'run_id':RUN_ID, 'authorized':True, 'scene_id':SCENE,
            'authorization_source':'current_user_chat', 'user_instruction':INSTRUCTION,
            'authorized_at':utc_now(), 'baseline_head':view['head'],
            'baseline_snapshot_hash':canonical_hash(view['snapshot']), 'targets':TARGETS,
            'replace_asset_ids':REPLACE, 'add_object_ids':ADD,
            'visual_policy':'structure_first_relaxed_appearance', 'browser_confirmation_click_claimed':False})
        capture = self.runtime/SCENE/'captures'/(view['snapshot']['capture_revision']+'.json')
        immutable_json(self.root/'capture_binding.json', {'path':str(capture),'sha256':file_hash(capture)})
        self.status('running','localizing','正在核对门、花瓶的位置与原始图像；家具和已连接墙体的布局保持不变。')

    def frozen(self):
        base = read(self.root/'baseline_snapshot.json')
        auth = read(self.root/'authorization.json')
        if canonical_hash(base) != auth['baseline_snapshot_hash']:
            raise RuntimeError('Baseline changed')
        capture = read(self.root/'capture_binding.json')
        if file_hash(capture['path']) != capture['sha256']:
            raise RuntimeError('Frozen capture changed')
        return base, auth

    def prepare_author(self, specification):
        base, auth = self.frozen()
        work = self.assets/'author_00'
        work.mkdir(parents=True, exist_ok=False)
        specs = read(specification)
        objects = {o['instance_id']:o for o in base['objects']}
        targets = [copy.deepcopy(objects[i]) for i in REPLACE] + specs['additions']
        if [o['instance_id'] for o in targets] != TARGETS:
            raise RuntimeError('Unexpected author scope')
        source = Path(__file__).resolve().parents[2]/'data/redwood_livingroom1_rgb_20261028/rgb'
        refs = []
        for index in [300, 456, 1026, 1120, 1148, 1339, 2104, 2565]:
            dest=work/f'ref_{index:05d}.jpg'
            shutil.copyfile(source/f'{index:05d}.jpg',dest)
            refs.append(dest)
        payload = {'baseline_head':auth['baseline_head'], 'baseline_snapshot_hash':auth['baseline_snapshot_hash'],
            'user_instruction':INSTRUCTION, 'visual_policy':auth['visual_policy'],
            'objects':[{**o, 'local_bounds_blender':local_blender_bounds(o['local_bounds_m'])} for o in targets],
            'modeling_constraints':specs['modeling_constraints'], 'placement_evidence':specs.get('placement_evidence'),
            'reference_images':{p.name:file_hash(p) for p in refs}}
        payload['author_input_hash']=canonical_hash(payload)
        write(work/'contract.json',payload)
        immutable_json(self.root/'author_contract.json',payload)
        return work, payload, refs

    async def author(self, specification):
        work, contract, refs = self.prepare_author(specification)
        protected = {p.name:file_hash(p) for p in [work/'contract.json', *refs]}
        prompt = f'''You are the GPT modeller completing an already accepted Redwood living room.
Actually write and run {work/'object.py'} with Blender executable {self.backend.blender_path}.
Read contract.json. Construct precisely SEVEN separate native assets, saved to {{instance_id}}/asset.blend under this workspace, one empty factory scene per asset. The script must generate all seven and save them. Execute Blender --background --factory-startup --threads 4 --python-exit-code 1 --python object.py; one technical execution correction is allowed. Return complete only after successful real execution. Do not run a visual-review loop; the independent host will render and review exported GLBs.

Each asset is in LOCAL Blender XYZ, Z up. Its exact local bounds are in contract.objects[*].local_bounds_blender. Scene local Y-up becomes Blender (x,-z,y). Never apply accepted world transform/scale: the host applies it once. Root transforms, extents, existing furniture and room layout are frozen. Use the modeling_constraints for aperture position, visible-facing surfaces and dimensions. No other room objects, cameras, lighting, outdoor photo planes, or whole-room asset.

floor: warm natural wood plank floor seen in references; real shallow board seams or a packed image texture with staggered joints and fine wood grain. Cover the complete rectangular footprint, keep exact top surface height and overall bounds. Major plank geometry should follow room axes. Avoid thousands of unnecessary disconnected cubes and thick grooves.
wall_tv / wall_sofa_color / wall_sofa_green: continuous light warm gray/beige painted plaster/wallpaper with modest variation and white skirting. Keep exact outer footprint/height and joined corners; skirting must fit WITHIN the locked extent. Wall_sofa_color MUST have the true rectangular door aperture specified in its asset-local coordinates: construct side piers and lintel or boolean away that volume. Other walls remain solid. An opaque door painted onto a solid wall is not accepted. Model the cutout only at the defined location; preserve top and side wall continuity. Do not include the separate door asset in the wall.
interior_door: observed closed white flat door with frame, jamb/reveal and modest handle where not seen (record inferred detail). Coordinate the leaf/frame with the supplied wall cutout. It must sit at floor level, fill that aperture, preserve specified outer size/depth, and face into the room.
balcony_door: visible white metal double sliding frame, top rail, narrow center stile, floor threshold and TWO translucent glazed panes, as seen in frame2104. Keep supports connected. Use glTF-compatible glass via explicit alpha/roughness/IOR or transmission that survives GLB. Never paste an outdoor photograph. Curtains and outside balcony are out of scope.
vase_tv_stand: single ivory/light ceramic vase, ovoid rounded belly narrowing to foot, short narrow neck, flared open rim and a hollow dark interior. No flowers. It sits on the existing cabinet in the final assembly; do NOT include the cabinet or TV. Match the source silhouette and proportions in frame00300. Mesh interior/rim thickness must be real, not a solid sphere or black photo disk.

Colors, exact texture/roughness can differ moderately per user's permission; preserve major structure. Assets must be named editable meshes/curves/materials. Use glTF-compatible Principled materials and PACKED image textures, as shader-only procedural textures do not survive export. Any generated texture must be created here. Prefer low-frequency subtle plaster over noise clutter. Start each asset with bpy.ops.wm.read_factory_settings(use_empty=True); scene.world may be None, no need to use it. Do not access siblings/parents, existing fine assets, network, credentials, nested Codex, or other agents. Preserve contract.json and ref_* exactly. No unsupported Blender API assumptions. Return Chinese descriptions of each asset's actual parts and assumptions, and exact saved paths.
'''
        self.status('running','authoring','GPT6 正在构建墙面、木地板、两类门和陶瓷花瓶。')
        response = await asyncio.to_thread(self.backend._invoke,prompt,refs,work,self.context,
            timeout=1200,schema=AUTHOR_SCHEMA,sandbox='workspace-write',max_tool_calls=48)
        if response['status']!='complete' or {x['instance_id'] for x in response['assets']}!=set(TARGETS):
            raise RuntimeError('Author did not complete the exact scope')
        for name,digest in protected.items():
            if file_hash(work/name)!=digest: raise RuntimeError('Protected author input changed')
        script=Path(response['script_path']).resolve()
        if script!=work/'object.py' or not script.is_file(): raise RuntimeError('Wrong author script')
        events=[json.loads(line) for line in (work/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        commands=[e['item'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='command_execution'
            and e['item'].get('exit_code')==0 and 'blender' in e['item'].get('command','').lower()
            and '--python' in e['item'].get('command','') and 'object.py' in e['item'].get('command','')]
        if not commands: raise RuntimeError('No actual successful author-issued Blender execution')
        paths={}
        for row in response['assets']:
            path=Path(row['asset_blend']).resolve()
            if path!=work/row['instance_id']/'asset.blend' or not path.is_file(): raise RuntimeError('Wrong authored asset path')
            paths[row['instance_id']]={'path':str(path),'sha256':file_hash(path)}
        proof={'author_input_hash':contract['author_input_hash'], 'script_sha256':file_hash(script),
            'protected_inputs':protected,'assets':paths,'successful_blender_commands':commands,
            'provider_receipt':provider_receipt(work,'gpt-6-astra'),'actual_author_execution':True,'host_generated_geometry':False}
        write(work/'execution_proof.json',proof)
        immutable_json(self.root/'author_execution.json',proof)
        self.status('running','authored','7 项资产已由 GPT6 实际生成，开始独立导出与复核。')

    async def validate_assets(self):
        self.frozen()
        work=self.assets/'author_00'
        contract=read(work/'contract.json'); proof=read(work/'execution_proof.json')
        if contract['author_input_hash']!=proof['author_input_hash']: raise RuntimeError('Author contract mismatch')
        reviewed=[]
        for target in contract['objects']:
            identity=target['instance_id']; output=self.assets/('validated_'+identity)
            if output.exists():
                result=read(output/'result.json')
                if result['authored_sha256']!=proof['assets'][identity]['sha256']: raise RuntimeError('Stale validated native asset')
                for name,digest in result['files'].items():
                    if file_hash(output/name)!=digest: raise RuntimeError('Validated asset changed')
            else:
                self.status('running','exporting',f'正在复核 {identity} 的导出、尺寸和材质。',active_target=identity)
                asset_contract={'instance_id':identity, 'local_bounds_m':target['local_bounds_m'],
                    'local_bounds_blender':target['local_bounds_blender'],
                    'candidate':{'candidate_hash':contract['author_input_hash']},
                    'observation_transform':transform_to_matrix(target['transform']).tolist(),
                    'frames':[], 'crop_rectangles':[]}
                result=await asyncio.to_thread(self.backend._validate_authored,work/identity,output,asset_contract,
                    {'authored_blend_sha256':proof['assets'][identity]['sha256']},self.context,300)
            self.backend._montages(work/identity,output,{'crop_rectangles':[]})
            artifact=self.backend._artifact(output/'asset.glb',SCENE,'object_geometry','model/gltf-binary')
            reviewed.append({'instance_id':identity,'asset':{'asset_version':'gpt-room-'+artifact['sha256'][:24],
                'artifact':artifact,'native_frame':'asset_y_up','native_to_object':np.eye(4).tolist()},'result':result})
        immutable_json(self.root/'validated_assets.json',reviewed)
        self.status('running','exported','7 项 GLB 已重开并生成四视图，等待整体装配与独立视觉复核。')

    async def assemble(self):
        base, auth=self.frozen()
        if (await self.view())['head']!=auth['baseline_head']: raise RuntimeError('Scene changed during modeling')
        assets={row['instance_id']:row['asset'] for row in read(self.root/'validated_assets.json')}
        contract=read(self.root/'author_contract.json')
        objects={row['instance_id']:copy.deepcopy(row) for row in contract['objects']}
        additions=[]
        for identity in ADD:
            item=objects[identity];item.pop('local_bounds_blender',None);item['asset']=assets[identity]
            additions.append(item)
        request={'contract_version':CONTRACT_VERSION,'request_id':RUN_ID+'_extension','scene_id':SCENE,
            'tool':'extend_scene',**{'base_revision':auth['baseline_head']['revision'],'base_head_seq':auth['baseline_head']['head_seq']},
            'params':{'intent':'design_edit','decision_ref':RUN_ID,'additions':additions,
                'asset_replacements':[{'instance_id':i,'asset':assets[i]} for i in REPLACE]}}
        immutable_json(self.root/'extension_request.json',request)
        candidate=await self.host.repo.prepare_extension(request,self.context)
        immutable_json(self.root/'candidate.json',candidate)
        snapshot=await self.host.repo.read_candidate(candidate['candidate_id'],candidate['candidate_hash'],self.context)
        self.status('running','assembling','正在装配完整房间，核查门洞和花瓶支撑关系。')
        build=await self.backend.build(snapshot,self.context)
        # Register actual author/export proof files with the assembled candidate.
        paths=[(self.assets/'author_00'/'object.py','refinement_source','text/x-python'),
               (self.assets/'author_00'/'contract.json','author_contract','application/json'),
               (self.assets/'author_00'/'execution_proof.json','author_execution','application/json')]
        for identity in TARGETS:
            folder=self.assets/('validated_'+identity)
            paths.extend([(folder/'asset.glb','object_geometry','model/gltf-binary'),
                (folder/'asset.blend','object_blend','application/x-blender'),
                (folder/'result.json','refinement_manifest','application/json'),
                (folder/'turntable.jpg','object_turntable','image/jpeg')])
        build['artifact_refs'].extend(self.backend._artifact(path,SCENE,role,mime) for path,role,mime in paths)
        immutable_json(self.root/'build_output.json',build)
        source=next(ref for ref in build['artifact_refs'] if ref['role']=='scene_blend')
        folder=self.assets/'inspection';folder.mkdir(exist_ok=False)
        script=folder/'room_finish_inspect.py';shutil.copyfile(Path(__file__).with_name('room_finish_inspect.py'),script)
        inp={'scene_blend':str(self.backend.resolver.resolve(source,scene_id=SCENE)),
            'scene_blend_sha256':source['sha256'],'worker_sha256':file_hash(script),'output_dir':str(folder),
            'candidate_scene':snapshot['scene'],'baseline_snapshot':base,'replace_ids':REPLACE,
            'aperture':contract['modeling_constraints']['wall_sofa_color_aperture']}
        write(folder/'input.json',inp)
        await asyncio.to_thread(self.backend._process,[str(self.backend.blender_path),'--background','--factory-startup',
            '--threads','4','--python-exit-code','1','--python',str(script),'--',str(folder/'input.json')],
            folder,self.context,600,log_name='inspection.log')
        inspection=read(folder/'inspection.json')
        if not inspection['passed'] or inspection['source_scene_sha256']!=source['sha256']: raise RuntimeError('Room inspection failed')
        immutable_json(self.root/'assembly_inspection.json',inspection)
        self.status('running','assembled','整屋已重开验证，门洞贯通、花瓶落在柜面，进入 GPT6 独立视觉复核。')

    async def critic(self):
        self.frozen()
        author=self.assets/'author_00';output=self.assets/'critic_00'
        images=[]
        # Label every reviewed GLB so a reviewer cannot confuse the two walls or doors.
        for identity in TARGETS:
            source=self.assets/('validated_'+identity)/'turntable.jpg'
            with Image.open(source) as im:
                sheet=Image.new('RGB',(im.width,im.height+36),'#edf2ee');sheet.paste(im,(0,36))
                ImageDraw.Draw(sheet).text((12,10),identity,fill='#173a32')
                path=self.assets/('review_'+identity+'.jpg');sheet.save(path,quality=94);images.append(path)
        images.extend(self.assets/'inspection'/name for name in ['overview.png','top.png','door.png','vase.png'])
        images.extend(sorted(author.glob('ref_*.jpg')))
        prompt=f'''Independently review the seven new assets in an accepted Redwood room.
Images: first seven labeled four-view sheets rendered from REOPENED final GLBs, then actual assembled-room overview/top/door/vase renders, then original RGB reference photos. Review EACH of these identities exactly once: {', '.join(TARGETS)}.
The user's request is: {INSTRUCTION}. They allow moderate colors/wood grain/roughness differences. Existing furniture, room axes, dimensions and layout are intentionally fixed and verified unchanged. The photo camera geometry and approximate accepted room layout are not exact metric truth; do not demand moving furniture or changing room bounds.
Check: floor has recognizable warm wood planks; three continuous lightly colored painted/plaster walls retain connected room outline with skirting; wall_sofa_color contains an actual aperture aligned to its separate closed white interior door/frame; balcony door has connected white frame/rails/two glazed panes (outdoors/curtains out of scope); vase has ovoid ceramic belly, narrow short neck/open hollow rim and rests on the TV cabinet. A host mesh-ray check already verifies the real wall aperture, and a support check verifies vase/cabinet contact. Assess the visible result independently.
Reject only clear major missing/wrong structure, disconnected supports, door floating/in wall blocker, solid/no-opening vase mouth, gross wrong proportions, texture/billboard contamination, or an unusable rendering. A closed door leaf appropriately fills an opening. White/light warm gray paint still counts as intentional source-matched material; do not force saturated wall colors. Color/texture differences alone are minor. Record uncertain hidden details and approximate placement as limitations. Do not judge refinement from polygon count.
Return Chinese JSON with all seven objects individually accepted/revise/unclear and specific evidence. Overall accept requires every object accepted and no major issues. No tools, no code or unsupported claims of exact reconstruction. You are reviewing the actual supplied exported assets, not the author's description.'''
        self.status('running','visual_review','GPT6 正逐项检查墙面、地板、门与花瓶，并对照原始图像。')
        result=await asyncio.to_thread(self.backend._invoke,prompt,images,output,self.context,
            timeout=600,schema=CRITIC_SCHEMA,sandbox='read-only',max_tool_calls=0)
        write(output/'critic.json',result)
        receipt=provider_receipt(output,'gpt-6-astra')
        immutable_json(self.root/'critic_receipt.json',receipt)
        immutable_json(self.root/'visual_review.json',result)
        if (result['status']!='accept' or len(result['objects'])!=7 or
            {x['instance_id'] for x in result['objects']}!=set(TARGETS) or
            any(x['status']!='accept' or any(i['severity']=='major' for i in x['issues']) for x in result['objects'])):
            self.status('failed','visual_revision_needed','独立复核发现需要修正的项目，当前场景尚未替换。')
            raise RuntimeError('Independent visual review did not accept every asset')
        self.status('running','review_accepted','7 项资产均通过独立视觉复核，准备保存为新版本。')

    async def commit(self):
        base,auth=self.frozen();before=await self.view()
        if before['head']!=auth['baseline_head'] or before['snapshot']!=base: raise RuntimeError('Scene changed before save')
        review=read(self.root/'visual_review.json')
        if (review['status']!='accept' or {x['instance_id'] for x in review['objects']}!=set(TARGETS)
            or len(review['objects'])!=len(TARGETS) or any(x['status']!='accept' or
                any(i['severity']=='major' for i in x['issues']) for x in review['objects'])): raise RuntimeError('Missing independent acceptance')
        if review!=read(self.assets/'critic_00'/'critic.json'): raise RuntimeError('Critic report changed')
        if provider_receipt(self.assets/'critic_00','gpt-6-astra')!=read(self.root/'critic_receipt.json'): raise RuntimeError('Critic receipt changed')
        author_proof=read(self.root/'author_execution.json')
        if provider_receipt(self.assets/'author_00','gpt-6-astra')!=author_proof['provider_receipt']: raise RuntimeError('Author receipt changed')
        for folder in [self.assets/'author_00',self.assets/'critic_00']:
            record=read(folder/'request.json')
            for item in record['images']:
                if file_hash(item['path'])!=item['sha256']: raise RuntimeError('Actual model input image changed')
            for filename,key in [('prompt.txt','prompt_sha256'),('schema.json','schema_sha256')]:
                if file_hash(folder/filename)!=record[key]: raise RuntimeError('Actual model instruction changed')
        if file_hash(self.assets/'author_00'/'object.py')!=author_proof['script_sha256']: raise RuntimeError('Author script changed')
        for name,digest in author_proof['protected_inputs'].items():
            if file_hash(self.assets/'author_00'/name)!=digest: raise RuntimeError('Protected source changed')
        for identity,item in author_proof['assets'].items():
            if file_hash(item['path'])!=item['sha256']: raise RuntimeError('Authored native changed')
        candidate=read(self.root/'candidate.json'); build=read(self.root/'build_output.json')
        for ref in build['artifact_refs']: self.backend.resolver.resolve(ref,scene_id=SCENE)
        inspection=read(self.assets/'inspection'/'inspection.json')
        if inspection!=read(self.root/'assembly_inspection.json') or not inspection['passed']: raise RuntimeError('Inspection changed')
        if inspection['source_scene_sha256']!=next(r['sha256'] for r in build['artifact_refs'] if r['role']=='scene_blend'):
            raise RuntimeError('Reviewed assembly mismatch')
        for filename,digest in inspection['renders'].items():
            if file_hash(self.assets/'inspection'/filename)!=digest: raise RuntimeError('Reviewed assembly render changed')
        # The critic's provider request binds hashes of every actual supplied image.
        for path,role in [(self.assets/'critic_00'/'critic.json','room_visual_review'),
                          (self.assets/'inspection'/'inspection.json','room_geometry_inspection')]:
            build['artifact_refs'].append(self.backend._artifact(path,SCENE,role,'application/json'))
        frame_ids=['rgb-00300','rgb-00456','rgb-01034','rgb-01171','rgb-01327','rgb-02104','rgb-02557']
        evidence_request={'contract_version':CONTRACT_VERSION,'request_id':RUN_ID+'_evidence','scene_id':SCENE,
            'tool':'select_evidence','params':{'revision':auth['baseline_head']['revision'],
                'target_ids':TARGETS,'purpose':'validation','n':len(frame_ids)}}
        evidence=await self.host.evidence.select_for_extension(evidence_request,candidate['candidate_id'],candidate['candidate_hash'],
            self.context,frame_ids=frame_ids)
        immutable_json(self.root/'evidence.json',evidence)
        built=await self.host.repo.attach_build(candidate['candidate_id'],candidate['candidate_hash'],build,self.context)
        immutable_json(self.root/'built_candidate.json',built)
        persisted=await self.host.repo.read_candidate(built['candidate_id'],built['candidate_hash'],self.context)
        report=await self.host.validator.validate(persisted,evidence,'tabletop_design_v1',self.context)
        immutable_json(self.root/'validation.json',report)
        if not report['technical_pass'] or not report['committable']: raise RuntimeError('Scene extension validation failed')
        self.frozen()
        current=await self.view()
        if current!=before: raise RuntimeError('Scene changed before commit')
        self.status('running','committing','门、花瓶与饰面已通过逐项复核，正在保存整屋新版本。')
        request={'contract_version':CONTRACT_VERSION,'request_id':RUN_ID+'_commit','scene_id':SCENE,
            'tool':'commit_candidate','base_revision':auth['baseline_head']['revision'],'base_head_seq':auth['baseline_head']['head_seq'],
            'params':{'candidate_id':built['candidate_id'],'candidate_hash':built['candidate_hash'],
                'validation_id':report['validation_id'],'decision_ref':RUN_ID}}
        immutable_json(self.root/'commit_request.json',request)
        committed=await self.host.repo.commit(request,self.context)
        immutable_json(self.root/'commit_result.json',committed)
        await self.host.sync_outbox(SCENE)
        after=await self.view();scene=after['snapshot'];old={o['instance_id']:o for o in base['objects']};new={o['instance_id']:o for o in scene['objects']}
        if after['head']!={'revision':committed['revision'],'head_seq':committed['head_seq']}: raise RuntimeError('Head changed after commit')
        if set(new)!=set(old)|set(ADD): raise RuntimeError('Wrong committed object set')
        for identity,previous in old.items():
            if identity not in REPLACE and new[identity]!=previous: raise RuntimeError('Existing furnishing changed')
            for key in ['transform','local_bounds_m','support_id','category','kind']:
                if new[identity][key]!=previous[key]: raise RuntimeError('Existing layout changed')
        immutable_json(self.root/'final_snapshot.json',scene)
        completion={'status':'complete','scene_id':SCENE,'run_id':RUN_ID,
            'authorization_sha256':file_hash(self.root/'authorization.json'),'baseline_head':auth['baseline_head'],
            'head':after['head'],'final_snapshot_sha256':file_hash(self.root/'final_snapshot.json'),
            'assets':{i:new[i]['asset'] for i in TARGETS},'preservation':{'existing_layout':True,'non_targets':True},
            'artifact_refs':build['artifact_refs'],'model_calls':2,
            'provider_receipts':[author_proof['provider_receipt'],read(self.root/'critic_receipt.json')],
            'visual_review':review,'geometry_inspection':inspection,'completed_at':utc_now(),
            'scope':'Redwood room with frozen accepted approximate layout. Door/vase placement uses fresh SAM and existing ViPE; scale and hidden details remain approximate.'}
        immutable_json(self.root/'completion.json',completion)
        self.status('complete','saved','墙面、木地板、室内门、阳台玻璃门和电视柜花瓶均已复核保存。',
            current_head=after['head'],completed=[{'instance_id':i,'status':'accepted'} for i in TARGETS],
            completion_sha256=file_hash(self.root/'completion.json'))

    async def audit(self):
        self.frozen()
        saved=read(self.root/'completion.json'); current=await self.view()
        if current['head']!=saved['head'] or current['snapshot']!=read(self.root/'final_snapshot.json'):
            raise RuntimeError('Current scene differs from the saved finish')
        for ref in saved['artifact_refs']:self.backend.resolver.resolve(ref,scene_id=SCENE)
        for receipt in saved['provider_receipts']:
            if provider_receipt(receipt['directory'],'gpt-6-astra')!=receipt:raise RuntimeError('Provider proof changed')
        live=read(self.root/'live_actual_assets.json')
        if (live['head']!=saved['head'] or len(live['objects'])!=18 or live['errors'] or live['pageErrors']
            or live['pending'] or any(o['mode']!='asset' or not o['meshes'] for o in live['objects'])
            or live['refresh']['before']!=live['refresh']['after']):raise RuntimeError('Actual browser verification failed')
        report=read(self.root/'validation.json')
        if not report['technical_pass'] or not report['committable'] or saved['visual_review']['status']!='accept':
            raise RuntimeError('Saved candidate did not pass both technical and independent visual checks')
        scene_blend=next(r for r in saved['artifact_refs'] if r['role']=='scene_blend')
        result={'status':'complete','audited_at':utc_now(),'head':saved['head'],'objects':18,
            'accepted_targets':TARGETS,'model_calls':len(saved['provider_receipts']),
            'preserved_existing_furnishings':11,'all_existing_transforms_bounds_support_preserved':True,
            'capture_unchanged':True,'actual_browser_all_assets_loaded':True,'unchanged_refresh_keeps_assets':True,
            'technical_validation_id':report['validation_id'],'independent_visual_review':str(self.root/'visual_review.json'),
            'scene_blend':str(self.backend.resolver.resolve(scene_blend,scene_id=SCENE)),
            'completion_sha256':file_hash(self.root/'completion.json'),'browser_record_sha256':file_hash(self.root/'live_actual_assets.json')}
        write(self.root/'final_audit.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)


async def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('action',choices=['init','author','validate','assemble','critic','commit','finish','audit']);p.add_argument('--spec',type=Path)
    args=p.parse_args();task=RoomFinish(args.run_root)
    if args.action=='init': await task.initialize()
    elif args.action=='author': await task.author(args.spec)
    elif args.action=='validate': await task.validate_assets()
    elif args.action=='assemble': await task.assemble()
    elif args.action=='critic': await task.critic()
    elif args.action=='commit': await task.commit()
    elif args.action=='audit': await task.audit()
    elif args.action=='finish':
        await task.validate_assets()
        await task.assemble()
        await task.critic()
        await task.commit()


if __name__=='__main__': asyncio.run(main())
