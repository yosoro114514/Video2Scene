"""Persistent project, human-event and tool-job host for the Pi room workflow."""
from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import uuid

from video2scene.api_v1.__main__ import create_host
from video2scene.contracts_v1 import ContractError, canonical_hash, file_hash, utc_now
from video2scene.config import blender_path, load_settings
from .pi_runtime import PiBridge, ROOT, save
from .scene_actions import SceneActions, TOOL_SCHEMAS
from .video_tools import prepare_video, run_vipe_tool, run_sam_tool
from .batch import BatchWorkflowMixin, remaining_objects
from .resources import ResourceBudget
from .rolling_jobs import RollingJobs
from .process_owner import process_birth,owner_live


def identifier(prefix):return prefix+'_'+uuid.uuid4().hex[:20]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def schema(properties=None,required=None):
    properties=properties or {}
    return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}
S={'type':'string'}
EXTRA_TOOLS={
    'start_refinement_jobs':{'description':'Authorize rolling background dispatch for the exact current human-approved batch. Persist an operation_id and all batch target_ids, then return immediately. Completed packages wake Pi; do not busy-poll or restart jobs.',
        'parameters':schema({'acceptance_id':S,'operation_id':S,'target_ids':{'type':'array','minItems':1,'maxItems':128,'items':S}})},
    'refinement_jobs_status':{'description':'Read durable background tasks, packages, errors and events after a cursor. No new work is launched. Save ready packages with save_refinement; if none are ready, end the turn and wait for a host wakeup.',
        'parameters':schema({'after':{'type':'integer','minimum':0}},[])},
    'workflow_status':{'description':'Read authoritative project, jobs, current scene head, human acceptance and unresolved work. Always use after resume.', 'parameters':schema()},
    'inspect_video':{'description':'Decode the project input video and inspect real uniformly sampled RGB keyframes. Returns actual image attachments and a host observation receipt for record_prior.',
        'parameters':schema({'frame_budget':{'type':'integer','minimum':24,'maximum':160},'keyframe_count':{'type':'integer','minimum':6,'maximum':16}},[])},
    'view_frames':{'description':'Inspect more already prepared original-source frames, by exact source index. Returns images and a merged verified observation receipt.',
        'parameters':schema({'source_frame_indices':{'type':'array','items':{'type':'integer'},'minItems':1,'maxItems':12}})},
    'estimate_geometry':{'description':'Run ViPE on a separate dense 10 FPS temporal video, including every observed source frame. It does NOT use the sparse GPT/SAM keyframe clip. A real GPU task; completed identical input is reused only after hash verification.',
        'parameters':schema({'reason':S,'retry_reason':S},['reason'])},
    'segment_instances':{'description':'Run SAM3.1 segmentation/propagation for the currently recorded Pi prior objects. Prompts/points come from that prior. Returns real mask previews.',
        'parameters':schema({'reason':S,'retry_reason':S},['reason'])},
    'request_human_review':{'description':'Show the newly saved whitebox and ask the real user to edit/save/accept. Stop modeling until a verified acceptance exists.',
        'parameters':schema({'message':S})},
    'repair_whitebox_supports':{'description':'Retry an exact previously rejected build_whitebox draft using its workflow_status job id, correcting invalid support IDs only. Use an existing layout object_id or literal unknown for no parent. Preserves every other draft field and never overwrites an existing scene.',
        'parameters':schema({'failed_job_id':S,'support_corrections':{'type':'array','minItems':1,'maxItems':32,
            'items':schema({'object_id':S,'support_object_id':S})}})},
    'finish_workflow':{'description':'Mark delivery complete only after a real human acceptance, at least one accepted refined-asset commit, and fresh full-scene export.',
        'parameters':schema({'summary':S})},
}

SYSTEM_PROMPT='''你是视频房间建模 Agent，由 Pi 维护真正的模型—工具循环，使用项目配置的模型和用户现有 Codex 订阅。
当batch.mode=rolling时，先start_refinement_jobs授权既有完整批次自动补位，工具立即返回。后台对象独立生成和复核，Pi用refinement_jobs_status读取就绪包，用save_refinement逐件合并、验证、保存。等待期间不循环空查，结束本轮，宿主会在包就绪后唤醒。同一场景只有一个保存通道；宿主只补位已授权对象。全部目标保存且后台收尾后再export_scene和finish_workflow。不要调用旧refine_object或refine_objects_parallel；不要重放失败或结果不明任务。
通过已有结构化工具查看视频、形成先验、按需运行 ViPE/SAM、测量、构建白模、接收真实人工修正、精修、验证并保存。所有场景变化只能通过这些工具；没有shell/文件写工具。
GPT观察帧与ViPE时序输入分开：inspect_video用于看图和SAM对齐；estimate_geometry独立从原视频准备10fps密集序列，并包含所有观察源帧。旧96帧稀疏ViPE结果仅供问题诊断，不能作为密集输入成功凭据。先检查新的floor跨视角一致性和物体支撑，若仍严重漂移应说明并请求补充证据，不能只靠先验掩盖几何失败。
每次开始/恢复先读 workflow_status。新项目用 inspect_video 看真实图像，依据看到的图像用 record_prior 提交 PRIOR_SCHEMA 完整先验，observation_receipt 必须来自看图工具。帧号必须是实际看到的 source_frame_index；不要使用采样序号冒充原帧号。对于遗漏视角可view_frames检查已准备序列中的帧。第一次白模应覆盖主要可见家具；门窗、墙、地板也是结构清单的一部分。禁止将没有看到的物体标记为观测事实。
PRIOR: 所有 dimensions 长度3，米、约略值；每对象object_id稳定小写ID，seed正点必须落在该实例内部，负点排除同类邻居，hint_box为[左,上,宽,高]归一化(x,y,width,height)。floor_observations选择清晰可见地面区域(避开地毯/家具)，wall_observations选择墙面区域(避开挂画/电视/门)；两者每条polygon至少3点。类别用 furniture/chair, furniture/sofa, furniture/table, furniture/cabinet, electronics/tv, furnishing/rug, lighting/floor_lamp, decoration/potted_plant 等。第一版SAM最多12个实例；额外小物体/开口可列入room关系和uncertainties。建议先选最清晰桌子作为精修目标。
依据需要显式调用 estimate_geometry、segment_instances；同输入完成结果可缓存，但失败不可自动假冒完成。工具等待时会向页面报告进度，不需循环空查。取得两者后 measure，查看返回图像与结构化测量，再由你提交build_whitebox完整layout。当前几何融合需要相机/深度及掩码，因此未取得两者不要构建。objects必须逐一对应prior对象，unsupported应放rejected_objects并解释；structures包含floor和有证据的wall/opening。世界Y向上，floor顶面y=0；局部薄墙z为法线，yaw围绕Y。应让有证据相接的墙连续共角，保持地板水平；明显非直角墙不能强制矩形。合理隐藏厚度记mixed/inferred。家具在地板或支撑上，不能悬空或严重重叠。
build_whitebox完成后必须request_human_review并清楚告诉用户在页面修改/保存/采用，结束当前轮。模型绝不能生成acceptance或假装用户点击；已有真实批准才可继续。用户数值编辑会保存新版本；恢复时以最新head为准。
非rolling批次的单对象流程：有acceptance_id后，对获准target_ids逐个refine_object，再validate_candidate，再commit_candidate。refine_object内是有界Codex作者+独立critic工具工作进程，Pi仍决定整体推进。所有模式均保持已接受的根变换、边界、支撑、其他对象；颜色/纹理允许适度偏差，重大结构问题不可放宽。失败时先解释实际原因，不自动重放失败或结果不明的作者调用；需要修正原因并核对已有产物后再决定后续操作。完成授权目标后export_scene，再finish_workflow。
仅当batch.mode不是rolling且batch.parallelism=2时遵循宿主run_targets：用refine_objects_parallel并发生成最多两个独立对象资产包，各自经过Codex作者和critic。然后逐个stage_refinement→validate_candidate→commit_candidate，一个保存后再合并另一个。对象失败不影响其他成功资产；已有package/candidate必须复用，禁止重做已保存对象或重放结果不明的调用。本轮结束由宿主衔接下一轮，全部保存后才导出交付。所有布局和已经精修的资产必须保留。
白模阶段不要宣称已完成精模。收到用户解释问题时读场景和证据后回答，不在仅查询时改场景。编辑请求有human_event时仅在其授权字段对象内apply_scene_patch→validate→commit；没有授权不能自行改已接受布局。
精简中文沟通：当前发现、正在处理、还需何种真实反馈。因限额/认证失败保留状态，禁止改用API Key或别的模型。依据工具结果作决策，不凭文件名宣称执行成功。每个新工具调用给出具体参数，普通场景总工具预算64。
'''


class WorkflowService(BatchWorkflowMixin):
    def __init__(self,root,port=8794,*,resource_limits=None):
        self.settings=load_settings()
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self._closing=False
        receipt=self.root/'service.json'
        if receipt.exists():
            prior=read(receipt)
            if prior.get('pid')!=os.getpid() and (owner_live(prior) if prior.get('process_birth') else process_birth(prior.get('pid',0)) is not None):
                raise ContractError('RESOURCE_EXHAUSTED','Existing workflow service is still alive; stop it before opening the same project root')
        self.runtime=self.root/'runtime';self.port=port;self.token=secrets.token_urlsafe(32)
        self.lock=threading.RLock();self.bridges={};self.actions={};self.cancel={};self.db_path=self.root/'workflow.sqlite'
        with self.db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,project_id TEXT,type TEXT,data TEXT,at TEXT);
            CREATE TABLE IF NOT EXISTS requests(scope TEXT,id TEXT,hash TEXT,result TEXT,PRIMARY KEY(scope,id));
            CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,project_id TEXT,data TEXT);
            CREATE TABLE IF NOT EXISTS human_events(id TEXT PRIMARY KEY,project_id TEXT,data TEXT);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,project_id TEXT,state TEXT,data TEXT);''')
        self.host=create_host(self.runtime,blender=blender_path(self.settings.blender))
        self.resources=ResourceBudget(**(resource_limits or {'model':3,'blender':1}))
        if self.host.backend:self.host.backend.resource_budget=self.resources
        self.rolling=RollingJobs(self,start=False)
        for p in self.projects():
            if p.get('active_run') or p.get('batch',{}).get('mode')=='rolling' and p['batch']['status']=='running':
                p.update(active_run=None,phase='paused',error='服务重启，等待恢复并核对已有工具产物。');self.put(p)
        with self.db() as db:db.execute("UPDATE jobs SET state='uncertain' WHERE state='running'")
        # A restarted host cannot adopt an in-flight GPU subprocess. Ask its own
        # recorded supervisor to stop; never signal a bare or recycled PID.
        for process_record in (self.root/'projects').glob('*/tools/*/*/process.json'):
            folder=process_record.parent
            if not (folder/'process_result.json').exists():
                save(folder/'cancel.request',{'reason':'workflow_host_restarted','at':utc_now()})
        self.rolling.start_supervisor()

    def db(self):
        db=sqlite3.connect(self.db_path,timeout=30);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');return db
    def get(self,identity):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}',identity or ''):raise ValueError('Invalid project ID')
        with self.db() as db:row=db.execute('SELECT data FROM projects WHERE id=?',(identity,)).fetchone()
        if not row:raise KeyError('Project not found')
        return json.loads(row['data'])
    def put(self,p):
        p['updated_at']=utc_now()
        with self.db() as db:db.execute('INSERT OR REPLACE INTO projects VALUES(?,?)',(p['id'],json.dumps(p,ensure_ascii=False)))
    def projects(self):
        with self.db() as db:return [json.loads(r['data']) for r in db.execute('SELECT data FROM projects ORDER BY rowid DESC')]
    def event(self,pid,kind,**data):
        with self.db() as db:
            cur=db.execute('INSERT INTO events(project_id,type,data,at) VALUES(?,?,?,?)',(pid,kind,json.dumps(data,ensure_ascii=False),utc_now()));return cur.lastrowid
    def events(self,pid,after=0):
        self.get(pid)
        with self.db() as db:rows=db.execute('SELECT * FROM events WHERE project_id=? AND seq>? ORDER BY seq LIMIT 250',(pid,after)).fetchall()
        result=[{'seq':r['seq'],'type':r['type'],'data':json.loads(r['data']),'at':r['at']} for r in rows]
        for r in result:r['text']=r['data'].get('text')
        return {'events':result,'cursor':result[-1]['seq'] if result else after}
    def run_root(self,pid):return self.root/'projects'/self.get(pid)['id']

    def _dedup(self,scope,request,perform):
        rid=request.get('request_id')
        if not isinstance(rid,str) or not rid or len(rid)>128:raise ValueError('request_id is required')
        digest=canonical_hash(request)
        with self.lock:
            with self.db() as db:row=db.execute('SELECT hash,result FROM requests WHERE scope=? AND id=?',(scope,rid)).fetchone()
            if row:
                if row['hash']!=digest:raise ValueError('request_id reused with different input')
                if row['result']=='null':raise ValueError('This request already started; inspect project state before retrying')
                return json.loads(row['result'])
            with self.db() as db:db.execute('INSERT INTO requests VALUES(?,?,?,?)',(scope,rid,digest,'null'))
            result=perform()
            with self.db() as db:db.execute('UPDATE requests SET result=? WHERE scope=? AND id=?',(json.dumps(result,ensure_ascii=False),scope,rid))
            return result

    def create_project(self,request):
        def perform():
            source=Path(request.get('source_path','')).resolve()
            if not source.is_file() or source.suffix.lower() not in {'.mp4','.mov','.avi','.mkv','.webm','.m4v'}:raise ValueError('请上传或指定一个可读取的视频文件')
            pid=identifier('room');run=self.root/'projects'/pid;run.mkdir(parents=True)
            p={'id':pid,'scene_id':pid,'name':str(request.get('name') or source.stem)[:160],
                'source_path':str(source),'source_sha256':file_hash(source),'phase':'imported','head':None,'has_scene':False,
                'model':self.settings.model,'effort':self.settings.effort,'provider':'codex_subscription','messages':[],'frames':[],'active_run':None,
                'created_at':utc_now(),'acceptance':None,'latest_observation':None}
            self.put(p);save(run/'project.json',p)
            self.event(pid,'project_created',text='项目已创建，输入视频已登记。')
            if request.get('instruction'):
                self.message(pid,{'request_id':request['request_id']+'_initial','message':request['instruction'],'selected_ids':[]})
            return self.get(pid)
        return self._dedup('create',request,perform)

    def _verify(self,table,pid,token):
        identity=token if isinstance(token,str) else token.get('event_id') or token.get('receipt_id')
        with self.db() as db:row=db.execute(f'SELECT data FROM {table} WHERE id=? AND project_id=?',(identity,pid)).fetchone()
        if not row:raise ContractError('FORBIDDEN_SCOPE','Host verification record not found')
        record=json.loads(row['data'])
        if table=='observations' and not record.get('images_delivered'):raise ContractError('EVIDENCE_MISMATCH','Pi has not acknowledged receiving those images')
        return record
    def action(self,pid):
        with self.lock:return self._action(pid)

    def _action(self,pid):
        if pid not in self.actions:
            self.actions[pid]=SceneActions(pid,self.run_root(pid),self.runtime,host=self.host,
                blender=self.host.backend.blender_path,model=self.get(pid)['model'],effort=self.get(pid).get('effort','high'),
                human_event_verifier=lambda token:self._verify('human_events',pid,token),
                prior_observation_verifier=lambda token:self._verify('observations',pid,token))
            self.actions[pid].project_cancel_event=self.cancel.setdefault(pid,threading.Event())
            self.actions[pid].resource_budget=self.resources
            self.actions[pid].worker_callback=lambda target,event:self._worker_update(pid,target,event)
        return self.actions[pid]

    def status(self,pid):
        with self.lock:
            p=self.get(pid)
            try:
                view=asyncio.run(self.action(pid).inspect_scene());p.update(head=view['head'],has_scene=True)
                self.put(p)
            except ContractError as error:
                if error.code!='NOT_FOUND':raise
            return p

    def _on_pi_event(self,pid,event):
        with self.lock:
            advance_batch=False
            p=self.get(pid);kind=event['type'];rid=event.get('run_id')
            if kind=='text_delta':
                message_id=f'{rid}_assistant_{event.get("turn_id",p.get("turn_id",0))}'
                message=next((m for m in p['messages'] if m['id']==message_id),None)
                if message is None:message={'id':message_id,'role':'assistant','text':'','at':utc_now(),'run_id':rid};p['messages'].append(message)
                message['text']+=event['text']
            elif kind=='turn_start':
                p['turn_id']=event['turn_id']
                if p.get('model_retry',{}).get('status')=='waiting':p['model_retry']['status']='retrying'
            elif kind=='assistant_message':
                mid=f'{rid}_assistant_{event.get("turn_id",p.get("turn_id",0))}'
                m=next((m for m in p['messages'] if m['id']==mid),None)
                if m:m['text']=event.get('text','')
                elif event.get('text'):p['messages'].append({'id':mid,'role':'assistant','text':event['text'],'at':utc_now(),'run_id':rid})
                if event.get('stopReason')=='error' or (event.get('stopReason')=='aborted' and p['phase']!='paused'):
                    p['error']=event.get('error') or event['stopReason'];p['run_failed']=True
            elif kind=='auto_retry_start':
                data=event.get('data',{})
                p['model_retry']={'run_id':rid,'status':'cancelled' if p['phase']=='paused' else 'waiting',
                    'attempt':data.get('attempt',0),'max_attempts':data.get('maxAttempts',5),
                    'delay_ms':data.get('delayMs',0),'error':data.get('errorMessage'),'at':utc_now()}
                # The error belongs to a failed model attempt, not a completed run.
                if p['phase']!='paused':p.update(error=None,run_failed=False)
            elif kind=='auto_retry_end':
                data=event.get('data',{});retry=p.get('model_retry',{})
                cancelled=p['phase']=='paused' or data.get('finalError')=='Retry cancelled'
                retry.update(run_id=rid,status='cancelled' if cancelled else 'recovered' if data.get('success') else 'exhausted',
                    attempt=data.get('attempt',retry.get('attempt',0)),at=utc_now())
                p['model_retry']=retry
                if data.get('success') and not cancelled:p.update(error=None,run_failed=False)
                elif not cancelled:p.update(error=data.get('finalError') or retry.get('error') or 'Model reconnect attempts exhausted',run_failed=True)
            elif kind=='tool_receipt':
                details=event.get('result',{}).get('details',{})
                token=details.get('observation_receipt')
                if token:
                    with self.db() as db:
                        row=db.execute('SELECT data FROM observations WHERE id=? AND project_id=?',(token,pid)).fetchone()
                        if row:
                            obs=json.loads(row['data']);obs['images_delivered']=True
                            obs['pi_receipt_hash']=canonical_hash(event);db.execute('UPDATE observations SET data=? WHERE id=?',(json.dumps(obs),token))
            elif kind=='ready':
                p['pi_session_file']=event['session_file'];p['pi_ready']=True
                p['retry_policy']=event.get('retry')
            elif kind in ('settled','run_error','aborted','process_exit','fatal'):
                p['active_run']=None;p['active_tool']=None
                if p.get('model_retry',{}).get('status') in ('waiting','retrying'):
                    p['model_retry']['status']='cancelled' if p['phase']=='paused' or kind=='aborted' else 'exhausted'
                if kind in ('run_error','fatal') or (kind=='process_exit' and event.get('exit_code')):
                    p['error']=event.get('error') or 'Pi process exited';p['phase']='needs_auth' if 'CODEX_LOGIN' in p['error'] else 'failed'
                elif p['phase']=='paused':pass
                elif p.get('run_failed'):p['phase']='needs_auth' if 'CODEX_LOGIN' in (p.get('error') or '') else 'failed'
                elif kind=='aborted':p['phase']='paused'
                elif p['phase']=='running':
                    if p.get('has_scene') and not p.get('acceptance'):p['phase']='needs_review'
                    elif p.get('phase_before_run')=='complete' and p.get('export',{}).get('head')==p.get('head'):p['phase']='complete'
                    else:p['phase']='idle'
            if kind in ('settled','run_error','aborted','process_exit','fatal'):
                advance_batch=self._batch_settled(p)
            if kind not in ('text_delta',):self.event(pid,kind,**{k:v for k,v in event.items() if k!='type'})
            self.put(p)
            if kind=='auto_retry_start':self.rolling.reconnect_event(pid,event)
            if advance_batch:threading.Thread(target=self._advance_batch,args=(pid,),daemon=True).start()

    def _bridge(self,pid):
        bridge=self.bridges.get(pid)
        if bridge and bridge.process.poll() is None:return bridge
        run=self.run_root(pid);p=self.get(pid)
        tools=[{'name':name,**value} for name,value in {**EXTRA_TOOLS,**TOOL_SCHEMAS}.items()]
        config={'project_id':pid,'model':p['model'],'effort':p.get('effort','high'),'cwd':str(run),'session_dir':str(run/'pi/sessions'),
            'session_file':p.get('pi_session_file'),'system_prompt':SYSTEM_PROMPT,'tools':tools,
            'tool_endpoint':f'http://127.0.0.1:{self.port}/workflow/internal/tool','max_tool_calls':64}
        bridge=PiBridge(config,run/'pi',token=self.token,on_event=lambda e:self._on_pi_event(pid,e));self.bridges[pid]=bridge
        bridge.start();return bridge

    def _start(self,pid,rid,prompt,steer=False):
        if self._closing:return
        try:
            bridge=self._bridge(pid)
            bridge.send({'type':'steer' if steer else 'prompt','id':rid,'run_id':rid,'message':prompt})
        except Exception as error:self._on_pi_event(pid,{'type':'run_error','run_id':rid,'error':str(error)})

    def message(self,pid,request,*,source='user'):
        def perform():
            p=self.status(pid);text=str(request.get('message','')).strip()
            if p['phase']=='paused' or not p.get('active_run'):
                with self.db() as db:pending=db.execute("SELECT 1 FROM jobs WHERE project_id=? AND state='running' LIMIT 1",(pid,)).fetchone()
                native_pending=[path for path in self.run_root(pid).glob('tools/*/*/process.json') if not (path.parent/'process_result.json').exists()]
                if pending or native_pending:raise ValueError('暂停正在收尾，等待当前工具退出后再继续。')
            if not text or len(text)>12000:raise ValueError('Message must be 1..12000 characters')
            if request.get('base_revision') and (not p['head'] or request['base_revision']!=p['head']['revision'] or request.get('base_head_seq')!=p['head']['head_seq']):
                raise ContractError('STALE_REVISION','Scene changed; refresh the saved scene before sending')
            intent=request.get('intent','chat');selected=request.get('selected_ids',[])
            if intent not in ('chat','edit_selected','refine_selected'):raise ValueError('Unknown message intent')
            context={'project_id':pid,'scene_id':pid,'head':p['head'],'selected_ids':selected,'intent':intent}
            if intent!='chat':
                if p.get('active_run'):raise ValueError('Wait for the current task before authorizing another object change')
                if not p.get('head') or request.get('base_revision')!=p['head']['revision'] or request.get('base_head_seq')!=p['head']['head_seq']:
                    raise ContractError('STALE_REVISION','An object change must refer to the current saved scene')
                view=asyncio.run(self.action(pid).inspect_scene());objects={o['instance_id']:o for o in view['snapshot']['objects']}
                if not isinstance(selected,list) or not selected or any(i not in objects for i in selected):raise ValueError('Select existing scene objects first')
                from .scene_actions import PATCH_FIELDS
                payload={'user_instruction':text}
                payload.update({'allowed_object_ids':selected,'allowed_fields':sorted(PATCH_FIELDS)} if intent=='edit_selected' else
                               {'target_ids':selected,'visual_policy':'structure_first_relaxed_appearance'})
                if intent=='refine_selected':payload['include_structures']=any(objects[i]['kind'] in {'wall','floor','opening','ceiling'} for i in selected)
                human={'event_id':identifier('human'),'scene_id':pid,'actor':'user','source':'browser',
                       'kind':'scene_patch' if intent=='edit_selected' else 'accept_whitebox','head':p['head'],
                       'snapshot_hash':canonical_hash(view['snapshot']),'payload':payload}
                with self.db() as db:db.execute('INSERT INTO human_events VALUES(?,?,?)',(human['event_id'],pid,json.dumps(human,ensure_ascii=False)))
                if intent=='edit_selected':context['human_event']=human['event_id'];context['authorized_edit']=payload
                else:
                    p['acceptance']=asyncio.run(self.action(pid).accept_whitebox(payload,human['event_id']))
                    context['acceptance']=p['acceptance']
            rid=identifier('run');steer=bool(p.get('active_run'))
            if not steer:p['phase_before_run']=p['phase']
            p['messages'].append({'id':identifier('msg'),'role':source,'text':text,'at':utc_now(),'run_id':rid})
            p.update(phase='running',active_run=p.get('active_run') or rid,error=None,run_failed=False)
            if not steer:p.pop('model_retry',None)
            self.cancel.setdefault(pid,threading.Event()).clear()
            self.put(p);self.event(pid,'user_message',text=text,context=context)
            threading.Thread(target=self._start,args=(pid,rid,text+'\n当前页面上下文：'+json.dumps(context,ensure_ascii=False),steer),daemon=True).start()
            return {'accepted':True,'run_id':rid,'project':p}
        return self._dedup('message:'+pid,request,perform)

    def control(self,pid,request):
        if request.get('action')=='set_refinement_concurrency':return self._rolling_control(pid,request)
        if request.get('action')=='start_refine_all':return self.start_refine_all(pid,request)
        if request.get('action') in {'pause_object','resume_object'}:return self.control_batch_object(pid,request)
        with self.lock:
            p=self.status(pid);action=request.get('action')
            if action=='pause':
                self.cancel.setdefault(pid,threading.Event()).set();p['phase']='paused'
                if pid in self.actions:self.actions[pid].cancel_workers()
                if p.get('batch') and p['batch']['status']!='complete':p['batch']['status']='paused'
                self.put(p)
                self.rolling.kick()
                bridge=self.bridges.get(pid)
                if bridge and bridge.process.poll() is None:bridge.send({'type':'abort','id':identifier('abort')})
                self.event(pid,'human_pause',text='用户请求暂停。');return self.get(pid)
            if action=='approve_whitebox':
                if p.get('active_run') or p['phase']!='needs_review':raise ValueError('Wait for the whitebox review stage before accepting')
                head={'revision':request.get('revision'),'head_seq':request.get('head_seq')}
                if head!=p['head']:raise ContractError('STALE_REVISION','Review must refer to the exact saved scene head')
                view=asyncio.run(self.action(pid).inspect_scene());layout=read(self.run_root(pid)/'layout.json')
                ids=request.get('target_ids')
                if not ids:
                    remaining=remaining_objects(view['snapshot'])
                    preferred=next((o for o in remaining if o['instance_id']==layout['recommended_refinement_target']),None)
                    if not remaining:raise ValueError('所有对象已精修；如需返修，请选择对象并发送精修指令。')
                    ids=[(preferred or remaining[0])['instance_id']]
                human={'event_id':identifier('human'),'scene_id':pid,'actor':'user','source':'browser','kind':'accept_whitebox',
                    'head':head,'snapshot_hash':canonical_hash(view['snapshot']),
                    'payload':{'target_ids':ids,'include_structures':any(o['instance_id'] in ids and o['kind'] in {'wall','floor','opening','ceiling'} for o in view['snapshot']['objects']),
                               'user_instruction':'采用当前已保存的白模布局，保持位置、尺寸和支撑，精修选定对象并复核保存。颜色纹理允许适度偏差。','visual_policy':'structure_first_relaxed_appearance'}}
                with self.db() as db:db.execute('INSERT INTO human_events VALUES(?,?,?)',(human['event_id'],pid,json.dumps(human,ensure_ascii=False)))
                acceptance=asyncio.run(self.action(pid).accept_whitebox(human['payload'],human['event_id']))
                p['acceptance']=acceptance;p['phase']='idle';self.put(p);self.event(pid,'human_accepted',text='人工已采用当前保存布局。',acceptance=acceptance)
                self.message(pid,{'request_id':human['event_id']+'_continue','message':'我已在页面采用当前保存布局。请按真实 acceptance 精修获准目标，复核通过后保存并导出。','selected_ids':ids})
                return self.get(pid)
            if action=='resume':
                if p.get('active_run'):raise ValueError('Current task has not stopped yet')
                if p.get('batch') and p['batch']['status'] not in ('complete','stale'):return self.resume_batch(pid,request.get('parallelism'))
                self.message(pid,{'request_id':request.get('request_id') or identifier('resume'),'message':'继续当前项目。先核对 workflow_status 与已保存版本、真实任务产物，再推进尚未完成的工作。','selected_ids':[]})
                return self.get(pid)
            raise ValueError('Unknown control action')

    def _observation(self,pid,request,rows):
        p=self.get(pid);previous=[]
        if p.get('latest_observation'):
            try:previous=self._verify('observations',pid,p['latest_observation'])['frames']
            except ContractError:pass
        frames={r['source_frame_index']:r for r in previous}
        frames.update({r['source_frame_index']:{'source_frame_index':r['source_frame_index'],'path':r['path'],'sha256':r['sha256']} for r in rows})
        obs={'receipt_id':identifier('obs'),'scene_id':pid,'provider':'openai-codex','model':self.get(pid)['model'],
            'run_id':request['run_id'],'turn_id':request['turn_id'],'images_delivered':False,'frames':list(frames.values())}
        with self.db() as db:db.execute('INSERT INTO observations VALUES(?,?,?)',(obs['receipt_id'],pid,json.dumps(obs)))
        p['latest_observation']=obs['receipt_id'];self.put(p);return obs['receipt_id']

    def _images(self,pid,paths):
        content=[];run=self.run_root(pid)
        for value in paths[:16]:
            p=Path(value).resolve()
            if not p.is_relative_to(run) or not p.is_file():raise ValueError('Preview outside project evidence')
            if p.suffix.lower() not in ('.png','.jpg','.jpeg'):raise ValueError('Unsupported evidence image format')
            if p.stat().st_size>5_000_000:raise ValueError('Evidence image exceeds the supported size')
            content.append({'type':'image','mimeType':'image/png' if p.suffix.lower()=='.png' else 'image/jpeg',
                'data':base64.b64encode(p.read_bytes()).decode('ascii')})
        return content

    def _sam_attempt(self,run,requests,retry_reason=None):
        worker=ROOT/'video2scene/agent_scene/sam_worker.py'
        base='sam_'+canonical_hash([requests,file_hash(worker)])[:20]
        return self._native_attempt(run,'sam',base,retry_reason)

    def _native_attempt(self,run,toolname,base,retry_reason=None):
        if any(not (p.parent/'process_result.json').exists() for p in (run/'tools'/toolname).glob('*/process.json')):
            raise ValueError('A prior '+toolname.upper()+' worker has not exited; wait for its process receipt')
        for attempt in range(3):
            identity=base if not attempt else base+'_retry'+str(attempt)
            receipt=run/'tools'/toolname/identity/'receipt.json'
            if not receipt.exists():
                if attempt and not str(retry_reason or '').strip():raise ValueError(toolname.upper()+' failed previously; provide a concrete retry_reason after fixing the cause')
                return identity
            if read(receipt)['status']=='complete':return identity
            if read(receipt)['status'] in ('running','uncertain'):raise ValueError(toolname.upper()+' attempt is still running or interrupted; inspect its receipt')
        raise ValueError(toolname.upper()+' retry budget exhausted; inspect prompts and worker diagnostics')

    def _vipe_attempt(self,run,retry_reason=None):
        script=ROOT/'experiments/redwood_v1/run_rgb_only.sh'
        base='vipe_dense_'+canonical_hash([file_hash(run/'frames/manifest.json'),'dense_10fps_v1',file_hash(script)])[:20]
        receipt=run/'tools/vipe'/base/'receipt.json'
        if receipt.exists() and read(receipt).get('source_script_sha256') not in (None,file_hash(script)):
            base+='_'+file_hash(script)[:8]
        return self._native_attempt(run,'vipe',base,retry_reason)

    def _repair_supports(self,pid,args):
        with self.db() as db:rows=db.execute("SELECT data FROM jobs WHERE project_id=? AND state='failed'",(pid,)).fetchall()
        draft=next((json.loads(r['data']) for r in rows if json.loads(r['data']).get('id')==args['failed_job_id']),None)
        if not draft or draft.get('request',{}).get('name')!='build_whitebox':raise ValueError('Select a failed build_whitebox job from this project')
        if canonical_hash(draft['request'])!=draft['request_hash']:raise ValueError('Failed draft request changed')
        original=copy.deepcopy(draft['request']['args']);layout=original['layout']
        objects={item['object_id']:item for item in layout['objects']+layout['structures']};seen=set()
        for patch in args['support_corrections']:
            identity=patch['object_id'];parent=patch['support_object_id']
            if identity not in objects or identity in seen:raise ValueError('Unknown or duplicate draft object')
            if parent not in objects and parent!='unknown':raise ValueError("support_object_id must name a layout object or be literal 'unknown'")
            old=objects[identity]['support_object_id']
            if old in objects or old=='unknown':raise ValueError('This helper repairs invalid support identities only')
            objects[identity]['support_object_id']=parent;seen.add(identity)
        return original

    def tool(self,request):
        pid=request['project_id'];name=request['name'];args=request.get('args',{});run=self.run_root(pid)
        definitions={**EXTRA_TOOLS,**TOOL_SCHEMAS}
        if name not in definitions:raise ValueError('Unknown workflow tool')
        from jsonschema import Draft202012Validator
        Draft202012Validator(definitions[name]['parameters']).validate(args)
        jid=identifier('job');cancel=self.cancel.setdefault(pid,threading.Event())
        if cancel.is_set():raise ContractError('CANCELLED','Project paused')
        key=pid+':'+request['tool_call_id'];digest=canonical_hash(request)
        with self.db() as db:
            prior=db.execute('SELECT data,state FROM jobs WHERE id=?',(key,)).fetchone()
            if prior:
                old=json.loads(prior['data'])
                if old['request_hash']!=digest:raise ValueError('Tool call ID collision')
                if prior['state']=='complete':
                    if file_hash(old['result_path'])!=old['result_sha256']:raise ValueError('Completed tool result hash changed')
                    return read(old['result_path'])
                raise ValueError('Tool call already started; reconcile its recorded outcome before retry')
            db.execute('INSERT INTO jobs VALUES(?,?,?,?)',(key,pid,'running',json.dumps({'id':jid,'tool':name,'request_hash':digest,'request':request,'started_at':utc_now()})))
        with self.lock:
            p=self.get(pid);p['active_tool']=name;self.put(p)
        self.event(pid,'tool_started',text='正在执行 '+name,job_id=jid,args=args)
        images=[];details={};batch_tool_started=False
        try:
            self._batch_tool_start(pid,name,args)
            batch_tool_started=True
            if name=='workflow_status':
                result=self.status(pid)
                with self.db() as db:
                    result['jobs']=[{'state':r['state'],**{k:v for k,v in json.loads(r['data']).items() if k in ('id','tool','error','result_path')}} for r in db.execute('SELECT state,data FROM jobs WHERE project_id=? ORDER BY rowid DESC LIMIT 30',(pid,))]
                result['messages']=result['messages'][-6:]
                state=run/'scene_actions/state.json'
                result['scene_progress']={k:v for k,v in read(state).items() if k in ('phase','current_head','active_operation','candidate','target_id','backend_progress')} if state.exists() else None
                result['tool_outputs']={}
                for toolname in ('sam','vipe'):
                    pointer=run/'tools'/toolname/'latest.json'
                    if pointer.exists():result['tool_outputs'][toolname]=read(pointer)
            elif name=='start_refinement_jobs':result=self.rolling.start(pid,**args)
            elif name=='refinement_jobs_status':result=self.rolling.status(pid,**args)
            elif name=='inspect_video':
                if file_hash(self.get(pid)['source_path'])!=self.get(pid)['source_sha256']:raise ValueError('Original input video changed after project creation')
                result=prepare_video(run,self.get(pid)['source_path'],**args)
                rows=result['keyframes'];details['observation_receipt']=self._observation(pid,request,rows)
                result['observation_receipt']=details['observation_receipt'];images=[r['path'] for r in rows]
                manifest=read(result['manifest_path']);result['available_source_frame_indices']=[f['source_frame_index'] for f in manifest['frames']]
                p=self.get(pid);p['frames']=[{'path':str(Path(r['path']).relative_to(run)).replace('\\','/'),'label':f"frame {r['source_frame_index']}",'source_frame_id':r['source_frame_index']} for r in rows];self.put(p)
            elif name=='view_frames':
                manifest=read(run/'frames/manifest.json');by_id={r['source_frame_index']:r for r in manifest['frames']}
                rows=[by_id[i] for i in args['source_frame_indices']]
                details['observation_receipt']=self._observation(pid,request,rows);result={'frames':rows,**details};images=[r['path'] for r in rows]
            elif name=='estimate_geometry':
                result=run_vipe_tool(run,{'job_id':self._vipe_attempt(run,args.get('retry_reason')),'reason':args['reason'],'timeout_s':1800},cancel=cancel)
                images=result.get('previews',[])
            elif name=='segment_instances':
                prior=read(run/'prior.json')
                requests=[{'object_id':o['object_id'],'category':o['category'],'prompt':o['sam_prompt'],'seed_source_frame_index':o['seed_source_frame_index'],
                    'positive_points_normalized':o['positive_points_normalized'],'negative_points_normalized':o['negative_points_normalized'],'hint_box_normalized':o['hint_box_normalized']} for o in prior['objects']]
                result=run_sam_tool(run,requests,self._sam_attempt(run,requests,args.get('retry_reason')),cancel=cancel);images=result.get('previews',[])[:12]
            elif name=='request_human_review':
                view=asyncio.run(self.action(pid).inspect_scene());p=self.get(pid);p.update(phase='needs_review',head=view['head'],has_scene=True);self.put(p)
                result={'status':'needs_review','head':view['head'],'message':args['message'],'instruction':'Wait for real browser accept_whitebox; do not call refine_object yet.'}
            elif name=='finish_workflow':
                p=self.status(pid)
                if not p.get('acceptance') or not p.get('export'):raise ValueError('Completion requires actual human acceptance, refinement commit and export')
                result=asyncio.run(self.action(pid).verify_delivery(p['acceptance']['acceptance_id'],p['export']))
                p.update(phase='complete',summary=args['summary']);self.put(p)
                if (run/'completion.json').exists():
                    previous=read(run/'completion.json');save(run/'completion_history'/(canonical_hash(previous)+'.json'),previous)
                save(run/'completion.json',{'project_id':pid,'model':self.get(pid)['model'],'provider':'openai-codex','authentication':'Codex subscription','result':result,'at':utc_now()})
            else:
                dispatch_name,dispatch_args=name,args
                if name=='repair_whitebox_supports':dispatch_name,dispatch_args='build_whitebox',self._repair_supports(pid,args)
                if dispatch_name=='build_whitebox':
                    pointer=read(run/'tools/vipe/latest.json')
                    manifest=Path(pointer.get('vipe_manifest_path','')).resolve()
                    if not manifest.is_relative_to(run) or not manifest.is_file():
                        raise ValueError('Whitebox needs the dense temporal ViPE result and fresh measurement; sparse keyframe ViPE is insufficient')
                result=asyncio.run(self.action(pid).dispatch(dispatch_name,dispatch_args))
                if name=='measure':images=[result['preview']]
                if dispatch_name=='build_whitebox':
                    p=self.status(pid);p.update(has_scene=True,phase='needs_review');self.put(p)
                if name in ('commit_candidate','save_refinement'):
                    p=self.status(pid)
                    if result.get('kind')=='patch':
                        p['acceptance']=None;p['phase']='needs_review'
                        if p.get('batch'):p['batch']['status']='stale'
                        self.put(p)
                if name=='export_scene':
                    p=self.status(pid);p['export']=result;self.put(p)
            self._batch_tool_result(pid,name,result,args=args)
            if cancel.is_set():raise ContractError('CANCELLED','Project paused after tool execution; inspect recorded artifacts')
            model_result=result
            if name in ('commit_candidate','save_refinement') and self.get(pid).get('batch'):
                model_result={k:v for k,v in result.items() if k not in {'snapshot','model_proof','artifact_refs'}}
            payload={'content':[{'type':'text','text':json.dumps(model_result,ensure_ascii=False)},*self._images(pid,images)],'details':{'job_id':jid,'tool':name,**details}}
            output=run/'tool_results'/(jid+'.json');save(output,payload)
            with self.db() as db:db.execute('UPDATE jobs SET state=?,data=? WHERE id=?',('complete',json.dumps({'id':jid,'tool':name,'request_hash':digest,'result_path':str(output),'result_sha256':file_hash(output),'finished_at':utc_now()}),key))
            self.event(pid,'tool_completed',text=name+' 已完成',job_id=jid,result=copy.deepcopy(result))
            return payload
        except Exception as error:
            if batch_tool_started:self._batch_tool_result(pid,name,error=error,args=args)
            with self.db() as db:db.execute('UPDATE jobs SET state=?,data=? WHERE id=?',('cancelled' if cancel.is_set() else 'failed',json.dumps({'id':jid,'tool':name,'request_hash':digest,'error':str(error),'request':request}),key))
            self.event(pid,'tool_failed',text=name+'：'+str(error),job_id=jid);raise
        finally:
            with self.lock:p=self.get(pid);p['active_tool']=None;self.put(p)

    def close(self):
        self._closing=True;self.rolling.stop.set();self.rolling.signal.set()
        for event in self.cancel.values():event.set()
        for actions in self.actions.values():actions.cancel_workers()
        self.rolling.close()
        for bridge in self.bridges.values():bridge.close()
        if self.host.jobs:self.host.jobs.shutdown(wait=False)
