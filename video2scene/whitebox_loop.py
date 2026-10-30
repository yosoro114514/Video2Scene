"""Local, versioned tabletop editing prototype with a transactional Blender compiler."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from video2scene.config import blender_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/uw_whitebox_loop_20261028'
SOURCE = ROOT / 'outputs/uw_direct_gpt6_20261027'
BLENDER = blender_path()
BUILDER = ROOT / 'scripts/whitebox_loop_blender.py'
C = np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
COLORS = {'t000':(224,141,53),'t001':(65,78,225),'t002':(83,170,36),'t003':(182,87,154),'table_00':(199,159,25)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    write(temporary,value)
    os.replace(temporary,path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EditError(ValueError):
    def __init__(self,message,status=400):
        super().__init__(message)
        self.status = status


def blender(args, log):
    start = time.monotonic()
    with Path(log).open('w',encoding='utf8') as stream:
        run = subprocess.run([str(BLENDER),'--background','--python-exit-code','1','--python',str(BUILDER),'--',*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,timeout=240)
    if run.returncode:
        raise RuntimeError('Blender failed: '+str(log))
    return round(time.monotonic()-start,3)


def geometry(directory):
    rows = read(directory/'geometry_parts.json')
    data = np.load(directory/'geometry.npz')
    groups = {}
    for row in rows:
        i = str(row['index'])
        groups.setdefault(row['identity'],[]).append((data['v'+i],data['f'+i]))
    result = {}
    for identity,parts in groups.items():
        vs,fs,offset = [],[],0
        for v,f in parts:
            vs.append(v)
            fs.append(f+offset)
            offset += len(v)
        result[identity] = (np.concatenate(vs),np.concatenate(fs))
    return result


def check_contacts(directory):
    groups = geometry(directory)
    compiled = read(directory/'compiled.json')
    table_inv = np.linalg.inv(np.array(compiled['objects']['table_00']['matrix_world']))
    local = {key:(np.c_[v,np.ones(len(v))]@table_inv.T)[:,:3] for key,(v,_) in groups.items()}
    hulls = {key:cv2.convexHull(p[:,:2].astype(np.float32)).reshape(-1,2) for key,p in local.items()}
    records = {}
    for key,p in local.items():
        if key=='table_00':
            continue
        gap = float(p[:,2].min())
        edge = min(cv2.pointPolygonTest(hulls['table_00'],(float(q[0]),float(q[1])),True) for q in hulls[key])
        records[key] = dict(support_gap_m=gap,inside_table=edge>=-1e-5,table_edge_clearance_m=edge,
                            supported=abs(gap)<.002,collision_candidates=[])
    keys = list(records)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            area,_ = cv2.intersectConvexConvex(hulls[a],hulls[b])
            if area>1e-6:
                records[a]['collision_candidates'].append(b)
                records[b]['collision_candidates'].append(a)
    report = dict(objects=records,method='Contact distance in table-local coordinates; conservative convex-footprint containment/collision screening, not full mesh physics.',
                  passed=all(r['supported'] and r['inside_table'] and not r['collision_candidates'] for r in records.values()))
    write(directory/'contacts.json',report)
    return report


def make_inspection(out, revision):
    import open3d as o3d
    directory = out/'revisions'/revision
    groups = geometry(directory)
    overall = o3d.t.geometry.RaycastingScene()
    separate, gid_to_key = {},{}
    for key,(v,f) in groups.items():
        vertices = o3d.core.Tensor(v.astype(np.float32))
        faces = o3d.core.Tensor(f.astype(np.uint32))
        gid = overall.add_triangles(vertices,faces)
        gid_to_key[gid] = key
        single = o3d.t.geometry.RaycastingScene()
        single.add_triangles(vertices,faces)
        separate[key] = single
    review = directory/'inspection'
    review.mkdir(exist_ok=True)
    rows = []
    for camera in read(out/'cameras.json'):
        frame = camera['frame']
        if frame not in (0,40,79):
            continue
        w,h = camera['resolution']
        fx,fy,cx,cy = camera['intrinsics']
        c2w = np.array(camera['c2w'])
        yy,xx = np.mgrid[:h,:w]
        directions = np.stack(((xx+.5-cx)/fx,(yy+.5-cy)/fy,np.ones_like(xx)),axis=-1) @ (C@c2w[:3,:3]).T
        origins = np.broadcast_to(C@c2w[:3,3],directions.shape)
        rays = o3d.core.Tensor(np.concatenate((origins,directions),axis=-1).astype(np.float32))
        result = overall.cast_rays(rays)
        ids = result['geometry_ids'].numpy()
        source = cv2.imread(str(out/'source_frames'/f'{frame:03d}.png'))
        visible_all,xray_all = source.copy(),source.copy()
        for gid,key in gid_to_key.items():
            visible = ids==gid
            xray = np.isfinite(separate[key].cast_rays(rays)['t_hit'].numpy())
            for kind,mask,canvas in [('visible',visible,visible_all),('xray',xray,xray_all)]:
                # External silhouettes intentionally omit holes; ordinary close-up shows mug opening/handle.
                contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contours = [c for c in contours if cv2.contourArea(c)>3]
                image = source.copy()
                image[mask] = (image[mask]*.84+np.array(COLORS[key])*.16).astype(np.uint8)
                cv2.drawContours(image,contours,-1,COLORS[key],2,cv2.LINE_AA)
                cv2.drawContours(canvas,contours,-1,COLORS[key],2,cv2.LINE_AA)
                cv2.imwrite(str(review/f'{key}_{frame:03d}_{kind}.png'),image)
            rows.append(dict(instance_id=key,frame=frame,visible_pixels=int(visible.sum()),full_projection_pixels=int(xray.sum()),
                             occluded_fraction=round(1-float(visible.sum())/max(1,float(xray.sum())),4)))
        cv2.imwrite(str(review/f'all_{frame:03d}_visible.png'),visible_all)
        cv2.imwrite(str(review/f'all_{frame:03d}_xray.png'),xray_all)
    write(review/'report.json',dict(revision=revision,camera_sha256=digest(out/'cameras.json'),objects=rows,
        note='Frozen source cameras. Visible uses first-hit scene occlusion; X-ray renders each object alone. Source RGB is a reference, not a post-edit design target.'))


def compare_revisions(out, before, after, target, operation):
    a = read(out/'revisions'/before/'compiled.json')
    b = read(out/'revisions'/after/'compiled.json')
    targets={target} if isinstance(target,str) else set(target)
    unchanged = {k:a['objects'][k]['geometry_sha256']==b['objects'][k]['geometry_sha256'] for k in a['objects'] if k not in targets}
    same_ids = set(a['objects'])==set(b['objects'])
    checks = dict(before=before,after=after,target=target,operation=operation,identities_preserved=same_ids,
        unaffected_geometry_unchanged=unchanged,cameras_unchanged=a['camera_sha256']==b['camera_sha256'])
    if operation=='refine':
        old,new = a['objects'][target],b['objects'][target]
        checks.update(root_matrix_max_error=float(np.abs(np.array(old['matrix_world'])-new['matrix_world']).max()),
                      local_bounds_max_error_m=float(np.abs(np.array(old['local_bounds'])-new['local_bounds']).max()),
                      support_preserved=old['support_id']==new['support_id'],locks_preserved=old['locked_fields']==new['locked_fields'],
                      before_vertices=old['vertices'],after_vertices=new['vertices'])
    checks['passed'] = same_ids and all(unchanged.values()) and checks['cameras_unchanged']
    if operation=='refine':
        checks['passed'] &= checks['root_matrix_max_error']<1e-6 and checks['local_bounds_max_error_m']<1e-5 and checks['support_preserved'] and checks['locks_preserved']
    if operation=='generated':
        checks['object_constraints']={}
        for key in targets:
            old,new=a['objects'][key],b['objects'][key]
            item=dict(root_matrix_max_error=float(np.abs(np.array(old['matrix_world'])-new['matrix_world']).max()),
                      local_bounds_max_error_m=float(np.abs(np.array(old['local_bounds'])-new['local_bounds']).max()),
                      support_preserved=old['support_id']==new['support_id'],
                      locks_preserved=set(old['locked_fields']).issubset(set(new['locked_fields'])))
            item['passed']=item['root_matrix_max_error']<1e-6 and item['local_bounds_max_error_m']<1e-5 and item['support_preserved'] and item['locks_preserved']
            checks['object_constraints'][key]=item
        checks['passed'] &= all(x['passed'] for x in checks['object_constraints'].values())
    write(out/'revisions'/after/'transition_validation.json',checks)
    return checks


class SceneStore:
    def __init__(self, out=DEFAULT_OUT):
        self.out = Path(out).resolve()
        self.lock = threading.Lock()

    def head(self):
        return read(self.out/'head.json')

    def snapshot(self):
        head = self.head()
        directory = self.out/'revisions'/head['current']
        result = dict(head=head,scene=read(directory/'scene.json'),compiled=read(directory/'compiled.json'),contacts=read(directory/'contacts.json'))
        result['display_glb']='model_web.glb' if (directory/'model_web.glb').exists() else 'model.glb'
        if (directory/'transition_validation.json').exists():
            result['transition'] = read(directory/'transition_validation.json')
        result['history'] = [read(p/'scene.json')['event'] | {'revision':p.name} for p in sorted((self.out/'revisions').glob('r*')) if (p/'complete.json').exists()]
        return result

    def compile(self, revision):
        directory = self.out/'revisions'/revision
        elapsed = blender(['compile','--output',self.out,'--revision',revision],directory/'build.log')
        contacts = check_contacts(directory)
        if not contacts['passed']:
            raise EditError('修改后物体越出桌面、失去支撑或与其他物体相交；未保存到当前版本。')
        make_inspection(self.out,revision)
        blender(['verify','--output',self.out,'--revision',revision],directory/'reopen.log')
        return elapsed

    def _commit_agent_state(self,state,head,request,targets,operation):
        existing=[int(p.name[1:]) for p in (self.out/'revisions').glob('r*') if p.name[1:].isdigit()]
        revision=f'r{max(existing,default=-1)+1:03d}';directory=self.out/'revisions'/revision;directory.mkdir()
        state.update(revision=revision,parent_revision=head['current'],event=dict(operation=operation,target=targets,
            intent='asset_refinement' if operation=='generated' else 'design_edit',actor='gpt_agent',created_at=time.time(),
            description=request.get('description',''),job_id=request.get('job_id')))
        write(directory/'scene.json',state);write(directory/'request.json',request)
        try:
            elapsed=self.compile(revision)
            checks=compare_revisions(self.out,head['current'],revision,targets,operation)
            if not checks['passed']:raise RuntimeError('Agent revision violates scene constraints')
            if self.head()['current']!=head['current']:
                raise EditError('构建期间保存版本已变化，候选已保留，未覆盖较新的修改。',409)
            write(directory/'complete.json',dict(compiled_seconds=elapsed,completed_at=time.time()))
        except Exception as exc:
            write(directory/'failure.json',dict(error=str(exc),time=time.time()));raise
        head['undo'].append(head['current']);head['current'],head['redo']=revision,[]
        atomic_write(self.out/'head.json',head)
        return self.snapshot()

    def commit_generated(self,request):
        with self.lock:
            head=self.head()
            if request['base_revision']!=head['current']:
                raise EditError('精修期间场景已变化，生成资产已保留，未覆盖较新的用户修改。',409)
            state=copy.deepcopy(read(self.out/'revisions'/head['current']/'scene.json'))
            assets=request['assets']
            if not assets or set(assets)-set(state['objects']):raise EditError('Invalid asset targets')
            for key,record in assets.items():
                p=(self.out/record['asset_file']).resolve()
                if not p.is_relative_to(self.out/'jobs') or not p.is_file() or digest(p)!=record['asset_sha256']:
                    raise EditError('Generated asset is missing or changed')
                obj=state['objects'][key]
                obj.update(variant='generated',asset_id='fresh/'+request['job_id']+'/'+key,
                           asset_file=record['asset_file'],asset_sha256=record['asset_sha256'])
                obj['locks']=sorted(set(obj['locks'])|{'offset_m','scale','yaw_deg','support_id'})
            return self._commit_agent_state(state,head,request,list(assets),'generated')

    def apply_language_edits(self,request):
        """Apply a schema-validated GPT patch atomically. Natural-language text is never code."""
        with self.lock:
            head=self.head()
            if request['base_revision']!=head['current']:raise EditError('场景版本已更新，请重新提交指令。',409)
            state=copy.deepcopy(read(self.out/'revisions'/head['current']/'scene.json'))
            targets=[]
            for edit in request['edits']:
                key=edit['instance_id']
                if key not in state['objects'] or not state['objects'][key]['editable']:raise EditError('该对象不支持布局编辑')
                if key in targets:raise EditError('同一对象有重复修改')
                fields=edit['fields'];values=edit['values'];obj=state['objects'][key]
                if not fields or set(fields)-{'offset_m','yaw_deg','scale'}:raise EditError('Invalid edit fields')
                for field in fields:
                    value=values[field];n={'offset_m':2,'scale':3}.get(field)
                    if n and (not isinstance(value,list) or len(value)!=n):raise EditError('Invalid dimensions')
                    nums=value if n else [value]
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in nums):raise EditError('Non-finite edit')
                    if field=='offset_m' and any(abs(v)>.35 for v in nums):raise EditError('位移超过 ±35 cm 范围')
                    if field=='scale' and any(not .5<=v<=1.75 for v in nums):raise EditError('缩放超出 0.5–1.75 范围')
                    if field=='yaw_deg' and abs(value)>180:raise EditError('旋转超过 ±180° 范围')
                    obj[field]=value
                obj['locks']=sorted(set(obj['locks'])|set(fields)|{'support_id'});targets.append(key)
            if not targets:raise EditError('No layout edits')
            return self._commit_agent_state(state,head,request,targets,'edit_batch')

    def apply(self, request):
        with self.lock:
            head = self.head()
            if request.get('base_revision')!=head['current']:
                raise EditError('场景版本已更新，请刷新后重新应用。',409)
            operation = request.get('operation')
            if operation in ('undo','redo'):
                source,destination = ('undo','redo') if operation=='undo' else ('redo','undo')
                if not head[source]:
                    raise EditError('没有可以'+('撤销' if operation=='undo' else '重做')+'的操作。')
                head[destination].append(head['current'])
                head['current'] = head[source].pop()
                atomic_write(self.out/'head.json',head)
                return self.snapshot()
            if operation not in ('edit','refine'):
                raise EditError('不支持的操作。')
            state = copy.deepcopy(read(self.out/'revisions'/head['current']/'scene.json'))
            target = request.get('instance_id')
            if target not in state['objects'] or not state['objects'][target]['editable']:
                raise EditError('请选择一个可编辑的桌面物体；本次测试的支撑桌面固定。')
            obj = state['objects'][target]
            if operation=='edit':
                fields = request.get('values')
                if not isinstance(fields,dict) or not fields or set(fields)-{'offset_m','yaw_deg','scale'}:
                    raise EditError('修改字段无效。')
                for key,value in fields.items():
                    n = {'offset_m':2,'scale':3}.get(key)
                    if n and (not isinstance(value,list) or len(value)!=n):
                        raise EditError('参数维度不正确。')
                    values = value if n else [value]
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
                        raise EditError('参数必须是有限数值。')
                    if key=='offset_m' and any(abs(v)>.35 for v in values):
                        raise EditError('位移超过本次桌面测试范围（±35 厘米）。')
                    if key=='scale' and any(not .5<=v<=1.75 for v in values):
                        raise EditError('尺寸比例范围为 0.5–1.75。')
                    if key=='yaw_deg' and abs(value)>180:
                        raise EditError('旋转范围为 ±180°。')
                    obj[key] = value
                obj['locks'] = sorted(set(obj['locks']) | set(fields) | {'support_id'})
            else:
                if target!='t001':
                    raise EditError('本轮只为马克杯接入了精模替换。')
                if obj['variant']=='fine':
                    raise EditError('这个杯子已经使用精细资产。')
                obj['variant'],obj['asset_id'] = 'fine','fine/t001/archived-gpt-v1'
                # The refinement operation has no authority to change pose, size or support.
            if obj==read(self.out/'revisions'/head['current']/'scene.json')['objects'][target]:
                raise EditError('参数没有变化。')
            existing = [int(p.name[1:]) for p in (self.out/'revisions').glob('r*') if p.name[1:].isdigit()]
            revision = f'r{max(existing,default=-1)+1:03d}'
            directory = self.out/'revisions'/revision
            directory.mkdir()
            state.update(revision=revision,parent_revision=head['current'],event=dict(operation=operation,target=target,
                intent='design_edit' if operation=='edit' else 'asset_refinement',actor='local_browser',created_at=time.time(),
                description=str(request.get('description',''))[:300]))
            write(directory/'scene.json',state)
            write(directory/'request.json',request)
            try:
                elapsed = self.compile(revision)
                checks = compare_revisions(self.out,head['current'],revision,target,operation)
                if not checks['passed']:
                    raise RuntimeError('Revision invariant checks failed')
                write(directory/'complete.json',dict(compiled_seconds=elapsed,completed_at=time.time()))
            except Exception as exc:
                write(directory/'failure.json',dict(error=str(exc),time=time.time()))
                raise
            head['undo'].append(head['current'])
            head['current'],head['redo'] = revision,[]
            atomic_write(self.out/'head.json',head)
            return self.snapshot()


def prepare(out=DEFAULT_OUT):
    out = Path(out).resolve()
    if (out/'head.json').exists():
        return SceneStore(out).snapshot()
    out.mkdir(parents=True,exist_ok=True)
    (out/'source_frames').mkdir(exist_ok=True)
    for frame in (0,40,79):
        shutil.copyfile(SOURCE/'source_frames'/f'{frame:03d}.png',out/'source_frames'/f'{frame:03d}.png')
    shutil.copyfile(SOURCE/'cameras_all.json',out/'cameras.json')
    shutil.copytree(SOURCE/'vendor',out/'vendor',dirs_exist_ok=True)
    shutil.copyfile(ROOT/'video2scene/whitebox_loop.html',out/'index.html')
    shutil.copyfile(ROOT/'video2scene/whitebox_agent_ui.js',out/'whitebox_agent_ui.js')
    write(out/'provenance.json',dict(aha_commit='9b80f78bd8ee293d4da14c86f307a73584aa3169',
        reused_code='third_party/aha3d_semantics/semantics.py: tag_root, tag_support, roots, descendants',
        inspired_by=['object-centered visible/X-ray inspection','stable semantic roots','asset variants'],
        added_locally=['versioned authoritative scene JSON','transactional browser-to-Blender writes','undo/redo','revision-bound evidence','constraint checks'],
        source_blend=str(SOURCE/'geometry_guided/result/scene.blend'),source_blend_sha256=digest(SOURCE/'geometry_guided/result/scene.blend'),
        fine_asset_origin='Archived previous GPT model; no new LLM or Pi3X invocation',scale='monocular estimate; centimetres are model units'))
    blender(['initialize','--output',out,'--source',SOURCE],out/'initialize.log')
    initial = read(out/'initial_scene.json')
    initial.update(revision='r000',parent_revision=None,event=dict(operation='initialize',target=None,intent='reconstruction_reference',actor='prototype',created_at=time.time(),description='原布局白模'))
    write(out/'revisions/r000/scene.json',initial)
    store = SceneStore(out)
    elapsed = store.compile('r000')
    write(out/'revisions/r000/complete.json',dict(compiled_seconds=elapsed,completed_at=time.time()))
    atomic_write(out/'head.json',dict(current='r000',undo=[],redo=[]))
    return store.snapshot()
