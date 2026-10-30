"""Pi authorizes rolling dispatch and chooses packages; host owns durable work."""
import asyncio
import json
import threading
import uuid

from video2scene.contracts_v1 import ContractError,canonical_hash,utc_now
from .rolling_jobs import LIVE,READY
from .process_owner import archive_dead_claim


class RollingBatchMixin:
    def _advance_rolling_batch(self,pid):
        p=self.get(pid);b=p['batch'];rows=b['targets']
        if not b.get('dispatch_authorization'):
            key='authorize'
            prompt=('先 workflow_status。当前持续并发批次已由真实用户采用。调用 start_refinement_jobs 一次授权完整既有批次自动补位：'
                f"acceptance_id={b['acceptance_id']}，operation_id=dispatch_{b['batch_id']}，target_ids="+json.dumps([r['target_id'] for r in rows])+ '。'
                '工具立即返回，后台独立作者和critic继续运行。然后读取 refinement_jobs_status；有package就调用save_refinement逐个保存，'
                '没有就结束本轮，由宿主在资产就绪后唤醒。不循环空查，不重启已有任务。')
        else:
            ready=[r for r in rows if r['state'] in READY and not r.get('pause_requested')]
            if ready:
                key=canonical_hash([[r['target_id'],r.get('package'),r.get('candidate')] for r in ready])
                prompt=('持续并发任务产生了可保存资产。先 refinement_jobs_status，逐个调用 save_refinement(package_id,package_hash)。'
                    '该工具完整执行合并→验证→保存，无需三次模型往返。其他对象继续生成，宿主自动补位。已有candidate的旧批次目标继续validate_candidate→commit_candidate。'
                    '每次保存后可再读状态处理新就绪包；无就绪包时结束本轮，等待宿主唤醒。失败对象不要重新建模，继续其他成功包。'
                    '禁止refine_object/refine_objects_parallel。当前可保存目标：'+json.dumps([r['target_id'] for r in ready]))
            elif any(r['state'] in LIVE for r in rows) or any(r['state']=='pending' and not r.get('pause_requested') for r in rows):
                # Dependency-blocked pending rows need user diagnosis once every
                # possible producer has drained, not an endless idle Pi loop.
                runnable=any(r['state']=='pending' and not any(d['target_id'] in r.get('depends_on',[]) and d['state']!='saved' for d in rows) for r in rows)
                if not any(r['state'] in LIVE for r in rows) and not runnable:
                    b.update(status='waiting',error='剩余目标的支撑对象尚未保存，请核对失败或暂停项。');p['phase']='idle';self.put(p)
                return
            elif all(r['state']=='saved' for r in rows):
                if self.rolling.active(pid):return
                key='export_'+str(p.get('head'))
                prompt='持续并发批次所有目标已保存且后台已收尾。先 refinement_jobs_status 核对，再 export_scene→finish_workflow，导出必须绑定最终head。'
            else:
                b.update(status='waiting',error='成功对象已保留，剩余失败／暂停对象需要核对。');p['phase']='idle';self.put(p);return
        if b.get('wake_key')==key:return
        b['wake_key']=key;self.put(p)
        self.message(pid,{'request_id':'rolling_wake_'+uuid.uuid4().hex,'message':prompt,'selected_ids':[]},source='system')

    def _rolling_tool_start(self,pid,name,args):
        p=self.get(pid);b=p['batch']
        if name in ('export_scene','finish_workflow'):
            if not all(r['state']=='saved' for r in b['targets']) or self.rolling.active(pid):
                raise ContractError('FORBIDDEN_SCOPE','Export only after every target is saved and all background workers settle')
            return
        if name in ('refine_object','refine_objects_parallel'):
            raise ContractError('FORBIDDEN_SCOPE','Rolling batches use durable start_refinement_jobs, never duplicate authors')
        if name not in ('start_refinement_jobs','save_refinement','stage_refinement','validate_candidate','commit_candidate'):return
        if b['status']!='running' or p['phase']=='paused':raise ContractError('CANCELLED','Batch is not running')
        if name=='start_refinement_jobs':return
        rows=b['targets']
        if name in ('save_refinement','stage_refinement'):
            row=next((r for r in rows if r.get('package',{}).get('package_id')==args['package_id']),None)
            if not row or row.get('pause_requested') or row['state'] not in ('package_ready','candidate_ready','validated') or row['package']['package_hash']!=args['package_hash']:
                raise ContractError('FORBIDDEN_SCOPE','Only a ready authorized package may enter the save lane')
            if row['package']['target_id']!=row['target_id'] or row['package'].get('acceptance_id',b['acceptance_id'])!=b['acceptance_id']:
                raise ContractError('FORBIDDEN_SCOPE','Package target or acceptance differs from this batch')
        else:
            row=next((r for r in rows if r.get('candidate',{}).get('candidate_id')==args['candidate_id']),None)
            if not row or row['state']=='saved' or row['candidate']['candidate_hash']!=args['candidate_hash']:
                raise ContractError('FORBIDDEN_SCOPE','Only the recorded integration candidate may be saved')
        if b.get('active_save_target') not in (None,row['target_id']):raise ContractError('RESOURCE_EXHAUSTED','Another object holds the scene save lane')
        b['active_save_target']=row['target_id'];row.update(state='staging' if name in ('save_refinement','stage_refinement') else 'validating' if name=='validate_candidate' else 'saving',save_started_at=utc_now())
        self.put(p)

    def _rolling_tool_result(self,pid,name,result,error,args):
        p=self.get(pid);b=p['batch'];row=next((r for r in b['targets'] if r['target_id']==b.get('active_save_target')),None)
        if error and row and name in ('save_refinement','stage_refinement','validate_candidate','commit_candidate'):
            row.update(state='needs_review',error=str(error));b.pop('active_save_target',None)
        elif name in ('save_refinement','commit_candidate') and row:
            row.update(state='saved',head=result['head'],saved_at=utc_now());row.pop('error',None);b.pop('active_save_target',None)
            b['completed_count']=sum(r['state']=='saved' for r in b['targets'])
            self.event(pid,'batch_object_saved',text=f"已保存 {row['target_id']}（{b['completed_count']}/{b['total']}）")
        elif name=='stage_refinement' and row:row.update(state='candidate_ready',candidate=result['candidate'])
        elif name=='validate_candidate' and row:
            row.update(state='validated' if result['committable'] else 'needs_review',validation_id=result['validation_id'])
            if not result['committable']:b.pop('active_save_target',None)
        elif name=='finish_workflow' and not error:
            b.update(status='complete',completed_at=utc_now());b.pop('error',None)
        self.put(p);self.rolling.kick()

    def _rolling_settled(self,p):
        b=p['batch']
        if p['phase']=='paused':b['status']='paused';return False
        if p.get('run_failed') or p['phase'] in ('failed','needs_auth'):
            b.update(status='waiting',error=p.get('error'));return False
        return b['status']=='running'

    def _resume_rolling_batch(self,pid):
        p=self.status(pid);b=p['batch'];actions=self.action(pid)
        if p.get('active_run'):raise ValueError('当前 Pi 回合尚未结束')
        if not self.rolling.active(pid):
            for name in ('active.lock','integration_lane.lock'):
                if not archive_dead_claim(actions.root/name):raise ContractError('RESOURCE_EXHAUSTED','旧场景事务仍被占用或身份不明，需先核对')
        # A committed repository transaction can survive a lost response or a
        # crash before completion.json. Reconcile that exact commit first.
        for path in (actions.root/'integrations').glob('*/transaction.json'):
            transaction=json.loads(path.read_text(encoding='utf-8'))
            if transaction.get('candidate') and transaction.get('validation_id'):
                ref={k:transaction['candidate'][k] for k in ('candidate_id','candidate_hash')}
                record,_,_=asyncio.run(actions._candidate_record(**ref))
                previous=asyncio.run(actions.host.repo.get_request_result(pid,record['decision_ref']+'_commit'))
                if previous:asyncio.run(actions.commit_candidate(**ref,validation_id=transaction['validation_id']))
        acceptance=json.loads((actions.root/'acceptances'/(b['acceptance_id']+'.json')).read_text(encoding='utf-8'))
        _,_,completed=asyncio.run(actions._accepted_view(acceptance))
        self.rolling.reconcile(pid,{r['target_id']:r for r in completed})
        p=self.get(pid);b=p['batch'];b.update(status='running');b.pop('error',None);b.pop('wake_key',None)
        p.update(phase='idle',error=None);self.put(p);self.cancel.setdefault(pid,threading.Event()).clear()
        self.rolling.kick();self._advance_batch(pid);return self.get(pid)

    def _rolling_control(self,pid,request):
        with self.lock:
            p=self.get(pid);b=p['batch'];action=request['action']
            if action=='set_refinement_concurrency':
                value=request.get('parallelism')
                if type(value) is not int or not 1<=value<=4:raise ValueError('对象并发数必须为 1–4')
                b['parallelism']=value;b['effective_concurrency']=min(value,b.get('effective_concurrency',value)) if b.get('throttle_reason') else value
                self.put(p);self.rolling.kick();return self.get(pid)
            row=next((r for r in b['targets'] if r['target_id']==request.get('target_id')),None)
            if not row or row['state']=='saved':raise ValueError('目标不可暂停或恢复')
            if action=='pause_object':
                row['pause_requested']=True
                if row['state'] in ('pending','package_ready'):row['state']='paused'
                # In-flight generation drains into a package; a save already in
                # progress completes its transaction before stopping.
            else:
                row.pop('pause_requested',None)
                if row['state']=='paused':row['state']='package_ready' if row.get('package') else 'pending'
                if row['state']=='needs_review':self.rolling.recover_unstarted(pid,p,row)
                if b['status']=='waiting':b.update(status='running');b.pop('error',None);p['error']=None
                b.pop('wake_key',None)
            self.put(p);self.rolling.kick();return self.get(pid)
