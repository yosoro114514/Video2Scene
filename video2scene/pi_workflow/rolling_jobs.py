"""Durable rolling jobs owned by the service, not by a temporary tool event loop."""
import asyncio
import copy
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from video2scene.contracts_v1 import ContractError,canonical_hash,utc_now
from .pi_runtime import save
from .parallel_refinement import identity
from .process_owner import owner_record,archive_dead_claim

LIVE={'registered','preparing','generating'}
READY={'package_ready','candidate_ready','staging','validating','validated','saving'}
POLICY={'failure_threshold':2,'cooldown_seconds':30,'recovery_seconds':60,'ready_package_limit':3}


def error_kind(error):
    value=str(error).lower()
    if any(t in value for t in ('codex_login','unauthorized','forbidden','usage limit','insufficient_quota','subscription_sharing_usage_limit_exceeded')):return 'auth_or_quota'
    if any(t in value for t in ('fetch failed','terminated','connection reset','econnreset','network','stream disconnected','429','rate limit','502','503','504','at capacity')):return 'transient_connection'
    if any(t in value for t in ('stale','target or support','changed')):return 'stale'
    if 'cancel' in value.lower() or 'paused' in value.lower():return 'cancelled'
    return 'needs_review'


class RollingJobs:
    def __init__(self,service,*,start=True):
        self.service=service;self.owner=uuid.uuid4().hex
        self.owner_path=service.root/'rolling_owner.json'
        if not archive_dead_claim(self.owner_path):raise ContractError('RESOURCE_EXHAUSTED','Another workflow service owns rolling dispatch; do not run two hosts on one root')
        from video2scene.agent_scene.batch_refinement import immutable_json
        immutable_json(self.owner_path,{**owner_record(),'owner':self.owner})
        self.executor=ThreadPoolExecutor(max_workers=16,thread_name_prefix='room-object')
        self.futures={};self.stop=threading.Event();self.signal=threading.Event()
        with service.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS refinement_tasks(id TEXT PRIMARY KEY,project_id TEXT,state TEXT,data TEXT)')
        self.thread=threading.Thread(target=self._loop,name='room-rolling-dispatch',daemon=True)
        if start:self.start_supervisor()

    def start_supervisor(self):self.thread.start()

    def _put(self,job):
        job['updated_at']=utc_now()
        with self.service.db() as db:db.execute('INSERT OR REPLACE INTO refinement_tasks VALUES(?,?,?,?)',
            (job['job_id'],job['project_id'],job['state'],json.dumps(job,ensure_ascii=False)))

    def jobs(self,pid):
        with self.service.db() as db:return [json.loads(r['data']) for r in db.execute('SELECT data FROM refinement_tasks WHERE project_id=? ORDER BY rowid',(pid,))]

    def _publish(self,pid,row,kind,**data):
        seq=self.service.event(pid,kind,target_id=row['target_id'],job_id=row.get('job_id'),**data)
        row['event_seq']=seq

    def start(self,pid,acceptance_id,operation_id,target_ids):
        """Pi explicitly grants mechanical refill within the existing human scope."""
        identity(operation_id);s=self.service
        with s.lock:
            p=s.get(pid);b=p['batch']
            binding={'acceptance_id':acceptance_id,'operation_id':operation_id,'target_ids':target_ids}
            if b.get('mode')!='rolling' or acceptance_id!=b['acceptance_id'] or set(target_ids)!={r['target_id'] for r in b['targets']} or len(target_ids)!=len(set(target_ids)):
                raise ContractError('FORBIDDEN_SCOPE','Dispatch authorization must exactly match the existing batch')
            if b.get('dispatch_authorization'):
                if b['dispatch_authorization']['binding']!=binding:raise ContractError('IDEMPOTENCY_CONFLICT','This batch already has a dispatch authorization')
            else:
                b['dispatch_authorization']={'binding':binding,'pi_run_id':p.get('active_run'),'at':utc_now()}
                save(s.run_root(pid)/'batches'/b['batch_id']/'dispatch_authorization.json',b['dispatch_authorization'])
                for row in b['targets']:
                    row.setdefault('operation_id','rolling_'+canonical_hash([b['batch_id'],operation_id,row['target_id']])[:40])
                    row.setdefault('job_id','object_'+canonical_hash([pid,row['operation_id']])[:32])
                s.put(p)
        self.kick();return self.status(pid)

    def status(self,pid,after=0):
        p=self.service.get(pid);b=p.get('batch',{})
        kinds=('object_registered','object_package_ready','object_failed','object_worker_progress','batch_object_saved')
        with self.service.db() as db:
            rows=db.execute('SELECT seq,type,data,at FROM events WHERE project_id=? AND seq>? AND type IN (?,?,?,?,?) ORDER BY seq LIMIT 100',(pid,after,*kinds)).fetchall()
        events=[{'seq':r['seq'],'type':r['type'],'data':json.loads(r['data']),'at':r['at']} for r in rows]
        return {'batch_id':b.get('batch_id'),'mode':b.get('mode'),'status':b.get('status'),
            'requested_concurrency':b.get('parallelism'),'effective_concurrency':b.get('effective_concurrency'),
            'objects':b.get('targets',[]),'events':{'events':events,'cursor':events[-1]['seq'] if events else after},
            'resources':self.service.resources.snapshot(),'saved':b.get('completed_count',0)}

    def kick(self):self.signal.set()

    def active(self,pid):
        return any(not future.done() for job in self.jobs(pid) if (future:=self.futures.get(job['job_id'])) is not None)

    def _loop(self):
        while not self.stop.is_set():
            self.signal.wait(.5);self.signal.clear()
            if self.stop.is_set():break
            for p in self.service.projects():
                if p.get('batch',{}).get('mode')!='rolling':continue
                try:self._dispatch(p['id'])
                except Exception as error:
                    with self.service.lock:
                        current=self.service.get(p['id']);b=current['batch'];b.update(status='waiting',error=str(error));self.service.put(current)
                        self.service.event(p['id'],'rolling_dispatch_failed',text=str(error))

    def _dispatch(self,pid):
        s=self.service;selected=[]
        with s.lock:
            p=s.get(pid);b=p['batch']
            if b['status']!='running' or not b.get('dispatch_authorization') or p['phase'] in ('paused','failed','needs_auth'):return
            now=time.time()
            if now<b.get('cooldown_until',0):return
            requested=b['parallelism'];effective=b.get('effective_concurrency',requested)
            if effective<requested and now-b.get('last_connection_failure',now)<POLICY['recovery_seconds']:pass
            elif effective<requested:
                b['effective_concurrency']=effective=min(requested,effective+1);b['last_connection_failure']=now;b.pop('throttle_reason',None)
            effective=min(requested,effective)
            active=sum(r['state'] in LIVE for r in b['targets'])
            ready=sum(r['state'] in READY for r in b['targets'])
            if ready<b.get('ready_package_limit',3):
                for row in b['targets']:
                    if active>=effective:break
                    if row['state']!='pending' or row.get('pause_requested'):continue
                    if any(r['target_id'] in row.get('depends_on',[]) and r['state']!='saved' for r in b['targets']):continue
                    row.update(state='registered',attempts=row.get('attempts',0)+1,queued_at=utc_now())
                    job={**copy.deepcopy(row),'project_id':pid,'batch_id':b['batch_id'],'acceptance_id':b['acceptance_id'],
                        'owner':self.owner,'service_pid':os.getpid(),'state':'registered'}
                    self._put(job);self._publish(pid,row,'object_registered');selected.append(job);active+=1
            s.put(p)
            # Register futures before releasing the state lock: resume cannot mistake
            # an admitted worker for an orphan during this interval.
            for job in selected:self.futures[job['job_id']]=self.executor.submit(self._run,job)
        self.service._advance_batch(pid)

    def _run(self,job):
        s=self.service;pid=job['project_id'];actions=s.action(pid);result=None
        try:
            # The durable per-target claim survives a service crash. No lease timeout
            # removes it: an unknown process outcome always requires reconciliation.
            claim=actions.root/'workers'/(identity(job['target_id'])+'.claim.json')
            from video2scene.agent_scene.batch_refinement import immutable_json
            immutable_json(claim,{'job_id':job['job_id'],'owner':self.owner,**owner_record(),'at':utc_now()})
            try:
                context=s.cancel.setdefault(pid,threading.Event())
                while True:
                    if context.is_set() or self.stop.is_set():raise ContractError('CANCELLED','Project paused before preparation')
                    # Preparation never overlaps a scene integration transaction.
                    if actions.preparation_gate.acquire(timeout=.1):break
                try:
                    job['state']='preparing';self._put(job)
                    acceptance=json.loads((actions.root/'acceptances'/(job['acceptance_id']+'.json')).read_text(encoding='utf-8'))
                    prepared=asyncio.run(actions._prepare_package(job,acceptance))
                finally:actions.preparation_gate.release()
                job.update(state='generating',started_at=utc_now());self._put(job)
                with s.lock:
                    p=s.get(pid);row=next(r for r in p['batch']['targets'] if r['job_id']==job['job_id'])
                    row.update(state='generating',started_at=job['started_at']);s.put(p)
                result=asyncio.run(actions._generate_package(prepared))
            finally:
                # Only the owning worker removes its own claim, after its child
                # processes have returned. A competing manager cannot remove it.
                if claim.exists() and json.loads(claim.read_text())['owner']==self.owner:claim.unlink()
        except Exception as error:result={'state':'failed','error':str(error),'error_code':getattr(error,'code',None),'target_id':job['target_id']}
        with s.lock:
            p=s.get(pid);b=p['batch'];row=next(r for r in b['targets'] if r.get('job_id')==job['job_id'])
            job.update(state=result['state'],result=result,finished_at=utc_now());self._put(job)
            if result['state']=='package_ready':
                row.update(state='paused' if row.get('pause_requested') else 'package_ready',package=result,ready_at=job['finished_at']);row.pop('error',None)
            else:
                category=error_kind(str(result.get('error_code',''))+' '+result['error']);row.update(state='needs_review',error=result['error'],error_kind=category)
                if category=='transient_connection':self._connection_failure(b,result['error'])
                elif category=='stale':b.update(status='stale',error=result['error']);p.update(phase='needs_review',error=result['error'])
                elif category=='auth_or_quota':b.update(status='waiting',error=result['error']);p.update(phase='needs_auth' if 'login' in result['error'].lower() or '401' in result['error'] else 'failed',error=result['error'])
            self._publish(pid,row,'object_package_ready' if result['state']=='package_ready' else 'object_failed',result=result)
            s.put(p)
        self.kick()

    def _connection_failure(self,b,error):
        b['connection_failures']=b.get('connection_failures',0)+1;b['last_connection_failure']=time.time()
        if b['connection_failures']>=POLICY['failure_threshold']:
            b.update(effective_concurrency=max(1,b.get('effective_concurrency',b['parallelism'])-1),
                cooldown_until=time.time()+POLICY['cooldown_seconds'],throttle_reason=str(error));b['connection_failures']=0

    def reconnect_event(self,pid,event):
        with self.service.lock:
            p=self.service.get(pid);b=p.get('batch',{})
            if b.get('mode')=='rolling' and event['type']=='auto_retry_start':
                self._connection_failure(b,event.get('data',{}).get('errorMessage','Model reconnecting'));self.service.put(p)

    def reconcile(self,pid,saved):
        """Called only on explicit resume; never restart an unknown invocation."""
        s=self.service;actions=s.action(pid);p=s.get(pid);b=p['batch']
        for row in b['targets']:
            target=row['target_id'];future=self.futures.get(row.get('job_id'))
            if target in saved:row.update(state='saved',head=saved[target]['head']);continue
            if future and not future.done():continue
            claim=actions.root/'workers'/(identity(target)+'.claim.json')
            if not archive_dead_claim(claim):row.update(state='needs_review',error='旧工作进程占用尚未核对；不会自动重放。');continue
            package=actions.root/'packages'/row.get('operation_id','not_started')
            if (package/'package.json').exists():
                record=json.loads((package/'package.json').read_text(encoding='utf-8'))
                if record['allowed_object_ids']!=[target] or record['acceptance_id']!=b['acceptance_id']:
                    raise ContractError('STALE_ARTIFACT','Recovered package belongs to another target or acceptance')
                value=actions._package_result(record)
                asyncio.run(actions._read_package(value['package_id'],value['package_hash']))
                row.update(state='paused' if row.get('pause_requested') else 'package_ready',package=value)
            elif package.exists():row.update(state='needs_review',error='作者或准备过程已有记录，结果需核对；不会自动重放。')
            elif not row.get('pause_requested'):row['state']='pending'
        b['completed_count']=sum(r['state']=='saved' for r in b['targets']);b.pop('active_save_target',None)
        s.put(p)

    def recover_unstarted(self,pid,p,row):
        """An explicit resume_object request, with fresh IDs and preserved history."""
        from .job_recovery import verified_unstarted_failure
        from video2scene.agent_scene.batch_refinement import immutable_json
        future=self.futures.get(row.get('job_id'))
        if future and not future.done():raise ContractError('RESOURCE_EXHAUSTED','旧工作进程尚未收尾')
        job=next((j for j in self.jobs(pid) if j['job_id']==row.get('job_id')),None)
        if not job:raise ContractError('RESOURCE_EXHAUSTED','缺少可核对的失败作业记录')
        actions=self.service.action(pid);b=p['batch']
        if not archive_dead_claim(actions.root/'workers'/(identity(row['target_id'])+'.claim.json')):
            raise ContractError('RESOURCE_EXHAUSTED','旧工作进程身份尚未核对')
        proof=verified_unstarted_failure(self.service,actions,b,row,job)
        acceptance=json.loads((actions.root/'acceptances'/(b['acceptance_id']+'.json')).read_text(encoding='utf-8'))
        asyncio.run(actions._accepted_view(acceptance))
        history=copy.deepcopy(row.get('attempt_history',[]));old=copy.deepcopy(row);old.pop('attempt_history',None)
        new_op='rolling_recovery_'+canonical_hash([b['batch_id'],row['operation_id'],len(history)])[:32]
        receipt={**proof,'at':utc_now(),'target_id':row['target_id'],'acceptance_id':b['acceptance_id'],
                 'old_job_id':row['job_id'],'old_operation_id':row['operation_id'],'new_operation_id':new_op}
        receipt_path=actions.root/'recoveries'/(row['operation_id']+'.json')
        if receipt_path.exists():
            previous=json.loads(receipt_path.read_text(encoding='utf-8'))
            if {k:v for k,v in previous.items() if k!='at'}!={k:v for k,v in receipt.items() if k!='at'}:
                raise ContractError('STALE_ARTIFACT','Recovery checkpoint no longer matches the verified failure')
            receipt=previous
        else:immutable_json(receipt_path,receipt)
        history.append({'previous':old,'recovery':receipt})
        base={k:row[k] for k in ('target_id','kind','category','depends_on','attempts') if k in row}
        row.clear();row.update(base,state='pending',operation_id=new_op,
            job_id='object_'+canonical_hash([pid,new_op])[:32],attempt_history=history)
        self.service.event(pid,'object_recovery_authorized',target_id=row['target_id'],text=proof['reason'])

    def close(self):
        self.stop.set();self.signal.set();self.thread.join(timeout=3)
        self.executor.shutdown(wait=True,cancel_futures=True)
        if self.owner_path.exists() and json.loads(self.owner_path.read_text())['owner']==self.owner:self.owner_path.unlink()
