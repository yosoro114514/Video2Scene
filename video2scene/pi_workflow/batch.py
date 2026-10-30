"""Persistent all-remaining queue. Pi still decides and executes each tool loop."""
import asyncio
import json
import threading
import uuid

from video2scene.contracts_v1 import ContractError, canonical_hash, utc_now
from .scene_actions import ROOM_KINDS
from .pi_runtime import save
from .parallel_batch import ParallelBatchMixin
from .rolling_batch import RollingBatchMixin


def remaining_objects(snapshot):
    kinds = {'object', 'support'} | ROOM_KINDS
    return [obj for obj in snapshot['objects'] if obj['kind'] in kinds
            and obj['asset']['asset_version'].startswith('whitebox-')]


class BatchWorkflowMixin(RollingBatchMixin,ParallelBatchMixin):
    def start_refine_all(self, pid, request):
        def perform():
            mode=request.get('mode','waves' if 'parallelism' in request else 'rolling')
            if mode not in ('waves','rolling'):raise ValueError('Unknown batch mode')
            parallelism = request.get('parallelism', 3 if mode=='rolling' else 2)
            if type(parallelism) is not int or parallelism not in (range(1,5) if mode=='rolling' else (1,2)): raise ValueError('对象并发数必须为 1–4；旧分轮模式仅支持 1 或 2')
            p = self.status(pid)
            self._require_no_pending(pid)
            if p.get('active_run'):
                raise ValueError('请等待当前任务结束后开始全部精修')
            if p.get('batch') and p['batch']['status'] not in {'complete', 'stale'}:
                raise ValueError('已有批次，请使用继续；不会重新生成已保存对象')
            view = asyncio.run(self.action(pid).inspect_scene())
            if view['head'] != {'revision': request.get('revision'), 'head_seq': request.get('head_seq')}:
                raise ContractError('STALE_REVISION', '全部精修必须绑定当前保存布局，请刷新页面')
            targets = remaining_objects(view['snapshot'])
            if not targets:
                raise ValueError('当前场景没有待精修的白模对象')
            ids = [o['instance_id'] for o in targets]
            payload = {'target_ids': ids, 'include_structures': True,
                       'visual_policy': 'structure_first_relaxed_appearance',
                       'user_instruction': '采用当前保存布局，一键精修全部剩余家具、陈设、墙体、地板和门。逐件独立复核、验证后保存。'
                       '跳过已精修资产，保持所有对象的位置、旋转、尺寸、支撑和房间轮廓、墙角、门洞。墙面和门楣使用一致暖白饰面，'
                       '地板采用合理木质饰面，门根据原画面建模。颜色纹理允许合理偏差；每个结构段只在自己的边界内建模，不封闭相邻门洞。'}
            human = {'event_id': 'human_' + uuid.uuid4().hex[:20], 'scene_id': pid, 'actor': 'user',
                     'source': 'browser', 'kind': 'accept_whitebox', 'head': view['head'],
                     'snapshot_hash': canonical_hash(view['snapshot']), 'payload': payload}
            with self.db() as db:
                db.execute('INSERT INTO human_events VALUES(?,?,?)', (human['event_id'], pid, json.dumps(human, ensure_ascii=False)))
            acceptance = asyncio.run(self.action(pid).accept_whitebox(payload, human['event_id']))
            batch = {'batch_id': 'batch_' + uuid.uuid4().hex[:20], 'acceptance_id': acceptance['acceptance_id'],
                     'base_head': view['head'], 'status': 'running', 'created_at': utc_now(), 'parallelism': parallelism,
                     'skipped_ids': [o['instance_id'] for o in view['snapshot']['objects'] if o['instance_id'] not in ids],
                     'targets': [{'target_id': o['instance_id'], 'kind': o['kind'], 'category': o['category'],
                                  'state': 'pending', 'attempts': 0,
                                  'depends_on': [o['support_id']] if o.get('support_id') in ids else []} for o in targets],
                     'completed_count': 0, 'total': len(ids)}
            if mode=='rolling':batch.update(mode='rolling',schema_version=2,effective_concurrency=parallelism,ready_package_limit=3,
                resources=self.resources.snapshot()['limits'])
            p.update(batch=batch, acceptance=acceptance, phase='idle', error=None)
            self.put(p)
            save(self.run_root(pid) / 'batches' / batch['batch_id'] / 'authorization.json', human)
            self.event(pid, 'batch_started', text=f'已采用当前布局，开始逐件精修 {len(ids)} 个对象', batch_id=batch['batch_id'], targets=ids)
            self._advance_batch(pid)
            return self.get(pid)
        return self._dedup('refine_all:' + pid, request, perform)

    def _require_no_pending(self, pid):
        if self.rolling.active(pid):raise ValueError('等待后台对象任务收尾后再开始新的批次')
        with self.db() as db:
            pending = db.execute("SELECT 1 FROM jobs WHERE project_id=? AND state='running' LIMIT 1", (pid,)).fetchone()
        native = [p for p in self.run_root(pid).glob('tools/*/*/process.json') if not (p.parent / 'process_result.json').exists()]
        if pending or native:
            raise ValueError('等待当前工具收尾后再继续')

    def _advance_batch(self, pid):
        with self.lock:
            if self._closing:return
            p = self.get(pid); batch = p.get('batch')
            if not batch or batch['status'] != 'running' or p.get('active_run') or p['phase'] == 'paused':
                return
            if batch.get('mode')=='rolling':return self._advance_rolling_batch(pid)
            if batch.get('parallelism') == 2:
                return self._advance_parallel_batch(pid)
            pending = next((r for r in batch['targets'] if r['state'] != 'saved'), None)
            if pending:
                instruction = ('当前已授权全部精修批次：本轮只处理下一项 ' + pending['target_id'] +
                               '，使用 acceptance_id=' + batch['acceptance_id'] + '。先 workflow_status。'
                               '已有 candidate 时复用其 candidate_id/hash，验证并提交；否则 refine_object→validate_candidate→commit_candidate。'
                               '不得重新生成已保存对象。保存此项后结束本轮，宿主自动衔接同一 Pi 会话的下一项，不要再请求人工采用。'
                               '技术或视觉失败应说明原因并停止；不可跳过失败项或在本轮重新运行作者。')
            else:
                instruction = '当前全部精修批次的所有目标已经保存。先核对 workflow_status，再 export_scene 和 finish_workflow 完成交付。'
            batch['run_start_completed'] = batch['completed_count']
            self.put(p)
            self.message(pid, {'request_id': 'batch_step_' + uuid.uuid4().hex, 'message': instruction,
                              'selected_ids': [pending['target_id']] if pending else []}, source='system')

    def _batch_tool_start(self, pid, name, args):
        with self.lock:
            p = self.get(pid); batch = p.get('batch')
            if not batch or batch['status'] in {'complete', 'stale'}: return
            if batch.get('mode')=='rolling':return self._rolling_tool_start(pid,name,args)
            if batch.get('parallelism') == 2:
                return self._parallel_tool_start(pid, name, args)
            if batch['status'] != 'running':
                if name in {'refine_object', 'validate_candidate', 'commit_candidate'}:
                    raise ContractError('CANCELLED', 'Batch stopped; reconcile and resume before another object operation')
                return
            row = next((r for r in batch['targets'] if r['state'] != 'saved'), None)
            if name == 'refine_object':
                if not row or args['target_id'] != row['target_id'] or args['acceptance_id'] != batch['acceptance_id']:
                    raise ContractError('FORBIDDEN_SCOPE', 'Batch must refine the next pending authorized object')
                if row.get('candidate') or row['state'] not in {'pending'}:
                    raise ContractError('IDEMPOTENCY_CONFLICT', 'Reconcile the existing candidate or failed attempt before another author call')
                row.update(state='generating', operation_id=args['operation_id'], attempts=row['attempts'] + 1)
            elif name in {'validate_candidate', 'commit_candidate'}:
                if not row or row.get('candidate', {}).get('candidate_id') != args['candidate_id']:
                    raise ContractError('FORBIDDEN_SCOPE', 'Batch candidate must belong to the current pending object')
                row['state'] = 'validating' if name == 'validate_candidate' else 'saving'
            self.put(p)

    def _batch_tool_result(self, pid, name, result=None, error=None, args=None):
        with self.lock:
            p = self.get(pid); batch = p.get('batch')
            if not batch or batch['status'] in {'complete', 'stale'}: return
            if batch.get('mode')=='rolling':return self._rolling_tool_result(pid,name,result,error,args)
            if batch.get('parallelism') == 2:
                return self._parallel_tool_result(pid, name, result, error, args)
            row = next((r for r in batch['targets'] if r['state'] != 'saved'), None)
            if error:
                batch.update(status='failed', error=str(error))
                if row and name in {'refine_object', 'validate_candidate', 'commit_candidate'}:
                    row.update(state='failed', error=str(error))
            elif name == 'refine_object' and row:
                row.update(state='candidate_ready', candidate=result['candidate'])
            elif name == 'validate_candidate' and row:
                row.update(state='validated' if result['committable'] else 'failed', validation_id=result['validation_id'])
                if not result['committable']: batch.update(status='failed', error='当前对象验证未通过，批次停止，已保存对象保留。')
            elif name == 'commit_candidate' and result.get('kind') == 'refine' and row:
                row.update(state='saved', head=result['head']); row.pop('error', None)
                batch['completed_count'] = sum(r['state'] == 'saved' for r in batch['targets'])
                self.event(pid, 'batch_object_saved', text=f"已保存 {row['target_id']}（{batch['completed_count']}/{batch['total']}）")
            elif name == 'finish_workflow':
                batch.update(status='complete', completed_at=utc_now())
            self.put(p)

    def _batch_settled(self, p):
        batch = p.get('batch')
        if not batch or batch['status'] == 'complete': return False
        if batch.get('mode')=='rolling':return self._rolling_settled(p)
        if batch.get('parallelism') == 2: return self._parallel_settled(p)
        if p['phase'] == 'paused': batch['status'] = 'paused'; return False
        if p.get('run_failed') or p['phase'] in {'failed', 'needs_auth'}:
            batch.update(status='failed', error=p.get('error')); return False
        if batch['status'] != 'running': return False
        if batch['completed_count'] > batch.get('run_start_completed', 0): return True
        batch.update(status='waiting', error='本轮未完成新对象，请查看运行记录后继续。')
        return False

    def resume_batch(self, pid, parallelism=None):
        if self.get(pid).get('batch',{}).get('mode')=='rolling':return self._resume_rolling_batch(pid)
        p = self.status(pid); self._require_no_pending(pid)
        if p.get('active_run'): raise ValueError('当前任务尚未停止')
        batch = p['batch']; actions = self.action(pid)
        acceptance = json.loads((actions.root / 'acceptances' / (batch['acceptance_id'] + '.json')).read_text(encoding='utf-8'))
        try:
            _, _, completed = asyncio.run(actions._accepted_view(acceptance))
        except ContractError:
            batch['status'] = 'stale'; self.put(p)
            raise
        saved = {r['target_id']: r for r in completed}
        if parallelism is not None:
            if parallelism not in (1, 2): raise ValueError('对象并发数仅支持 1 或 2')
            batch['parallelism'] = parallelism
        if batch.get('parallelism') == 2:
            objects = {o['instance_id']: o for o in asyncio.run(actions.inspect_scene())['snapshot']['objects']}
            ids = {r['target_id'] for r in batch['targets']}
            for row in batch['targets']:
                support = objects[row['target_id']].get('support_id')
                row['depends_on'] = [support] if support in ids else []
            self._parallel_resume_rows(actions, batch, saved)
        for row in ([] if batch.get('parallelism') == 2 else batch['targets']):
            if row['target_id'] in saved:
                row.update(state='saved', head=saved[row['target_id']]['head']); continue
            operation = actions.root / 'operations' / row.get('operation_id', 'not_started')
            if (operation / 'result.json').exists():
                result = json.loads((operation / 'result.json').read_text(encoding='utf-8'))
                row.update(state='candidate_ready', candidate=result['candidate'])
            elif (operation / 'started.json').exists():
                # No new provider call while the outcome of an old one is unknown.
                raise ContractError('RESOURCE_EXHAUSTED', '该对象作者调用已开始但没有可复用候选，请先核对失败／中断产物；不会自动重放模型调用')
            else: row['state'] = 'pending'
        batch.update(status='running', completed_count=len(saved)); batch.pop('error', None)
        p.update(phase='idle', error=None); self.put(p)
        self.cancel.setdefault(pid, threading.Event()).clear()
        self._advance_batch(pid)
        return self.get(pid)
