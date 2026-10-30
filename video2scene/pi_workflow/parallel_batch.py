"""Pi-directed two-object waves with one scene integration lane."""
import asyncio
import json
import threading
import uuid

from video2scene.contracts_v1 import ContractError, canonical_hash, utc_now


class ParallelBatchMixin:
    def _worker_update(self, pid, target, progress):
        with self.lock:
            p = self.get(pid)
            row = next((r for r in p.get('batch', {}).get('targets', []) if r['target_id'] == target), None)
            if row:
                row['worker'] = progress
                self.put(p)
        self.event(pid, 'object_worker_progress', target_id=target, progress=progress)

    def _advance_parallel_batch(self, pid):
        p = self.get(pid)
        batch = p['batch']
        usable = [r for r in batch['targets'] if r['state'] not in {'saved', 'failed', 'paused'}]
        ready = [r for r in usable if r.get('package') or r.get('candidate')]
        chosen = ready[:2]
        if not chosen:
            pending = [r for r in usable if r['state'] == 'pending']
            for row in pending:
                deps = row.get('depends_on', [])
                if any(r['target_id'] in deps and r['state'] != 'saved' for r in batch['targets']):
                    continue
                chosen.append(row)
                if len(chosen) == 2:
                    break
        if chosen:
            batch['run_targets'] = [r['target_id'] for r in chosen]
            instruction = ('当前批次使用双对象并发生成、串行保存。先 workflow_status。仅处理本轮目标：'
                + json.dumps(batch['run_targets'], ensure_ascii=False) + '。acceptance_id=' + batch['acceptance_id'] + '。'
                '对尚无package/candidate的目标，一次refine_objects_parallel提交本轮一至两个objects，使用各自唯一operation_id。'
                '该工具内部同时启动独立Codex作者和critic，返回每个对象的package_id/package_hash或错误。'
                '对成功package逐个stage_refinement→validate_candidate→commit_candidate，保存一个后才stage下一个；'
                '已有candidate先复用验证和保存，不重新生成。一个对象失败时记录原因并继续本轮其他成功对象。'
                '绝不重放结果不明的作者调用，绝不重做已保存对象，不调用refine_object。'
                '本轮目标处理后结束，由宿主安排下一轮；不要提前export_scene或finish_workflow。')
        elif any(r['state'] != 'saved' for r in batch['targets']):
            batch.update(status='failed', error='部分对象未完成；其他成功对象已保留，请核对各对象错误或暂停状态。')
            p.update(phase='idle', error=batch['error']); self.put(p)
            return
        else:
            batch['run_targets'] = []
            instruction = '全部授权对象已逐项保存。先workflow_status，再export_scene和finish_workflow，完成整屋交付。'
        batch['run_start_completed'] = batch['completed_count']
        batch['run_start_pending'] = sum(r['state'] == 'pending' for r in batch['targets'])
        batch.pop('active_save_target', None)
        self.put(p)
        self.message(pid, {'request_id': 'parallel_step_' + uuid.uuid4().hex, 'message': instruction,
                           'selected_ids': batch['run_targets']}, source='system')

    def _parallel_tool_start(self, pid, name, args):
        p = self.get(pid); batch = p['batch']
        mutating = {'refine_object', 'refine_objects_parallel', 'stage_refinement', 'validate_candidate', 'commit_candidate'}
        if name not in mutating:
            if name in {'finish_workflow', 'export_scene'} and batch['completed_count'] != batch['total']:
                raise ContractError('FORBIDDEN_SCOPE', 'Finish only after every authorized target is saved')
            return
        if batch['status'] != 'running':
            raise ContractError('CANCELLED', 'Batch paused; reconcile before object operations')
        rows = {r['target_id']: r for r in batch['targets']}
        allowed = set(batch.get('run_targets', []))
        if name == 'refine_object':
            raise ContractError('FORBIDDEN_SCOPE', 'Parallel batch uses independent packages, not direct scene-bound authoring')
        if name == 'refine_objects_parallel':
            ids = [r['target_id'] for r in args['objects']]
            if (args['acceptance_id'] != batch['acceptance_id'] or len(set(ids)) != len(ids)
                    or not set(ids) <= allowed or batch.get('active_save_target')
                    or any(rows[i]['state'] != 'pending' or rows[i].get('package') or rows[i].get('candidate') for i in ids)):
                raise ContractError('FORBIDDEN_SCOPE', 'Workers must use distinct assigned pending targets')
            for item in args['objects']:
                row = rows[item['target_id']]
                row.update(state='generating', operation_id=item['operation_id'], attempts=row['attempts'] + 1)
        elif name == 'stage_refinement':
            row = next((r for r in rows.values() if r.get('package', {}).get('package_id') == args['package_id']), None)
            if (not row or row['target_id'] not in allowed or row['state'] != 'package_ready'
                    or row['package']['package_hash'] != args['package_hash'] or batch.get('active_save_target')):
                raise ContractError('FORBIDDEN_SCOPE', 'Integrate exactly one reviewed assigned package at a time')
            batch['active_save_target'] = row['target_id']; row['state'] = 'staging'
        else:
            row = next((r for r in rows.values() if r.get('candidate', {}).get('candidate_id') == args['candidate_id']), None)
            if (not row or row['target_id'] not in allowed or row['state'] == 'saved'
                    or row['candidate']['candidate_hash'] != args['candidate_hash']
                    or batch.get('active_save_target') not in (None, row['target_id'])):
                raise ContractError('FORBIDDEN_SCOPE', 'Validation and save must use the assigned integration candidate')
            batch['active_save_target'] = row['target_id']
            row['state'] = 'validating' if name == 'validate_candidate' else 'saving'
        self.put(p)

    def _parallel_tool_result(self, pid, name, result, error, args):
        p = self.get(pid); batch = p['batch']; args = args or {}
        rows = {r['target_id']: r for r in batch['targets']}
        row = rows.get(batch.get('active_save_target'))
        if name == 'refine_objects_parallel':
            for item in (result or {}).get('objects', []):
                target = rows[item['target_id']]
                if item['state'] == 'package_ready':
                    target.update(state='paused' if target.get('pause_requested') else 'package_ready', package=item)
                    target.pop('error', None)
                else:
                    target.update(state='failed', error=item['error'])
            if error:
                for target in rows.values():
                    if target['state'] == 'generating':
                        target.update(state='failed', error=str(error))
        elif error and row:
            row.update(state='failed', error=str(error)); batch.pop('active_save_target', None)
        elif name == 'stage_refinement' and row:
            row.update(state='candidate_ready', candidate=result['candidate'], merge_operation_id=result['operation_id'])
        elif name == 'validate_candidate' and row:
            row.update(state='validated' if result['committable'] else 'failed', validation_id=result['validation_id'])
            if not result['committable']:
                row['error'] = '合并后验证未通过'; batch.pop('active_save_target', None)
        elif name == 'commit_candidate' and row:
            row.update(state='saved', head=result['head']); row.pop('error', None)
            batch['completed_count'] = sum(r['state'] == 'saved' for r in rows.values())
            batch.pop('active_save_target', None)
            self.event(pid, 'batch_object_saved', text=f"已保存 {row['target_id']}（{batch['completed_count']}/{batch['total']}）")
        elif name == 'finish_workflow' and not error:
            batch.update(status='complete', completed_at=utc_now())
        self.put(p)

    def _parallel_settled(self, p):
        batch = p['batch']
        if p['phase'] == 'paused':
            batch['status'] = 'paused'; return False
        if p.get('run_failed') or p['phase'] in {'failed', 'needs_auth'}:
            batch.update(status='failed', error=p.get('error')); return False
        if batch['status'] != 'running': return False
        if (batch['completed_count'] > batch.get('run_start_completed', 0)
                or sum(r['state'] == 'pending' for r in batch['targets']) < batch.get('run_start_pending', 0)):
            return True
        batch.update(status='waiting', error='本轮尚有未保存资产，请查看对象状态后继续。')
        return False

    def _parallel_resume_rows(self, actions, batch, saved):
        for row in batch['targets']:
            if row['target_id'] in saved:
                row.update(state='saved', head=saved[row['target_id']]['head']); continue
            package_dir = actions.root / 'packages' / row.get('operation_id', 'not_started')
            merge_dir = actions.root / 'operations' / row.get('merge_operation_id', row.get('operation_id', 'not_started'))
            if (merge_dir / 'result.json').exists():
                result = json.loads((merge_dir / 'result.json').read_text(encoding='utf-8'))
                row.update(state='candidate_ready', candidate=result['candidate'])
            elif (package_dir / 'package.json').exists():
                package = json.loads((package_dir / 'package.json').read_text(encoding='utf-8'))
                row.update(state='paused' if row.get('pause_requested') else 'package_ready', package=actions._package_result(package))
            elif (package_dir / 'started.json').exists() or (merge_dir / 'started.json').exists():
                row.update(state='failed', error='已有作者调用但无完整资产包，需核对产物；不会自动重放。')
            elif row['state'] != 'paused':
                row['state'] = 'pending'
        batch.pop('active_save_target', None)

    def control_batch_object(self, pid, request):
        if self.get(pid).get('batch',{}).get('mode')=='rolling':return self._rolling_control(pid,request)
        with self.lock:
            p = self.get(pid); batch = p.get('batch', {})
            row = next((r for r in batch.get('targets', []) if r['target_id'] == request.get('target_id')), None)
            if not row or batch.get('parallelism') != 2 or row['state'] == 'saved':
                raise ValueError('该对象不属于可暂停的并发任务')
            if request['action'] == 'pause_object':
                if row['state'] in {'staging', 'validating', 'validated', 'saving'}:
                    raise ValueError('当前对象正在串行验证保存，请等待该事务完成')
                row['pause_requested'] = True
                if row['state'] in {'pending', 'package_ready'}: row['state'] = 'paused'
                # Drain an in-flight author/critic into a durable package, then stop
                # before integration. Resuming never needs to regenerate that asset.
            else:
                if p.get('active_run'): raise ValueError('请等待当前轮结束后恢复对象')
                row.pop('pause_requested', None)
                if row['state'] == 'paused': row['state'] = 'pending'
            self.put(p)
        if request['action'] == 'resume_object': return self.resume_batch(pid, parallelism=2)
        return self.get(pid)
