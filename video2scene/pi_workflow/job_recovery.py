"""Explicit recovery only when durable evidence proves no author work occurred."""
import json

from video2scene.codex_capacity import capacity_rejection
from video2scene.contracts_v1 import ContractError,file_hash


def verified_unstarted_failure(service,actions,batch,row,job):
    def reject():raise ContractError('RESOURCE_EXHAUSTED','结果不明或已执行的作者不能重放；请核对原始产物')
    if job.get('state')!='failed' or job.get('operation_id')!=row['operation_id'] or job.get('target_id')!=row['target_id']:reject()
    folder=actions.root/'packages'/row['operation_id']
    if (folder/'package.json').exists():reject()
    started_path=folder/'started.json'
    if not started_path.exists():reject()
    started=json.loads(started_path.read_text(encoding='utf-8'))
    if started.get('acceptance_id')!=batch['acceptance_id'] or started.get('allowed_object_ids')!=[row['target_id']]:reject()
    prepared_path=folder/'prepared.json'
    files=[started_path]
    if not prepared_path.exists():
        result=job.get('result',{})
        target=next((o for o in started['before']['snapshot']['objects'] if o['instance_id']==row['target_id']),{})
        if result.get('error_code')!='EVIDENCE_MISMATCH' or result.get('error')!='Structure evidence must use original unmasked RGB and cameras' or target.get('kind')!='opening':reject()
        reason='Opening mask compatibility fixed; failure occurred before candidate preparation or model invocation'
    else:
        prepared=json.loads(prepared_path.read_text(encoding='utf-8'));files.append(prepared_path)
        matches=[]
        for path in (service.runtime/service.get(job['project_id'])['scene_id']/'assets').glob('gpt-direct-*/author_00/contract.json'):
            contract=json.loads(path.read_text(encoding='utf-8'))
            if contract.get('candidate')==prepared.get('source_candidate') and contract.get('instance_id')==row['target_id']:matches.append(path)
        if len(matches)!=1:reject()
        contract=matches[0];author=contract.parent;asset=author.parent
        request=author/'request.json';execution=asset/'author_00_control/invoke.log.execution.json'
        if not request.exists() or not execution.exists() or not capacity_rejection(author):reject()
        r=json.loads(request.read_text(encoding='utf-8'));e=json.loads(execution.read_text(encoding='utf-8'))
        if r.get('status')!='failed' or r.get('authentication')!='ChatGPT subscription' or e.get('state')!='failed' or not e.get('returncode') or not e.get('owned_subprocess'):reject()
        if any(p.name in ('object.py','asset.blend','asset.glb') for p in asset.rglob('*')):reject()
        if any(p.name!='author_00' for p in asset.glob('author_[0-9][0-9]')) or any(asset.glob('critic_*')):reject()
        for retry in (author/'preexecution_retries').glob('attempt_*'):
            if not capacity_rejection(retry):reject()
            files.append(retry/'events.jsonl')
        files.extend([contract,request,execution,author/'events.jsonl'])
        reason='Terminal model-capacity rejection before any assistant item, command or asset; exact failed invocation verified'
    return {'reason':reason,'evidence':[{'path':str(p),'sha256':file_hash(p)} for p in files]}
