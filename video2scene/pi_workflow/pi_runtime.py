"""Owned Pi SDK process, explicit JSONL commands and persisted subscription receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
BRIDGE=ROOT/'tools/pi-runtime/bridge.mjs'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


class PiBridge:
    def __init__(self,config,folder,*,token='',on_event=None):
        self.folder=Path(folder).resolve();self.folder.mkdir(parents=True,exist_ok=True)
        self.config=config;self.on_event=on_event;self.ready=threading.Event();self.idle=threading.Event();self.idle.set()
        self.events=[];self.error=None;self.lock=threading.RLock();self.process=None
        self.config_path=self.folder/'config.json';save(self.config_path,config)
        env=os.environ.copy();env['V2S_PI_TOOL_TOKEN']=token
        # Pi uses only the explicitly supplied subscription CredentialStore.
        env.pop('OPENAI_API_KEY',None)
        env.pop('CODEX_API_KEY',None)
        self.stderr=(self.folder/'stderr.log').open('a',encoding='utf-8')
        self.process=subprocess.Popen(['node',str(BRIDGE),str(self.config_path)],cwd=ROOT,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _read(self):
        try:
            for raw in self.process.stdout:
                event=json.loads(raw.decode('utf-8'))
                with self.lock:
                    self.events.append(event)
                    with (self.folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                if event['type']=='ready':self.ready.set()
                if event['type']=='prompt_started':self.idle.clear()
                if event['type'] in ('settled','run_error','fatal','aborted'):
                    self.idle.set()
                    if event['type'] in ('run_error','fatal'):self.error=event.get('error')
                if self.on_event:
                    try:self.on_event(event)
                    except Exception as error:
                        # Keep draining stdout: a callback failure must not deadlock Pi.
                        self.error=str(error)
                        with (self.folder/'callback_errors.log').open('a',encoding='utf-8') as f:f.write(str(error)+'\n')
        except Exception as error:
            self.error=str(error)
        finally:
            if self.process:self.process.wait()
            self.idle.set();self.ready.set()
            if self.on_event:self.on_event({'type':'process_exit','exit_code':self.process.poll(),'error':self.error})

    def start(self,timeout=40):
        if not self.ready.wait(timeout):raise TimeoutError('Pi startup did not complete')
        if not any(e['type']=='ready' for e in self.events):raise RuntimeError('Pi process startup failed; inspect its local stderr log')
        return next(e for e in self.events if e['type']=='ready')

    def send(self,value):
        with self.lock:
            if self.process.poll() is not None:raise RuntimeError('Pi process is no longer running')
            if value['type']=='prompt':self.idle.clear()
            self.process.stdin.write((json.dumps(value,ensure_ascii=False)+'\n').encode('utf-8'));self.process.stdin.flush()

    def close(self):
        if self.process and self.process.poll() is None:
            self.send({'type':'close'})
            try:self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.stderr.close()


def smoke():
    folder=ROOT/'outputs/pi_workflow_v1/development/subscription_smoke'
    if (folder/'result.json').exists():raise RuntimeError('Subscription probe already completed; do not replay it')
    config={'project_id':'subscription-smoke','model':'gpt-6-astra','effort':'high','cwd':str(folder),
        'session_dir':str(folder/'sessions'),'smoke':True,
        'smoke_tool_image':str(ROOT/'data/redwood_livingroom1_rgb_20261028/rgb/01148.jpg'),
        'system_prompt':'You are testing Pi image input and a real custom tool loop. Call workflow_probe once after inspecting the user image, then describe the returned image. Chinese, concise.',
        'tools':[{'name':'workflow_probe','description':'Returns a second actual image after recording your first-image observation.',
            'parameters':{'type':'object','properties':{'observation':{'type':'string'}},'required':['observation'],'additionalProperties':False}}]}
    bridge=PiBridge(config,folder)
    try:
        ready=bridge.start();print(json.dumps(ready,ensure_ascii=False),flush=True)
        bridge.send({'type':'prompt','id':'subscription-smoke-001','run_id':'subscription-smoke-001',
            'message':'请先描述第一张图的主要物体并调用 workflow_probe，然后看工具返回的图，再用一句话说明两图分别是什么。',
            'images':[str(ROOT/'data/redwood_livingroom1_rgb_20261028/rgb/00300.jpg')]})
        if not bridge.idle.wait(240):raise TimeoutError('Subscription probe still running')
        messages=[e for e in bridge.events if e['type']=='assistant_message']
        calls=[e for e in bridge.events if e['type']=='tool_receipt']
        if bridge.error or not calls or not messages or any(e.get('stopReason')=='error' for e in messages):
            raise RuntimeError(bridge.error or 'Pi model/tool/image probe did not complete')
        result={'passed':True,'provider':'openai-codex','authentication':'existing Codex ChatGPT subscription',
            'pi_version':'1.4.0','model':'gpt-6-astra','actual_tool_calls':len(calls),'model_turns':len(messages),
            'input_image':True,'tool_result_image':True,'response':messages[-1]['text'],'session_file':ready['session_file']}
        save(folder/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:bridge.close()


if __name__=='__main__':smoke()
