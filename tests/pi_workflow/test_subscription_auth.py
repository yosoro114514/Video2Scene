"""Startup failures are explicit and never fall back to API billing. No network calls."""
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import pytest


@pytest.mark.parametrize('auth', ['missing','expired','api_key_only'])
def test_subscription_login_failure_is_json_event_and_readonly(tmp_path,auth):
    root=Path(__file__).resolve().parents[2];config=tmp_path/'config.json'
    config.write_text(json.dumps({'project_id':'fixture-auth','model':'gpt-6-astra','cwd':str(tmp_path),
        'session_dir':str(tmp_path/'sessions'),'system_prompt':'fixture','tools':[]}),encoding='utf-8')
    auth_path=tmp_path/'auth.json'
    if auth=='expired':
        claim=base64.urlsafe_b64encode(json.dumps({'exp':1}).encode()).decode().rstrip('=')
        auth_path.write_text(json.dumps({'auth_mode':'chatgpt','tokens':{'access_token':'fixture.'+claim+'.signature','account_id':'fixture-account'}}))
    elif auth=='api_key_only':auth_path.write_text(json.dumps({'auth_mode':'apikey','OPENAI_API_KEY':'fixture-never-use'}))
    before=auth_path.read_bytes() if auth_path.exists() else None
    env={**os.environ,'CODEX_HOME':str(tmp_path),'OPENAI_API_KEY':'fixture-never-use'}
    result=subprocess.run(['node',str(root/'tools/pi-runtime/bridge.mjs'),str(config)],cwd=root,env=env,
        capture_output=True,timeout=25,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    events=[json.loads(row) for row in result.stdout.decode('utf-8').splitlines() if row.strip()]
    assert result.returncode!=0
    assert any(row.get('type')=='fatal' and 'CODEX_LOGIN' in row.get('error','') for row in events)
    assert not any(row.get('type') in ('ready','prompt_started','assistant_message') for row in events)
    assert (auth_path.read_bytes() if auth_path.exists() else None)==before
    assert b'fixture-never-use' not in result.stdout+result.stderr


@pytest.mark.parametrize('failure',['fetch failed','401 Unauthorized'])
def test_real_bridge_retry_is_compatible_with_running_host(tmp_path,failure):
    """Actual bridge/SDK, fake auth and model transport; no network or scene writes."""
    root=Path(__file__).resolve().parents[2]
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'project_id':'fixture-reconnect','model':'gpt-6-astra','cwd':str(tmp_path),
        'session_dir':str(tmp_path/'sessions'),'system_prompt':'fixture','tools':[]}),encoding='utf-8')
    claim=base64.urlsafe_b64encode(json.dumps({'exp':time.time()+3600}).encode()).decode().rstrip('=')
    (tmp_path/'auth.json').write_text(json.dumps({'auth_mode':'chatgpt','tokens':{'access_token':'fixture.'+claim+'.signature','account_id':'fixture-account'}}))
    preload=tmp_path/'offline.mjs'
    runtime=(root/'tools/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/core/model-runtime.js').as_uri()
    stream=(root/'tools/pi-runtime/node_modules/@earendil-works/pi-ai/dist/utils/event-stream.js').as_uri()
    preload.write_text(f'''import {{ModelRuntime}} from {json.dumps(runtime)};
import {{AssistantMessageEventStream}} from {json.dumps(stream)};
globalThis.fetch=()=>{{throw Error('No external network in bridge fixture')}};
let calls=0;
ModelRuntime.prototype.streamSimple=async function(model,context,options){{
  if(options.maxRetries!==0)throw Error('Nested provider retries must be disabled');
  const failed=++calls===1,s=new AssistantMessageEventStream();
  const m={{role:'assistant',content:failed?[]:[{{type:'text',text:'fixture recovered'}}],api:model.api,provider:model.provider,model:model.id,
    timestamp:Date.now(),usage:{{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}}},
    stopReason:failed?'error':'stop',...(failed?{{errorMessage:{json.dumps(failure)}}}:{{}})}};
  s.push(failed?{{type:'error',reason:'error',error:m}}:{{type:'done',reason:'stop',message:m}});s.end();return s;
}};
''',encoding='utf-8')
    driver=tmp_path/'driver.cjs'
    driver.write_text('''const {spawn}=require('node:child_process');
const child=spawn(process.execPath,['--import',process.argv[2],process.argv[3],process.argv[4]],{stdio:['pipe','pipe','pipe'],windowsHide:true});
const timer=setTimeout(()=>{child.kill();process.exitCode=1;},25000);
let buffer='';child.stdout.on('data',chunk=>{buffer+=chunk;let at;while((at=buffer.indexOf('\\n'))>=0){
const line=buffer.slice(0,at);buffer=buffer.slice(at+1);if(!line.trim())continue;process.stdout.write(line+'\\n');const e=JSON.parse(line);
if(e.type==='ready')child.stdin.write(JSON.stringify({type:'prompt',id:'fixture-prompt',run_id:'fixture-run',message:'Read-only fixture.'})+'\\n');
if(e.type==='settled')child.stdin.write(JSON.stringify({type:'close'})+'\\n');
}});child.stderr.pipe(process.stderr);child.on('exit',code=>{clearTimeout(timer);process.exitCode=code;});
''',encoding='utf-8')
    env={**os.environ,'CODEX_HOME':str(tmp_path),'OPENAI_API_KEY':'','CODEX_API_KEY':''}
    result=subprocess.run(['node',str(driver),preload.as_uri(),str(root/'tools/pi-runtime/bridge.mjs'),str(config)],
        cwd=root,env=env,capture_output=True,timeout=35,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
    events=[json.loads(line) for line in result.stdout.decode('utf-8').splitlines() if line.strip()]
    assert next(e for e in events if e['type']=='ready')['retry']['maxRetries']==5
    # Running hosts latch only assistant_message/error as terminal failure.
    terminal=[e for e in events if e['type']=='assistant_message' and e.get('stopReason')=='error']
    assert any(e['type']=='assistant_attempt_error' for e in events)
    assert any(e['type']=='settled' for e in events)
    if failure=='fetch failed':
        assert not terminal
        assert any(e['type']=='auto_retry_end' and e['data']['success'] for e in events)
        assert any(e['type']=='assistant_message' and e['text']=='fixture recovered' for e in events)
    else:
        assert len(terminal)==1 and terminal[0]['error']=='401 Unauthorized'
        assert not any(e['type']=='auto_retry_start' for e in events)
