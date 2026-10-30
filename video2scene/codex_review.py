"""Audited image review through the user's ChatGPT-authenticated Codex CLI.

No tokens are read by this module. Codex owns authentication and refresh. The
reviewer returns data only; its response is never executed as Python or shell.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .codex_capacity import capacity_rejection


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


STR={'type':'string'}
NUM={'type':'number'}
INT={'type':'integer'}
FRAMES={'type':'array','items':INT}
SCHEMA=obj({
    'source_video_sha256':STR,
    'status':{'type':'string','enum':['accept','revise','insufficient_evidence']},
    'summary':STR,
    'issues':{'type':'array','items':obj({'asset_id':STR,'priority':{'type':'string','enum':['P0','P1','P2']},'frames':FRAMES,'evidence':STR,'uncertainty':STR})},
    'edits':{'type':'array','items':obj({'asset_id':STR,'operation':{'type':'string','enum':['update','add','remove']},'patch_json':STR,'frames':FRAMES,'reason':STR})},
    'measurements':{'type':'array','items':obj({
        'asset_id':STR,'part':{'type':'string','enum':['screen','chair_back','desk_top','panel']},
        'use_texture':{'type':'boolean'},'reason':STR,
        'views':{'type':'array','items':obj({'frame':INT,'quad':{'type':'array','items':{'type':'array','items':NUM}},'visibility':{'type':'string','enum':['clear','partly_occluded']}})}
    })},
    'remaining_uncertainties':{'type':'array','items':STR}
})


def codex_command():
    # Avoid cmd.exe / .cmd quoting: invoke the installed package through Node.
    if os.name=='nt':
        installed=list((Path(os.environ['LOCALAPPDATA'])/'OpenAI/Codex/bin').glob('*/codex.exe'))
        if installed:return [str(max(installed,key=lambda p:p.stat().st_mtime))]
        path=Path(os.environ['APPDATA'])/'npm/node_modules/@openai/codex/bin/codex.js'
        node=shutil.which('node')
        if path.is_file() and node:return [node,str(path)]
    binary=shutil.which('codex')
    if binary and Path(binary).suffix.lower() not in ['.cmd','.bat','.ps1']:return [binary]
    raise RuntimeError('Codex CLI not found; install/sign in using the supported Codex CLI')


def _execute_cli(argv,prompt,output,env,timeout,max_tool_calls):
    with (output/'events.jsonl').open('w',encoding='utf8') as stdout,(output/'stderr.log').open('w',encoding='utf8') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env,text=True,encoding='utf8')
        process.stdin.write(prompt);process.stdin.close();started=time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic()-started>timeout:raise TimeoutError(f'Codex invocation exceeded {timeout}s')
                calls=set()
                for line in (output/'events.jsonl').read_text(encoding='utf8',errors='replace').splitlines():
                    try:event=json.loads(line)
                    except json.JSONDecodeError:continue
                    item=event.get('item',{})
                    if item.get('type') in ['command_execution','mcp_tool_call','web_search','file_change']:calls.add(item.get('id'))
                if len(calls)>max_tool_calls:raise RuntimeError('Codex tool-call budget exceeded')
                time.sleep(.5)
        except Exception:
            if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:process.kill()
            process.wait();raise
    return process.returncode


def invoke(prompt,images,output,model='gpt-6-astra',effort='max',timeout=1200,schema=None,sandbox='read-only',max_tool_calls=48):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    images=[Path(p).resolve() for p in images]
    if not images or any(not p.is_file() for p in images):raise ValueError('Review requires actual readable evidence images')
    env=os.environ.copy()
    for key in ['OPENAI_API_KEY','CODEX_API_KEY']:env.pop(key,None)
    profile={"read-only":":read-only","workspace-write":":workspace"}[sandbox]
    env['CODEX_PERMISSION_PROFILE']=profile
    command=codex_command()
    auth=subprocess.run(command+['login','status'],env=env,capture_output=True,text=True,encoding='utf8',errors='replace',check=True)
    if 'chatgpt' not in (auth.stdout+auth.stderr).lower():raise RuntimeError('ChatGPT subscription login required; refusing API-key fallback')
    version=subprocess.check_output(command+['--version'],env=env,text=True).strip()
    (output/'schema.json').write_text(json.dumps(schema or SCHEMA,indent=2),encoding='utf8')
    (output/'prompt.txt').write_text(prompt,encoding='utf8')
    request=dict(backend='codex-cli',authentication='ChatGPT subscription',requested_model=model,reasoning_effort=effort,cli_version=version,
                 prompt_sha256=digest(output/'prompt.txt'),schema_sha256=digest(output/'schema.json'),
                 images=[dict(path=str(p),sha256=digest(p)) for p in images],sandbox=sandbox,max_tool_calls=max_tool_calls,started=time.time(),status='running')
    def save(): (output/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf8')
    save()
    argv=command+['exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-c',f'default_permissions="{profile}"',
                  '--model',model,'-c',f'model_reasoning_effort="{effort}"','-c','approval_policy="never"','--color','never','--json',
                  '--output-schema',str(output/'schema.json'),'-o',str(output/'response.json'),'-C',str(output)]
    if os.name=='nt':argv+=['-c','windows.sandbox="elevated"']
    for image in images:argv+=['--image',str(image)]
    # --image accepts multiple values in current Codex. Explicitly end options
    # so the stdin prompt marker cannot be consumed as a fourth image argument.
    argv+=['--','-']
    (output/'argv.json').write_text(json.dumps(argv,ensure_ascii=False,indent=2),encoding='utf8')
    request['argv_sha256']=digest(output/'argv.json');save()
    try:
        deadline=time.monotonic()+timeout
        for attempt in range(3):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Codex invocation budget exhausted')
            returncode=_execute_cli(argv,prompt,output,env,remaining,max_tool_calls)
            if not returncode:break
            rejection=capacity_rejection(output)
            if rejection:(output/'provider_error.json').write_text(json.dumps(rejection,indent=2),encoding='utf-8')
            if not rejection or attempt==2:
                reason=rejection['message'] if rejection else 'see events.jsonl and stderr.log'
                raise RuntimeError(f'Codex review failed (exit {returncode}): {reason}; {output}')
            delay=(15,30)[attempt]
            if deadline-time.monotonic()<delay+5:raise TimeoutError('Capacity cooldown exceeds remaining invocation budget')
            archive=output/'preexecution_retries'/f'attempt_{attempt+1:02d}'
            archive.mkdir(parents=True,exist_ok=False)
            for name in ('events.jsonl','stderr.log','provider_error.json'):
                shutil.copyfile(output/name,archive/name)
            request.setdefault('preexecution_capacity_retries',[]).append({'attempt':attempt+1,'delay_s':delay,
                'events_sha256':digest(archive/'events.jsonl'),'archive':str(archive),'reason':rejection['message']})
            save();time.sleep(delay)
        events=[json.loads(line) for line in (output/'events.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
        completed=[e for e in events if e.get('type')=='turn.completed']
        if not completed or any(e.get('type') in ['turn.failed','error'] for e in events):raise RuntimeError('Codex did not complete the review turn')
        response=json.loads((output/'response.json').read_text(encoding='utf8'))
        request.update(status='complete',seconds=time.time()-request['started'],response_sha256=digest(output/'response.json'),
                       usage=completed[-1].get('usage'),thread_ids=[e.get('thread_id') for e in events if e.get('type')=='thread.started'],
                       model_resolution='Explicit CLI --model; no fallback. CLI JSONL does not independently report a server model identifier.')
        save();return response
    except Exception as error:
        request.update(status='failed',seconds=time.time()-request['started'],error=str(error));save();raise
