"""Offline subprocess check of image arguments and stdin transport, never Codex."""
import json
import sys

from video2scene import codex_review


def test_image_arguments_end_before_stdin_marker(tmp_path, monkeypatch):
    fake_cli=tmp_path/'local_transport_fixture.py'
    fake_cli.write_text('''import argparse,json,sys
from pathlib import Path
if sys.argv[1:]==['login','status']:
 print('Logged in using ChatGPT');sys.exit()
if sys.argv[1:]==['--version']:
 print('offline-parser-fixture');sys.exit()
p=argparse.ArgumentParser()
p.add_argument('command');p.add_argument('prompt',nargs='?')
for flag in ['--ignore-user-config','--ephemeral','--skip-git-repo-check','--json']:p.add_argument(flag,action='store_true')
for flag in ['--model','--color','--output-schema','-o','-C']:p.add_argument(flag)
p.add_argument('-c',action='append');p.add_argument('--image',action='append',nargs='+')
a=p.parse_args()
data={'image_arguments':[x for group in a.image for x in group],'prompt_argument':a.prompt,'stdin':sys.stdin.read()}
Path(a.o).write_text(json.dumps(data),encoding='utf8')
print(json.dumps({'type':'thread.started','thread_id':'offline-fixture'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':0,'output_tokens':0}}))
''',encoding='utf8')
    monkeypatch.setattr(codex_review,'codex_command',lambda:[sys.executable,str(fake_cli)])
    images=[]
    for i in range(3):
        path=tmp_path/f'source {i}.png';path.write_bytes(b'OFFLINE_IMAGE_FIXTURE');images.append(path)
    prompt='Offline fixture: preserve stdin exactly.\nNo model call.'
    out=tmp_path/'invocation'
    result=codex_review.invoke(prompt,images,out,timeout=10,max_tool_calls=0)
    assert result=={'image_arguments':[str(p.resolve()) for p in images],'prompt_argument':'-','stdin':prompt}
    recorded=json.loads((out/'argv.json').read_text(encoding='utf8'))
    assert recorded[-2:]==['--','-']
    assert recorded.count('--image')==3
    assert json.loads((out/'request.json').read_text(encoding='utf8'))['argv_sha256']==codex_review.digest(out/'argv.json')
