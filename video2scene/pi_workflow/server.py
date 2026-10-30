"""Local browser workbench and authenticated Pi tool endpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from video2scene.api_v1.server import make_server as editor_server
from video2scene.contracts_v1 import ContractError
from .pi_runtime import ROOT, save
from .service import WorkflowService, identifier
from .process_owner import owner_record


def make_server(service, port=8794):
    server=editor_server(service.host,port=port)
    parent=server.RequestHandlerClass
    static=Path(__file__).with_name('web').resolve()
    class Handler(parent):
        def failure(self,error):
            if isinstance(error,ContractError):self.error(error)
            elif isinstance(error,KeyError):self.error(ContractError('NOT_FOUND',str(error)))
            elif isinstance(error,(ValueError,TypeError)):self.error(ContractError('INVALID_ARGUMENT',str(error)))
            else:
                import traceback
                traceback.print_exc();self.error(ContractError('INTERNAL_ERROR','Workflow failed; inspect local host logs'))

        def do_GET(self):
            url=urlsplit(self.path);path=unquote(url.path);query=parse_qs(url.query)
            if not path.startswith('/workflow'):return super().do_GET()
            try:
                if path=='/workflow/api/projects':self.json({'projects':service.projects()})
                elif path=='/workflow/api/capabilities':self.json({'rolling_refinement':True,'default_object_concurrency':3,'max_object_concurrency':4,'resources':service.resources.snapshot()})
                elif (match:=re.fullmatch(r'/workflow/api/projects/([\w.-]+)(/events)?',path)):
                    pid=match[1]
                    self.json(service.events(pid,int(query.get('after',['0'])[0])) if match[2] else service.status(pid))
                elif path=='/workflow/artifacts':
                    run=service.run_root(query.get('project_id',[''])[0]);resource=(run/query.get('path',[''])[0]).resolve()
                    if not resource.is_relative_to(run):raise ContractError('FORBIDDEN_SCOPE','Artifact outside project')
                    if resource.suffix.lower() not in {'.png','.jpg','.jpeg','.glb','.blend','.mp4'}:
                        raise ContractError('FORBIDDEN_SCOPE','Only scene and visual artifacts are served')
                    self.file(resource)
                elif path in ('/workflow','/workflow/'):self.file(static/'index.html')
                elif path=='/workflow/guide':self.file(static/'guide.html')
                elif path in ('/workflow/app.mjs','/workflow/style.css'):self.file(static/path.rsplit('/',1)[1])
                else:raise ContractError('NOT_FOUND','Workflow route not found')
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as error:self.failure(error)

        def upload(self,query):
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):raise ContractError('FORBIDDEN_SCOPE','Cross-origin writes are not allowed')
            name=Path(query.get('name',['video.mp4'])[0]).name;suffix=Path(name).suffix.lower()
            if suffix not in {'.mp4','.mov','.avi','.mkv','.webm','.m4v'}:raise ValueError('Unsupported video format')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=2_000_000_000:raise ValueError('Video must be between 1 byte and 2 GB')
            folder=service.root/'uploads';folder.mkdir(exist_ok=True)
            target=folder/(identifier('video')+suffix);temp=target.with_suffix(suffix+'.partial')
            with temp.open('xb') as stream:
                remaining=size
                while remaining:
                    chunk=self.rfile.read(min(1024*1024,remaining))
                    if not chunk:raise ValueError('Incomplete video upload')
                    stream.write(chunk);remaining-=len(chunk)
            temp.replace(target);self.json({'source_path':str(target),'name':name,'bytes':size},201)

        def do_POST(self):
            url=urlsplit(self.path);path=unquote(url.path)
            if not path.startswith('/workflow/'):return super().do_POST()
            try:
                if path=='/workflow/api/upload':return self.upload(parse_qs(url.query))
                if path=='/workflow/internal/tool':
                    if not secrets.compare_digest(self.headers.get('X-Workflow-Token',''),service.token):
                        self.close_connection=True;raise ContractError('FORBIDDEN_SCOPE','Pi tool endpoint requires its process token')
                    return self.json(service.tool(self.body()))
                body=self.body()
                if not isinstance(body,dict):raise ValueError('Expected a JSON object')
                if path=='/workflow/api/projects':self.json(service.create_project(body),201)
                elif (match:=re.fullmatch(r'/workflow/api/projects/([\w.-]+)/(messages|control)',path)):
                    self.json(service.message(match[1],body) if match[2]=='messages' else service.control(match[1],body))
                else:raise ContractError('NOT_FOUND','Workflow route not found')
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as error:self.failure(error)
    server.RequestHandlerClass=Handler
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT/'outputs/pi_workflow_v1'));parser.add_argument('--port',type=int,default=8794)
    parser.add_argument('--model-concurrency',type=int,choices=range(1,5),default=3)
    parser.add_argument('--blender-concurrency',type=int,choices=(1,2),default=1)
    args=parser.parse_args();service=WorkflowService(args.root,args.port,resource_limits={'model':args.model_concurrency,'blender':args.blender_concurrency});server=make_server(service,args.port)
    import os
    save(service.root/'service.json',{**owner_record(),'url':f'http://127.0.0.1:{args.port}/workflow/','root':str(service.root),'provider':'codex_subscription','model':service.settings.model})
    print(json.dumps({'ready':True,'url':f'http://127.0.0.1:{args.port}/workflow/'}),flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:server.server_close();service.close()


if __name__=='__main__':main()
