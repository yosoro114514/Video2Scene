from __future__ import annotations
import asyncio
import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError
from video2scene.scene_tools import HTTP_ERRORS, TOOLS, response_status


def make_server(host, port=8791, bind="127.0.0.1", static_root=None):
    static_root=Path(static_root or Path(__file__).parents[1]/"editor_v1").resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"

        def json(self, payload, status=200):
            body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store"); self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def error(self, error):
            self.json({"contract_version":CONTRACT_VERSION,"error":error.as_dict()}, HTTP_ERRORS.get(error.code,422))

        def body(self):
            origin=self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                raise ContractError("FORBIDDEN_SCOPE","Cross-origin writes are not allowed")
            if self.headers.get_content_type()!="application/json":
                raise ContractError("INVALID_ARGUMENT","Content-Type must be application/json")
            size=int(self.headers.get("Content-Length","0"))
            if size<=0 or size>2_000_000: raise ContractError("INVALID_ARGUMENT","Invalid request body length")
            try:
                return json.loads(self.rfile.read(size),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
            except (ValueError,UnicodeDecodeError) as exc:
                raise ContractError("INVALID_ARGUMENT","Body must be finite UTF-8 JSON") from exc

        def do_POST(self):
            path=""; data=None
            try:
                path=urlsplit(self.path).path; data=self.body()
                if path=="/api/v1/tools/execute":
                    response=asyncio.run(host.execute(data)); self.json(response,response_status(response))
                elif path=="/api/v1/commands": self.json(host.submit_command(data),202)
                else: raise ContractError("NOT_FOUND","Route not found")
            except ContractError as exc:
                if path=="/api/v1/tools/execute" and isinstance(data,dict) and data.get("tool") in TOOLS:
                    def identity(key):
                        value=data.get(key)
                        return value if isinstance(value,str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",value) else "invalid"
                    response={"contract_version":CONTRACT_VERSION,"request_id":identity("request_id"),"scene_id":identity("scene_id"),"tool":data["tool"],"status":"error","result":None,"error":exc.as_dict(),"warnings":[]}
                    self.json(response,response_status(response))
                else:self.error(exc)
            except (ValueError,TypeError): self.error(ContractError("INVALID_ARGUMENT","Invalid request"))
            except Exception:
                import traceback
                traceback.print_exc(); self.error(ContractError("INTERNAL_ERROR","Request failed; inspect host logs"))

        def do_GET(self):
            try:
                url=urlsplit(self.path); path=url.path; query=parse_qs(url.query)
                scene=query.get("scene_id",[""])[0]
                if path.startswith("/api/v1/commands/"):
                    self.json(host.store.command(scene,unquote(path.rsplit("/",1)[1])))
                elif path.startswith("/api/v1/requests/"):
                    self.json(host.store.request(scene,unquote(path.rsplit("/",1)[1])))
                elif re.fullmatch(r"/api/v1/scenes/[^/]+/events",path):
                    scene=unquote(path.split("/")[4])
                    after=int(self.headers.get("Last-Event-ID") or query.get("after_seq",["0"])[0])
                    asyncio.run(host.sync_outbox(scene))
                    host.store.events(scene,after)  # Validate before sending headers.
                    self.send_response(200); self.send_header("Content-Type","text/event-stream")
                    self.send_header("Cache-Control","no-cache"); self.send_header("Connection","close"); self.end_headers()
                    self.close_connection=True
                    deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        for event in host.store.events(scene,after):
                            self.wfile.write(f'id: {event["seq"]}\ndata: {json.dumps(event,ensure_ascii=False)}\n\n'.encode("utf-8")); after=event["seq"]
                        self.wfile.write(b": heartbeat\n\n"); self.wfile.flush(); time.sleep(.5)
                elif path=="/api/v1/artifacts":
                    ref=host.store.artifact(query.get("uri",[""])[0])
                    resource=host.resolver.resolve(ref)
                    self.file(resource)
                elif path.startswith("/api/"): raise ContractError("NOT_FOUND","Route not found")
                else:
                    resource=(static_root/unquote(path).lstrip("/")).resolve()
                    if path=="/":resource=static_root/"index.html"
                    if not resource.is_relative_to(static_root): raise ContractError("FORBIDDEN_SCOPE","Static path escapes root")
                    self.file(resource)
            except (BrokenPipeError,ConnectionResetError): pass
            except ContractError as exc:self.error(exc)
            except (ValueError,TypeError):self.error(ContractError("INVALID_ARGUMENT","Invalid query"))
            except Exception:
                import traceback
                traceback.print_exc(); self.error(ContractError("INTERNAL_ERROR","Request failed; inspect host logs"))

        def file(self, resource):
            if not resource.is_file():raise ContractError("NOT_FOUND","Resource not found")
            size=resource.stat().st_size; start=0; end=size-1; status=200
            requested=self.headers.get("Range")
            if requested:
                match=re.fullmatch(r"bytes=(\d*)-(\d*)",requested)
                if not match or not any(match.groups()):raise ContractError("INVALID_ARGUMENT","Unsupported Range")
                a,b=match.groups()
                if a:start=int(a);end=int(b) if b else end
                else:start=max(0,size-int(b))
                if start>=size or end<start:
                    self.send_response(416);self.send_header("Content-Range",f"bytes */{size}");self.send_header("Content-Length","0");self.end_headers();return
                end=min(end,size-1);status=206
            self.send_response(status)
            media={".mjs":"text/javascript",".glb":"model/gltf-binary"}.get(resource.suffix,mimetypes.guess_type(resource.name)[0] or "application/octet-stream")
            self.send_header("Content-Type",media);self.send_header("X-Content-Type-Options","nosniff");self.send_header("Accept-Ranges","bytes")
            self.send_header("Content-Length",str(end-start+1))
            if status==206:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            with resource.open("rb") as stream:
                stream.seek(start); remaining=end-start+1
                while remaining:
                    data=stream.read(min(65536,remaining))
                    if not data:break
                    self.wfile.write(data);remaining-=len(data)

    server=ThreadingHTTPServer((bind,port),Handler); server.daemon_threads=True
    return server

