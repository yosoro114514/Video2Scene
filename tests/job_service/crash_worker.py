import asyncio
import json
import os
from pathlib import Path
import sys
from threading import Event

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from video2scene.job_service import SQLiteJobService
from video2scene.scene_state import SQLiteSceneRepository

root=Path(sys.argv[1]);state=json.loads((root/"job_crash.json").read_text(encoding="utf8"))

class Backend:
    async def refine(self,*args):
        os._exit(79)

repo=SQLiteSceneRepository(root/"repository")
service=SQLiteJobService(root/"jobs",repo,lambda key,ctx:state["evidence"],Backend(),autostart=False)
asyncio.run(service.submit(state["request"],service._ctx(state["context"],Event())))
service.run_next()
