"""Intentional hard process exit for scene commit durability tests."""
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video2scene.contracts_v1 import ExecutionContext
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.scene_state import SQLiteSceneRepository

root = Path(sys.argv[1])
state = json.loads((root / "crash.json").read_text(encoding="utf8"))


class Validator:
    async def get_report(self, key, context):
        return state["report"]


def fault(point):
    if point == state["point"]:
        os._exit(77)


repo = SQLiteSceneRepository(root / "repository", validator=Validator(),
    evidence_getter=lambda key, ctx: state["evidence"], criteria_getter=lambda key, ctx: state["criteria"],
    artifact_resolver=ArtifactResolver(root / "artifacts").resolve, fault_hook=fault)
ctx = ExecutionContext(actor="user", decision_ref="decision_1", allowed_object_ids=frozenset({"t001"}),
    allowed_fields=frozenset({"transform.scale"}), storage_root=root)
asyncio.run(repo.commit(state["request"], ctx))
