"""Data-only language plans; host remains responsible for authorization/commit."""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
from pathlib import Path
import re
from uuid import uuid4

from video2scene.codex_review import invoke
from video2scene.contracts_v1 import ContractError, canonical_hash, schema, validate


def _check_base(command, view):
    if view["head"] != {"revision": command["base_revision"], "head_seq": command["base_head_seq"]}:
        raise ContractError("STALE_REVISION", "Language plan must use the current scene head")


def _clarify(question):
    return {"action": "clarify", "question": question}


class DeterministicPlanner:
    """Narrow reliable commands; unsupported language is explicit clarification."""
    async def __call__(self, command, view, context):
        _check_base(command, view)
        text = command["text"].strip().lower()
        selected = command["selected_ids"]
        objects = {obj["instance_id"]: obj for obj in view["snapshot"]["objects"]}
        if any(word in text for word in ("不要", "别改", "do not", "don't")):
            return _clarify("这条指令包含禁止修改的描述，请明确需要执行的修改。")
        if len(selected) != 1 or selected[0] not in objects:
            return _clarify("请明确选中一个需要修改的对象。")
        target = selected[0]
        if text in ("精修", "精修选中对象", "细化选中对象", "refine", "refine selected object", "精修这个杯子",
                    "精修选中对象的几何与材质，保持根变换、支撑和局部边界。"):
            return {"action": "refine", "target_id": target}
        matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*[%％]", text))
        if len(matches) > 1:
            return _clarify("请分别提交每个字段的明确目标值。")
        match = matches[0] if matches else None
        height = any(word in text for word in ("高", "height", "taller"))
        increasing = any(word in text for word in ("增", "再高", "更高", "height by", "taller", "increase"))
        if match and height and increasing:
            percent = float(match.group(1))
            if not 0 < percent <= 500:
                return _clarify("请给出大于 0 且不超过 500% 的高度增量。")
            scale = copy.deepcopy(objects[target]["transform"]["scale"])
            scale[1] *= 1 + percent / 100  # object_y_up is the canonical semantic height axis.
            return {"action": "edit", "edits": [{"instance_id": target, "field": "transform.scale", "value": scale}]}
        return _clarify("请给出明确的目标字段和数值，例如“高度增加 5%”；位置描述也需要方向和距离。")


class CodexPlanner:
    """Uses the existing ChatGPT-authenticated CLI with actual image attachments.

The call ledger prevents automatic replay of uncertain external calls. No API-key
fallback, remote tools or executable model output are used.
"""
    def __init__(self, root, evidence_service, artifact_resolver, *, model="gpt-6-astra", effort="high",
                 timeout_s=120, invoke_fn=invoke):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_service, self.artifact_resolver = evidence_service, artifact_resolver
        self.model, self.effort, self.timeout_s, self.invoke_fn = model, effort, timeout_s, invoke_fn

    @staticmethod
    def plan_schema():
        return {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["edit", "clarify", "refine"]},
            "edits": {"type": "array", "items": {"type": "object", "properties": {
                "instance_id": {"type": "string"},
                "field": {"type": "string", "enum": ["category", "transform.translation_m", "transform.rotation_xyzw", "transform.scale", "support_id"]},
                "value_json": {"type": "string"}},
                "required": ["instance_id", "field", "value_json"], "additionalProperties": False}},
            "target_id": {"type": "string"}, "question": {"type": "string"},
            "observation": {"type": "string"}},
            "required": ["action", "edits", "target_id", "question", "observation"],
            "additionalProperties": False}

    async def __call__(self, command, view, context):
        _check_base(command, view)
        selected = command["selected_ids"]
        if not selected:
            return _clarify("请选择需要修改的对象。")
        key = canonical_hash([command["scene_id"], command["command_id"]])
        output = self.root / ("call_" + key[:32])
        output.mkdir(parents=True, exist_ok=True)
        ledger_path = output / "call_state.json"
        request_hash = canonical_hash(command)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf8"))
            if ledger["request_hash"] != request_hash:
                raise ContractError("IDEMPOTENCY_CONFLICT", "Command ID already has another model request")
            if ledger["status"] == "succeeded":
                return ledger["plan"]
            raise ContractError("RESOURCE_EXHAUSTED", "Prior model call needs reconciliation; no automatic repeated call",
                                {"outcome": ledger["status"], "call_directory": str(output)}, retryable=False)
        request = {"contract_version": "video2scene/1.0", "request_id": "model_evidence_" + uuid4().hex,
            "scene_id": command["scene_id"], "tool": "select_evidence", "params": {"revision": command["base_revision"],
            "target_ids": selected, "purpose": "review", "n": 3}}
        evidence = await self.evidence_service.select(request, context)
        resolver = getattr(self.artifact_resolver, "resolve", self.artifact_resolver)
        images = [resolver(frame["rgb"], scene_id=command["scene_id"]) for frame in evidence["frames"]]
        prompt = (
            "You are a data-only scene edit planner. Return one JSON plan matching the schema. "
            "Do not call tools, run commands, write code or change files. The attached images are visual evidence; "
            "use their actual contents and briefly describe an observable fact in observation. "
            "Only selected_ids may change. Use absolute target values. Scene/object axes are Y-up, so height is scale[1]. "
            "A relative 5% increase multiplies the CURRENT value by 1.05. Preserve other fields. "
            "For ambiguous directions/distances or uncertain targets choose clarify with a question and empty edits. "
            "For refinement choose refine with one selected target_id and empty edits. "
            "For edit supply only supported edits and leave question/target_id empty. "
            "Encode each absolute edit value as valid JSON in value_json, such as [1,1.05,1] for transform.scale. "
            "Do not add/delete/merge objects or invent hidden room boundaries.\n"
            + json.dumps({"command": command, "scene_view": view,
                "image_frames": [{"source_frame_id": f["source_frame_id"], "rgb": f["rgb"], "K": f["K"], "c2w": f["c2w"]} for f in evidence["frames"]]}, ensure_ascii=False)
        )
        ledger = {"request_hash": request_hash, "status": "running", "model": self.model, "effort": self.effort,
                  "maximum_calls": 1, "native_tool_budget": 0, "evidence_id": evidence["evidence_id"], "evidence_hash": evidence["evidence_hash"]}
        try:
            with ledger_path.open("x", encoding="utf8") as file:
                json.dump(ledger, file, ensure_ascii=False, indent=2)
        except FileExistsError:
            raise ContractError("RESOURCE_EXHAUSTED", "Identical model command is already in progress")
        try:
            result = await asyncio.to_thread(self.invoke_fn, prompt, images, output / "codex", model=self.model,
                effort=self.effort, timeout=self.timeout_s, schema=self.plan_schema(), sandbox="read-only", max_tool_calls=0)
            from jsonschema import Draft202012Validator
            Draft202012Validator(self.plan_schema()).validate(result)
            action = result["action"]
            if action == "edit":
                if not result["edits"]:
                    raise ContractError("NO_CHANGE", "Model returned no edits")
                edits = [{"instance_id": edit["instance_id"], "field": edit["field"], "value": json.loads(edit["value_json"])} for edit in result["edits"]]
                for edit in edits:
                    validate(edit, "Edit")
                    if edit["instance_id"] not in selected:
                        raise ContractError("FORBIDDEN_SCOPE", "Model selected an unauthorized object")
                plan = {"action": "edit", "edits": edits}
            elif action == "refine":
                if result["target_id"] not in selected:
                    raise ContractError("FORBIDDEN_SCOPE", "Model selected an unauthorized refinement target")
                plan = {"action": "refine", "target_id": result["target_id"]}
            else:
                plan = _clarify(result["question"] or "请明确要修改的对象、字段和目标值。")
            ledger.update(status="succeeded", plan=plan, observation=result["observation"])
            return plan
        except ContractError:
            ledger.update(status="failed", outcome="uncertain_or_rejected", automatic_retry=False)
            raise
        except Exception as exc:
            ledger.update(status="failed", outcome="uncertain_or_rejected", automatic_retry=False,
                          error={"type": type(exc).__name__, "message": str(exc)})
            code = "TIMEOUT" if isinstance(exc, TimeoutError) else "TOOL_UNAVAILABLE"
            raise ContractError(code, "Codex planning failed; inspect the preserved call record",
                                {"call_directory": str(output), "outcome": "uncertain", "automatic_retry": False}) from exc
        finally:
            temporary = ledger_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf8")
            temporary.replace(ledger_path)


class HybridPlanner:
    def __init__(self, codex=None):
        self.deterministic = DeterministicPlanner()
        self.codex = codex

    async def __call__(self, command, view, context):
        plan = await self.deterministic(command, view, context)
        if plan["action"] != "clarify" or self.codex is None:
            return plan
        return await self.codex(command, view, context)
