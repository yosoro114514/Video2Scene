"""Provider event interface; all scene effects still pass the shared dispatcher."""
from uuid import uuid4

from video2scene.contracts_v1 import ContractError, validate


class CodexProvider:
    capabilities = {"vision": True, "tool_calling": True, "streaming": False, "cancel": False, "resume": False}

    def __init__(self, planner, repository, dispatcher, context):
        self.planner, self.repository, self.dispatcher, self.context = planner, repository, dispatcher, context

    async def run(self, session):
        validate(session, "SessionSpec")
        identity = session["session_id"]
        def event(kind, data):
            return validate({"session_id": identity, "type": kind, "data": data}, "AgentEvent")
        try:
            if session["model"] != self.planner.model:
                raise ContractError("TOOL_UNAVAILABLE", "Requested model differs from configured provider")
            if self.planner.timeout_s > session["budget"]["timeout_s"]:
                raise ContractError("TOOL_UNAVAILABLE", "Configured model timeout exceeds this session budget")
            view = await self.repository.inspect({"contract_version": "video2scene/1.0", "request_id": "inspect_" + uuid4().hex,
                "scene_id": session["scene_id"], "tool": "inspect_scene", "params": {"revision": "head"}}, self.context)
            command = {"contract_version": "video2scene/1.0", "command_id": "session_" + identity,
                "session_id": identity, "scene_id": session["scene_id"], "base_revision": session["base_revision"],
                "base_head_seq": session["base_head_seq"], "selected_ids": session["target_ids"],
                "intent": "design_edit", "kind": "language", "text": session["instruction"]}
            plan = await self.planner(command, view, self.context)
            if plan["action"] == "clarify":
                yield event("needs_clarification", {"question": plan["question"]})
                yield event("session_end", {"state": "needs_clarification", "message": plan["question"]})
                return
            if plan["action"] != "edit" or "apply_scene_patch" not in session["allowed_tools"]:
                raise ContractError("TOOL_UNAVAILABLE", "This provider session supports patch planning; use host refinement commands")
            request = {"contract_version": "video2scene/1.0", "request_id": "provider_" + uuid4().hex,
                "scene_id": session["scene_id"], "tool": "apply_scene_patch", "base_revision": session["base_revision"],
                "base_head_seq": session["base_head_seq"], "params": {"intent": "design_edit", "edits": plan["edits"], "decision_ref": self.context.decision_ref}}
            yield event("tool_call", request)
            response = await self.dispatcher.execute(request, self.context)
            yield event("tool_result", response)
            yield event("session_end", {"state": "failed" if response["status"] == "error" else "succeeded",
                "message": "Candidate preparation completed; host validation and commit are separate"})
        except ContractError as exc:
            yield event("text_delta", {"text": exc.code + ": " + exc.message})
            yield event("session_end", {"state": "failed", "message": exc.message})


class PiProvider:
    """Legacy SessionSpec interface; the live Pi SDK host is pi_workflow.server."""
    capabilities = {"vision": False, "tool_calling": False, "streaming": False, "cancel": False, "resume": False}

    async def run(self, session):
        raise ContractError("TOOL_UNAVAILABLE", "Use video2scene.pi_workflow.server for the installed Pi SDK workflow; this legacy SessionSpec adapter is not configured")
        yield  # Preserve the async-iterator provider signature.
