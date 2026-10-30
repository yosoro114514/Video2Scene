"""Single validated dispatcher for host, HTTP and CLI callers."""
from __future__ import annotations
import inspect
from .contracts_v1 import CONTRACT_VERSION, ContractError, validate, schema

TOOLS = tuple(x["properties"]["tool"]["const"] for x in schema()["$defs"]["ToolRequest"]["oneOf"])
HTTP_ERRORS = {"INVALID_ARGUMENT": 400, "UNSUPPORTED_CONTRACT": 400, "NOT_FOUND": 404,
               "FORBIDDEN_SCOPE": 403, "LOCKED_FIELD": 403, "STALE_REVISION": 409,
               "IDEMPOTENCY_CONFLICT": 409, "CANCELLED": 409, "EVENT_CURSOR_EXPIRED": 409,
               "TOOL_UNAVAILABLE": 503, "RESOURCE_EXHAUSTED": 503, "TIMEOUT": 504,
               "INTERNAL_ERROR": 500}


def response_status(response):
    if response["status"] == "accepted": return 202
    if response["status"] == "ok": return 200
    return HTTP_ERRORS.get(response["error"]["code"], 422)


class Dispatcher:
    def __init__(self):
        self.handlers = {}

    def register(self, tool, handler):
        if tool not in TOOLS:
            raise ValueError(f"Unknown tool: {tool}")
        if tool in self.handlers:
            raise ValueError(f"Tool already registered: {tool}")
        self.handlers[tool] = handler

    async def execute(self, request, context):
        # Unknown tool names cannot be represented in the frozen response enum.
        # Transport adapters turn this pre-envelope error into an ApiError.
        if not isinstance(request, dict) or request.get("tool") not in TOOLS:
            raise ContractError("INVALID_ARGUMENT", "Unknown tool")
        envelope = {"contract_version": CONTRACT_VERSION,
                    "request_id": request.get("request_id", "invalid"),
                    "scene_id": request.get("scene_id", "invalid"), "tool": request["tool"],
                    "status": "ok", "result": None, "error": None, "warnings": []}
        try:
            if request.get("contract_version") != CONTRACT_VERSION:
                raise ContractError("UNSUPPORTED_CONTRACT", "Unsupported contract version")
            validate(request, "ToolRequest")
            context.check_cancelled()
            handler = self.handlers.get(request["tool"])
            if handler is None:
                raise ContractError("TOOL_UNAVAILABLE", "Tool has no registered implementation", retryable=True)
            result = handler(request, context)
            if inspect.isawaitable(result): result = await result
            envelope["result"] = result
            if isinstance(result, dict) and set(result) == {"job_id", "state"}:
                envelope["status"] = "accepted"
            validate(envelope, "ToolResponse")
        except ContractError as exc:
            envelope.update(status="error", result=None, error=exc.as_dict())
        except Exception:
            envelope.update(status="error", result=None,
                            error=ContractError("INTERNAL_ERROR", "Tool failed; inspect server logs").as_dict())
        return envelope

