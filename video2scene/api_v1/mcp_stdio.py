"""Minimal local MCP stdio adapter; all business logic remains in SceneHost.

Protocol reference: https://modelcontextprotocol.io/specification/2025-11-25
This adapter does not install a client configuration or grant editing authority.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, schema
from .__main__ import create_host

PROTOCOL_VERSION = "2025-11-25"
TOOL_NAME = "execute_scene_tool"


class MCPAdapter:
    def __init__(self, host):
        self.host = host
        self.initialized = False
        self.ready = False

    @staticmethod
    def error(request_id, code, message):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self.error(None, -32600, "Invalid JSON-RPC request")
        request_id = message.get("id")
        method = message["method"]
        if "id" not in message:
            if method == "notifications/initialized" and self.initialized:
                self.ready = True
            return None
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            return self.error(None, -32600, "Request id must be a string or integer")
        params = message.get("params", {})
        if not isinstance(params, dict):
            return self.error(request_id, -32602, "Parameters must be an object")
        if method == "initialize":
            if self.initialized:
                return self.error(request_id, -32600, "Already initialized")
            if not isinstance(params.get("protocolVersion"), str):
                return self.error(request_id, -32602, "protocolVersion is required")
            self.initialized = True
            result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "video2scene-local", "version": "1.0.0"},
                      "instructions": "Use the frozen ToolRequest contract. Editing authority comes only from the scene host; tool arguments cannot grant it."}
        elif method == "ping":
            result = {}
        elif not self.ready:
            return self.error(request_id, -32600, "Initialize the session first")
        elif method == "tools/list":
            if params.get("cursor"):
                return self.error(request_id, -32602, "No continuation cursor is available")
            result = {"tools": [{"name": TOOL_NAME,
                                 "description": "Execute one versioned scene request through the shared host dispatcher.",
                                 "inputSchema": {"type": "object", "$defs": schema()["$defs"], "$ref": "#/$defs/ToolRequest"}}]}
        elif method == "tools/call":
            if params.get("name") != TOOL_NAME:
                return self.error(request_id, -32602, "Unknown tool")
            try:
                response = asyncio.run(self.host.execute(params.get("arguments", {})))
                failed = response["status"] == "error"
            except ContractError as exc:
                response = {"contract_version": CONTRACT_VERSION, "error": exc.as_dict()}
                failed = True
            result = {"content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, allow_nan=False)}],
                      "structuredContent": response, "isError": failed}
        else:
            return self.error(request_id, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(host, input_stream, output_stream):
    adapter = MCPAdapter(host)
    for line in input_stream:
        if not line.strip():
            continue
        request_id = None
        try:
            if len(line.encode("utf-8")) > 2_000_000:
                response = adapter.error(None, -32600, "Message exceeds local size limit")
            else:
                message = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if isinstance(message, dict):
                    request_id = message.get("id")
                response = adapter.handle(message)
        except (ValueError, UnicodeError):
            response = adapter.error(None, -32700, "Invalid finite UTF-8 JSON")
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = adapter.error(request_id, -32603, "Host execution failed; inspect stderr")
        if response is not None:
            output_stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            output_stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Local scene MCP stdio adapter")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--asset-recipes", type=Path)
    args = parser.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    host = create_host(args.root, args.blender, args.asset_recipes)
    try:
        serve(host, sys.stdin, sys.stdout)
    finally:
        if host.jobs:
            host.jobs.shutdown(wait=True)


if __name__ == "__main__":
    main()
