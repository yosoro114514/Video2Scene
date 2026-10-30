"""Frozen contract examples. These never claim model, render or quality success."""
import copy
import json
from pathlib import Path


def examples():
    return json.loads(Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))


def example(name):
    return copy.deepcopy(next(item["value"] for item in examples()["positive"] if item["name"] == name))


def mock_handler(tool):
    async def handle(request, context):
        response = example("response_" + tool)
        return response["result"]
    return handle

