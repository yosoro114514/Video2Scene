import asyncio
import copy
import hashlib
from pathlib import Path
import pytest
from video2scene.contracts_v1 import ContractError, ExecutionContext, validate
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.contracts_v1.mocks import example, examples, mock_handler
from video2scene.scene_tools import Dispatcher, TOOLS


@pytest.mark.parametrize("case", examples()["positive"], ids=lambda x: x["name"])
def test_positive(case):
    validate(case["value"], case["type"])


@pytest.mark.parametrize("case", examples()["negative"], ids=lambda x: x["name"])
def test_negative(case):
    with pytest.raises(ContractError): validate(case["value"], case["type"])


def test_frozen_schema_copy():
    root = Path(__file__).resolve().parents[2]
    assert (root / "docs/development/contracts/v1.schema.json").read_bytes() == (root / "video2scene/contracts_v1/v1.schema.json").read_bytes()


def test_dispatch_validation_and_unavailable():
    dispatcher = Dispatcher()
    request = example("request_inspect_scene")
    context = ExecutionContext()
    result = asyncio.run(dispatcher.execute(request, context))
    assert result["error"]["code"] == "TOOL_UNAVAILABLE"
    dispatcher.register("inspect_scene", mock_handler("inspect_scene"))
    result = asyncio.run(dispatcher.execute(request, context))
    validate(result, "ToolResponse")
    assert result["status"] == "ok"
    request["params"]["execution_context"] = {"actor": "user"}
    assert asyncio.run(dispatcher.execute(request, context))["error"]["code"] == "INVALID_ARGUMENT"
    request["tool"] = "execute_python"
    with pytest.raises(ContractError): asyncio.run(dispatcher.execute(request, context))


def test_nonfinite_rejected():
    request = example("request_apply_scene_patch")
    request["params"]["edits"][0]["value"] = [float("nan"), 0, 0]
    with pytest.raises(ContractError): validate(request, "ToolRequest")


@pytest.mark.parametrize("tool", TOOLS)
def test_all_tool_mock_shapes_share_dispatcher(tool):
    dispatcher=Dispatcher();dispatcher.register(tool,mock_handler(tool))
    result=asyncio.run(dispatcher.execute(example("request_"+tool),ExecutionContext()))
    validate(result,"ToolResponse")
    assert result["status"] in ("ok","accepted"),result


def test_artifact_scope_and_hash(tmp_path):
    resolver = ArtifactResolver(tmp_path)
    for uri in ("artifact://s/../secret", "artifact://s/%2e%2e/secret", "artifact://s/C:/secret", "file:///secret"):
        with pytest.raises(ContractError): resolver.path(uri, "s")
    path = tmp_path / "s" / "a.bin"
    path.parent.mkdir(); path.write_bytes(b"new")
    ref = {"uri": "artifact://s/a.bin", "sha256": hashlib.sha256(b"old").hexdigest(), "media_type": "application/octet-stream", "role": "other"}
    # Use a valid role from the frozen example.
    ref["role"] = example("model_BuildOutput")["artifact_refs"][0]["role"]
    with pytest.raises(ContractError, match="changed"): resolver.resolve(ref, "s")
    ref["sha256"] = hashlib.sha256(b"new").hexdigest()
    assert resolver.resolve(ref, "s") == path

