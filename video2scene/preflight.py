"""Read-only installation checks; never logs in, loads weights or calls a model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import ROOT, blender_path, load_settings


def check(mode="demo", *, probe_gpu=False):
    settings = load_settings()
    checks = []

    def add(name, ok, detail, required=True):
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python", (3, 11) <= sys.version_info[:2] < (3, 14), sys.version.split()[0])
    for name in ("numpy", "opencv-python", "trimesh", "pillow", "jsonschema", "scipy"):
        try:
            add(name, True, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            add(name, False, "Install this source checkout: python -m pip install -e .")
    blender = blender_path(settings.blender)
    add("blender", blender.is_file(), str(blender) if blender.is_file() else "Set V2S_BLENDER or blender in video2scene.local.toml")
    for relative in ("scripts/asset_backend/blender_worker.py", "video2scene/editor_v1/index.html",
                     "video2scene/editor_v1/vendor/three/build/three.module.js"):
        add(relative, (ROOT / relative).is_file(), "Required source-checkout resource")
    if mode == "workbench":
        node = shutil.which("node")
        add("node", node is not None, node or "Install Node.js >=22.19.0")
        if node:
            try:
                version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
                add("node_version", tuple(map(int, version.lstrip("v").split("."))) >= (22, 19, 0), version)
            except (subprocess.SubprocessError, ValueError) as exc:
                add("node_version", False, type(exc).__name__)
        for relative in ("tools/pi-runtime/node_modules/@earendil-works/pi-coding-agent/package.json",
                         "tools/pi-runtime/bridge.mjs", "video2scene/pi_workflow/web/index.html",
                         "scripts/run_vipe_workflow.sh"):
            add(relative, (ROOT / relative).is_file(), "Run npm ci --prefix tools/pi-runtime --ignore-scripts for Node dependencies")
        from .codex_review import codex_command
        try:
            command = codex_command()
            add("codex_cli", True, command[0])
        except (RuntimeError, KeyError) as exc:
            add("codex_cli", False, str(exc))
        auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        add("codex_file_login", auth.is_file(), "Pi currently requires a file-backed ChatGPT login; contents/expiry/account access are not inspected")
        add("sam_checkpoint_configured", bool(settings.sam_checkpoint), "Set an absolute Linux sam_checkpoint path; the worker verifies its SHA-256")
        if os.name == "nt":
            add("wsl", shutil.which("wsl") is not None, "Required for GPU tools on Windows")
        if probe_gpu:
            # Explicit opt-in: probe the selected worker interpreter, with no model imports.
            script = "import importlib.util,json,pathlib,sys; print(json.dumps({'sam_importable':importlib.util.find_spec('sam3') is not None,'checkpoint_exists':pathlib.Path(sys.argv[1]).is_file()}))"
            command = (["wsl", "-d", settings.wsl_distro, "--exec"] if os.name == "nt" else []) + [settings.sam_python, "-c", script, settings.sam_checkpoint]
            try:
                # WSL can emit localized Windows diagnostics on stderr. Keep bytes
                # so those diagnostics cannot break an otherwise valid JSON probe.
                result = subprocess.run(command, capture_output=True, timeout=30,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                value = json.loads(result.stdout.decode("utf-8"))
                add("sam_worker_probe", result.returncode == 0 and all(value.values()), json.dumps(value))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                add("sam_worker_probe", False, type(exc).__name__)
    return {"mode": mode, "ok": all(row["ok"] for row in checks if row["required"]),
            "checks": checks, "model_calls": 0,
            "note": "Checks local prerequisites only. Does not establish model access, GPU capacity, token expiry or end-to-end success."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "workbench"), default="demo")
    parser.add_argument("--probe-gpu", action="store_true", help="Probe the configured SAM environment without loading models")
    args = parser.parse_args(argv)
    try:
        result = check(args.mode, probe_gpu=args.probe_gpu)
    except (ValueError, OSError) as exc:
        result = {"ok": False, "error": str(exc), "model_calls": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
