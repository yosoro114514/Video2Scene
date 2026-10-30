"""Machine settings for source checkouts. This module never reads credentials."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import shutil
import tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    blender: str = "blender"
    model: str = "gpt-6-astra"
    effort: str = "high"
    wsl_distro: str = "Ubuntu-22.04"
    sam_python: str = "python3"
    sam_checkpoint: str = ""
    hf_home: str = ""
    torch_home: str = ""
    conda_bin: str = ""
    conda_env: str = "vipe-scene"


def load_settings(path=None, *, environ=None):
    """Environment > local TOML > defaults; an explicitly named file must exist.

    Linux worker paths are literal absolute Linux paths, even on Windows.
    Only the host Blender path is expanded/resolved by ``blender_path``.
    """
    env = os.environ if environ is None else environ
    explicit = path or env.get("V2S_CONFIG")
    location = Path(explicit).expanduser() if explicit else ROOT / "video2scene.local.toml"
    values = {}
    if location.is_file():
        data = tomllib.loads(location.read_text(encoding="utf-8-sig"))
        if set(data) - {"video2scene"}:
            raise ValueError("Only the [video2scene] configuration table is supported")
        values = data.get("video2scene", {})
        if not isinstance(values, dict):
            raise ValueError("[video2scene] must be a table")
    elif explicit:
        raise FileNotFoundError(f"Configuration file not found: {location}")
    names = {field.name for field in fields(Settings)}
    if set(values) - names:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(set(values) - names)))
    values = {**values, **{name: env["V2S_" + name.upper()] for name in names if "V2S_" + name.upper() in env}}
    for name, value in values.items():
        if not isinstance(value, str) or any(c in value for c in "\n\r\x00"):
            raise ValueError(f"Configuration {name} must be a single-line string")
    settings = Settings(**values)
    if not all((settings.blender, settings.model, settings.wsl_distro, settings.sam_python, settings.conda_env)):
        raise ValueError("blender, model, wsl_distro, sam_python and conda_env must not be empty")
    if settings.effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning effort")
    for name in ("hf_home", "torch_home", "sam_checkpoint", "conda_bin"):
        value = getattr(settings, name)
        if value and not value.startswith("/"):
            raise ValueError(f"{name} must be an absolute Linux path (not ~ or a Windows path)")
    return settings


def blender_path(value=None):
    selected = str(value or load_settings().blender)
    return Path(shutil.which(selected) or os.path.expandvars(selected)).expanduser()


def gpu_environment(settings=None):
    settings = settings or load_settings()
    return [f"{name.upper()}={getattr(settings, name)}" for name in ("hf_home", "torch_home") if getattr(settings, name)]


def vipe_environment(settings=None):
    settings = settings or load_settings()
    return [*gpu_environment(settings), f"ENV_NAME={settings.conda_env}",
            *([f"CONDA_BIN={settings.conda_bin}"] if settings.conda_bin else [])]


def public_settings():
    """Paths and model settings only; no authentication cache is opened."""
    return asdict(load_settings())
