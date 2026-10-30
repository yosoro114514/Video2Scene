from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

CONTRACT_VERSION = "video2scene/1.0"


class ContractError(Exception):
    def __init__(self, code: str, message: str, details=None, retryable=False):
        super().__init__(message)
        self.code, self.message = code, message
        self.details, self.retryable = details or {}, retryable

    def as_dict(self):
        return {"code": self.code, "message": self.message,
                "retryable": self.retryable, "details": self.details}


def canonical_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(data: Any) -> str:
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def file_hash(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def camera_hash(frames, frame_ids=None):
    if frame_ids is not None:
        indexed = {frame["source_frame_id"]: frame for frame in frames}
        try:
            frames = [indexed[key] for key in frame_ids]
        except KeyError as exc:
            raise ContractError("FRAME_MISMATCH", "Requested frame is absent from evidence") from exc
    fields = ("source_frame_id", "K", "c2w", "width", "height", "source_to_image")
    return canonical_hash([{key: frame[key] for key in fields} for frame in frames])


@lru_cache
def schema():
    return json.loads((Path(__file__).with_name("v1.schema.json")).read_text(encoding="utf-8"))


@lru_cache
def _validator(type_name):
    source = schema()
    if type_name not in source["$defs"]:
        raise ValueError(f"Unknown contract type: {type_name}")
    return Draft202012Validator({"$schema": source["$schema"], "$defs": source["$defs"],
                                 "$ref": "#/$defs/" + type_name}, format_checker=FormatChecker())


def validate(data, type_name):
    try:
        canonical_bytes(data)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_ARGUMENT", "JSON must contain finite values", {"reason": str(exc)}) from exc
    errors = list(_validator(type_name).iter_errors(data))
    if errors:
        raise ContractError("INVALID_ARGUMENT", f"Invalid {type_name}",
                            {"errors": [e.message for e in errors[:5]]})
    return data


@dataclass(frozen=True)
class ExecutionContext:
    actor: str = "agent"
    session_id: str = "local"
    decision_ref: str | None = None
    allowed_object_ids: frozenset[str] = field(default_factory=frozenset)
    allowed_fields: frozenset[str] = field(default_factory=frozenset)
    storage_root: Path = field(default_factory=Path.cwd)
    cancel_event: Event = field(default_factory=Event, compare=False)

    def authorize(self, instance_id: str, field: str):
        if instance_id not in self.allowed_object_ids or field not in self.allowed_fields:
            raise ContractError("FORBIDDEN_SCOPE", "Object or field is outside the user decision",
                                {"instance_id": instance_id, "field": field})

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise ContractError("CANCELLED", "Execution was cancelled")

