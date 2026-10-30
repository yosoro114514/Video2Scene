"""Wire DTOs generated from frozen v1 schema; validate at runtime boundaries."""
from __future__ import annotations
from typing import Any, Literal, NotRequired, TypedDict

class BaseToken(TypedDict):
    revision: str
    head_seq: int

class ArtifactRef(TypedDict):
    uri: str
    sha256: str
    media_type: str
    role: str

class Transform(TypedDict):
    translation_m: list[float]
    rotation_xyzw: list[float]
    scale: list[float]

class Bounds(TypedDict):
    min: list[float]
    max: list[float]

class AssetRef(TypedDict):
    asset_version: str
    artifact: ArtifactRef
    native_frame: Literal['asset_y_up', 'blender_z_up']
    native_to_object: list[list[float]]

class FieldState(TypedDict):
    source: Literal['observed', 'derived', 'user', 'model', 'unknown']
    locked: bool
    decision_ref: str | None

class SceneObject(TypedDict):
    instance_id: str
    kind: Literal['object', 'support', 'wall', 'opening', 'floor', 'ceiling']
    category: str
    transform: Transform
    local_bounds_m: Bounds
    support_id: str | None
    asset: AssetRef
    field_state: dict[str, Any]
    evidence_ids: list[str]
    geometry_source: Literal['observed', 'inferred', 'mixed']

class SceneSnapshot(TypedDict):
    contract_version: Literal['video2scene/1.0']
    scene_id: str
    revision: str
    parent_revision: str | None
    capture_revision: str
    geometry_revision: str
    world_frame: Literal['scene_y_up']
    object_frame: Literal['object_y_up']
    unit: Literal['m']
    scale_status: Literal['estimated', 'calibrated']
    scope: Literal['partial_scene', 'room']
    objects: list[SceneObject]

class SceneView(TypedDict):
    head: BaseToken
    snapshot: SceneSnapshot

class EvidenceFrame(TypedDict):
    source_frame_id: str
    source_time_s: float | None
    preview_time_s: float
    width: int
    height: int
    rgb: ArtifactRef
    mask: ArtifactRef | None
    K: list[list[float]]
    c2w: list[list[float]]
    source_to_image: list[list[float]]
    selection_reason: str

class EvidenceBundle(TypedDict):
    evidence_id: str
    scene_id: str
    scene_revision: str
    capture_revision: str
    geometry_revision: str
    target_ids: list[str]
    purpose: Literal['review', 'refine', 'validation']
    frames: list[EvidenceFrame]
    evidence_hash: str

class CandidateRef(TypedDict):
    candidate_id: str
    scene_id: str
    base_revision: str
    base_head_seq: int
    candidate_hash: str
    state: Literal['staged', 'built', 'checked', 'committed', 'rejected', 'stale']

class SceneDraft(TypedDict):
    contract_version: Literal['video2scene/1.0']
    scene_id: str
    capture_revision: str
    geometry_revision: str
    world_frame: Literal['scene_y_up']
    object_frame: Literal['object_y_up']
    unit: Literal['m']
    scale_status: Literal['estimated', 'calibrated']
    scope: Literal['partial_scene', 'room']
    objects: list[SceneObject]

class CandidateSnapshot(TypedDict):
    candidate: CandidateRef
    scene: SceneDraft

class AssetUpdate(TypedDict):
    instance_id: str
    asset: AssetRef
    local_bounds_m: Bounds

class BuildOutput(TypedDict):
    artifact_refs: list[ArtifactRef]
    asset_updates: list[AssetUpdate]
    input_fingerprint: str

class BuildResult(TypedDict):
    candidate: CandidateRef
    artifact_refs: list[ArtifactRef]

class RenderReport(TypedDict):
    candidate_id: str
    candidate_hash: str
    evidence_id: str
    evidence_hash: str
    frame_ids: list[str]
    view_mode: Literal['normal', 'visible_mask', 'xray']
    camera_hash: str
    artifact_refs: list[ArtifactRef]

class Check(TypedDict):
    name: str
    passed: bool | None
    severity: Literal['info', 'warning', 'error']
    details: str
    artifact_refs: list[ArtifactRef]

class ValidationReport(TypedDict):
    validation_id: str
    candidate_id: str
    candidate_hash: str
    evidence_id: str
    evidence_hash: str
    criteria_id: str
    criteria_hash: str
    intent: Literal['reconstruction_correction', 'design_edit']
    technical_pass: bool
    visual_pass: bool | None
    committable: bool
    checks: list[Check]

class CollisionReport(TypedDict):
    candidate_id: str
    candidate_hash: str
    method: Literal['mesh', 'box']
    passed: bool
    checks: list[Check]

class CommitResult(TypedDict):
    revision: str
    previous_revision: str
    head_seq: int
    event_id: str

class HistoryEntry(TypedDict):
    revision: str
    parent_revision: str | None
    operation: Literal['import', 'patch', 'refine']
    created_at: str

class HistoryResult(TypedDict):
    head: BaseToken
    entries: list[HistoryEntry]
    can_undo: bool
    can_redo: bool

class Error(TypedDict):
    code: Literal['INVALID_ARGUMENT', 'UNSUPPORTED_CONTRACT', 'NOT_FOUND', 'FORBIDDEN_SCOPE', 'LOCKED_FIELD', 'STALE_REVISION', 'IDEMPOTENCY_CONFLICT', 'NO_CHANGE', 'EVIDENCE_MISMATCH', 'FRAME_MISMATCH', 'CONSTRAINT_VIOLATION', 'BUILD_FAILED', 'STALE_ARTIFACT', 'TOOL_UNAVAILABLE', 'RESOURCE_EXHAUSTED', 'TIMEOUT', 'CANCELLED', 'INTERNAL_ERROR', 'EVENT_CURSOR_EXPIRED']
    message: str
    retryable: bool
    details: dict[str, Any]

class ApiError(TypedDict):
    contract_version: Literal['video2scene/1.0']
    error: Error

class JobReceipt(TypedDict):
    job_id: str
    state: Literal['queued', 'running']

class JobRecord(TypedDict):
    job_id: str
    scene_id: str
    base_revision: str
    base_head_seq: int
    request_id: str
    state: Literal['queued', 'running', 'cancel_requested', 'succeeded', 'failed', 'cancelled', 'stale']
    progress: float | None
    result: CandidateRef | BuildResult | RenderReport | ValidationReport | None
    error: Error | None

class Edit(TypedDict):
    instance_id: str
    field: Literal['category'] | Literal['transform.translation_m'] | Literal['transform.rotation_xyzw'] | Literal['transform.scale'] | Literal['support_id']
    value: str | list[float] | list[float] | list[float] | str | None

class Budget(TypedDict):
    max_iterations: int
    timeout_s: int
    max_model_calls: NotRequired[int]

class Constraints(TypedDict):
    keep_root_transform: bool
    keep_support: bool
    keep_bounds: bool

class ToolRequest(TypedDict):
    contract_version: Literal['video2scene/1.0']
    request_id: str
    scene_id: str
    tool: Literal['inspect_scene'] | Literal['select_evidence'] | Literal['apply_scene_patch'] | Literal['build_whitebox'] | Literal['refine_asset'] | Literal['render_compare'] | Literal['validate_candidate'] | Literal['commit_candidate'] | Literal['revision_history'] | Literal['job_control']
    params: dict[str, Any]
    base_revision: NotRequired[str]
    base_head_seq: NotRequired[int]

class ToolResponse(TypedDict):
    contract_version: Literal['video2scene/1.0']
    request_id: str
    scene_id: str
    tool: Literal['inspect_scene'] | Literal['select_evidence'] | Literal['apply_scene_patch'] | Literal['build_whitebox'] | Literal['render_compare'] | Literal['validate_candidate'] | Literal['commit_candidate'] | Literal['revision_history'] | Literal['job_control'] | Literal['build_whitebox', 'refine_asset', 'render_compare', 'validate_candidate'] | Literal['inspect_scene', 'select_evidence', 'apply_scene_patch', 'build_whitebox', 'refine_asset', 'render_compare', 'validate_candidate', 'commit_candidate', 'revision_history', 'job_control']
    status: Literal['ok'] | Literal['accepted'] | Literal['error']
    result: SceneView | EvidenceBundle | CandidateRef | BuildResult | RenderReport | ValidationReport | CommitResult | HistoryResult | CommitResult | JobRecord | JobReceipt | None
    error: None | Error
    warnings: list[str]

class UserCommand(TypedDict):
    contract_version: Literal['video2scene/1.0']
    command_id: str
    session_id: str
    scene_id: str
    base_revision: str
    base_head_seq: int
    selected_ids: list[str]
    intent: Literal['reconstruction_correction', 'design_edit']
    kind: Literal['language'] | Literal['patch']
    text: NotRequired[str]
    edits: NotRequired[list[Edit]]

class CommandReceipt(TypedDict):
    command_id: str
    operation_id: str
    state: Literal['accepted']

class CommandRecord(TypedDict):
    receipt: CommandReceipt
    state: Literal['accepted', 'running', 'needs_clarification', 'succeeded', 'failed', 'cancelled']
    request_ids: list[str]
    message: str
    error: Error | None

class SceneEvent(TypedDict):
    contract_version: Literal['video2scene/1.0']
    event_id: str
    scene_id: str
    seq: int
    type: Literal['command.accepted'] | Literal['tool.completed'] | Literal['job.updated'] | Literal['scene.head_changed'] | Literal['agent.text'] | Literal['agent.clarification'] | Literal['agent.finished']
    occurred_at: str
    request_id: str | None
    data: CommandReceipt | dict[str, Any] | JobRecord | CommitResult | dict[str, Any] | dict[str, Any] | dict[str, Any]

class SessionSpec(TypedDict):
    session_id: str
    model: str
    instruction: str
    scene_id: str
    base_revision: str
    base_head_seq: int
    allowed_tools: list[Literal['inspect_scene', 'select_evidence', 'apply_scene_patch', 'build_whitebox', 'refine_asset', 'render_compare', 'validate_candidate', 'commit_candidate', 'revision_history', 'job_control']]
    target_ids: list[str]
    evidence_ids: list[str]
    budget: Budget

class ProviderCapabilities(TypedDict):
    vision: bool
    tool_calling: bool
    streaming: bool
    cancel: bool
    resume: bool

class AgentEvent(TypedDict):
    session_id: str
    type: Literal['text_delta'] | Literal['tool_call'] | Literal['tool_result'] | Literal['needs_clarification'] | Literal['session_end']
    data: dict[str, Any] | ToolRequest | ToolResponse | dict[str, Any] | dict[str, Any]
