"""Conservative technical validation; visual evidence is never fabricated.

This module owns no head mutation. It reads the repository's frozen authorization
metadata and stores a report which commit must bind again to current inputs.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import json
import re
import sqlite3
import uuid
from pathlib import Path

import numpy as np

from video2scene.contracts_v1 import ContractError, canonical_hash, camera_hash, validate


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DEFAULT_CRITERIA = (
    {"criteria_id": "tabletop_design_v1", "intent": "design_edit",
     "allow_visual_null": True, "support_tolerance_m": 0.04,
     "collision_penetration_m": 0.01, "collision_fraction": 0.20,
     "preserve_existing_geometry_issues": True,
     "description": "Design target, authorized changes, support and no new gross box intersections. Visual quality not asserted."},
    {"criteria_id": "tabletop_reconstruction_v1", "intent": "reconstruction_correction",
     "allow_visual_null": False, "support_tolerance_m": 0.04,
     "collision_penetration_m": 0.01, "collision_fraction": 0.20,
     "preserve_existing_geometry_issues": True,
     "description": "Technical constraints plus independently executed fixed-camera visual comparison to source images."},
)


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def _check(name, passed, details, *, severity=None, artifacts=()):
    return {"name": name, "passed": passed,
            "severity": severity or ("info" if passed else "error"),
            "details": str(details), "artifact_refs": list(artifacts)}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def camera_fingerprint(evidence, frame_ids=None):
    """Hash the exact projection inputs in the requested frame order."""
    return camera_hash(evidence["frames"], frame_ids)


class _ImmutableStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "validation.sqlite"
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, sha256 TEXT, PRIMARY KEY(kind,id))")

    def put(self, kind, identity, value):
        if not _ID.fullmatch(identity):
            raise ContractError("INVALID_ARGUMENT", "Invalid validation record identifier")
        digest = canonical_hash(value)
        folder = self.root / kind
        folder.mkdir(exist_ok=True)
        path = folder / (identity + ".json")
        with sqlite3.connect(self.db, timeout=30) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT sha256 FROM records WHERE kind=? AND id=?", (kind, identity)).fetchone()
            if row:
                if row[0] != digest:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Immutable validation record already exists")
                return self.get(kind, identity)
            # An unreferenced complete file from a pre-commit crash can be reused.
            if path.exists():
                if canonical_hash(_read_json(path)) != digest:
                    raise ContractError("STALE_ARTIFACT", "Unreferenced validation file differs")
            else:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
            con.execute("INSERT INTO records VALUES (?,?,?)", (kind, identity, digest))
        return copy.deepcopy(value)

    def get(self, kind, identity):
        if not isinstance(identity, str) or not _ID.fullmatch(identity):
            raise ContractError("INVALID_ARGUMENT", "Invalid validation record identifier")
        with sqlite3.connect(self.db) as con:
            row = con.execute("SELECT sha256 FROM records WHERE kind=? AND id=?", (kind, identity)).fetchone()
        if not row:
            raise ContractError("NOT_FOUND", f"Unknown {kind} record: {identity}")
        try:
            value = _read_json(self.root / kind / (identity + ".json"))
            if canonical_hash(value) != row[0]:
                raise ValueError("content changed")
        except (ValueError, OSError) as exc:
            raise ContractError("STALE_ARTIFACT", f"Stored {kind} record has changed") from exc
        return value


class CriteriaRegistry:
    """Server-owned criteria. An existing identifier cannot change meaning."""

    def __init__(self, root):
        self.store = _ImmutableStore(root)
        for criteria in _DEFAULT_CRITERIA:
            self.register(criteria)

    def register(self, criteria):
        criteria = copy.deepcopy(criteria)
        criteria.pop("criteria_hash", None)
        if criteria.get("intent") not in ("design_edit", "reconstruction_correction"):
            raise ContractError("INVALID_ARGUMENT", "Criteria intent is required")
        if not isinstance(criteria.get("allow_visual_null"), bool):
            raise ContractError("INVALID_ARGUMENT", "Criteria must explicitly allow or reject visual null")
        if criteria["intent"] == "reconstruction_correction" and criteria["allow_visual_null"]:
            raise ContractError("INVALID_ARGUMENT", "Reconstruction correction requires executed visual validation")
        criteria["criteria_hash"] = canonical_hash(criteria)
        return self.store.put("criteria", criteria["criteria_id"], criteria)

    def get(self, criteria_id, context=None):
        return self.store.get("criteria", criteria_id)


def _value(obj, field):
    if field.startswith("transform."):
        return obj["transform"][field.split(".", 1)[1]]
    if field.startswith("asset."):
        return obj["asset"]
    return obj[field]


def _world_obb(obj):
    """World center, orthonormal column axes and half extents of local bounds."""
    t = obj["transform"]
    quaternion = np.asarray(t["rotation_xyzw"], dtype=float)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid object quaternion")
    # Accepted quaternions may contain serialization roundoff. Keep the SAT axes
    # orthonormal, consistent with the renderer's normalized quaternion rotation.
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    bounds = obj["local_bounds_m"]
    low, high = np.asarray(bounds["min"], float), np.asarray(bounds["max"], float)
    scale, translation = np.asarray(t["scale"], float), np.asarray(t["translation_m"], float)
    if not np.isfinite([low, high, scale, translation]).all() or np.any(high < low):
        raise ValueError("Invalid object bounds or transform")
    center = rotation @ ((low + high) * .5 * scale) + translation
    half = (high - low) * .5 * np.abs(scale)
    return center, rotation, half


def _world_bounds(obj):
    center, axes, half = _world_obb(obj)
    extent = np.abs(axes) @ half
    return center - extent, center + extent


def _obb_penetrates(left, right):
    """15-axis SAT: True only for positive-volume OBB intersection.

    Face/edge contact is allowed. The metre tolerance only absorbs floating-point
    roundoff, far below the existing centimetre-scale collision threshold. This
    tests bounding boxes, not mesh surfaces or intersection volume.
    """
    ac, aa, ah = left
    bc, ba, bh = right
    delta = bc - ac
    tolerance = 1e-9 * max(1., float(np.sum(ah) + np.sum(bh)), float(np.linalg.norm(delta)))
    axes = [aa[:, i] for i in range(3)] + [ba[:, i] for i in range(3)]
    axes.extend(np.cross(aa[:, i], ba[:, j]) for i in range(3) for j in range(3))
    for axis in axes:
        length = np.linalg.norm(axis)
        # The cross product of parallel axes is zero and supplies no new plane.
        # Avoid normalizing numerical noise from nearly parallel orientations.
        if length <= 1e-12:
            continue
        axis = axis / length
        radius = float(np.dot(ah, np.abs(aa.T @ axis)) + np.dot(bh, np.abs(ba.T @ axis)))
        if abs(float(np.dot(delta, axis))) >= radius - tolerance:
            return False
    return True


def _obb_contact_gap(left, right):
    """Largest signed SAT gap: positive separation, negative penetration.

    Wall supports attach at a face or edge (mounted TV, adjoining lintel),
    unlike horizontal supports whose top must meet the object's bottom.
    Testing every SAT axis also rejects contact inferred only from world AABBs.
    This remains a bounds check, not a claim about mesh contact or fastening.
    """
    ac, aa, ah = left
    bc, ba, bh = right
    axes = [aa[:, i] for i in range(3)] + [ba[:, i] for i in range(3)]
    axes.extend(np.cross(aa[:, i], ba[:, j]) for i in range(3) for j in range(3))
    gaps = []
    for axis in axes:
        length = np.linalg.norm(axis)
        if length <= 1e-12:
            continue
        axis = axis / length
        radius = float(np.dot(ah, np.abs(aa.T @ axis)) + np.dot(bh, np.abs(ba.T @ axis)))
        gaps.append(abs(float(np.dot(bc - ac, axis))) - radius)
    return max(gaps)


def _geometry_issues(scene, criteria):
    objects = {o["instance_id"]: o for o in scene["objects"]}
    boxes = {i: _world_bounds(o) for i, o in objects.items()}
    oriented_boxes = {i: _world_obb(o) for i, o in objects.items()}
    issues = {}
    tol = criteria["support_tolerance_m"]
    for identity, obj in objects.items():
        support = obj["support_id"]
        if support in objects:
            if objects[support]["kind"] == "wall":
                gap = _obb_contact_gap(oriented_boxes[identity], oriented_boxes[support])
                if abs(gap) > tol:
                    issues[f"support:{identity}:{support}"] = f"wall attachment signed OBB gap={gap:.4f}m (bounds contact, not mesh fastening)"
                continue
            low, high = boxes[identity]
            slow, shigh = boxes[support]
            overlap = np.minimum(high, shigh) - np.maximum(low, slow)
            gap = float(low[1] - shigh[1])
            if overlap[0] < -tol or overlap[2] < -tol or abs(gap) > tol:
                issues[f"support:{identity}:{support}"] = f"support box gap={gap:.4f}m, horizontal overlap={overlap[[0,2]].tolist()}"
    for left, right in itertools.combinations(objects, 2):
        a, b = objects[left], objects[right]
        if a["support_id"] == right or b["support_id"] == left:
            continue
        if a["kind"] in ("opening", "floor", "ceiling") or b["kind"] in ("opening", "floor", "ceiling"):
            continue
        if not _obb_penetrates(oriented_boxes[left], oriented_boxes[right]):
            continue
        alo, ahi = boxes[left]
        blo, bhi = boxes[right]
        overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
        if np.min(overlap) > criteria["collision_penetration_m"]:
            volume = float(np.prod(overlap))
            smaller = min(float(np.prod(ahi-alo)), float(np.prod(bhi-blo)))
            if smaller > 0 and volume / smaller > criteria["collision_fraction"]:
                issues[f"collision:{left}:{right}"] = f"AABB intersection fraction={volume/smaller:.3f} after OBB overlap (conservative box check, not mesh contact)"
    return issues


class CandidateValidator:
    def __init__(self, root, *, repository=None, evidence_service=None,
                 artifact_resolver=None, visual_checker=None):
        self.store = _ImmutableStore(Path(root) / "validation_v1")
        self.criteria = CriteriaRegistry(self.store.root)
        self.repository = repository
        self.evidence_service = evidence_service
        self.artifact_resolver = artifact_resolver
        self.visual_checker = visual_checker

    def get_criteria(self, criteria_id, context=None):
        return self.criteria.get(criteria_id, context)

    async def get_report(self, validation_id, context=None):
        return validate(self.store.get("reports", validation_id), "ValidationReport")

    async def _artifact(self, ref):
        if self.artifact_resolver is None:
            raise ContractError("TOOL_UNAVAILABLE", "Artifact resolver is required for actual validation")
        resolver = getattr(self.artifact_resolver, "resolve", self.artifact_resolver)
        path = Path(await _await(resolver(ref)))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ContractError("STALE_ARTIFACT", "Artifact content differs from its frozen hash")
        return path

    async def validate(self, candidate_snapshot, evidence, criteria, job_context):
        validate(candidate_snapshot, "CandidateSnapshot")
        validate(evidence, "EvidenceBundle")
        candidate, scene = candidate_snapshot["candidate"], candidate_snapshot["scene"]
        criteria_id = criteria if isinstance(criteria, str) else criteria["criteria_id"]
        stored_criteria = self.get_criteria(criteria_id, job_context)
        if isinstance(criteria, dict) and criteria != stored_criteria:
            raise ContractError("EVIDENCE_MISMATCH", "Criteria differ from the server-owned record")
        criteria = stored_criteria
        if self.repository is None or self.evidence_service is None:
            raise ContractError("TOOL_UNAVAILABLE", "Repository and evidence service are required")
        persisted = await self.repository.read_candidate(candidate["candidate_id"], candidate["candidate_hash"], job_context)
        if persisted != candidate_snapshot:
            raise ContractError("EVIDENCE_MISMATCH", "Candidate differs from repository snapshot")
        metadata = await _await(self.repository.candidate_metadata(candidate["candidate_id"], job_context))
        getter = getattr(self.evidence_service, "get", self.evidence_service)
        stored_evidence = await _await(getter(evidence["evidence_id"], job_context))
        if stored_evidence != evidence:
            raise ContractError("EVIDENCE_MISMATCH", "Evidence differs from the persisted bundle")
        expected = {"scene_id": scene["scene_id"], "scene_revision": candidate["base_revision"],
                    "capture_revision": scene["capture_revision"], "geometry_revision": scene["geometry_revision"]}
        if any(evidence[key] != value for key, value in expected.items()):
            raise ContractError("EVIDENCE_MISMATCH", "Evidence is not bound to this candidate base/capture/geometry")
        if metadata["intent"] != criteria["intent"]:
            raise ContractError("INVALID_ARGUMENT", "Criteria intent differs from the authorized candidate intent")
        if not set(metadata["target_fields"]).issubset(evidence["target_ids"]):
            raise ContractError("EVIDENCE_MISMATCH", "Evidence does not cover every changed target")
        checks = [_check("candidate_built", candidate["state"] in ("built", "checked"),
                         f"candidate state={candidate['state']}")]
        base = metadata["base_snapshot"]
        checks.extend(self._scope_checks(scene, base, metadata))
        checks.extend(self._structure_checks(scene))
        try:
            current_issues = _geometry_issues(scene, criteria)
            old_issues = _geometry_issues(base, criteria)
            # Keep a pre-existing issue only when both implicated objects retain
            # their exact geometry/transform/support. A larger bad edit is new.
            old_objects = {o["instance_id"]: o for o in base["objects"]}
            new_objects = {o["instance_id"]: o for o in scene["objects"]}
            newly_invalid = []
            for issue, detail in current_issues.items():
                ids = issue.split(":")[1:]
                same_geometry = all(i in old_objects and all(new_objects[i][k] == old_objects[i][k]
                    for k in ("transform", "local_bounds_m", "support_id")) for i in ids)
                preserved = issue in old_issues and same_geometry and criteria["preserve_existing_geometry_issues"]
                checks.append(_check(issue, None if preserved else False, detail,
                                     severity="warning" if preserved else "error"))
                if not preserved:
                    newly_invalid.append(issue)
            checks.append(_check("support_and_box_collisions", not newly_invalid,
                                 "No newly introduced gross box conflict" if not newly_invalid else ", ".join(newly_invalid)))
        except (KeyError, ValueError, TypeError) as exc:
            checks.append(_check("support_and_box_collisions", False, str(exc)))
        refs = [o["asset"]["artifact"] for o in scene["objects"]]
        refs.extend(metadata.get("artifact_refs", []))
        for frame in evidence["frames"]:
            refs.append(frame["rgb"])
            if frame["mask"]:
                refs.append(frame["mask"])
        unique_refs = {(ref["uri"], ref["sha256"]): ref for ref in refs}
        artifact_errors = []
        for ref in unique_refs.values():
            try:
                await self._artifact(ref)
            except (ContractError, OSError) as exc:
                artifact_errors.append(f"{ref['uri']}: {exc}")
        checks.append(_check("artifact_freshness", not artifact_errors,
                             "; ".join(artifact_errors) or f"Verified {len(unique_refs)} artifact content hashes"))
        technical_pass = not any(c["passed"] is False and c["severity"] == "error" for c in checks)
        visual_pass = None
        if self.visual_checker is not None and technical_pass:
            try:
                result = await _await(self.visual_checker(candidate_snapshot, evidence, criteria, job_context))
                visual_pass, visual_checks = await self._validate_visual(result, candidate, evidence)
                checks.extend(visual_checks)
            except Exception as exc:
                checks.append(_check("visual_execution", None, f"Visual check not completed: {type(exc).__name__}: {exc}", severity="error"))
                # A configured but failed visual check is never silently waived.
                visual_pass = None
                technical_pass = False
        else:
            checks.append(_check("visual_execution", None,
                "not_run: no visual checker configured" if self.visual_checker is None else "not_run: technical checks failed",
                severity="warning" if criteria["allow_visual_null"] else "error"))
        committable = technical_pass and (visual_pass is True or (visual_pass is None and criteria["allow_visual_null"]))
        report = {"validation_id": "val_" + uuid.uuid4().hex, "candidate_id": candidate["candidate_id"],
                  "candidate_hash": candidate["candidate_hash"], "evidence_id": evidence["evidence_id"],
                  "evidence_hash": evidence["evidence_hash"], "criteria_id": criteria["criteria_id"],
                  "criteria_hash": criteria["criteria_hash"], "intent": criteria["intent"],
                  "technical_pass": technical_pass, "visual_pass": visual_pass,
                  "committable": committable, "checks": checks}
        validate(report, "ValidationReport")
        return self.store.put("reports", report["validation_id"], report)

    def _scope_checks(self, scene, base, metadata):
        if metadata.get('operation') == 'extension':
            return self._extension_scope_checks(scene, base, metadata)
        old = {o["instance_id"]: o for o in base["objects"]}
        new = {o["instance_id"]: o for o in scene["objects"]}
        if set(old) != set(new):
            return [_check("authorization_scope", False, "Instance addition/deletion is outside v1 scope")]
        problems = []
        fields = ("category", "transform.translation_m", "transform.rotation_xyzw", "transform.scale", "support_id")
        for identity, obj in new.items():
            before = old[identity]
            permitted = set(metadata["target_fields"].get(identity, []))
            changed = {f for f in fields if _value(before, f) != _value(obj, f)}
            if before["asset"] != obj["asset"]:
                if not permitted.intersection(("asset.geometry", "asset.materials")):
                    problems.append(f"{identity}: unauthorized asset replacement")
            if before["local_bounds_m"] != obj["local_bounds_m"]:
                if "asset.geometry" not in permitted or metadata.get("constraints", {}).get("keep_bounds", False):
                    problems.append(f"{identity}: bounds changed outside authorization")
            for field in changed:
                if field not in permitted:
                    problems.append(f"{identity}.{field}: outside frozen authorization")
                if before["field_state"][field]["locked"] and not obj["field_state"][field]["locked"]:
                    problems.append(f"{identity}.{field}: lock removed")
            for field, state in before["field_state"].items():
                if state["locked"] and not obj["field_state"][field]["locked"]:
                    problems.append(f"{identity}.{field}: lock removed")
                if field not in permitted and state != obj["field_state"][field]:
                    problems.append(f"{identity}.{field}: unauthorized field-state change")
            if before["kind"] != obj["kind"]:
                problems.append(f"{identity}: kind is immutable in v1")
            constraints = metadata.get("constraints", {})
            if constraints.get("keep_root_transform") and before["transform"] != obj["transform"]:
                problems.append(f"{identity}: root transform constraint violated")
            if constraints.get("keep_support") and before["support_id"] != obj["support_id"]:
                problems.append(f"{identity}: support constraint violated")
            if not permitted and before != obj:
                problems.append(f"{identity}: non-target object changed")
        return [_check("authorization_scope_and_locks", not problems, "; ".join(problems) or "Frozen per-object field scope and locks preserved")]

    def _extension_scope_checks(self, scene, base, metadata):
        """A builder cannot broaden the exact host-authorized addition proposal."""
        old = {obj['instance_id']: obj for obj in base['objects']}
        new = {obj['instance_id']: obj for obj in scene['objects']}
        additions = metadata.get('exact_additions', {})
        replacements = metadata.get('exact_asset_replacements', {})
        problems = []
        if set(new) != set(old) | set(additions) or set(old) & set(additions):
            problems.append('Object addition/deletion differs from exact authorized identities')
        expected_fields = {identity: ['scene.objects.add'] for identity in additions}
        expected_fields.update({identity: ['asset.geometry', 'asset.materials'] for identity in replacements})
        if metadata.get('target_fields') != expected_fields or not set(replacements) <= set(old):
            problems.append('Extension field authorization differs from its frozen scope')
        if canonical_hash(scene) != metadata.get('expected_draft_hash'):
            problems.append('Extension candidate differs from the exact authored draft')
        base_meta = {key: value for key, value in base.items() if key not in ('revision', 'parent_revision', 'objects')}
        scene_meta = {key: value for key, value in scene.items() if key != 'objects'}
        if base_meta != scene_meta:
            problems.append('Extension changed scene/capture/coordinate metadata')
        for identity, previous in old.items():
            expected = copy.deepcopy(previous)
            if identity in replacements:
                expected['asset'] = replacements[identity]
                for field in ('asset.geometry', 'asset.materials'):
                    expected['field_state'][field]['source'] = 'model'
                    expected['field_state'][field]['decision_ref'] = metadata['decision_ref']
            if new.get(identity) != expected:
                problems.append(f'{identity}: existing object changed beyond its exact asset replacement')
        for identity, proposed in additions.items():
            if new.get(identity) != proposed:
                problems.append(f'{identity}: new object differs from its authorized specification')
        return [_check('authorization_scope_and_locks', not problems,
                       '; '.join(problems) or 'Exact additions and asset replacements preserve every existing layout/capture field')]

    def _structure_checks(self, scene):
        problems = []
        objects = {o["instance_id"]: o for o in scene["objects"]}
        if len(objects) != len(scene["objects"]):
            problems.append("duplicate object ids")
        for identity, obj in objects.items():
            transform = obj["transform"]
            if abs(np.linalg.norm(transform["rotation_xyzw"]) - 1) > 1e-5:
                problems.append(f"{identity}: quaternion is not unit length")
            if any(x <= 0 for x in transform["scale"]):
                problems.append(f"{identity}: nonpositive scale")
            if any(a > b for a,b in zip(obj["local_bounds_m"]["min"], obj["local_bounds_m"]["max"])):
                problems.append(f"{identity}: inverted bounds")
            seen = {identity}
            support = obj["support_id"]
            while support:
                if support not in objects or support in seen:
                    problems.append(f"{identity}: invalid support graph")
                    break
                seen.add(support)
                support = objects[support]["support_id"]
        return [_check("transforms_and_support_graph", not problems, "; ".join(problems) or "Transforms and support graph valid")]

    async def _validate_visual(self, result, candidate, evidence):
        if not isinstance(result, dict) or type(result.get("passed")) is not bool:
            raise ContractError("INVALID_ARGUMENT", "Visual checker must return passed plus a bound render_report")
        render = validate(result.get("render_report"), "RenderReport")
        bindings = {"candidate_id": candidate["candidate_id"], "candidate_hash": candidate["candidate_hash"],
                    "evidence_id": evidence["evidence_id"], "evidence_hash": evidence["evidence_hash"]}
        if any(render[k] != v for k,v in bindings.items()):
            raise ContractError("EVIDENCE_MISMATCH", "Visual report is bound to different inputs")
        if render["camera_hash"] != camera_fingerprint(evidence, render["frame_ids"]):
            raise ContractError("FRAME_MISMATCH", "Visual render camera fingerprint is stale")
        for ref in render["artifact_refs"]:
            await self._artifact(ref)
        details = result.get("details")
        if not isinstance(details, str) or not details.strip():
            raise ContractError("INVALID_ARGUMENT", "Visual result must describe the executed comparison")
        return result["passed"], [_check("visual_comparison", result["passed"], details, artifacts=render["artifact_refs"])]
