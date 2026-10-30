"""SQLite is authoritative for head, requests, candidate index and outbox.

JSON files are immutable and written before their references enter a transaction.
An interrupted transaction may leave an unreferenced file; it cannot expose a new
head. Services are injected so this module never invokes a model or Blender.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import inspect as pyinspect
import json
import math
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from video2scene.contracts_v1 import ContractError, canonical_hash, utc_now, validate
from video2scene.contracts_v1.extension import validate_extension_request


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(prefix):
    return prefix + uuid4().hex


def _fail(code, message, **details):
    raise ContractError(code, message, details=details)


def _draft(scene):
    return {k: copy.deepcopy(v) for k, v in scene.items() if k not in ("revision", "parent_revision")}


async def _invoke(fn, *args):
    result = fn(*args)
    return await result if pyinspect.isawaitable(result) else result


class SQLiteSceneRepository:
    def __init__(self, root, *, validator=None, evidence_getter=None, criteria_getter=None,
                 artifact_resolver=None, fault_hook=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "scene.sqlite"
        self.validator = validator
        self.evidence_getter = evidence_getter
        self.criteria_getter = criteria_getter
        self.artifact_resolver = artifact_resolver
        self.fault_hook = fault_hook
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT PRIMARY KEY, revision TEXT NOT NULL, head_seq INTEGER NOT NULL,
                    undo_json TEXT NOT NULL, redo_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS revisions (
                    scene_id TEXT NOT NULL, revision TEXT NOT NULL, path TEXT NOT NULL,
                    hash TEXT NOT NULL, parent_revision TEXT, operation TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY(scene_id,revision));
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY, scene_id TEXT NOT NULL, hash TEXT NOT NULL,
                    path TEXT NOT NULL, file_hash TEXT NOT NULL, state TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, build_hash TEXT);
                CREATE TABLE IF NOT EXISTS requests (
                    scene_id TEXT NOT NULL, request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL, PRIMARY KEY(scene_id,request_id));
                CREATE TABLE IF NOT EXISTS outbox (
                    scene_id TEXT NOT NULL, seq INTEGER NOT NULL, event_id TEXT UNIQUE NOT NULL,
                    payload_json TEXT NOT NULL, PRIMARY KEY(scene_id,seq));
                CREATE TABLE IF NOT EXISTS imports (
                    scene_id TEXT PRIMARY KEY, source_hash TEXT, imported_at TEXT NOT NULL);
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    def _hook(self, point):
        if self.fault_hook:
            self.fault_hook(point)

    def _immutable(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _json(value).encode("utf8")
        try:
            with path.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != body:
                _fail("STALE_ARTIFACT", "Immutable file already has different content", path=str(relative))
        return str(relative).replace("\\", "/"), hashlib.sha256(body).hexdigest()

    def _read_file(self, path, expected_hash):
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            _fail("FORBIDDEN_SCOPE", "Stored path leaves repository")
        try:
            body = resolved.read_bytes()
        except OSError as exc:
            _fail("STALE_ARTIFACT", "Immutable repository file is unavailable", path=path, reason=str(exc))
        if hashlib.sha256(body).hexdigest() != expected_hash:
            _fail("STALE_ARTIFACT", "Immutable repository file changed", path=path)
        return json.loads(body)

    @staticmethod
    def _scene(db, scene_id):
        row = db.execute("SELECT * FROM scenes WHERE scene_id=?", (scene_id,)).fetchone()
        if row is None:
            _fail("NOT_FOUND", "Scene does not exist", scene_id=scene_id)
        return row

    def _snapshot(self, db, scene_id, revision):
        row = db.execute("SELECT * FROM revisions WHERE scene_id=? AND revision=?", (scene_id, revision)).fetchone()
        if row is None:
            _fail("NOT_FOUND", "Revision does not exist", scene_id=scene_id, revision=revision)
        return self._read_file(row["path"], row["hash"])

    @staticmethod
    def _base(row, request):
        if row["revision"] != request["base_revision"] or row["head_seq"] != request["base_head_seq"]:
            _fail("STALE_REVISION", "Scene changed since this operation was authorized",
                  current_revision=row["revision"], current_head_seq=row["head_seq"])

    @staticmethod
    def _decision(request, context):
        decision = request["params"].get("decision_ref")
        if not decision or decision != context.decision_ref:
            _fail("FORBIDDEN_SCOPE", "Decision was not issued by this host context")

    @staticmethod
    def _authorize(context, instance_id, field):
        context.authorize(instance_id, field)

    @staticmethod
    def _semantics(scene):
        objects = scene["objects"]
        by_id = {obj["instance_id"]: obj for obj in objects}
        if len(by_id) != len(objects):
            _fail("INVALID_ARGUMENT", "Instance IDs must be unique")
        for obj in objects:
            q = obj["transform"]["rotation_xyzw"]
            if not math.isclose(sum(float(v) ** 2 for v in q), 1.0, abs_tol=1e-5):
                _fail("INVALID_ARGUMENT", "Rotation quaternion must be normalized", instance_id=obj["instance_id"])
            bounds = obj["local_bounds_m"]
            if any(a >= b for a, b in zip(bounds["min"], bounds["max"])):
                _fail("CONSTRAINT_VIOLATION", "Local bounds must have positive extent", instance_id=obj["instance_id"])
            seen = {obj["instance_id"]}
            support = obj["support_id"]
            while support is not None:
                if support not in by_id or support in seen:
                    _fail("CONSTRAINT_VIOLATION", "Support graph has a missing node or cycle", instance_id=obj["instance_id"])
                seen.add(support)
                support = by_id[support]["support_id"]

    @staticmethod
    def _replay(db, request):
        row = db.execute("SELECT * FROM requests WHERE scene_id=? AND request_id=?",
                         (request["scene_id"], request["request_id"])).fetchone()
        if row is None:
            return None
        if row["request_hash"] != canonical_hash(request):
            _fail("IDEMPOTENCY_CONFLICT", "Request ID already has different content")
        return json.loads(row["result_json"])

    @staticmethod
    def _save_request(db, request, result):
        db.execute("INSERT INTO requests VALUES (?,?,?,?)",
                   (request["scene_id"], request["request_id"], canonical_hash(request), _json(result)))

    async def get_request_result(self, scene_id, request_id):
        with self._db() as db:
            row = db.execute("SELECT result_json FROM requests WHERE scene_id=? AND request_id=?", (scene_id, request_id)).fetchone()
            return json.loads(row[0]) if row else None

    async def import_scene(self, snapshot, context=None, *, source_hash=None):
        """Import a T02-converted snapshot into a new scene, never overwrite one."""
        validate(snapshot, "SceneSnapshot")
        self._semantics(snapshot)
        scene_id, revision = snapshot["scene_id"], snapshot["revision"]
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT 1 FROM scenes WHERE scene_id=?", (scene_id,)).fetchone():
                    _fail("IDEMPOTENCY_CONFLICT", "Import cannot overwrite an existing scene")
                path, digest = self._immutable(Path("scenes") / scene_id / "revisions" / (revision + ".json"), snapshot)
                db.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?)", (scene_id, revision, path, digest, snapshot["parent_revision"], "import", utc_now()))
                db.execute("INSERT INTO scenes VALUES (?,?,?,?,?)", (scene_id, revision, 0, "[]", "[]"))
                db.execute("INSERT INTO imports VALUES (?,?,?)", (scene_id, source_hash, utc_now()))
                self._hook("import_before_db_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return {"head": {"revision": revision, "head_seq": 0}, "snapshot": copy.deepcopy(snapshot)}

    seed = import_scene

    async def inspect(self, request, context=None):
        validate(request, "ToolRequest")
        with self._db() as db:
            db.execute("BEGIN")
            row = self._scene(db, request["scene_id"])
            requested = request["params"]["revision"]
            revision = row["revision"] if requested == "head" else requested
            result = {"head": {"revision": row["revision"], "head_seq": row["head_seq"]},
                      "snapshot": self._snapshot(db, request["scene_id"], revision)}
            db.commit()
            return result

    def _new_candidate(self, db, draft, request, metadata, state="staged", artifacts=None):
        candidate_id = _id("cand_")
        digest = canonical_hash({"serializer": "canonical-json-v1", "scene": draft,
                                 "base_revision": request["base_revision"], "base_head_seq": request["base_head_seq"],
                                 "artifacts": artifacts or []})
        ref = {"candidate_id": candidate_id, "scene_id": request["scene_id"],
               "base_revision": request["base_revision"], "base_head_seq": request["base_head_seq"],
               "candidate_hash": digest, "state": state}
        snapshot = {"candidate": ref, "scene": draft}
        validate(snapshot, "CandidateSnapshot")
        path, file_hash = self._immutable(Path("scenes") / request["scene_id"] / "candidates" / (candidate_id + ".json"), snapshot)
        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)", (candidate_id, request["scene_id"], digest, path, file_hash, state, _json(metadata), None))
        return ref

    def _metadata(self, request, snapshot, target_fields, operation):
        params = request["params"]
        return {"operation": operation, "intent": params["intent"], "decision_ref": params["decision_ref"],
                "target_fields": target_fields, "allowed_object_ids": sorted(target_fields),
                "allowed_fields": sorted({field for fields in target_fields.values() for field in fields}),
                "constraints": params.get("constraints", {}), "evidence_id": params.get("evidence_id"),
                "artifact_refs": [], "base_revision": request["base_revision"], "base_snapshot": snapshot}

    async def prepare_patch(self, request, context):
        validate(request, "ToolRequest")
        self._decision(request, context)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                replay = self._replay(db, request)
                if replay is not None:
                    db.commit()
                    return replay
                row = self._scene(db, request["scene_id"])
                self._base(row, request)
                original = self._snapshot(db, request["scene_id"], row["revision"])
                draft = _draft(original)
                objects = {obj["instance_id"]: obj for obj in draft["objects"]}
                seen, target_fields, changed = set(), {}, False
                for edit in request["params"]["edits"]:
                    instance_id, field, value = edit["instance_id"], edit["field"], edit["value"]
                    if (instance_id, field) in seen:
                        _fail("INVALID_ARGUMENT", "Duplicate object/field in patch")
                    seen.add((instance_id, field))
                    if instance_id not in objects:
                        _fail("NOT_FOUND", "Patch target does not exist", instance_id=instance_id)
                    self._authorize(context, instance_id, field)
                    obj = objects[instance_id]
                    target = obj["transform"] if field.startswith("transform.") else obj
                    key = field.split(".")[-1]
                    changed |= target[key] != value
                    target[key] = copy.deepcopy(value)
                    # The host's field-specific grant is the explicit user override;
                    # retaining locked preserves the user's lock after the edit.
                    state = obj["field_state"][field]
                    state["source"] = "user" if context.actor == "user" else "model"
                    state["decision_ref"] = context.decision_ref
                    target_fields.setdefault(instance_id, []).append(field)
                if not changed:
                    _fail("NO_CHANGE", "Patch does not change a value")
                validate(draft, "SceneDraft")
                self._semantics(draft)
                metadata = self._metadata(request, original, target_fields, "patch")
                result = self._new_candidate(db, draft, request, metadata)
                self._save_request(db, request, result)
                self._hook("prepare_before_db_commit")
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    async def prepare_refinement(self, request, context):
        validate(request, "ToolRequest")
        self._decision(request, context)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                replay = self._replay(db, request)
                if replay is not None:
                    db.commit()
                    return replay
                row = self._scene(db, request["scene_id"])
                self._base(row, request)
                snapshot = self._snapshot(db, request["scene_id"], row["revision"])
                params = request["params"]
                target = params["target_id"]
                if target not in {obj["instance_id"] for obj in snapshot["objects"]}:
                    _fail("NOT_FOUND", "Refinement target does not exist")
                for field in params["allowed_fields"]:
                    self._authorize(context, target, field)
                metadata = self._metadata(request, snapshot, {target: params["allowed_fields"]}, "refine")
                result = self._new_candidate(db, _draft(snapshot), request, metadata)
                self._save_request(db, request, result)
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    async def prepare_extension(self, request, context):
        """Stage final authored additions plus exact asset-only replacements.

        Host-only contract: ``tool='extend_scene'`` with complete ``additions``
        SceneObjects and ``asset_replacements=[{instance_id, asset}]``. Each new
        identity needs ``scene.objects.add``; each replacement needs both asset
        fields. Existing transforms, bounds, support and capture cannot change.
        The builder may attach artifacts, but cannot rewrite this frozen draft.
        """
        validate_extension_request(request)
        self._decision(request, context)
        params = request['params']
        for obj in params['additions']:
            self._authorize(context, obj['instance_id'], 'scene.objects.add')
        for replacement in params['asset_replacements']:
            for field in ('asset.geometry', 'asset.materials'):
                self._authorize(context, replacement['instance_id'], field)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                replay = self._replay(db, request)
                if replay is not None:
                    db.commit()
                    return replay
                head = self._scene(db, request['scene_id'])
                self._base(head, request)
                original = self._snapshot(db, request['scene_id'], head['revision'])
                draft = _draft(original)
                objects = {obj['instance_id']: obj for obj in draft['objects']}
                additions, replacements, target_fields = {}, {}, {}
                for proposed in params['additions']:
                    identity = proposed['instance_id']
                    if identity in objects or identity in additions:
                        _fail('INVALID_ARGUMENT', 'Addition identity already exists or is repeated', instance_id=identity)
                    additions[identity] = copy.deepcopy(proposed)
                    target_fields[identity] = ['scene.objects.add']
                for proposed in params['asset_replacements']:
                    identity = proposed['instance_id']
                    if identity not in objects or identity in replacements:
                        _fail('INVALID_ARGUMENT', 'Asset replacement must name one distinct existing object', instance_id=identity)
                    obj = objects[identity]
                    if proposed['asset'] == obj['asset'] or proposed['asset']['asset_version'] == obj['asset']['asset_version']:
                        _fail('STALE_ARTIFACT', 'Extension replacement must use a new asset version', instance_id=identity)
                    obj['asset'] = copy.deepcopy(proposed['asset'])
                    for field in ('asset.geometry', 'asset.materials'):
                        obj['field_state'][field]['source'] = 'model'
                        obj['field_state'][field]['decision_ref'] = context.decision_ref
                    replacements[identity] = copy.deepcopy(obj['asset'])
                    target_fields[identity] = ['asset.geometry', 'asset.materials']
                draft['objects'].extend(copy.deepcopy(list(additions.values())))
                validate(draft, 'SceneDraft')
                self._semantics(draft)
                self._verify_artifacts([obj['asset']['artifact'] for obj in draft['objects']], request['scene_id'])
                metadata = self._metadata(request, original, target_fields, 'extension')
                metadata.update(exact_additions=additions, exact_asset_replacements=replacements,
                                expected_draft_hash=canonical_hash(draft),
                                constraints={'keep_root_transform': True, 'keep_support': True, 'keep_bounds': True})
                result = self._new_candidate(db, draft, request, metadata)
                self._save_request(db, request, result)
                self._hook('prepare_before_db_commit')
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    def _candidate(self, db, candidate_id, candidate_hash=None):
        row = db.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            _fail("NOT_FOUND", "Candidate does not exist")
        if candidate_hash is not None and row["hash"] != candidate_hash:
            _fail("STALE_ARTIFACT", "Candidate hash does not match")
        value = self._read_file(row["path"], row["file_hash"])
        value["candidate"]["state"] = row["state"]
        return row, value

    async def read_candidate(self, candidate_id, candidate_hash, context=None):
        with self._db() as db:
            return self._candidate(db, candidate_id, candidate_hash)[1]

    async def candidate_metadata(self, candidate_id, context=None):
        with self._db() as db:
            row, _ = self._candidate(db, candidate_id)
            return json.loads(row["metadata_json"])

    def _verify_artifacts(self, refs, scene_id):
        if self.artifact_resolver is None:
            _fail("TOOL_UNAVAILABLE", "No trusted artifact verifier is configured")
        for ref in refs:
            resolver = getattr(self.artifact_resolver, "resolve", self.artifact_resolver)
            path = Path(resolver(ref, scene_id=scene_id))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                _fail("STALE_ARTIFACT", "Build artifact is absent or changed", uri=ref["uri"])

    @staticmethod
    def _grant_metadata(metadata, context):
        if metadata["decision_ref"] != context.decision_ref:
            _fail("FORBIDDEN_SCOPE", "Candidate belongs to another user decision")
        for instance_id, fields in metadata["target_fields"].items():
            for field in fields:
                context.authorize(instance_id, field)

    async def attach_build(self, candidate_id, candidate_hash, build_output, context):
        validate(build_output, "BuildOutput")
        if build_output["input_fingerprint"] != candidate_hash:
            _fail("STALE_ARTIFACT", "Build was produced from another candidate")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row, candidate = self._candidate(db, candidate_id, candidate_hash)
                metadata = json.loads(row["metadata_json"])
                self._grant_metadata(metadata, context)
                self._verify_artifacts(build_output["artifact_refs"], row["scene_id"])
                build_hash = canonical_hash(build_output)
                cached = db.execute("SELECT candidate_id,hash FROM candidates WHERE build_hash=? AND scene_id=?", (canonical_hash([candidate_id, build_hash]), row["scene_id"])).fetchone()
                if cached:
                    result = self._candidate(db, cached["candidate_id"], cached["hash"])[1]["candidate"]
                    db.commit()
                    return result
                if row["state"] != "staged":
                    _fail("INVALID_ARGUMENT", "Only a staged candidate accepts a build")
                head = self._scene(db, row["scene_id"])
                self._base(head, candidate["candidate"])
                self._verify_artifacts(build_output["artifact_refs"], row["scene_id"])
                draft = candidate["scene"]
                objects = {obj["instance_id"]: obj for obj in draft["objects"]}
                updates = build_output["asset_updates"]
                if metadata['operation'] == 'extension':
                    if updates:
                        _fail('FORBIDDEN_SCOPE', 'Extension build cannot rewrite frozen additions or asset replacements')
                    if canonical_hash(draft) != metadata['expected_draft_hash']:
                        _fail('STALE_ARTIFACT', 'Extension draft differs from its frozen authored specification')
                    self._verify_artifacts([obj['asset']['artifact'] for obj in draft['objects']], row['scene_id'])
                if metadata["operation"] == "patch" and updates:
                    _fail("FORBIDDEN_SCOPE", "Whitebox compilation cannot replace object assets")
                if metadata["operation"] == "refine" and not updates:
                    _fail("BUILD_FAILED", "Refinement did not produce a new asset")
                seen = set()
                for update in updates:
                    target = update["instance_id"]
                    if target not in metadata["target_fields"] or target in seen:
                        _fail("FORBIDDEN_SCOPE", "Build changed an unauthorized or repeated object")
                    seen.add(target)
                    obj = objects[target]
                    if set(metadata["target_fields"][target]) != {"asset.geometry", "asset.materials"}:
                        _fail("TOOL_UNAVAILABLE", "Separate geometry/material verification is not configured; both asset fields must be authorized")
                    if update["asset"] == obj["asset"] or update["asset"]["asset_version"] == obj["asset"]["asset_version"]:
                        _fail("STALE_ARTIFACT", "Refinement reused an old asset version")
                    if metadata["constraints"].get("keep_bounds") and update["local_bounds_m"] != obj["local_bounds_m"]:
                        _fail("FORBIDDEN_SCOPE", "Refinement changed locked bounds")
                    if "asset.geometry" not in metadata["target_fields"][target] and update["local_bounds_m"] != obj["local_bounds_m"]:
                        _fail("FORBIDDEN_SCOPE", "Material refinement changed geometry bounds")
                    self._verify_artifacts([update["asset"]["artifact"]], row["scene_id"])
                    obj["asset"] = copy.deepcopy(update["asset"])
                    obj["local_bounds_m"] = copy.deepcopy(update["local_bounds_m"])
                    for field in metadata["target_fields"][target]:
                        obj["field_state"][field]["source"] = "model"
                        obj["field_state"][field]["decision_ref"] = metadata["decision_ref"]
                validate(draft, "SceneDraft")
                self._semantics(draft)
                metadata["artifact_refs"] = build_output["artifact_refs"]
                metadata["source_candidate_id"] = candidate_id
                result = self._new_candidate(db, draft, candidate["candidate"], metadata, "built", build_output["artifact_refs"])
                db.execute("UPDATE candidates SET build_hash=? WHERE candidate_id=?", (canonical_hash([candidate_id, build_hash]), result["candidate_id"]))
                self._hook("attach_before_db_commit")
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    async def _validation(self, request, context, candidate, metadata):
        if self.validator is None or self.evidence_getter is None or self.criteria_getter is None:
            _fail("TOOL_UNAVAILABLE", "Commit requires trusted validation, evidence and criteria services")
        report = await _invoke(self.validator.get_report, request["params"]["validation_id"], context)
        validate(report, "ValidationReport")
        ref = candidate["candidate"]
        if report["candidate_id"] != ref["candidate_id"] or report["candidate_hash"] != ref["candidate_hash"]:
            _fail("EVIDENCE_MISMATCH", "Validation report belongs to another candidate")
        evidence = await _invoke(self.evidence_getter, report["evidence_id"], context)
        validate(evidence, "EvidenceBundle")
        criteria = await _invoke(self.criteria_getter, report["criteria_id"], context)
        if report["evidence_hash"] != evidence["evidence_hash"] or report["criteria_hash"] != criteria["criteria_hash"]:
            _fail("EVIDENCE_MISMATCH", "Evidence or criteria changed after validation")
        draft = candidate["scene"]
        if (evidence["scene_id"] != draft["scene_id"] or evidence["scene_revision"] != ref["base_revision"]
                or evidence["capture_revision"] != draft["capture_revision"] or evidence["geometry_revision"] != draft["geometry_revision"]):
            _fail("EVIDENCE_MISMATCH", "Evidence does not describe the candidate's base capture")
        if not set(metadata["allowed_object_ids"]).issubset(evidence["target_ids"]):
            _fail("EVIDENCE_MISMATCH", "Evidence does not cover authorized objects")
        if metadata["evidence_id"] is not None and metadata["evidence_id"] != evidence["evidence_id"]:
            _fail("EVIDENCE_MISMATCH", "Refinement used different evidence")
        if report["intent"] != metadata["intent"] or criteria["intent"] != metadata["intent"]:
            _fail("CONSTRAINT_VIOLATION", "Validation intent differs from user decision")
        visual_ok = report["visual_pass"] is True or (report["visual_pass"] is None and criteria.get("allow_visual_null") is True)
        if not (report["technical_pass"] and report["committable"] and visual_ok):
            _fail("CONSTRAINT_VIOLATION", "Candidate has not passed required validation")
        return report

    def _head_event(self, db, request, result):
        scene_id = request["scene_id"]
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM outbox WHERE scene_id=?", (scene_id,)).fetchone()[0]
        event = {"contract_version": "video2scene/1.0", "event_id": result["event_id"], "scene_id": scene_id,
                 "seq": seq, "type": "scene.head_changed", "occurred_at": utc_now(), "request_id": request["request_id"], "data": result}
        validate(event, "SceneEvent")
        db.execute("INSERT INTO outbox VALUES (?,?,?,?)", (scene_id, seq, result["event_id"], _json(event)))

    async def commit(self, request, context):
        validate(request, "ToolRequest")
        self._decision(request, context)
        with self._db() as db:
            replay = self._replay(db, request)
            if replay is not None:
                return replay
            row, candidate = self._candidate(db, request["params"]["candidate_id"], request["params"]["candidate_hash"])
            metadata = json.loads(row["metadata_json"])
        self._grant_metadata(metadata, context)
        if candidate["candidate"]["scene_id"] != request["scene_id"]:
            _fail("FORBIDDEN_SCOPE", "Candidate belongs to another scene")
        if candidate["candidate"]["base_revision"] != request["base_revision"] or candidate["candidate"]["base_head_seq"] != request["base_head_seq"]:
            _fail("STALE_REVISION", "Commit token differs from candidate base")
        await self._validation(request, context, candidate, metadata)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                replay = self._replay(db, request)
                if replay is not None:
                    db.commit()
                    return replay
                head = self._scene(db, request["scene_id"])
                self._base(head, request)
                row, candidate = self._candidate(db, request["params"]["candidate_id"], request["params"]["candidate_hash"])
                if row["state"] not in ("built", "checked"):
                    _fail("CONSTRAINT_VIOLATION", "Only a built and validated candidate can commit")
                self._verify_artifacts(metadata["artifact_refs"], request["scene_id"])
                for obj in candidate["scene"]["objects"]:
                    self._verify_artifacts([obj["asset"]["artifact"]], request["scene_id"])
                revision = _id("rev_")
                snapshot = dict(candidate["scene"], revision=revision, parent_revision=head["revision"])
                validate(snapshot, "SceneSnapshot")
                path, digest = self._immutable(Path("scenes") / request["scene_id"] / "revisions" / (revision + ".json"), snapshot)
                self._hook("commit_after_artifacts")
                result = {"revision": revision, "previous_revision": head["revision"], "head_seq": head["head_seq"] + 1, "event_id": _id("evt_")}
                db.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?)", (request["scene_id"], revision, path, digest, head["revision"], metadata["operation"], utc_now()))
                undo = json.loads(head["undo_json"]) + [head["revision"]]
                db.execute("UPDATE scenes SET revision=?,head_seq=?,undo_json=?,redo_json='[]' WHERE scene_id=?", (revision, result["head_seq"], _json(undo), request["scene_id"]))
                db.execute("UPDATE candidates SET state='committed' WHERE candidate_id=?", (row["candidate_id"],))
                self._save_request(db, request, result)
                self._head_event(db, request, result)
                self._hook("commit_before_db_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
        self._hook("commit_after_db_commit")
        return result

    async def history(self, request, context=None):
        validate(request, "ToolRequest")
        action = request["params"]["action"]
        if action != "list":
            self._decision(request, context)
        with self._db() as db:
            db.execute("BEGIN" if action == "list" else "BEGIN IMMEDIATE")
            try:
                if action != "list":
                    replay = self._replay(db, request)
                    if replay is not None:
                        db.commit()
                        return replay
                row = self._scene(db, request["scene_id"])
                undo, redo = json.loads(row["undo_json"]), json.loads(row["redo_json"])
                if action == "list":
                    rows = db.execute("SELECT revision,parent_revision,operation,created_at FROM revisions WHERE scene_id=? ORDER BY rowid DESC LIMIT ?", (request["scene_id"], request["params"].get("limit", 100))).fetchall()
                    result = {"head": {"revision": row["revision"], "head_seq": row["head_seq"]}, "entries": [dict(r) for r in rows], "can_undo": bool(undo), "can_redo": bool(redo)}
                else:
                    self._base(row, request)
                    source, destination = (undo, redo) if action == "undo" else (redo, undo)
                    if not source:
                        _fail("NO_CHANGE", "No revision available for " + action)
                    revision = source.pop()
                    destination.append(row["revision"])
                    self._snapshot(db, request["scene_id"], revision)
                    result = {"revision": revision, "previous_revision": row["revision"], "head_seq": row["head_seq"] + 1, "event_id": _id("evt_")}
                    db.execute("UPDATE scenes SET revision=?,head_seq=?,undo_json=?,redo_json=? WHERE scene_id=?", (revision, result["head_seq"], _json(undo), _json(redo), request["scene_id"]))
                    self._save_request(db, request, result)
                    self._head_event(db, request, result)
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    async def read_outbox(self, scene_id, after_seq=0):
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload_json FROM outbox WHERE scene_id=? AND seq>? ORDER BY seq", (scene_id, after_seq))]
