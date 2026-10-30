"""Durable one-worker queue. A completed build never commits scene head."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from video2scene.contracts_v1 import ContractError, ExecutionContext, canonical_hash, utc_now, validate

TERMINAL = frozenset(("succeeded", "failed", "cancelled", "stale"))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def _alive(pid):
    """Read-only process check; never use Windows os.kill(signal=0)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied means do not steal its lease.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class SQLiteJobService:
    def __init__(self, root, repository, evidence_service, backend, *, on_update=None,
                 resource="cpu", autostart=True, lease_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "job.sqlite"
        self.lease_path = Path(lease_root or self.root).resolve() / "resource_leases.sqlite"
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        self.repository, self.evidence_service, self.backend = repository, evidence_service, backend
        self.on_update, self.resource = on_update, resource
        self.owner = "worker_" + uuid4().hex
        self.stop_event, self.wake_event = threading.Event(), threading.Event()
        self._cancel = {}
        self._thread = None
        self._worker_lock = threading.Lock()
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, scene_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, request_json TEXT NOT NULL, context_json TEXT NOT NULL,
                    record_json TEXT NOT NULL, receipt_json TEXT NOT NULL, owner TEXT, owner_pid INTEGER,
                    phase TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(scene_id,request_id));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, record_json TEXT NOT NULL);
            """)
        with self._db(self.lease_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS leases (resource TEXT PRIMARY KEY, owner TEXT NOT NULL, pid INTEGER NOT NULL, acquired_at TEXT NOT NULL)")
        self._recover()
        if autostart:
            self.start()

    @contextmanager
    def _db(self, path=None):
        db = sqlite3.connect(path or self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _ctx_data(ctx):
        return {"actor": ctx.actor, "session_id": ctx.session_id, "decision_ref": ctx.decision_ref,
                "allowed_object_ids": sorted(ctx.allowed_object_ids), "allowed_fields": sorted(ctx.allowed_fields),
                "storage_root": str(ctx.storage_root)}

    @staticmethod
    def _ctx(data, event):
        return ExecutionContext(actor=data["actor"], session_id=data["session_id"], decision_ref=data["decision_ref"],
            allowed_object_ids=frozenset(data["allowed_object_ids"]), allowed_fields=frozenset(data["allowed_fields"]),
            storage_root=Path(data["storage_root"]), cancel_event=event)

    def _save(self, db, record, phase=None):
        validate(record, "JobRecord")
        if phase is None:
            db.execute("UPDATE jobs SET record_json=? WHERE job_id=?", (_json(record), record["job_id"]))
        else:
            db.execute("UPDATE jobs SET record_json=?,phase=? WHERE job_id=?", (_json(record), phase, record["job_id"]))
        db.execute("INSERT INTO events(job_id,record_json) VALUES (?,?)", (record["job_id"], _json(record)))

    def _notify(self, record):
        if self.on_update is not None:
            try:
                self.on_update(record)
            except Exception:
                # Durable events remain available for host replay after callback failure.
                pass

    def _recover(self):
        recovered = []
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM jobs").fetchall():
                record = json.loads(row["record_json"])
                if record["state"] not in ("running", "cancel_requested") or (row["owner_pid"] and _alive(row["owner_pid"])):
                    continue
                record.update(state="failed", result=None, error=ContractError("INTERNAL_ERROR",
                    "Worker stopped with an uncertain outcome; do not resubmit automatically",
                    {"outcome": "uncertain", "phase": row["phase"], "reconciliation_required": True}).as_dict())
                self._save(db, record, "recovery_uncertain")
                recovered.append(record)
            db.commit()
        for record in recovered:
            self._notify(record)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name=self.owner, daemon=True)
        self._thread.start()

    def shutdown(self, wait=True):
        self.stop_event.set()
        self.wake_event.set()
        if wait and self._thread:
            self._thread.join(timeout=10)
        return not self._thread or not self._thread.is_alive()

    def _loop(self):
        while not self.stop_event.is_set():
            if not self.run_next():
                self.wake_event.wait(.2)
                self.wake_event.clear()

    def _lease(self):
        with self._db(self.lease_path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM leases WHERE resource=?", (self.resource,)).fetchone()
            if row and row["owner"] != self.owner and _alive(row["pid"]):
                db.rollback()
                return False
            db.execute("INSERT OR REPLACE INTO leases VALUES (?,?,?,?)", (self.resource, self.owner, os.getpid(), utc_now()))
            db.commit()
            return True

    def _release(self):
        with self._db(self.lease_path) as db:
            db.execute("DELETE FROM leases WHERE resource=? AND owner=?", (self.resource, self.owner))

    async def submit(self, request, context):
        validate(request, "ToolRequest")
        if request["tool"] != "refine_asset":
            raise ContractError("TOOL_UNAVAILABLE", "This worker currently schedules only asset refinement")
        if request["params"]["decision_ref"] != context.decision_ref:
            raise ContractError("FORBIDDEN_SCOPE", "Decision does not match host context")
        for field in request["params"]["allowed_fields"]:
            context.authorize(request["params"]["target_id"], field)
        context.check_cancelled()
        job_id = "job_" + uuid4().hex
        receipt = {"job_id": job_id, "state": "queued"}
        record = {"job_id": job_id, "scene_id": request["scene_id"], "base_revision": request["base_revision"],
            "base_head_seq": request["base_head_seq"], "request_id": request["request_id"], "state": "queued",
            "progress": 0, "result": None, "error": None}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE scene_id=? AND request_id=?", (request["scene_id"], request["request_id"])).fetchone()
            if old:
                if old["request_hash"] != canonical_hash(request):
                    db.rollback()
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Job request ID already has other content")
                db.commit()
                return json.loads(old["receipt_json"])
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (job_id, request["scene_id"], request["request_id"],
                canonical_hash(request), _json(request), _json(self._ctx_data(context)), _json(record), _json(receipt), None, None, "queued", utc_now()))
            self._save(db, record)
            db.commit()
        self._notify(record)
        self.wake_event.set()
        return receipt

    async def get_by_request(self, scene_id, request_id):
        with self._db() as db:
            row = db.execute("SELECT receipt_json FROM jobs WHERE scene_id=? AND request_id=?", (scene_id, request_id)).fetchone()
            return json.loads(row[0]) if row else None

    async def control(self, request, context=None):
        validate(request, "ToolRequest")
        job_id = request["params"]["job_id"]
        changed = False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE job_id=? AND scene_id=?", (job_id, request["scene_id"])).fetchone()
            if row is None:
                db.rollback()
                raise ContractError("NOT_FOUND", "Job does not exist in this scene")
            record = json.loads(row["record_json"])
            if request["params"]["action"] == "cancel" and record["state"] not in TERMINAL:
                if record["state"] == "queued":
                    record.update(state="cancelled", error=ContractError("CANCELLED", "Cancelled before execution").as_dict())
                else:
                    record.update(state="cancel_requested")
                self._save(db, record)
                changed = True
            db.commit()
        if changed:
            event = self._cancel.get(job_id)
            if event:
                event.set()
            self._notify(record)
        return record

    async def read_events(self, after_seq=0):
        with self._db() as db:
            return [{"seq": row["seq"], "job": json.loads(row["record_json"])} for row in db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq", (after_seq,))]

    def _record(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return row, json.loads(row["record_json"])

    def _phase(self, job_id, phase, progress):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT record_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            record = json.loads(row[0])
            record["progress"] = progress
            self._save(db, record, phase)
            db.commit()
        self._notify(record)

    def run_next(self):
        if not self._worker_lock.acquire(blocking=False):
            return False
        acquired = False
        try:
            acquired = self._lease()
            if not acquired:
                return False
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                row = next((r for r in db.execute("SELECT * FROM jobs ORDER BY rowid") if json.loads(r["record_json"])["state"] == "queued"), None)
                if row is None:
                    db.commit()
                    return False
                record = json.loads(row["record_json"])
                record.update(state="running", progress=.05)
                db.execute("UPDATE jobs SET owner=?,owner_pid=? WHERE job_id=?", (self.owner, os.getpid(), record["job_id"]))
                self._save(db, record, "preparing")
                db.commit()
            event = threading.Event()
            self._cancel[record["job_id"]] = event
            context = self._ctx(json.loads(row["context_json"]), event)
            self._notify(record)
            asyncio.run(self._execute(json.loads(row["request_json"]), context, record["job_id"]))
            return True
        finally:
            if acquired:
                self._release()
            self._worker_lock.release()

    async def _execute(self, request, context, job_id):
        result, error, state = None, None, "failed"
        timed_out = False
        try:
            if self._record(job_id)[1]["state"] == "cancel_requested":
                context.cancel_event.set()
            context.check_cancelled()
            candidate = await self.repository.prepare_refinement(request, context)
            frozen = await self.repository.read_candidate(candidate["candidate_id"], candidate["candidate_hash"], context)
            meta = await self.repository.candidate_metadata(candidate["candidate_id"], context)
            getter = getattr(self.evidence_service, "get", self.evidence_service)
            evidence = await _await(getter(request["params"]["evidence_id"], context))
            validate(evidence, "EvidenceBundle")
            if (evidence["scene_id"] != request["scene_id"] or evidence["scene_revision"] != request["base_revision"]
                    or request["params"]["target_id"] not in evidence["target_ids"]):
                raise ContractError("EVIDENCE_MISMATCH", "Refinement evidence does not match frozen target/base")
            self._phase(job_id, "backend_running", .15)
            spec = dict(request["params"], candidate=candidate, input_fingerprint=candidate["candidate_hash"])
            task = asyncio.create_task(self.backend.refine(spec, meta["base_snapshot"], evidence, context))
            deadline = time.monotonic() + request["params"]["budget"]["timeout_s"]
            while not task.done():
                if self._record(job_id)[1]["state"] == "cancel_requested":
                    context.cancel_event.set()
                if time.monotonic() > deadline and not timed_out:
                    timed_out = True
                    context.cancel_event.set()
                    await self.control({"contract_version": "video2scene/1.0", "request_id": "timeout_" + job_id,
                        "scene_id": request["scene_id"], "tool": "job_control", "params": {"job_id": job_id, "action": "cancel"}}, context)
                await asyncio.sleep(.05)
            output = await task  # Child termination must be acknowledged before terminal cancellation.
            context.check_cancelled()
            self._phase(job_id, "attaching", .85)
            result = await self.repository.attach_build(candidate["candidate_id"], candidate["candidate_hash"], output, context)
            state = "succeeded"
        except ContractError as exc:
            state = "cancelled" if exc.code == "CANCELLED" else "stale" if exc.code == "STALE_REVISION" else "failed"
            error = exc.as_dict()
        except asyncio.CancelledError:
            state = "cancelled"
            error = ContractError("CANCELLED", "Backend coroutine confirmed cancellation").as_dict()
        except Exception as exc:
            error = ContractError("INTERNAL_ERROR", "Refinement worker failed", {"exception": type(exc).__name__, "message": str(exc)}).as_dict()
        finally:
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT record_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                record = json.loads(row[0])
                if record["state"] not in TERMINAL:
                    if timed_out:
                        state, result, error = "failed", None, ContractError("TIMEOUT", "Refinement stopped after timeout", retryable=True).as_dict()
                    elif record["state"] == "cancel_requested":
                        state, result, error = "cancelled", None, ContractError("CANCELLED", "Worker confirmed execution has stopped").as_dict()
                    record.update(state=state, progress=1 if state == "succeeded" else record["progress"], result=result if state == "succeeded" else None, error=error)
                    self._save(db, record, "terminal")
                db.commit()
            self._cancel.pop(job_id, None)
            self._notify(record)
