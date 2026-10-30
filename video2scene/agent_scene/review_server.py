"""Human-confirmed Agent refinement service layered over the existing v1 API.

The confirmation is immutable, version-bound, and recorded before authoring.
Restarting the server never replays an uncertain external model operation.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit

from video2scene.api_v1.__main__ import create_host
from video2scene.api_v1.server import make_server
from video2scene.api_v1.service import SceneHost
from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, canonical_hash, file_hash, utc_now, validate
from video2scene.job_service import SQLiteJobService

IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
REFINEMENT = re.compile(r"refin|remodel|精修|精模|细化|细节|重建", re.I)
NUMERIC_FIELDS = frozenset(("transform.translation_m", "transform.rotation_xyzw", "transform.scale"))
CONFIRM_FIELDS = frozenset(("scene_id", "revision", "head_seq", "target_id"))
BATCH_ID = "batch_furniture_001"
BATCH_TARGET_COUNT = 10
ROOM_FINISH_ID = "room_finish_001"
ROOM_REPLACE_IDS = ("floor", "wall_tv", "wall_sofa_color", "wall_sofa_green")
ROOM_ADD_IDS = ("interior_door", "balcony_door", "vase_tv_stand")

def refinable_object(obj):
    return obj is not None and (obj['kind']=='object' or
        (obj['kind']=='support' and obj['category'].startswith('furniture/')))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class ReviewSceneHost(SceneHost):
    """Keeps generic editing, but reserves authoring for one confirmed command."""

    def __init__(self, *args, run_root, initial_snapshot, require_edit=True, **kwargs):
        self.run_root = Path(run_root).resolve()
        self.initial_snapshot = copy.deepcopy(initial_snapshot)
        self.scene_id = initial_snapshot["scene_id"]
        self.require_edit = require_edit
        self.confirmation_path = self.run_root / "confirmation.json"
        self._confirmation_lock = threading.RLock()
        self._event_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _batch_exists(self):
        root = self.run_root / BATCH_ID
        return (root / "authorization.json").exists() or (root / "status.json").exists()

    async def _batch_status(self, view):
        """Read the runner's immutable proofs; never create authorization or work."""
        root = self.run_root / BATCH_ID
        if not self._batch_exists():
            return None
        batch = {"batch_id": BATCH_ID, "phase": "invalid", "editing_locked": True,
                 "completed_count": 0, "total": BATCH_TARGET_COUNT, "targets": [],
                 "current_target_id": None, "layout_preserved": False}

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        def checked_record(record, directory):
            path = Path(record["path"]).resolve()
            require(path.is_relative_to(directory.resolve()) and path.is_file(), "证据文件缺失或路径不符")
            require(file_hash(path) == record["sha256"], "证据文件哈希不符")
            return path

        async def historical(revision):
            if view["snapshot"]["revision"] == revision:
                return view["snapshot"]
            result = await self.repo.inspect(self._request(self.scene_id, "inspect_scene", {"revision": revision}), self.context())
            require(result["snapshot"]["revision"] == revision, "场景历史版本不符")
            return result["snapshot"]

        try:
            auth_path = root / "authorization.json"
            auth = read_json(auth_path)
            auth_hash = file_hash(auth_path)
            require(auth.get("authorized") is True and auth.get("authorization_source") == "current_user_chat"
                    and auth.get("scene_id") == self.scene_id and auth.get("batch_id") == BATCH_ID, "批次授权不属于当前场景")
            ids = auth["target_ids"]
            require(isinstance(ids, list) and len(ids) == BATCH_TARGET_COUNT and len(set(ids)) == len(ids)
                    and all(isinstance(identity, str) and IDENTITY.fullmatch(identity) for identity in ids), "授权对象清单不完整")
            baseline_head = auth["baseline_head"]
            require(isinstance(baseline_head, dict) and type(baseline_head.get("head_seq")) is int
                    and baseline_head["head_seq"] >= 0 and isinstance(baseline_head.get("revision"), str), "授权基线版本无效")
            baseline_scene = await historical(baseline_head["revision"])
            require(canonical_hash(baseline_scene) == auth["baseline_snapshot_hash"], "授权基线与场景历史不一致")
            base_objects = {obj["instance_id"]: obj for obj in baseline_scene["objects"]}
            require(all(identity in base_objects and base_objects[identity]["kind"] not in ("wall", "floor", "ceiling") for identity in ids), "授权包含未知对象或房间结构")
            batch.update(baseline_head=baseline_head, authorization_verified=True,
                         targets=[{"target_id": identity, "state": "waiting"} for identity in ids])
            status_path = root / "status.json"
            if not status_path.exists():
                require(view["head"] == baseline_head and canonical_hash(view["snapshot"]) == auth["baseline_snapshot_hash"], "等待运行时场景已偏离授权基线")
                batch.update(phase="ready", layout_preserved=True, message="批量精修已获授权，等待运行器开始处理。")
                return batch
            from .batch_refinement import locked_layout
            baseline = read_json(root / "baseline.json")
            baseline_file = read_json(root / "baseline_snapshot.json")
            require(baseline["scene_id"] == self.scene_id and baseline["head"] == baseline_head
                    and baseline["authorization_sha256"] == auth_hash and baseline["target_ids"] == ids
                    and baseline["snapshot_hash"] == canonical_hash(baseline_scene) == canonical_hash(baseline_file), "批次基线证明不一致")
            layout_hash = canonical_hash(locked_layout(baseline_scene, ids))
            require(baseline["layout_hash"] == layout_hash, "锁定布局哈希不符")
            for key in ("capture_record", "initial_snapshot", "geometry_proposals"):
                checked_record(baseline[key], self.run_root)
            record = read_json(status_path)
            require(record["scene_id"] == self.scene_id and record["batch_id"] == BATCH_ID
                    and record["authorization_sha256"] == auth_hash and record["target_ids"] == ids
                    and record["baseline_head"] == baseline_head
                    and record["baseline_snapshot_hash"] == baseline["snapshot_hash"]
                    and record["baseline_layout_hash"] == layout_hash, "批次状态与授权或基线不一致")
            phase = record["phase"]
            require(phase in ("ready", "running", "complete", "failed", "needs_reconciliation"), "未知批次阶段")
            completed = record.get("completed", [])
            require(isinstance(completed, list) and len(completed) <= len(ids), "完成记录数量无效")
            done_ids = [entry["target_id"] for entry in completed]
            require(done_ids == ids[:len(done_ids)], "完成对象未按授权顺序逐个提交")
            require(record.get("progress") == {"completed": len(completed), "total": len(ids)}, "进度与完成记录不一致")
            last_head = baseline_head
            for entry in completed:
                require(entry["before_head"] == last_head and entry["head"]["head_seq"] == last_head["head_seq"] + 1, "对象提交版本链不连续")
                completion_path = checked_record({"path": entry["completion_path"], "sha256": entry["completion_sha256"]}, root)
                require(completion_path.name == "completion.json", "对象完成证据文件名不符")
                proof = read_json(completion_path)
                require(proof["status"] == "complete" and proof["target_id"] == entry["target_id"]
                        and proof["before_head"] == entry["before_head"] and proof["head"] == entry["head"]
                        and proof["asset"] == entry["asset"] and proof.get("locked_fields_unchanged") is True
                        and proof.get("non_targets_unchanged") is True and proof.get("camera_evidence_unchanged") is True,
                        "对象完成证据与保存资产不符")
                last_head = entry["head"]
            require(record["current_head"] == last_head, "批次当前版本与提交链不一致")
            if phase == "complete":
                require(len(completed) == len(ids), "尚未完成全部十个对象")
            # A completed batch remains historical evidence after a later manual
            # edit; only the exact final scene may be labelled current complete.
            batch_scene = view["snapshot"]
            changed_after = phase == "complete" and view["head"] != last_head
            if changed_after:
                require(view["head"]["head_seq"] > last_head["head_seq"], "场景当前版本早于批次完成版本")
                batch_scene = await historical(last_head["revision"])
            else:
                require(view["head"] == last_head, "保存版本正在同步，请等待下一次状态更新")
            require(canonical_hash(locked_layout(batch_scene, ids)) == layout_hash, "相连墙体、布局或非目标资产已改变")
            objects = {obj["instance_id"]: obj for obj in batch_scene["objects"]}
            for entry in completed:
                require(objects[entry["target_id"]]["asset"] == entry["asset"], "已完成对象与场景资产哈希不一致")
                self.resolver.resolve(entry["asset"]["artifact"], self.scene_id)
            for identity in ids[len(completed):]:
                require(objects[identity]["asset"] == base_objects[identity]["asset"], "未完成对象的资产已被更改")
            active = record.get("active_target_id")
            require(active is None or active in ids[len(completed):], "当前精修对象不在剩余清单")
            stage = record.get("stage", "")
            validating = stage in ("validating", "validating_geometry", "independent_review", "candidate_built", "committing")
            batch.update(phase="scene_changed" if changed_after else phase,
                         editing_locked=phase != "complete", completed_count=len(completed),
                         current_target_id=active, layout_preserved=not changed_after,
                         baseline_layout_hash=layout_hash, completed_head=last_head if phase == "complete" else None,
                         stage=stage, updated_at=record.get("updated_at", ""),
                         message=record.get("message") or "按已授权的顺序逐对象精修。")
            if changed_after:
                batch["message"] = "十个对象的批量精修已保存；当前场景在批次完成后又有修改。"
            for item in batch["targets"]:
                if item["target_id"] in done_ids:
                    item["state"] = "accepted"
                elif item["target_id"] == active:
                    item["state"] = ("failed" if phase == "failed" else "needs_reconciliation" if phase == "needs_reconciliation"
                                     else "validating" if validating else "refining")
                    item["message"] = record.get("message", "")
            return batch
        except (ValueError, OSError, KeyError, TypeError, ContractError, ImportError) as exc:
            batch.update(phase="invalid", editing_locked=True, message=f"批次记录校验未通过：{getattr(exc, 'message', str(exc))}")
            return batch

    def _batch_edit_guard(self):
        if self._batch_exists() or self._room_finish_exists():
            workflow = asyncio.run(self._current_workflow(self.current_view()))
            if workflow["editing_locked"]:
                raise ContractError("FORBIDDEN_SCOPE", "当前场景任务执行期间布局已锁定；完成后可继续编辑。")

    def _room_finish_exists(self):
        root = self.run_root / ROOM_FINISH_ID
        return (root / "authorization.json").exists() or (root / "status.json").exists()

    async def _current_workflow(self, view):
        return await self._room_finish_status(view) if self._room_finish_exists() else await self._batch_status(view)

    async def _room_finish_status(self, view):
        """Room finishing is a later single commit, independent of furniture history."""
        if not self._room_finish_exists():
            return None
        root = self.run_root / ROOM_FINISH_ID
        result = {"run_id": ROOM_FINISH_ID, "phase": "invalid", "editing_locked": True,
                  "completed_count": 0, "total": 7, "targets": [], "current_target_id": None,
                  "layout_preserved": False}

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        async def historical(revision):
            if view["snapshot"]["revision"] == revision:
                return view["snapshot"]
            old = await self.repo.inspect(self._request(self.scene_id, "inspect_scene", {"revision": revision}), self.context())
            require(old["snapshot"]["revision"] == revision, "历史场景版本不符")
            return old["snapshot"]

        try:
            auth_path = root / "authorization.json"
            auth, auth_hash = read_json(auth_path), file_hash(auth_path)
            require(auth.get("authorized") is True and auth.get("authorization_source") == "current_user_chat"
                    and auth.get("scene_id") == self.scene_id and auth.get("run_id") == ROOM_FINISH_ID
                    and isinstance(auth.get("user_instruction"), str) and auth["user_instruction"].strip()
                    and isinstance(auth.get("authorized_at"), str) and auth["authorized_at"], "房间任务授权不完整或不属于当前场景")
            ids, replacements, additions = auth["targets"], auth["replace_asset_ids"], auth["add_object_ids"]
            require(isinstance(ids, list) and len(ids) == 7 and set(ids) == set(ROOM_REPLACE_IDS + ROOM_ADD_IDS)
                    and len(replacements) == 4 and set(replacements) == set(ROOM_REPLACE_IDS)
                    and len(additions) == 3 and set(additions) == set(ROOM_ADD_IDS), "房间任务目标与本次授权范围不符")
            base_head = auth["baseline_head"]
            require(isinstance(base_head, dict) and isinstance(base_head.get("revision"), str)
                    and type(base_head.get("head_seq")) is int and base_head["head_seq"] >= 0, "房间任务基线版本无效")
            base = await historical(base_head["revision"])
            require(canonical_hash(base) == auth["baseline_snapshot_hash"], "房间任务基线快照与授权不符")
            require(read_json(root / "baseline_snapshot.json") == base, "房间任务保存的基线快照不符")
            capture = read_json(root / "capture_binding.json")
            capture_path = Path(capture["path"]).resolve()
            require(capture_path.is_relative_to(self.run_root) and capture_path.is_file()
                    and file_hash(capture_path) == capture["sha256"], "房间任务相机证据已改变")
            old_objects = {obj["instance_id"]: obj for obj in base["objects"]}
            require(all(identity in old_objects for identity in replacements)
                    and all(identity not in old_objects for identity in additions), "既有饰面目标或新增对象身份不符")
            result.update(baseline_head=base_head, authorization_verified=True,
                          targets=[{"target_id": identity, "state": "waiting"} for identity in ids])
            status_path = root / "status.json"
            if not status_path.exists():
                require(view["head"] == base_head and view["snapshot"] == base, "房间任务尚未开始但场景已改变")
                result.update(phase="ready", message="房间饰面与门 / 花瓶任务已授权，等待构建。", layout_preserved=True)
                return result
            record = read_json(status_path)
            require(record["scene_id"] == self.scene_id and record["run_id"] == ROOM_FINISH_ID
                    and record["authorization_sha256"] == auth_hash and record["baseline_head"] == base_head
                    and record["targets"] == ids, "房间任务状态与授权不一致")
            phase = record["phase"]
            require(phase in ("running", "complete", "failed"), "未知房间任务阶段")
            rows = record.get("completed", [])
            require(isinstance(rows, list) and len(rows) <= len(ids), "房间对象记录无效")
            done_ids = [row["instance_id"] for row in rows]
            require(len(set(done_ids)) == len(done_ids) and set(done_ids) <= set(ids)
                    and all(row["status"] in ("prepared", "accepted") for row in rows), "房间对象完成记录不符")
            active, stage = record.get("active_target"), record.get("stage", "")
            require(active is None or active in ids, "房间当前处理对象不在授权清单")
            result.update(phase=phase, current_target_id=active, stage=stage,
                          updated_at=record.get("updated_at", ""), message=record.get("message") or "房间饰面与门 / 花瓶处理中。")
            if phase != "complete":
                require(view["head"] == base_head == record["current_head"] and view["snapshot"] == base,
                        "房间提交正在同步或场景已改变，请核对当前版本")
                require(all(row["status"] == "prepared" for row in rows), "保存前不能把对象标为已完成")
                result["layout_preserved"] = True
            else:
                completion_path = root / "completion.json"
                require(record["completion_sha256"] == file_hash(completion_path), "房间完成证明哈希不符")
                proof = read_json(completion_path)
                require(proof["status"] == "complete" and proof["scene_id"] == self.scene_id
                        and proof["run_id"] == ROOM_FINISH_ID and proof["authorization_sha256"] == auth_hash
                        and proof["baseline_head"] == base_head and proof["head"] == record["current_head"], "房间完成证明与授权或版本不符")
                final_head = proof["head"]
                require(final_head["head_seq"] == base_head["head_seq"] + 1, "房间任务应当一次提交")
                final_path = root / "final_snapshot.json"
                require(file_hash(final_path) == proof["final_snapshot_sha256"], "房间最终快照文件哈希不符")
                final = read_json(final_path)
                require(final == await historical(final_head["revision"]) and final["parent_revision"] == base_head["revision"],
                        "房间最终快照与场景历史不一致")
                require(set(done_ids) == set(ids) and all(row["status"] == "accepted" for row in rows), "尚未保存全部房间对象")
                objects = {obj["instance_id"]: obj for obj in final["objects"]}
                require(set(objects) == set(old_objects) | set(additions), "场景对象清单超出本次新增范围")
                require({k: v for k, v in final.items() if k not in ("revision", "parent_revision", "objects")}
                        == {k: v for k, v in base.items() if k not in ("revision", "parent_revision", "objects")}, "场景相机或非目标元数据已改变")
                for identity, previous in old_objects.items():
                    current, expected = copy.deepcopy(objects[identity]), copy.deepcopy(previous)
                    if identity in replacements:
                        require(current["asset"] != expected["asset"], "房间目标没有生成新资产")
                        current.pop("asset"); expected.pop("asset")
                        for field in ("asset.geometry", "asset.materials"):
                            current["field_state"].pop(field, None); expected["field_state"].pop(field, None)
                    require(current == expected, "既有家具、根变换、边界或支撑关系已改变")
                require(proof["preservation"].get("existing_layout") is True and proof["preservation"].get("non_targets") is True,
                        "房间完成证明缺少保留布局与家具结果")
                require(set(proof["assets"]) == set(ids), "房间最终资产清单不完整")
                for identity in ids:
                    require(objects[identity]["asset"] == proof["assets"][identity], "房间目标资产与完成证明不符")
                    self.resolver.resolve(proof["assets"][identity]["artifact"], self.scene_id)
                changed = view["head"] != final_head
                require(not changed or view["head"]["head_seq"] > final_head["head_seq"], "当前场景版本早于房间任务完成版本")
                result.update(phase="scene_changed" if changed else "complete", editing_locked=False,
                              completed_count=len(ids), completed_head=final_head, layout_preserved=not changed)
                if changed:
                    result["message"] = "房间饰面、门与花瓶已保存；当前场景包含后续修改。"
            prepared = {row["instance_id"]: row["status"] for row in rows}
            validating = stage in ("validating", "independent_review", "critic", "validation", "committing")
            for item in result["targets"]:
                identity = item["target_id"]
                if identity in prepared:
                    item["state"] = prepared[identity]
                elif identity == active:
                    item["state"] = "failed" if phase == "failed" else "validating" if validating else "refining"
                    item["message"] = record.get("message", "")
            return result
        except (ValueError, OSError, KeyError, TypeError, ContractError) as exc:
            result.update(phase="invalid", editing_locked=True, message=f"房间任务记录校验未通过：{getattr(exc, 'message', str(exc))}")
            return result

    async def execute(self, request, context=None):
        if (self._batch_exists() or self._room_finish_exists()) and isinstance(request, dict):
            readonly = request.get("tool") in ("inspect_scene", "revision_history", "select_evidence")
            readonly = readonly or (request.get("tool") == "job_control" and request.get("params", {}).get("action") == "get")
            if not readonly:
                view = await self.repo.inspect(self._request(self.scene_id, "inspect_scene", {"revision": "head"}), self.context())
                if (await self._current_workflow(view))["editing_locked"]:
                    raise ContractError("FORBIDDEN_SCOPE", "场景任务执行期间布局已锁定。")
        return await super().execute(request, context)

    def _confirmation(self):
        if not self.confirmation_path.exists():
            return None
        try:
            value = read_json(self.confirmation_path)
            if (not isinstance(value, dict) or value.get("confirmed") is not True
                    or any(not isinstance(value.get(key), str) or not IDENTITY.fullmatch(value[key])
                           for key in ("scene_id", "revision", "target_id", "command_id", "session_id"))
                    or value["scene_id"] != self.scene_id or type(value.get("head_seq")) is not int
                    or value["head_seq"] < 0 or not isinstance(value.get("confirmed_at"), str) or not value["confirmed_at"]):
                raise ValueError("Invalid confirmation")
            return value
        except (ValueError, OSError) as exc:
            raise ContractError("RESOURCE_EXHAUSTED", "确认记录不完整，需要核对原执行结果；不会自动重试。",
                                {"outcome": "uncertain", "path": "confirmation.json"}) from exc

    @staticmethod
    def _request(scene, tool, params):
        return {"contract_version": CONTRACT_VERSION, "request_id": "review_read", "scene_id": scene,
                "tool": tool, "params": params}

    def current_view(self):
        return asyncio.run(self.repo.inspect(self._request(self.scene_id, "inspect_scene", {"revision": "head"}), self.context()))

    def submit_command(self, command):
        validate(command, "UserCommand")
        self._batch_edit_guard()
        if command["scene_id"] != self.scene_id:
            raise ContractError("FORBIDDEN_SCOPE", "该工作台只处理当前 Redwood 场景。")
        if command["kind"] == "language" and REFINEMENT.search(command["text"]):
            raise ContractError("FORBIDDEN_SCOPE", "请先保存白模修改，再通过 Agent 页面明确确认精修对象。")
        return super().submit_command(command)

    async def _refine(self, request, context):
        confirmation = self._confirmation()
        if confirmation is None:
            raise ContractError("FORBIDDEN_SCOPE", "精修需要用户确认当前白模版本。")
        command_id = confirmation["command_id"]
        expected_request = "req_" + canonical_hash([self.scene_id, command_id, "refine"])[:40]
        expected_decision = "decision_" + canonical_hash([self.scene_id, command_id])[:32]
        # Trust comes from host-created command context, never client arguments.
        if (context.actor != "user" or context.session_id != confirmation["session_id"]
                or context.decision_ref != expected_decision or request["request_id"] != expected_request
                or request["scene_id"] != self.scene_id or request["params"]["target_id"] != confirmation["target_id"]):
            raise ContractError("FORBIDDEN_SCOPE", "该精修请求不属于已经确认的用户操作。")
        token = {"revision": request.get("base_revision"), "head_seq": request.get("base_head_seq")}
        expected = {"revision": confirmation["revision"], "head_seq": confirmation["head_seq"]}
        view = await self.repo.inspect(self._request(self.scene_id, "inspect_scene", {"revision": "head"}), context)
        if token != expected or view["head"] != expected:
            raise ContractError("STALE_REVISION", "白模已改变，已确认的旧版本不能启动新的精修。")
        bounded = copy.deepcopy(request)
        bounded["params"]["budget"] = {"max_model_calls": 4, "max_iterations": 2, "timeout_s": 1200}
        validate(bounded, "ToolRequest")
        return await super()._refine(bounded, context)

    def human_corrections(self, snapshot):
        """Read actual persisted user commands and still-effective numeric edits."""
        base = {obj["instance_id"]: obj for obj in self.initial_snapshot["objects"]}
        changed = []
        for obj in snapshot["objects"]:
            before = base.get(obj["instance_id"])
            if before is None:
                continue
            for field in NUMERIC_FIELDS:
                key = field.split(".", 1)[1]
                if obj["transform"][key] != before["transform"][key]:
                    changed.append({"instance_id": obj["instance_id"], "field": field,
                                    "before": before["transform"][key], "after": obj["transform"][key]})
        history = []
        with self.store.connect() as db:
            rows = db.execute("SELECT command,record FROM commands WHERE scene=? ORDER BY rowid", (self.scene_id,)).fetchall()
        for raw, result in rows:
            command, record = json.loads(raw), json.loads(result)
            if record["state"] != "succeeded" or command["kind"] != "patch":
                continue
            edits = [edit for edit in command["edits"] if edit["field"] in NUMERIC_FIELDS]
            if edits:
                history.append({"command_id": command["command_id"], "base_revision": command["base_revision"],
                                "base_head_seq": command["base_head_seq"], "edits": edits})
        recorded = {(edit["instance_id"], edit["field"]) for item in history for edit in item["edits"]}
        effective = [edit for edit in changed if (edit["instance_id"], edit["field"]) in recorded]
        return {"saved_numeric_commands": history, "effective_numeric_changes": effective}

    def _existing_receipt(self, confirmation):
        try:
            record = self.store.command(self.scene_id, confirmation["command_id"])
            return dict(record["receipt"], status="accepted", duplicate=True, command_state=record["state"])
        except ContractError as exc:
            if exc.code != "NOT_FOUND":
                raise
            return {"status": "uncertain", "duplicate": True, "command_id": confirmation["command_id"],
                    "message": "确认已记录，但命令结果尚不明确；不会重新启动。"}

    def confirm(self, payload):
        if self._batch_exists() or self._room_finish_exists():
            raise ContractError("FORBIDDEN_SCOPE", "当前使用已授权的批量精修流程，原单对象确认记录保持不变。")
        if not isinstance(payload, dict) or set(payload) != CONFIRM_FIELDS:
            raise ContractError("INVALID_ARGUMENT", "确认只接受场景、版本、head_seq 和目标对象。")
        for key in ("scene_id", "revision", "target_id"):
            if not isinstance(payload[key], str) or not IDENTITY.fullmatch(payload[key]):
                raise ContractError("INVALID_ARGUMENT", f"Invalid {key}")
        if type(payload["head_seq"]) is not int or payload["head_seq"] < 0:
            raise ContractError("INVALID_ARGUMENT", "Invalid head_seq")
        if payload["scene_id"] != self.scene_id:
            raise ContractError("FORBIDDEN_SCOPE", "确认属于另一个场景。")
        with self._confirmation_lock:
            previous = self._confirmation()
            if previous is not None:
                if any(previous[key] != payload[key] for key in CONFIRM_FIELDS):
                    raise ContractError("IDEMPOTENCY_CONFLICT", "本次运行已确认另一个版本或对象；不会追加第二个精修任务。")
                return self._existing_receipt(previous)
            view = self.current_view()
            if view["head"] != {"revision": payload["revision"], "head_seq": payload["head_seq"]}:
                raise ContractError("STALE_REVISION", "场景已更新，请刷新并重新核对当前白模。")
            target = next((obj for obj in view["snapshot"]["objects"] if obj["instance_id"] == payload["target_id"]), None)
            if not refinable_object(target):
                raise ContractError("INVALID_ARGUMENT", "请选择一个家具或物体；房间结构不能作为本次精修对象。")
            if self.backend is None or self.jobs is None:
                raise ContractError("TOOL_UNAVAILABLE", "精修后端尚未配置。")
            if self.store.pending_commands():
                raise ContractError("RESOURCE_EXHAUSTED", "仍有未完成的工作台操作，请先等待保存完成。", retryable=True)
            corrections = self.human_corrections(view["snapshot"])
            if self.require_edit and (payload["head_seq"] <= 0 or not corrections["effective_numeric_changes"]):
                raise ContractError("FORBIDDEN_SCOPE", "请先在工作台实际修正并保存至少一项位置、旋转或缩放，再确认精修。")
            fingerprint = canonical_hash(payload)
            command_id, session_id = "confirm_" + fingerprint[:40], "human_" + fingerprint[:32]
            confirmation = dict(payload, confirmed=True, confirmed_at=utc_now(), command_id=command_id,
                                session_id=session_id, snapshot_hash=canonical_hash(view["snapshot"]),
                                human_corrections=corrections,
                                user_instruction="保持用户确认后的根变换、局部边界和支撑关系，仅精修所选对象的几何与材质。")
            authorization_path=self.run_root/'confirmation_authorization.json'
            if authorization_path.exists():
                authorization=read_json(authorization_path)
                if authorization.get('bound_confirmation_payload')==payload:
                    confirmation['authorization_context']={'path':authorization_path.name,
                        'canonical_sha256':canonical_hash(authorization),
                        'type':authorization.get('authorization_type'),
                        'executed_by':authorization.get('executed_by')}
            self.run_root.mkdir(parents=True, exist_ok=True)
            try:
                # Exclusive creation is the cross-process idempotency boundary.
                with self.confirmation_path.open("x", encoding="utf-8") as stream:
                    json.dump(confirmation, stream, ensure_ascii=False, allow_nan=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                existing = self._confirmation()
                if any(existing[key] != payload[key] for key in CONFIRM_FIELDS):
                    raise ContractError("IDEMPOTENCY_CONFLICT", "另一请求已确认不同版本。")
                return self._existing_receipt(existing)
            self.record_event("human_confirmed", "用户确认白模和精修对象。", revision=payload["revision"], target_id=payload["target_id"])
            command = {"contract_version": CONTRACT_VERSION, "command_id": command_id, "session_id": session_id,
                       "scene_id": self.scene_id, "base_revision": payload["revision"], "base_head_seq": payload["head_seq"],
                       "selected_ids": [payload["target_id"]], "intent": "design_edit", "kind": "language",
                       "text": "精修选中对象的几何与材质，保持根变换、支撑和局部边界。"}
            # Only this trusted path bypasses the public language-refinement gate.
            receipt = super().submit_command(command)
            self.jobs.start()
            return dict(receipt, status="accepted", duplicate=False)

    def record_event(self, stage, message, **details):
        event = {"time": utc_now(), "stage": stage, "message": message, **details}
        with self._event_lock:
            with (self.run_root / "review_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            _atomic_json(self.run_root / "review_progress.json", event)

    def backend_update(self, event):
        labels = {"authoring": "GPT 正在编写并执行对象建模脚本。", "validating_geometry": "正在检查新资产几何与边界。",
                  "independent_review": "GPT 正在独立复核源图与新资产渲染。", "candidate_built": "新资产候选已生成，等待宿主验证与提交。"}
        event = copy.deepcopy(event)
        stage = event.pop("stage", "refining")
        self.record_event(stage, labels.get(stage, stage), **event)

    async def run_command(self, command):
        await super().run_command(command)
        confirmation = self._confirmation()
        if not confirmation or command["command_id"] != confirmation["command_id"]:
            return
        record = self.store.command(self.scene_id, command["command_id"])
        if record["state"] == "succeeded":
            record["message"] = "已提交 GPT 新建的单对象精模，保留用户确认的布局与约束。"
            self.store.save_command(self.scene_id, command["command_id"], record)
        self.record_event(record["state"], record["message"] or record["state"], command_id=command["command_id"])

    def recover(self):
        """Resume local edits only; external authoring needs outcome reconciliation."""
        for command in self.store.pending_commands():
            if command["kind"] == "language" and REFINEMENT.search(command["text"]):
                record = self.store.command(command["scene_id"], command["command_id"])
                record.update(state="failed", message="服务曾中断，精修执行结果待核对；没有自动重新调用模型。",
                              error=ContractError("INTERNAL_ERROR", "External refinement outcome requires reconciliation",
                                                  {"outcome": "uncertain", "automatic_retry": False}).as_dict())
                self.store.save_command(command["scene_id"], command["command_id"], record)
            else:
                self._start_command(command)

    def review_status(self):
        view = self.current_view()
        corrections = self.human_corrections(view["snapshot"])
        path = self.run_root / "status.json"
        initial = read_json(path) if path.is_file() else {}
        response = {"scene_id": self.scene_id, "phase": initial.get("phase", "needs_user"),
                    "message": initial.get("message", "白模已载入，请在工作台修正、保存并确认。"),
                    "ready_for_review": initial.get("ready_for_review", True), "head": view["head"],
                    "confirmation": None, "require_edit": self.require_edit,
                    "human_edit_saved": view["head"]["head_seq"] > 0 and bool(corrections["effective_numeric_changes"]),
                    "refinable_target_ids": [obj["instance_id"] for obj in view["snapshot"]["objects"] if refinable_object(obj)],
                    "events": []}
        layout_path, prior_path = self.run_root / "layout.json", self.run_root / "prior.json"
        layout = read_json(layout_path) if layout_path.is_file() else {}
        prior = read_json(prior_path) if prior_path.is_file() else {}
        response['layout_summary'] = layout.get('summary','')
        response['uncertainties'] = layout.get('uncertainties', initial.get('uncertainties',[]))
        response['rejected_objects'] = layout.get('rejected_objects',[])
        recommended = (layout.get("recommended_refinement_target") or initial.get("recommended_refinement_target")
                       or prior.get("suggested_refinement_target"))
        if recommended in response["refinable_target_ids"]:
            response["recommended_target_id"] = recommended
        events_path = self.run_root / "review_events.jsonl"
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines()[-15:]:
                try:
                    response["events"].append(json.loads(line))
                except ValueError:
                    continue
        try:
            confirmation = self._confirmation()
            response["confirmation"] = confirmation
            if confirmation:
                response["ready_for_review"] = False
                try:
                    record = self.store.command(self.scene_id, confirmation["command_id"])
                except ContractError:
                    response.update(phase="failed", message="确认已记录，命令结果待核对；没有自动重新启动。",
                                    error={"message": "确认和命令之间发生中断，需要检查执行记录。", "outcome": "uncertain"})
                else:
                    response["command"] = record
                    if record["state"] == "succeeded":
                        response.update(phase="complete", message=record["message"])
                    elif record["state"] in ("failed", "cancelled", "needs_clarification"):
                        response.update(phase="failed", message=record["message"], error=record.get("error"))
                    else:
                        latest = response["events"][-1] if response["events"] else {}
                        validating = latest.get("stage") in ("validating_geometry", "independent_review", "candidate_built")
                        response.update(phase="validating" if validating else "refining",
                                        message=latest.get("message", "确认已接收，正在精修。"))
        except ContractError as exc:
            response.update(phase="failed", ready_for_review=False, message=exc.message, error=exc.as_dict())
        # A technical recovery keeps the original failed command immutable.
        # Only records bound to its exact confirmation may describe a new attempt.
        confirmation=response.get('confirmation')
        recovery_path=self.run_root/'recovery_status.json'
        if confirmation and recovery_path.exists():
            recovery=read_json(recovery_path)
            expected={'revision':confirmation['revision'],'head_seq':confirmation['head_seq']}
            matches=(recovery.get('confirmed_base')==expected
                and recovery.get('target_id')==confirmation['target_id']
                and recovery.get('confirmation_sha256')==file_hash(self.confirmation_path))
            if matches:
                response['recovery']=recovery
                response['original_attempt']=response.get('command')
                if recovery.get('phase') in ('refining','validating') and view['head']==expected:
                    response.update(phase=recovery['phase'],message=recovery['message'],ready_for_review=False)
                    response.pop('error',None)
                elif recovery.get('phase')=='complete':
                    completion_path=self.run_root/'recovery_completion.json'
                    if completion_path.exists():
                        completion=read_json(completion_path)
                        target=next(o for o in view['snapshot']['objects'] if o['instance_id']==confirmation['target_id'])
                        assets={ref['sha256'] for ref in completion.get('artifact_refs',[]) if ref.get('role')=='object_geometry'}
                        if (completion.get('status')=='complete' and completion.get('target_id')==confirmation['target_id']
                            and completion.get('head')==view['head'] and completion.get('confirmed_base')==expected
                            and completion.get('confirmation_sha256')==file_hash(self.confirmation_path)
                            and target['asset']['artifact']['sha256'] in assets):
                            response.update(phase='complete',message='已完成技术修复和独立复核，茶几精模已保存，墙体修正保持不变。',ready_for_review=False)
                            response.pop('error',None)
                elif recovery.get('phase')=='failed':
                    response.update(phase='failed',message=recovery['message'],error={'message':recovery['message']})
                response['events'].append({'time':recovery.get('updated_at',''),'message':recovery.get('message','恢复状态已更新')})
        batch = asyncio.run(self._batch_status(view))
        if batch is not None:
            response.update(batch=batch, ready_for_review=False, editing_locked=batch["editing_locked"], head=view["head"])
            response.pop("recommended_target_id", None)
            if batch["current_target_id"]:
                response["recommended_target_id"] = batch["current_target_id"]
            phase = batch["phase"]
            public_phase = ("complete" if phase == "complete" else "needs_user" if phase == "scene_changed"
                            else "failed" if phase in ("invalid", "failed", "needs_reconciliation")
                            else "validating" if batch.get("stage") in ("validating", "validating_geometry", "independent_review", "candidate_built", "committing")
                            else "refining")
            response.update(phase=public_phase, message=batch["message"])
            if public_phase == "failed":
                response["error"] = {"message": batch["message"]}
            else:
                response.pop("error", None)
            response["events"].append({"time": batch.get("updated_at", ""), "message": batch["message"]})
        room = asyncio.run(self._room_finish_status(view))
        if room is not None:
            response.update(room_finish=room, ready_for_review=False, editing_locked=room["editing_locked"], head=view["head"])
            if batch is not None:
                response["batch"]["history_only"] = True
            response.pop("recommended_target_id", None)
            if room["current_target_id"]:
                response["recommended_target_id"] = room["current_target_id"]
            phase = room["phase"]
            public_phase = ("complete" if phase == "complete" else "needs_user" if phase == "scene_changed"
                            else "failed" if phase in ("invalid", "failed")
                            else "validating" if room.get("stage") in ("validating", "independent_review", "critic", "validation", "committing")
                            else "refining")
            response.update(phase=public_phase, message=room["message"])
            if public_phase == "failed":
                response["error"] = {"message": room["message"]}
            else:
                response.pop("error", None)
            response["events"].append({"time": room.get("updated_at", ""), "message": room["message"]})
        return response


def create_review_host(run_root, blender=None, *, evidence_override=None, backend_override=None, require_edit=True):
    run_root = Path(run_root).resolve()
    snapshot = read_json(run_root / "snapshot.json")
    validate(snapshot, "SceneSnapshot")
    runtime = run_root / "runtime"
    base = create_host(runtime)  # Local repository/evidence/validator; no worker is started.
    if evidence_override is None:
        from .evidence import MaskEvidenceService
        evidence = MaskEvidenceService(runtime, base.repo)
    else:
        evidence = evidence_override(runtime, base.repo) if callable(evidence_override) else evidence_override
    base.validator.evidence_service = evidence
    base.repo.evidence_getter = evidence.get
    backend = backend_override
    if backend is None and blender:
        from .direct_asset import DirectAssetBackend
        backend = DirectAssetBackend(runtime, blender_path=blender, confirmation_path=run_root / "confirmation.json",
                                     observation_objects={obj["instance_id"]: obj for obj in snapshot["objects"]})
    host = ReviewSceneHost(runtime, base.repo, evidence, base.validator, backend=backend, planner=base.planner,
                           run_root=run_root, initial_snapshot=snapshot, require_edit=require_edit)
    try:
        host.current_view()
    except ContractError as exc:
        if exc.code != "NOT_FOUND":
            raise
        asyncio.run(host.repo.import_scene(snapshot, host.context(), source_hash=canonical_hash(snapshot)))
    if backend is not None:
        def on_update(job):
            host.store.emit(job["scene_id"], "job.updated", job, job["request_id"], "job_" + canonical_hash(job))
        host.jobs = SQLiteJobService(runtime, host.repo, evidence, backend, on_update=on_update, resource="cpu", autostart=False)
        if hasattr(backend, "status_callback"):
            backend.status_callback = host.backend_update
    return host


def make_review_server(host, port=8793, bind="127.0.0.1"):
    server = make_server(host, port, bind=bind)
    parent = server.RequestHandlerClass
    static_root = Path(__file__).resolve().parent

    class ReviewHandler(parent):
        def do_GET(self):
            url = urlsplit(self.path)
            if not url.path.startswith("/agent"):
                return super().do_GET()
            try:
                if url.path == "/agent/status":
                    requested = parse_qs(url.query).get("scene_id", [host.scene_id])[0]
                    if requested != host.scene_id:
                        raise ContractError("FORBIDDEN_SCOPE", "该服务不包含请求的场景。")
                    return self.json(host.review_status())
                paths = {"/agent/": "review.html", "/agent/review.html": "review.html",
                         "/agent/review.mjs": "review.mjs", "/agent/review.css": "review.css"}
                if url.path == "/agent":
                    self.send_response(302)
                    self.send_header("Location", "/agent/" + ("?" + url.query if url.query else ""))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if url.path not in paths:
                    raise ContractError("NOT_FOUND", "Agent route not found")
                return self.file(static_root / paths[url.path])
            except (BrokenPipeError, ConnectionResetError):
                return
            except ContractError as exc:
                return self.error(exc)

        def do_POST(self):
            if urlsplit(self.path).path != "/agent/confirm":
                return super().do_POST()
            try:
                response = host.confirm(self.body())
                self.json(response, 202)
            except ContractError as exc:
                self.error(exc)
            except (ValueError, TypeError):
                self.error(ContractError("INVALID_ARGUMENT", "确认请求格式无效。"))
            except Exception:
                import traceback
                traceback.print_exc()
                self.error(ContractError("INTERNAL_ERROR", "确认结果尚不明确，请查询原运行记录；不要重复启动。"))

    server.RequestHandlerClass = ReviewHandler
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--blender", type=Path, required=True)
    args = parser.parse_args()
    host = create_review_host(args.run_root, args.blender)
    host.recover()
    server = make_review_server(host, args.port)
    print(f"Agent whitebox review: http://127.0.0.1:{args.port}/agent/?scene_id={host.scene_id}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if host.jobs:
            host.jobs.shutdown(wait=True)


if __name__ == "__main__":
    main()
