from __future__ import annotations
import asyncio
import inspect
import re
import threading
import uuid
from pathlib import Path

from video2scene.contracts_v1 import CONTRACT_VERSION, ContractError, ExecutionContext, canonical_hash, validate
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.scene_tools import Dispatcher
from .store import HostStore


async def maybe(value):
    return await value if inspect.isawaitable(value) else value


class SceneHost:
    def __init__(self, root, repository, evidence, validator, backend=None, jobs=None, planner=None):
        self.root=Path(root).resolve(); self.repo=repository; self.evidence=evidence
        self.validator=validator; self.backend=backend; self.jobs=jobs; self.planner=planner
        self.store=HostStore(self.root); self.resolver=ArtifactResolver(self.root)
        self.dispatcher=Dispatcher(); self._command_threads={}; self._thread_lock=threading.Lock()
        for name,handler in {"inspect_scene":repository.inspect,"apply_scene_patch":repository.prepare_patch,
                             "commit_candidate":repository.commit,"revision_history":repository.history,
                             "select_evidence":evidence.select,"build_whitebox":self._build,
                             "render_compare":self._render,"validate_candidate":self._validate,
                             "refine_asset":self._refine,"job_control":self._job_control}.items():
            self.dispatcher.register(name,handler)

    def context(self, **kwargs):
        return ExecutionContext(storage_root=self.root, **kwargs)

    async def _build(self, request, context):
        if self.backend is None: raise ContractError("TOOL_UNAVAILABLE","Blender backend unavailable",retryable=True)
        p=request["params"]; candidate=await self.repo.read_candidate(p["candidate_id"],p["candidate_hash"],context)
        output=await self.backend.build(candidate,context)
        ref=await self.repo.attach_build(p["candidate_id"],p["candidate_hash"],output,context)
        return {"candidate":ref,"artifact_refs":output["artifact_refs"]}

    async def _render(self, request, context):
        if self.backend is None: raise ContractError("TOOL_UNAVAILABLE","Render backend unavailable",retryable=True)
        p=request["params"]; candidate=await self.repo.read_candidate(p["candidate_id"],p["candidate_hash"],context)
        req={"contract_version":CONTRACT_VERSION,"request_id":"ev_"+uuid.uuid4().hex,"scene_id":request["scene_id"],"tool":"select_evidence",
             "params":{"revision":candidate["candidate"]["base_revision"],"target_ids":[o["instance_id"] for o in candidate["scene"]["objects"]],"purpose":"validation","n":10}}
        evidence=await self.evidence.select(req,context)
        by_id={f["source_frame_id"]:f for f in evidence["frames"]}
        if any(fid not in by_id for fid in p["frame_ids"]): raise ContractError("FRAME_MISMATCH","Render requested frames missing from evidence")
        return await self.backend.render(candidate,evidence,p["view_mode"],context,frame_ids=p["frame_ids"])

    async def _validate(self, request, context):
        p=request["params"]; candidate=await self.repo.read_candidate(p["candidate_id"],p["candidate_hash"],context)
        evidence=await self.evidence.get(p["evidence_id"],context)
        criteria=await maybe(self.validator.get_criteria(p["criteria_id"],context))
        if criteria["intent"]!=p["intent"]: raise ContractError("INVALID_ARGUMENT","Validation intent does not match criteria")
        return await self.validator.validate(candidate,evidence,criteria,context)

    async def _refine(self, request, context):
        if self.jobs is None: raise ContractError("TOOL_UNAVAILABLE","Job service unavailable",retryable=True)
        return await self.jobs.submit(request,context)

    async def _job_control(self, request, context):
        if self.jobs is None: raise ContractError("TOOL_UNAVAILABLE","Job service unavailable",retryable=True)
        return await self.jobs.control(request,context)

    async def execute(self, request, context=None):
        context=context or self.context()
        if isinstance(request,dict) and request.get("contract_version")!=CONTRACT_VERSION:
            raise ContractError("UNSUPPORTED_CONTRACT","Unsupported contract version")
        validate(request,"ToolRequest")
        old,fresh=self.store.reserve_request(request)
        if old: return old
        if not fresh:
            # Scene transactions save their results atomically. Recover the original
            # result after a crash instead of inferring success from the current head.
            if request["tool"]=="refine_asset" and self.jobs is not None:
                original=await self.jobs.get_by_request(request["scene_id"],request["request_id"])
                status="accepted"
            else:
                original=await self.repo.get_request_result(request["scene_id"],request["request_id"])
                status="ok"
            if original is None: raise ContractError("RESOURCE_EXHAUSTED","Request pending or outcome uncertain; query original request",retryable=True)
            response={"contract_version":CONTRACT_VERSION,"request_id":request["request_id"],"scene_id":request["scene_id"],"tool":request["tool"],"status":status,"result":original,"error":None,"warnings":[]}
            validate(response,"ToolResponse")
        else:
            response=await self.dispatcher.execute(request,context)
        self.store.save_request(request,response)
        self.store.emit(request["scene_id"],"tool.completed",{"request_id":request["request_id"],"status":response["status"]},request["request_id"],"tool_"+canonical_hash([request["scene_id"],request["request_id"]]))
        await self.sync_outbox(request["scene_id"])
        return response

    async def sync_outbox(self, scene_id):
        for event in await maybe(self.repo.read_outbox(scene_id,0)):
            self.store.emit(scene_id,event.get("type","scene.head_changed"),event.get("data",event.get("payload")),event.get("request_id"),event.get("event_id"))
        if self.jobs is not None:
            for entry in await self.jobs.read_events(0):
                job=entry["job"]
                if job["scene_id"]==scene_id:
                    self.store.emit(scene_id,"job.updated",job,job["request_id"],"job_"+canonical_hash(job))

    def submit_command(self, command):
        record,fresh=self.store.accept_command(command)
        if fresh:
            self.store.emit(command["scene_id"],"command.accepted",record["receipt"])
            self._start_command(command)
        return record["receipt"]

    def _start_command(self, command):
        key=(command["scene_id"],command["command_id"])
        with self._thread_lock:
            if key in self._command_threads and self._command_threads[key].is_alive(): return
            thread=threading.Thread(target=lambda:asyncio.run(self.run_command(command)),daemon=True)
            self._command_threads[key]=thread; thread.start()

    def recover(self):
        for command in self.store.pending_commands(): self._start_command(command)

    async def run_command(self, command):
        scene=command["scene_id"]; cid=command["command_id"]; record=self.store.command(scene,cid)
        record["state"]="running"; self.store.save_command(scene,cid,record)
        decision="decision_"+canonical_hash([scene,cid])[:32]
        selected=set(command["selected_ids"]); context=self.context(actor="user",session_id=command["session_id"],decision_ref=decision)

        async def call(tool, params, suffix, mutate=False, ctx=None):
            req={"contract_version":CONTRACT_VERSION,"request_id":"req_"+canonical_hash([scene,cid,suffix])[:40],"scene_id":scene,"tool":tool,"params":params}
            if mutate: req.update(base_revision=command["base_revision"],base_head_seq=command["base_head_seq"])
            if req["request_id"] not in record["request_ids"]:
                record["request_ids"].append(req["request_id"]); self.store.save_command(scene,cid,record)
            result=await self.execute(req,ctx or context)
            if result["status"]=="error": raise ContractError(**result["error"])
            if result["status"]=="accepted":
                job_id=result["result"]["job_id"]
                while True:
                    status=await self.jobs.control({"contract_version":CONTRACT_VERSION,"request_id":"poll_"+uuid.uuid4().hex,"scene_id":scene,"tool":"job_control","params":{"job_id":job_id,"action":"status"}},ctx or context)
                    if status["state"]=="succeeded": return status["result"]
                    if status["state"] in ("failed","stale","cancelled"):
                        raise ContractError(**(status["error"] or {"code":"CANCELLED","message":"Job cancelled"}))
                    await asyncio.sleep(.2)
            return result["result"]

        try:
            edits=command.get("edits"); refinement=None; language_scope=None
            if command["kind"]=="language":
                text=command["text"].strip().lower()
                if text in ("undo","redo","撤销","重做"):
                    action="undo" if text in ("undo","撤销") else "redo"
                    await call("revision_history",{"action":action,"decision_ref":decision},action,True)
                    record.update(state="succeeded",message=action); return
                if self.planner is None: raise ContractError("TOOL_UNAVAILABLE","Language provider is not configured",retryable=True)
                view=await call("inspect_scene",{"revision":"head"},"inspect")
                if view["head"]!={"revision":command["base_revision"],"head_seq":command["base_head_seq"]}:
                    raise ContractError("STALE_REVISION","Scene changed before language planning")
                plan=await maybe(self.planner(command,view,context))
                if plan.get("action") in ("clarify","needs_clarification"):
                    record.update(state="needs_clarification",message=plan.get("question","Please clarify the requested edit"))
                    self.store.emit(scene,"agent.clarification",{"operation_id":record["receipt"]["operation_id"],"session_id":command["session_id"],"command_id":cid,"question":record["message"]})
                    return
                edits=plan.get("edits")
                if plan.get("action")=="refine": refinement=plan
                # A model proposal cannot itself grant access to locked fields.
                # The host derives a conservative field scope from the user text.
                language_scope=set()
                for field,pattern in {
                    "transform.scale":r"height|scale|size|width|bigger|smaller|taller|shorter|高|宽|大|小|尺寸|缩放",
                    "transform.translation_m":r"move|position|translate|left|right|向|位置|移动",
                    "transform.rotation_xyzw":r"rotat|turn|yaw|旋转|转动",
                    "category":r"category|label|类别|类型|命名",
                    "support_id":r"support|place on|放在|支撑",
                }.items():
                    if re.search(pattern,text):language_scope.add(field)
            if refinement:
                target=refinement["target_id"]
                if target not in selected: raise ContractError("FORBIDDEN_SCOPE","Refinement target was not selected")
                if not re.search(r"refin|remodel|detail|精修|精模|重建|细化|细节",command["text"],re.I):
                    raise ContractError("FORBIDDEN_SCOPE","User did not request asset refinement")
                if re.search(r"only|仅|只",command["text"],re.I):
                    raise ContractError("TOOL_UNAVAILABLE","Single-field asset refinement is not yet supported")
                fields=frozenset(("asset.geometry","asset.materials"))
                context=self.context(actor="user",session_id=command["session_id"],decision_ref=decision,allowed_object_ids=frozenset(selected),allowed_fields=fields)
                evidence=await call("select_evidence",{"revision":command["base_revision"],"target_ids":[target],"purpose":"refine","n":3},"evidence")
                ref=await call("refine_asset",{"target_id":target,"intent":command["intent"],"allowed_fields":sorted(fields),"evidence_id":evidence["evidence_id"],"constraints":{"keep_root_transform":True,"keep_support":True,"keep_bounds":True},"budget":{"max_iterations":1,"timeout_s":600,"max_model_calls":1},"decision_ref":decision},"refine",True)
            else:
                if not edits: raise ContractError("NO_CHANGE","Command contains no effective edit")
                targets={e["instance_id"] for e in edits}
                if not targets<=selected: raise ContractError("FORBIDDEN_SCOPE","Edit target was not selected")
                if language_scope is not None and not {e["field"] for e in edits}<=language_scope:
                    record.update(state="needs_clarification",message="Please specify which position, size, rotation, category or support field to change.")
                    self.store.emit(scene,"agent.clarification",{"operation_id":record["receipt"]["operation_id"],"session_id":command["session_id"],"command_id":cid,"question":record["message"]})
                    return
                context=self.context(actor="user",session_id=command["session_id"],decision_ref=decision,allowed_object_ids=frozenset(targets),allowed_fields=frozenset(e["field"] for e in edits))
                candidate=await call("apply_scene_patch",{"intent":command["intent"],"edits":edits,"decision_ref":decision},"patch",True)
                built=await call("build_whitebox",{"candidate_id":candidate["candidate_id"],"candidate_hash":candidate["candidate_hash"]},"build")
                ref=built["candidate"]
                evidence=await call("select_evidence",{"revision":command["base_revision"],"target_ids":sorted(targets),"purpose":"validation","n":3},"evidence")
            criteria="tabletop_design_v1" if command["intent"]=="design_edit" else "tabletop_reconstruction_v1"
            report=await call("validate_candidate",{"candidate_id":ref["candidate_id"],"candidate_hash":ref["candidate_hash"],"evidence_id":evidence["evidence_id"],"intent":command["intent"],"criteria_id":criteria},"validate")
            await call("commit_candidate",{"candidate_id":ref["candidate_id"],"candidate_hash":ref["candidate_hash"],"validation_id":report["validation_id"],"decision_ref":decision},"commit",True)
            if refinement:
                mode=getattr(self.backend,"refinement_mode","procedural_fallback")
                message={"image_recipe":"已保存图像建模方案生成的新资产；视觉质量以独立检查为准。",
                         "observed_surface":"已保存 RGB 推断的局部可见表面；缺失部分尚未补全，照片质量未验收。"}.get(mode,"已保存新的固定程序化资产；照片一致性和纹理质量尚未验证。")
            else:message="Scene saved"
            record.update(state="succeeded",message=message)
        except ContractError as exc:
            record.update(state="cancelled" if exc.code=="CANCELLED" else "failed",message=exc.message,error=exc.as_dict())
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error=ContractError("INTERNAL_ERROR",f"Command failed: {type(exc).__name__}")
            record.update(state="failed",message=error.message,error=error.as_dict())
        finally:
            self.store.save_command(scene,cid,record)
            self.store.emit(scene,"agent.finished",{"operation_id":record["receipt"]["operation_id"],"session_id":command["session_id"],"state":record["state"],"request_ids":record["request_ids"]})

