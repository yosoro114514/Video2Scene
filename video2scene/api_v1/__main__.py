import argparse
import asyncio
import json
from pathlib import Path
from video2scene.contracts_v1 import ContractError
from video2scene.contracts_v1.artifacts import ArtifactResolver
from video2scene.scene_state import SQLiteSceneRepository
from video2scene.capture_evidence import EvidenceService, migrate_prototype
from video2scene.validation_v1 import CandidateValidator
from .service import SceneHost
from .server import make_server


def create_host(root, blender=None, asset_recipes=None, *, observed_scan=None, observed_scene=None, observed_snapshot=None):
    observed=(observed_scan,observed_scene,observed_snapshot)
    if any(observed) and not all(observed):
        raise ValueError("Observed-surface mode requires scan, source scene and source snapshot paths")
    if any(observed) and asset_recipes:
        raise ValueError("Choose exactly one refinement source: image recipe or observed surface")
    if (asset_recipes or any(observed)) and not blender:
        raise ValueError("Configured refinement sources require an explicit Blender executable")
    root=Path(root).resolve(); resolver=ArtifactResolver(root)
    repo=SQLiteSceneRepository(root,artifact_resolver=resolver.resolve)
    evidence=EvidenceService(root,repo)
    validator=CandidateValidator(root,repository=repo,evidence_service=evidence,artifact_resolver=resolver.resolve)
    repo.validator=validator; repo.evidence_getter=evidence.get; repo.criteria_getter=validator.get_criteria
    backend=None
    if blender:
        from video2scene.asset_backend import BlenderAssetBackend
        kwargs={}
        if asset_recipes:
            from video2scene.asset_backend import FileRecipeProvider
            kwargs={"recipe_provider":FileRecipeProvider(asset_recipes),"refinement_mode":"image_recipe"}
        if all(observed):
            from video2scene.asset_backend import ObservedSurfaceBackend
            backend=ObservedSurfaceBackend(root,observed_scan,observed_scene,observed_snapshot,blender_path=blender)
        else:
            backend=BlenderAssetBackend(root,blender_path=blender,**kwargs)
    from video2scene.agent_runtime import DeterministicPlanner
    host=SceneHost(root,repo,evidence,validator,backend=backend,planner=DeterministicPlanner())
    if backend is not None:
        from video2scene.job_service import SQLiteJobService
        from video2scene.contracts_v1 import canonical_hash
        def on_update(job):
            host.store.emit(job["scene_id"],"job.updated",job,job["request_id"],"job_"+canonical_hash(job))
        host.jobs=SQLiteJobService(root,repo,evidence,backend,on_update=on_update,resource="cpu")
    return host


def main():
    p=argparse.ArgumentParser(description="Independent local v1 scene host")
    p.add_argument("--root",type=Path,required=True);p.add_argument("--scene-id",default="tabletop_v1")
    p.add_argument("--import-prototype",type=Path);p.add_argument("--revision",default="r005")
    p.add_argument("--blender",type=Path);p.add_argument("--port",type=int,default=8791)
    p.add_argument("--asset-recipes",type=Path,help="Validated image-author recipe bundle; never falls back to fixed recipes")
    p.add_argument("--observed-scan",type=Path,help="RGB-inferred colored scan for partial surface refinement")
    p.add_argument("--observed-scene",type=Path,help="Source reconstruction scene.json bound to the scan")
    p.add_argument("--observed-snapshot",type=Path,help="Frozen v1 snapshot defining original observed object poses")
    p.add_argument("--execute",type=Path,help="Execute a ToolRequest JSON through the same dispatcher")
    args=p.parse_args();host=create_host(args.root,args.blender,args.asset_recipes,observed_scan=args.observed_scan,observed_scene=args.observed_scene,observed_snapshot=args.observed_snapshot)
    if args.import_prototype:
        draft=migrate_prototype(args.import_prototype,args.root,args.scene_id,revision=args.revision)
        snapshot={**draft,"revision":"import_"+args.revision,"parent_revision":None}
        asyncio.run(host.repo.import_scene(snapshot,host.context()))
    if args.execute:
        result=asyncio.run(host.execute(json.loads(args.execute.read_text(encoding="utf-8"))))
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    host.recover();server=make_server(host,args.port)
    print(f"Scene editor: http://127.0.0.1:{args.port}/?scene_id={args.scene_id}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if host.jobs:host.jobs.shutdown(wait=True)


if __name__=="__main__":main()

