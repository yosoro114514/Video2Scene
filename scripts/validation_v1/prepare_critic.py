"""Prepare a correctly matched immutable review manifest, without external calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from video2scene.contracts_v1 import camera_hash,canonical_hash
from video2scene.contracts_v1.artifacts import ArtifactResolver


def prepare(run_root,output_name="critic_proposal_002.json"):
    run_root=Path(run_root).resolve()
    old=json.loads((run_root/"critic_proposal.json").read_text(encoding="utf8"))
    render=json.loads((run_root/"final_render_report.json").read_text(encoding="utf8"))
    evidence=json.loads((run_root/"final_evidence.json").read_text(encoding="utf8"))
    assert camera_hash(evidence["frames"],render["frame_ids"])==render["camera_hash"]
    resolver=ArtifactResolver(run_root/"runtime")
    refs=[ref for ref in render["artifact_refs"] if ref["media_type"]=="image/png"]
    mapped={resolver.resolve(ref).name.split("--",1)[0]:ref for ref in refs}
    if set(mapped)!=set(render["frame_ids"]):raise ValueError("Frame-to-file mapping is not unique")
    proposal={k:v for k,v in old.items() if k not in ("images","manifest_hash")}
    proposal["images"]=[image for image in old["images"] if image["kind"]=="source"]
    proposal["images"] += [{"kind":"render","frame_id":frame_id,"artifact":mapped[frame_id],"path":str(resolver.resolve(mapped[frame_id]))} for frame_id in render["frame_ids"]]
    proposal["supersedes_manifest_hash"]=old["manifest_hash"]
    proposal["preparation_note"]="Images explicitly matched by stable frame ID in the backend filename; artifact array order is not assumed. Prior proposal was not uploaded."
    proposal["manifest_hash"]=canonical_hash(proposal)
    target=run_root/output_name
    with target.open("x",encoding="utf8") as file:json.dump(proposal,file,ensure_ascii=False,indent=2)
    print(json.dumps({"path":str(target),"manifest_hash":proposal["manifest_hash"],"status":"prepared_not_uploaded","pairs":[{"frame_id":i["frame_id"],"filename":Path(i["path"]).name} for i in proposal["images"]]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run_root");args=parser.parse_args();prepare(args.run_root)
