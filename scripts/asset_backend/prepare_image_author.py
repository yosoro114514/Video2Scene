"""Offline by default. --invoke consumes the separately recorded proposal approval."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from video2scene.asset_backend.recipe_author import prepare_author_request,invoke_author_once
from video2scene.contracts_v1 import canonical_hash,file_hash,ContractError


def prepare(args):
    root=Path(args.storage_root).resolve()
    with sqlite3.connect((root/'scene.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute('SELECT r.path,r.hash FROM scenes s JOIN revisions r ON r.scene_id=s.scene_id AND r.revision=s.revision WHERE s.scene_id=?',(args.scene_id,)).fetchone()
    if row is None:raise ContractError('NOT_FOUND','Scene is not registered')
    scene_path=(root/row[0]).resolve()
    if not scene_path.is_relative_to(root) or file_hash(scene_path)!=row[1]:raise ContractError('STALE_ARTIFACT','Scene snapshot changed')
    scene=json.loads(scene_path.read_text(encoding='utf-8'))
    capture=json.loads((root/args.scene_id/'captures'/(scene['capture_revision']+'.json')).read_text(encoding='utf-8'))
    if canonical_hash({k:v for k,v in capture.items() if k!='capture_hash'})!=capture['capture_hash']:raise ContractError('STALE_ARTIFACT','Capture changed')
    wanted={'frame-000000','frame-000040','frame-000079'}
    evidence={'scene_id':args.scene_id,'scene_revision':scene['revision'],'capture_revision':scene['capture_revision'],
              'geometry_revision':scene['geometry_revision'],'target_ids':[o['instance_id'] for o in scene['objects']],
              'purpose':'refine','frames':[f for f in capture['frames'] if f['source_frame_id'] in wanted]}
    evidence['evidence_id']='author-'+canonical_hash(evidence)[:24];evidence['evidence_hash']=canonical_hash(evidence)
    manifest=prepare_author_request(scene,evidence,root,args.source_root,args.out)
    print(json.dumps({'proposal':str(Path(args.out).resolve()/'upload_proposal.json'),'model':manifest['model'],'budget':manifest['budget'],
                      'images':manifest['images'],'object_ids':manifest['object_ids'],'network_request_sent':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--storage-root');parser.add_argument('--scene-id',default='tabletop_v1')
    parser.add_argument('--source-root',default='outputs/uw_whitebox_loop_20261028');parser.add_argument('--out',required=True)
    parser.add_argument('--invoke',action='store_true')
    args=parser.parse_args()
    if args.invoke:print(invoke_author_once(args.out))
    else:
        if not args.storage_root:parser.error('--storage-root is required for offline preparation')
        prepare(args)
