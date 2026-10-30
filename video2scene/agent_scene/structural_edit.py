"""Save the user's connected, orthogonal-wall correction through the v1 API."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import urllib.request
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from video2scene.capture_evidence import bounds_corners, transform_points, transform_to_matrix
from video2scene.contracts_v1 import canonical_hash, validate
from .planning import read, save

FEEDBACK='墙可以连在一起 而且怎么是歪的'
WALLS=('wall_tv','wall_sofa_color','wall_sofa_green')


def connected_wall_patch(snapshot):
    objects={obj['instance_id']:obj for obj in snapshot['objects']}
    if not set(WALLS+('floor',)) <= objects.keys():
        raise ValueError('This correction requires the three named observed Redwood walls and floor')
    angles=[]
    for identity in WALLS:
        matrix=Rotation.from_quat(objects[identity]['transform']['rotation_xyzw']).as_matrix()
        angles.append(np.arctan2(-matrix[2,0],matrix[0,0]))
    # Axes differing by 90/180 degrees vote for the same rectangular-room frame.
    yaw=float(np.angle(np.mean(np.exp(4j*np.array(angles))))/4)
    axes=Rotation.from_euler('y',yaw).as_matrix()
    furniture=[o for o in snapshot['objects'] if o['kind'] in ('object','support')]
    corners=np.concatenate([transform_points(transform_to_matrix(o['transform']),bounds_corners(o['local_bounds_m'])) for o in furniture])
    room_corners=corners@axes
    lo,hi=room_corners.min(0),room_corners.max(0)
    center={key:np.array(objects[key]['transform']['translation_m'])@axes for key in WALLS}
    thickness=.1;height=2.6;clearance=.04
    xmin=float(lo[0]-.10)
    xmax=float(max(center['wall_sofa_color'][0]-thickness/2,hi[0]+clearance))
    zmin=float(min(center['wall_tv'][2]+thickness/2,lo[2]-clearance))
    zmax=float(max(center['wall_sofa_green'][2]-thickness/2,hi[2]+clearance))
    if not (.5<xmax-xmin<15 and .5<zmax-zmin<15):raise ValueError('Implausible room envelope')
    # Butt joints share faces exactly; the side wall fills the space between
    # the inner faces of the two parallel walls. The window side stays open.
    proposals={
      'wall_tv':([(xmin+xmax+thickness)/2,height/2,zmin-thickness/2],
                 [xmax+thickness-xmin,height,thickness],yaw),
      'wall_sofa_green':([(xmin+xmax+thickness)/2,height/2,zmax+thickness/2],
                 [xmax+thickness-xmin,height,thickness],yaw),
      'wall_sofa_color':([xmax+thickness/2,height/2,(zmin+zmax)/2],
                 [zmax-zmin,height,thickness],yaw-np.pi/2),
      'floor':([(xmin+xmax+thickness)/2,-.025,(zmin+zmax)/2],
                 [xmax+thickness-xmin+.10,.05,zmax-zmin+2*thickness+.10],yaw),
    }
    edits=[];after=copy.deepcopy(snapshot)
    for identity,(center_local,dimensions,angle) in proposals.items():
        obj=objects[identity]
        old_dims=np.array(obj['local_bounds_m']['max'])-obj['local_bounds_m']['min']
        values={'translation_m':(np.array(center_local)@axes.T).tolist(),
                'rotation_xyzw':Rotation.from_euler('y',angle).as_quat().tolist(),
                'scale':(np.array(dimensions)/old_dims).tolist()}
        for name,value in values.items():
            edits.append({'instance_id':identity,'field':'transform.'+name,'value':value})
        next(o for o in after['objects'] if o['instance_id']==identity)['transform']=values
    validate(after,'SceneSnapshot')
    return edits,after,{'room_yaw_deg':float(np.degrees(yaw)),
        'interior_xz_bounds':[[xmin,zmin],[xmax,zmax]],'wall_thickness_m':thickness,
        'height_m':height,'junctions':['wall_tv / wall_sofa_color','wall_sofa_color / wall_sofa_green'],
        'source':'Explicit user instruction interpreted as a rectangular connected three-wall layout',
        'unchanged':['all furniture','frozen capture cameras','asset geometry','object local bounds'],
        'unknown_boundary':'Window/glass-door side remains open; no new wall or door location invented.'}


def api(base,path,value=None):
    request=urllib.request.Request(base+path,data=None if value is None else json.dumps(value).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8793')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();run=args.run_root.resolve();folder=run/'user_edits/walls_001'
    request={'contract_version':'video2scene/1.0','request_id':'structural-inspect-'+uuid.uuid4().hex,
        'scene_id':'redwood-agent','tool':'inspect_scene','params':{'revision':'head'}}
    response=api(args.url,'/api/v1/tools/execute',request)
    if response['status']=='error':raise RuntimeError(response['error'])
    view=response['result'];edits,after,diagnostics=connected_wall_patch(view['snapshot'])
    command={'contract_version':'video2scene/1.0','command_id':'human_walls_'+canonical_hash([view['head'],FEEDBACK])[:32],
        'session_id':'human_chat_wall_feedback','scene_id':'redwood-agent',
        'base_revision':view['head']['revision'],'base_head_seq':view['head']['head_seq'],
        'selected_ids':list(WALLS)+['floor'],'intent':'design_edit','kind':'patch','edits':edits}
    if (folder/'receipt.json').exists():raise RuntimeError('Wall edit already submitted; inspect recorded command before any new action')
    save(folder/'user_feedback.json',{'text':FEEDBACK,'channel':'Codex chat','interpretation':diagnostics})
    save(folder/'before.json',view);save(folder/'proposed.json',after);save(folder/'command.json',command)
    if not args.apply:
        print(json.dumps(diagnostics,ensure_ascii=False));return
    receipt=api(args.url,'/api/v1/commands',command);save(folder/'receipt.json',receipt)
    for _ in range(180):
        record=api(args.url,'/api/v1/commands/'+command['command_id']+'?scene_id=redwood-agent')
        if record['state'] in ('succeeded','failed','cancelled','needs_clarification'):
            save(folder/'result.json',record)
            if record['state']!='succeeded':raise RuntimeError(record.get('error') or record['message'])
            updated=api(args.url,'/api/v1/tools/execute',dict(request,request_id='structural-result-'+uuid.uuid4().hex))['result']
            save(folder/'after.json',updated)
            print(json.dumps({'state':record['state'],'head':updated['head'],'diagnostics':diagnostics},ensure_ascii=False));return
        time.sleep(.5)
    raise TimeoutError('Command outcome still pending; inspect original command, do not resubmit')


if __name__=='__main__':main()
