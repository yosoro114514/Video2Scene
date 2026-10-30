"""Select the original observed masks for stable agent scene object identities."""
import copy
import inspect

import cv2
import numpy as np

from video2scene.capture_evidence.service import EvidenceService, check_frame
from video2scene.capture_evidence.storage import immutable_json, resolve_artifact
from video2scene.contracts_v1 import canonical_hash, validate, ContractError


class MaskEvidenceService(EvidenceService):
    async def select(self, request, context):
        validate(request,'ToolRequest')
        params=request['params'];scene_id=request['scene_id']
        result=self.repository.inspect({**{k:request[k] for k in ('contract_version','request_id','scene_id')},
            'tool':'inspect_scene','params':{'revision':params['revision']}},context)
        view=await result if inspect.isawaitable(result) else result
        snapshot=view['snapshot']
        targets=set(params['target_ids'])
        if not targets <= {o['instance_id'] for o in snapshot['objects']}:
            raise ContractError('NOT_FOUND','Requested evidence object absent')
        capture=self._capture(scene_id,snapshot['capture_revision'])
        if capture['geometry_revision'] != snapshot['geometry_revision']:
            raise ContractError('EVIDENCE_MISMATCH','Capture geometry revision differs from snapshot')
        frame_targets=capture['metadata'].get('frame_targets',{})
        rows=[]
        for frame in capture['frames']:
            if len(targets)==1 and frame_targets.get(frame['source_frame_id']) in targets:
                check_frame(frame,self.storage_root)
                mask=cv2.imread(str(resolve_artifact(self.storage_root,frame['mask'])),cv2.IMREAD_GRAYSCALE)
                rows.append((frame,float(np.count_nonzero(mask)/(mask.size))))
        if not rows:
            return await GenericEvidenceService(self.storage_root,self.repository).select(request,context)
        selected=[]
        while rows and len(selected)<params['n']:
            def score(row):
                position=np.array(row[0]['c2w'])[:3,3]
                diversity=min((np.linalg.norm(position-np.array(old['c2w'])[:3,3]) for old in selected),default=0.)
                return row[1]+min(float(diversity),2.)*.12
            best=max(range(len(rows)),key=lambda i:score(rows[i]))
            frame=copy.deepcopy(rows.pop(best)[0])
            frame['selection_reason']='Observed SAM3.1 target mask; exact original source frame and frozen ViPE camera, ranked by observed area and view diversity.'
            selected.append(frame)
        payload={'scene_id':scene_id,'scene_revision':snapshot['revision'],'capture_revision':snapshot['capture_revision'],
                 'geometry_revision':snapshot['geometry_revision'],'target_ids':params['target_ids'],'purpose':params['purpose'],'frames':selected}
        payload['evidence_id']='ev-'+canonical_hash(payload)[:32]
        payload['evidence_hash']=canonical_hash(payload)
        validate(payload,'EvidenceBundle')
        immutable_json(self.storage_root/scene_id/'evidence'/f"{payload['evidence_id']}.json",payload)
        return payload


class GenericEvidenceService(EvidenceService):
    """Project structure/multiple targets only into unmasked original frames."""
    def _capture(self, scene_id, capture_revision):
        capture=super()._capture(scene_id,capture_revision)
        target_frames=capture.get('metadata',{}).get('frame_targets',{})
        capture['frames']=[frame for frame in capture['frames']
            if frame['mask'] is None and frame['source_frame_id'] not in target_frames]
        return capture
