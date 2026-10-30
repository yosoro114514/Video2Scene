import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video2scene.agent_scene.structural_edit import connected_wall_patch
from video2scene.capture_evidence import bounds_corners, transform_points, transform_to_matrix


def scene():
    examples=json.loads((Path(__file__).resolve().parents[2]/'docs/development/contracts/examples.json').read_text(encoding='utf-8'))
    snapshot=copy.deepcopy(next(row['value'] for row in examples['positive'] if row['name']=='model_SceneSnapshot'))
    source=snapshot['objects'][0];objects=[]
    rows=[('floor','floor',[0,-.025,1],[6,.05,7],0),
          ('wall_tv','wall',[0,1.3,-1],[5,2.6,.1],161),
          ('wall_sofa_color','wall',[2,1.3,1.5],[3.2,2.6,.1],-109),
          ('wall_sofa_green','wall',[0,1.3,4],[3.8,2.6,.1],-14),
          ('chair','object',[0,.5,1],[1,1,1],8)]
    for identity,kind,center,dims,yaw in rows:
        obj=copy.deepcopy(source);obj.update(instance_id=identity,kind=kind,support_id=None if kind=='floor' else 'floor')
        obj['transform']={'translation_m':center,'rotation_xyzw':Rotation.from_euler('y',yaw,degrees=True).as_quat().tolist(),'scale':[1,1,1]}
        obj['local_bounds_m']={'min':(-np.array(dims)/2).tolist(),'max':(np.array(dims)/2).tolist()}
        objects.append(obj)
    snapshot['objects']=objects
    return snapshot


def test_wall_correction_has_exact_right_angles_and_face_junctions():
    original=scene();edits,after,diagnostics=connected_wall_patch(original)
    before={o['instance_id']:o for o in original['objects']};objects={o['instance_id']:o for o in after['objects']}
    assert objects['chair']==before['chair']
    assert len(edits)==12
    axes=Rotation.from_euler('y',diagnostics['room_yaw_deg'],degrees=True).as_matrix()
    bounds={}
    for identity in ('wall_tv','wall_sofa_color','wall_sofa_green'):
        obj=objects[identity]
        points=transform_points(transform_to_matrix(obj['transform']),bounds_corners(obj['local_bounds_m']))@axes
        bounds[identity]=(points.min(0),points.max(0))
        assert points[:,1].min()==pytest.approx(0)
        assert points[:,1].max()==pytest.approx(2.6)
        assert obj['asset']==before[identity]['asset']
        assert obj['local_bounds_m']==before[identity]['local_bounds_m']
    tv,side,green=[bounds[key] for key in ('wall_tv','wall_sofa_color','wall_sofa_green')]
    assert tv[1][2]==pytest.approx(side[0][2])
    assert side[1][2]==pytest.approx(green[0][2])
    assert tv[1][0]==pytest.approx(side[1][0])
    assert green[1][0]==pytest.approx(side[1][0])
    tv_axis=Rotation.from_quat(objects['wall_tv']['transform']['rotation_xyzw']).as_matrix()[:,0]
    side_axis=Rotation.from_quat(objects['wall_sofa_color']['transform']['rotation_xyzw']).as_matrix()[:,0]
    assert tv_axis@side_axis==pytest.approx(0,abs=1e-12)
    assert after['capture_revision']==original['capture_revision']
    assert after['geometry_revision']==original['geometry_revision']
