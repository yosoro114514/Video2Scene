import json
from pathlib import Path

import pytest

from video2scene.pipeline import Runner, make_recipe


def test_failed_stage_is_never_cached(tmp_path):
    runner=Runner(tmp_path);target=tmp_path/'artifact.txt';calls=[]
    def fail():
        calls.append(1);target.write_text('partial');raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):runner.stage('build',{'input':'one'},fail,[target])
    assert json.loads((tmp_path/'pipeline.json').read_text())['stages']['build']['status']=='failed'
    runner.stage('build',{'input':'one'},lambda:target.write_text('complete'),[target])
    runner.stage('build',{'input':'one'},lambda:pytest.fail('valid output should be cached'),[target])
    target.write_text('corrupted')
    runner.stage('build',{'input':'one'},lambda:target.write_text('repaired'),[target])
    assert target.read_text()=='repaired'


def test_changed_source_requires_new_version(tmp_path):
    runner=Runner(tmp_path);target=tmp_path/'artifact.txt'
    runner.stage('build',{'video':'A'},lambda:target.write_text('A'),[target])
    with pytest.raises(ValueError,match='fresh output'):
        runner.stage('build',{'video':'B'},lambda:target.write_text('B'),[target])
    assert target.read_text()=='A'


def test_camera_frame_mismatch_is_rejected(tmp_path):
    import cv2
    import numpy as np
    video=tmp_path/'input.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    writer.write(np.zeros((48,64,3),np.uint8));writer.release()
    scene=tmp_path/'scene.json';scene.write_text(json.dumps({'cameras':[{'index':10}],'objects':[]}))
    with pytest.raises(ValueError,match='frame IDs'):
        make_recipe(scene,video,tmp_path/'output')


def test_reports_can_refresh_without_overwriting_models(tmp_path):
    runner=Runner(tmp_path);model=tmp_path/'model.glb';model.write_bytes(b'original model')
    report=tmp_path/'report.json'
    runner.stage('presentation',{'code':'v1'},lambda:report.write_text('v1'),[report])
    runner.stage('presentation',{'code':'v2'},lambda:report.write_text('v2'),[report])
    assert model.read_bytes()==b'original model'
    assert report.read_text()=='v2'


def test_portrait_evidence_keeps_objects_below_landscape_boundary():
    from scripts.present_refinement import corners
    camera={'c2w':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            'intrinsics':[400,400,240,320],'resolution':[480,640]}
    obj={'center':[0,1.35,2],'dimensions':[.2,.2,.2],
         'rotation':[[1,0,0],[0,1,0],[0,0,1]]}
    box=corners(obj,camera)
    assert box is not None
    assert box[1]>480 and box[3]<=640
