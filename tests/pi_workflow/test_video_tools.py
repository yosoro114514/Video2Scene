"""CPU fixtures for original frame identity and explicit tool receipts."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from video2scene.pi_workflow import video_tools as tools


@pytest.fixture(autouse=True)
def mock_worker_config(monkeypatch):
    # GPU workers are replaced below. Tests must not depend on a maintainer's settings.
    monkeypatch.setenv('V2S_SAM_CHECKPOINT', '/test-fixtures/sam3.1_multiplex.pt')


def video(path, count=25, width=80, height=60, fps=12.):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,height))
    assert writer.isOpened()
    for index in range(count):
        image=np.full((height,width,3),(index*7)%255,np.uint8)
        image[:,width//2:,1]=(index*11)%255
        writer.write(image)
    writer.release()
    return path


@pytest.fixture
def prepared(tmp_path):
    source=video(tmp_path/'source.avi')
    run=tmp_path/'project'
    result=tools.prepare_video(run,source,frame_budget=7,keyframe_count=3)
    return run,source,result


def test_video_import_freezes_exact_original_frame_mapping(prepared):
    run,source,result=prepared
    manifest=tools._read(result['manifest_path'])
    assert result['cached'] is False
    assert result['source_frame_count']==25
    assert [r['source_frame_index'] for r in manifest['frames']]==[0,4,8,12,16,20,24]
    assert [r['source_frame_index'] for r in result['keyframes']]==[0,12,24]
    assert [r['source_time_s'] for r in manifest['frames']]==pytest.approx(np.arange(0,25,4)/12.)
    assert all(r['source_width']==80 and r['source_height']==60 for r in manifest['frames'])
    assert all(r['width']==80 and r['height']==60 for r in manifest['frames'])
    cap=cv2.VideoCapture(str(source))
    try:
        for row in manifest['frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_index'])
            ok,original=cap.read();assert ok
            frozen=cv2.imread(str(run/'frames'/row['file']))
            assert np.abs(original.astype(float)-frozen.astype(float)).mean()<3
    finally:cap.release()
    cached=tools.prepare_video(run,source,frame_budget=7,keyframe_count=3)
    assert cached['cached'] is True
    assert Path(result['previews'][0]).is_file()


def test_import_rejects_mutated_frame_or_new_configuration(prepared):
    run,source,result=prepared
    with pytest.raises(ValueError,match='already frozen'):
        tools.prepare_video(run,source,frame_budget=6,keyframe_count=3)
    path=run/'frames/000000.jpg';path.write_bytes(path.read_bytes()+b'mutated')
    with pytest.raises(ValueError,match='changed'):
        tools.prepare_video(run,source,frame_budget=7,keyframe_count=3)


def test_resize_records_original_to_prepared_coordinates(tmp_path,monkeypatch):
    monkeypatch.setattr(tools,'MAX_IMAGE_EDGE',64)
    source=video(tmp_path/'wide.avi',count=4,width=120,height=60)
    result=tools.prepare_video(tmp_path/'project',source,frame_budget=4,keyframe_count=2)
    row=result['keyframes'][0]
    assert (row['width'],row['height'])==(64,32)
    assert row['source_to_prepared'][0][0]==pytest.approx(64/120)
    assert row['source_to_prepared'][1][1]==pytest.approx(32/60)


def test_cancelled_tool_never_starts_a_gpu_worker(prepared,monkeypatch):
    run,_,_=prepared
    event=threading.Event();event.set()
    monkeypatch.setattr(tools,'_run_wsl',lambda *a:pytest.fail('A cancelled call must not spawn'))
    with pytest.raises(tools.VideoToolCancelled):tools.run_vipe_tool(run,cancel=event)
    with pytest.raises(tools.VideoToolCancelled):tools.run_sam_tool(run,[],'cancelled',cancel=event)
    assert not (run/'tools/vipe').exists()
    assert not (run/'tools/sam').exists()


def test_sam_explicit_request_outputs_and_cache_binding(prepared,monkeypatch):
    run,_,_=prepared
    calls=[]
    def fake_worker(argv,job,timeout,cancel):
        calls.append(argv)
        assert timeout==1200
        request=tools._read(job/'output/request.json')
        assert request['requests'][0]['seed_source_frame_index']==12
        assert request['requests'][0]['prompt']=='chair'
        output=job/'output'
        tools.write_json(output/'result.json',{'status':'complete','all_targets_resolved':True,'objects':[
            {'object_id':'chair_1','category':'furniture/chair','status':'complete','frames_with_target':4,
             'selected_sam_id':2,'seed_sample_index':3,'previews':[]}]})
        tools.write_json(job/'process_result.json',{'status':'complete','returncode':0})
    monkeypatch.setattr(tools,'_run_wsl',fake_worker)
    requests=[{'object_id':'chair_1','category':'furniture/chair','prompt':'chair',
               'seed_source_frame_index':12,'positive_points_normalized':[[.5,.5]]}]
    result=tools.run_sam_tool(run,requests,'sam_001')
    assert result['cached'] is False
    assert Path(result['sam_root'])==run/'tools/sam/sam_001/output'
    assert result['objects'][0]['frames_with_target']==4
    assert tools.run_sam_tool(run,requests,'sam_001')['cached'] is True
    assert len(calls)==1
    altered=[{**requests[0],'prompt':'armchair'}]
    with pytest.raises(ValueError,match='already bound'):tools.run_sam_tool(run,altered,'sam_001')
    path=Path(result['result_path']);path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):tools.run_sam_tool(run,requests,'sam_001')


def test_sam_failure_is_retained_and_not_reused(prepared,monkeypatch):
    run,_,_=prepared
    def fake_worker(*args):raise RuntimeError('fixture inference failure')
    monkeypatch.setattr(tools,'_run_wsl',fake_worker)
    requests=[{'object_id':'chair_1','category':'furniture/chair','prompt':'chair','seed_source_frame_index':12}]
    with pytest.raises(RuntimeError,match='fixture'):tools.run_sam_tool(run,requests,'failed_001')
    receipt=tools._read(run/'tools/sam/failed_001/receipt.json')
    assert receipt['status']=='failed'
    with pytest.raises(ValueError,match='incomplete'):tools.run_sam_tool(run,requests,'failed_001')


def test_sam_worker_cause_is_reported_without_all_frames(prepared,monkeypatch):
    run,_,_=prepared
    def failed_worker(argv,job,timeout,cancel):
        tools.write_json(job/'output/result.json',{'status':'failed','error_type':'RuntimeError',
            'error':'SAM object cabinet seed source 24 (sample 6/6), direction=backward: KeyError: 5',
            'traceback':'fixture worker traceback','objects':[{'object_id':'rug','status':'complete','frames':[{'huge':'not returned'}]}]})
        raise RuntimeError('Tool worker failed')
    monkeypatch.setattr(tools,'_run_wsl',failed_worker)
    requests=[{'object_id':'cabinet','category':'furniture/cabinet','prompt':'cabinet','seed_source_frame_index':24}]
    with pytest.raises(RuntimeError,match='SAM object cabinet seed source 24'):
        tools.run_sam_tool(run,requests,'failed_diagnostic')
    receipt=tools._read(run/'tools/sam/failed_diagnostic/receipt.json')
    assert receipt['worker_failure']['completed_object_ids']==['rug']
    assert 'frames' not in json.dumps(receipt['worker_failure'])
    assert receipt['status']=='failed'


def test_vipe_receipt_uses_immutable_job_and_explicit_source_mapping(prepared,monkeypatch):
    import video2scene.artifacts
    from video2scene.artifacts import Frame
    run,_,_=prepared
    calls=[]
    def fake_worker(argv,job,timeout,cancel):
        calls.append(argv)
        assert argv[-1]=='inference'
        assert argv[:2]==['env','VIPE_BUFFER=176']
        assert tools._hash(job/'prepared/input.mp4')!=tools._hash(run/'prepared/input.mp4')
        assert tools._read(job/'prepared/manifest.json')['sampling_version']==tools.VIPE_SAMPLING_VERSION
        (job/'vipe').mkdir()
        tools.write_json(job/'vipe/fake.json',{'fixture':True})
    class FakeArtifacts:
        def __init__(self,path):
            assert Path(path).name=='vipe'
            self.indices=list(range(len(tools._read(Path(path).parent/'prepared/manifest.json')['frames'])))
        def frames(self,indices=None):
            for index in self.indices if indices is None else indices:
                yield Frame(index,np.full((60,80,3),120,np.uint8),np.full((60,80),2,np.float32),
                            np.array([50,50,40,30]),np.eye(4))
    monkeypatch.setattr(tools,'_run_wsl',fake_worker)
    monkeypatch.setattr(video2scene.artifacts,'ViPEArtifacts',FakeArtifacts)
    result=tools.run_vipe_tool(run,{'job_id':'vipe_001','reason':'need camera and depth'})
    assert result['cached'] is False
    assert Path(result['vipe_dir'])==run/'tools/vipe/vipe_001/vipe'
    assert tools._read(run/'tools/vipe/latest.json')['vipe_dir']==result['vipe_dir']
    assert result['frame_count']==25 and result['observation_frame_count']==7
    assert tools._read(run/'tools/vipe/latest.json')['vipe_manifest_sha256']==tools._hash(result['vipe_manifest_path'])
    assert tools.run_vipe_tool(run)['cached'] is True
    assert len(calls)==1
    assert Path(result['previews'][0]).is_file()


def test_dense_video_contains_every_observation_without_overwriting_it(tmp_path):
    source=video(tmp_path/'dense_source.avi',count=61,fps=30.)
    run=tmp_path/'project';imported=tools.prepare_video(run,source,frame_budget=9,keyframe_count=3)
    original_manifest_hash=tools._hash(imported['manifest_path'])
    original_preview_hash=tools._hash(imported['prepared_video'])
    observations=tools._read(imported['manifest_path'])
    dense=tools.prepare_vipe_dense_video(run,run/'tools/dense_fixture/prepared')
    regular=set(range(0,61,3));required={row['source_frame_index'] for row in observations['frames']}
    actual=[row['source_frame_index'] for row in dense['frames']]
    assert actual==sorted(regular|required)
    assert [row['frame_index'] for row in dense['frames']]==list(range(len(actual)))
    matched={row['source_frame_index']:row for row in dense['frames'] if row['is_observation_frame']}
    assert set(matched)==required
    for row in observations['frames']:
        observed=matched[row['source_frame_index']]
        assert observed['observation_frame_index']==row['frame_index']
        assert observed['source_time_s']==pytest.approx(row['source_time_s'])
        assert observed['observation_rgb_sha256']==row['sha256']
    assert dense['source_max_gap_s']<=.100001
    assert dense['observation_manifest_sha256']==original_manifest_hash
    assert dense['source_sha256']==observations['source_sha256']
    assert len(dense['frames'])/dense['prepared_fps']==pytest.approx(observations['duration_s'])
    assert tools._hash(imported['manifest_path'])==original_manifest_hash
    assert tools._hash(imported['prepared_video'])==original_preview_hash
    with pytest.raises(ValueError,match='overwrite'):
        tools.prepare_vipe_dense_video(run,run/'prepared')


def test_dense_plan_is_separate_from_sparse_budget_and_reserves_infill_slots():
    sources=np.linspace(0,2869,96).round().astype(int).tolist()
    plan=tools._dense_plan({'source_fps':30.,'source_frame_count':2870,
                            'frames':[{'source_frame_index':i} for i in sources]},10.)
    assert plan['dense_frame_count']==1020
    assert plan['slam_buffer']==1056
    assert set(sources)<=set(plan['source_indices'])
    assert max(np.diff(plan['source_indices']))==3
    assert plan['encoded_fps']==pytest.approx(1020/(2870/30))


def test_dense_cache_does_not_relabel_sparse_result(prepared,monkeypatch):
    import video2scene.artifacts
    run,_,_=prepared
    previous=run/'tools/vipe/old_sparse';previous.mkdir(parents=True)
    old={'status':'complete','job_id':'old_sparse','binding':'old_sparse_fingerprint',
         'result':{'sampling_version':'sparse_observation_v0'}}
    tools.write_json(previous/'receipt.json',old)
    old_hash=tools._hash(previous/'receipt.json')
    tools.write_json(run/'tools/vipe/latest.json',{'receipt_path':str(previous/'receipt.json'),'job_id':'old_sparse','vipe_dir':str(previous/'vipe')})
    def fake_worker(argv,job,timeout,cancel):
        (job/'vipe').mkdir();tools.write_json(job/'vipe/fake.json',{'fixture':True})
    class FakeArtifacts:
        def __init__(self,path):self.indices=list(range(len(tools._read(Path(path).parent/'prepared/manifest.json')['frames'])))
    monkeypatch.setattr(tools,'_run_wsl',fake_worker)
    monkeypatch.setattr(video2scene.artifacts,'ViPEArtifacts',FakeArtifacts)
    monkeypatch.setattr(tools,'_geometry_preview',lambda *args:([],{}))
    result=tools.run_vipe_tool(run)
    assert result['cached'] is False and result['job_id']!='old_sparse'
    assert result['sampling_version']==tools.VIPE_SAMPLING_VERSION
    assert tools._hash(previous/'receipt.json')==old_hash
    receipt=tools._read(result['receipt_path'])
    assert receipt['previous_vipe_result']['sampling_version']=='sparse_observation_v0'
    assert tools.run_vipe_tool(run)['cached'] is True
    changed=tools.run_vipe_tool(run,{'dense_fps':6.})
    assert changed['cached'] is False and changed['job_id']!=result['job_id']


def test_dense_input_rejects_changed_original_source(prepared):
    run,source,_=prepared
    source.write_bytes(source.read_bytes()+b'mutated')
    with pytest.raises(ValueError,match='Original source video'):
        tools.run_vipe_tool(run,{'job_id':'must_not_start'})
    assert not (run/'tools/vipe/must_not_start').exists()


def test_job_id_cannot_escape_project(prepared):
    run,_,_=prepared
    with pytest.raises(ValueError,match='job_id'):
        tools.run_vipe_tool(run,{'job_id':'../outside'})


@pytest.mark.skipif(os.environ.get('RUN_PI_WSL_SMOKE')!='1',reason='Explicit WSL CPU process smoke')
def test_actual_wsl_cancellation_stops_own_process_group(tmp_path):
    job=tmp_path/'cpu_cancel';job.mkdir()
    event=threading.Event()
    child_file=tools.to_wsl(job/'fixture_child_pid.txt')
    program="import pathlib,subprocess,time;p=subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(40)']);pathlib.Path("+repr(child_file)+").write_text(str(p.pid));time.sleep(40)"
    def cancel_after_child_starts():
        end=time.monotonic()+20
        while not (job/'fixture_child_pid.txt').exists() and time.monotonic()<end:
            time.sleep(.1)
        event.set()
    timer=threading.Thread(target=cancel_after_child_starts);timer.start()
    try:
        with pytest.raises(tools.VideoToolCancelled):
            tools._run_wsl(['/usr/bin/python3','-c',program],job,30,event)
    finally:event.set();timer.join()
    result=tools._read(job/'process_result.json')
    assert result['status']=='cancelled'
    assert result['seconds']<12
    # A terminated child may briefly remain a zombie until PID 1 reaps it;
    # either absence or Z confirms it cannot keep executing GPU/CPU work.
    pid=int((job/'fixture_child_pid.txt').read_text())
    probe="import pathlib,sys;p=pathlib.Path('/proc')/sys.argv[1]/'stat';print(p.read_text().split()[2] if p.exists() else 'absent')"
    state=subprocess.check_output(['wsl','-d',tools.DISTRO,'--exec','/usr/bin/python3','-c',probe,str(pid)],text=True).strip()
    assert state in {'Z','absent'}
