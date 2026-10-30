import copy
import json
from pathlib import Path
import numpy as np
import pytest
from video2scene.review_geometry import apply_review,fit_rectangle,project,rectangle,validate_review


def fixture():
    cameras=[]
    for i,x in enumerate([-.3,0,.3]):
        t=np.eye(4);t[0,3]=x
        cameras.append(dict(index=i,c2w=t.tolist(),intrinsics=[450,450,240,320],resolution=[480,640]))
    recipe=dict(source_video_sha256='source',output='unused',cameras=cameras,review_frames=[0,1,2],assets=[dict(id='monitor',template='monitor',center=[0,0,2],dimensions=[.5,.3,.03],rotation=np.eye(3).tolist(),color=[20,20,20],assumptions=[])])
    response=dict(source_video_sha256='source',status='revise',summary='Review',issues=[],edits=[],measurements=[],remaining_uncertainties=[])
    return recipe,response


def test_review_rejects_wrong_source_and_arbitrary_patch(tmp_path):
    recipe,response=fixture();response['source_video_sha256']='different'
    with pytest.raises(ValueError,match='another source'):apply_review(recipe,response,tmp_path)
    response['source_video_sha256']='source'
    response['edits']=[dict(asset_id='monitor',operation='update',patch_json=json.dumps({'python':'print(1)'}),frames=[0,1,2],reason='bad')]
    with pytest.raises(ValueError,match='Disallowed'):apply_review(recipe,response,tmp_path)


def test_multiview_fit_recovers_projected_physical_rectangle():
    recipe,_=fixture();points=rectangle(np.array([.05,.1,2.2]),np.eye(3),.53,.32)
    views=[dict(frame=c['index'],quad=project(points,c)[0].tolist(),visibility='clear') for c in recipe['cameras']]
    result=fit_rectangle(dict(views=views),recipe['cameras'])
    assert result['rms_px']<.01
    assert np.allclose(result['center'],[.05,.1,2.2],atol=.001)
    assert np.allclose(result['dimensions'][:2],[.53,.32],atol=.001)
    with pytest.raises(ValueError,match='three distinct'):fit_rectangle(dict(views=views[:2]),recipe['cameras'])


def test_edits_preserve_original_and_cameras(tmp_path):
    recipe,response=fixture();original=copy.deepcopy(recipe)
    response['edits']=[dict(asset_id='monitor',operation='update',patch_json=json.dumps({'color':[25,25,28],'yaw_delta_deg':90}),frames=[0,1,2],reason='observed front')]
    result,audit=apply_review(recipe,response,tmp_path)
    assert recipe==original and result['cameras']==original['cameras']
    assert result['assets'][0]['color']==[25,25,28]
    assert np.allclose(np.array(result['assets'][0]['rotation'])[:,2],[1,0,0])
    assert len(audit)==1


def test_measurement_outside_image_is_not_silently_clipped():
    recipe,response=fixture()
    response['measurements']=[dict(asset_id='monitor',part='screen',use_texture=True,views=[dict(frame=0,quad=[[20,20],[481,20],[481,200],[20,200]],visibility='clear')])]
    with pytest.raises(ValueError,match='outside source'):validate_review(response,recipe)


def test_revision_limit_stops_even_when_reviewer_always_requests_changes(tmp_path,monkeypatch):
    import video2scene.review_loop as loop
    from video2scene.pipeline import Runner
    recipe,response=fixture();recipe['output']=str(tmp_path)
    recipe['assets']=[]
    response['edits']=[{'operation':'update'}]
    for folder in ['renders','evidence','reviews/pre']:(tmp_path/folder).mkdir(parents=True,exist_ok=True)
    for i in recipe['review_frames']:
        (tmp_path/f'renders/after_{i:03d}.png').write_bytes(b'render')
        (tmp_path/f'evidence/frame_{i:03d}.png').write_bytes(b'image')
    loop.write(tmp_path/'recipe.json',recipe);loop.write(tmp_path/'reviews/pre/response.json',response)
    calls=[]
    def fake_apply(recipe,response,target):
        updated=copy.deepcopy(recipe);updated['output']=str(target);return updated,[]
    def fake_review(recipe,folder,*args):
        calls.append(folder)
        folder.mkdir(parents=True,exist_ok=True)
        loop.write(folder/'response.json',response);loop.write(folder/'request.json',{'actual_call':True})
        (folder/'events.jsonl').write_text('{}')
    monkeypatch.setattr(loop,'apply_review',fake_apply);monkeypatch.setattr(loop,'review',fake_review)
    runner=Runner(tmp_path)
    def fake_command(name,argv):
        if name.startswith('review_build'):
            target=Path(argv[argv.index('--recipe')+1]).parent
            for file in ['video_scene.blend','model.glb','scene.json']:(target/file).write_bytes(b'model')
            for i in recipe['review_frames']:(target/f'renders/after_{i:03d}.png').write_bytes(b'new render')
        elif name.startswith('review_validate'):
            loop.write(Path(argv[2]).parent/'native_validation.json',{})
        else:
            target=Path(argv[-1]).parent
            for file in ['index.html','comparison.jpg','review_report.json']:(target/file).write_bytes(b'report')
            loop.write(target/'viewer_data.json',{})
    runner.command=fake_command
    result=loop.run_loop(runner,tmp_path/'recipe.json','blender',8,'gpt-6-astra','max',2)
    assert len(calls)==3
    assert result['stop_reason']=='revision_limit_reached'
    assert result['visual_acceptance']=='needs_refinement'
    assert len(result['history'])==3
