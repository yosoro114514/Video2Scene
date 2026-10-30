import copy
import json
from pathlib import Path
import cv2
import numpy as np
from video2scene.pipeline import Runner
import video2scene.direct_blender as direct


def test_direct_authoring_hard_stops_after_two_revisions(tmp_path,monkeypatch):
    frames=[0,1,2];recipe=dict(source_video_sha256='source',output=str(tmp_path),review_frames=frames,cameras=[],assets=[])
    image=np.zeros((64,48,3),np.uint8)
    for folder in ['renders','evidence']:(tmp_path/folder).mkdir()
    for i in frames:
        cv2.imwrite(str(tmp_path/f'evidence/frame_{i:03d}.png'),image)
        cv2.imwrite(str(tmp_path/f'renders/after_{i:03d}.png'),image)
    direct.write(tmp_path/'recipe.json',recipe);(tmp_path/'video_scene.blend').write_bytes(b'input')
    authors=[];reviews=[]
    monkeypatch.setattr(direct,'references',lambda r,p:[p/f'evidence/frame_{i:03d}.png' for i in frames])
    def author(recipe,source,workspace,*args):
        authors.append(workspace);workspace.mkdir(parents=True)
        for file in ['candidate.blend','model_scene.py','events.jsonl']:(workspace/file).write_text('generated')
        for file in ['execution_proof.json','request.json','response.json']:direct.write(workspace/file,{})
    def review(prompt,images,folder,*args,**kwargs):
        reviews.append(folder)
        response=dict(source_video_sha256='source',status='revise',summary='needs work',major_regression=False,
                      issues=[dict(asset_id='chair',priority='P1',frames=frames,evidence='shape mismatch',requested_fix='repair shape')],uncertainties=[])
        direct.write(folder/'response.json',response);direct.write(folder/'request.json',{});(folder/'events.jsonl').write_text('{}')
        return response
    monkeypatch.setattr(direct,'run_author',author);monkeypatch.setattr(direct,'invoke',review)
    runner=Runner(tmp_path)
    def command(name,argv):
        if name.startswith('direct_render'):
            target=Path(argv[argv.index('--output')+1]);c=copy.deepcopy(recipe);c['output']=str(target)
            direct.write(target/'recipe.json',c);direct.write(target/'scene.json',{})
            for file in ['video_scene.blend','model.glb']:(target/file).write_bytes(b'candidate')
            for i in frames:cv2.imwrite(str(target/f'renders/after_{i:03d}.png'),image)
        elif name.startswith('direct_validate'):direct.write(Path(argv[2]).parent/'native_validation.json',{})
        else:
            target=Path(argv[-1]).parent;direct.write(target/'viewer_data.json',{});(target/'index.html').write_text('viewer')
    runner.command=command
    result=direct.run_direct(runner,tmp_path/'recipe.json',Path('blender'),4,'gpt-6-astra','max',2)
    assert len(authors)==len(reviews)==3
    assert result['stop_reason']=='revision_limit_reached'
    assert result['visual_acceptance']=='needs_refinement'
    assert len(result['history'])==3
    assert (tmp_path/'reviewed/video_scene.blend').read_bytes()==b'candidate'
