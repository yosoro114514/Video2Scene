"""Resumable video -> geometry -> editable Blender -> evidence -> viewer runner.

The default path is automatic and produces conservative instance assets. A
source-bound asset plan supplies the stronger visual decisions used in refinement.
Optional --reviewer codex invokes the supported CLI with ChatGPT authentication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');temp.replace(path)


def linux(path):
    value=str(Path(path).resolve())
    if len(value)>2 and value[1]==':':return '/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return value


class Runner:
    def __init__(self,output):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/'pipeline.json'
        self.state=json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else dict(schema_version=1,stages={})

    def stage(self,name,signature,action,outputs):
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        outputs=[Path(p) for p in outputs];previous=self.state['stages'].get(name,{})
        if previous.get('status')=='complete' and previous.get('signature')==key and all(p.is_file() and previous.get('artifacts',{}).get(str(p))==sha(p) for p in outputs):
            print('CACHED',name,flush=True);return
        if previous.get('status')=='complete' and previous.get('signature')!=key and name not in {'presentation','validation'}:
            raise ValueError(f'{name} inputs changed; use a fresh output directory to preserve the previous run')
        record=dict(status='running',signature=key,inputs=signature,started=time.time())
        self.state['status']='running';self.state['stages'][name]=record;write(self.path,self.state)
        try:
            action()
            if not all(p.is_file() for p in outputs):raise RuntimeError(f'{name} did not create all required outputs')
            record.update(status='complete',seconds=time.time()-record['started'],artifacts={str(p):sha(p) for p in outputs})
        except Exception as exc:
            record.update(status='failed',error=str(exc));self.state['status']='failed';write(self.path,self.state);raise
        write(self.path,self.state)

    def command(self,name,argv):
        logs=self.output/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def canonical(o,kind):
    r=np.array(o['rotation']);d=np.array(o['dimensions'])
    if kind in ['keyboard','laptop','laptop_assembly','telephone']:
        up=int(np.argmax(np.abs(r[1])));remaining=[i for i in range(3) if i!=up]
        width=max(remaining,key=lambda i:d[i]);depth=next(i for i in remaining if i!=width)
        y=r[:,up]*np.sign(r[1,up]);x=r[:,width];z=np.cross(x,y)
        return np.column_stack([x,y,z]).tolist(),d[[width,up,depth]].tolist()
    if kind=='monitor':
        normal=int(np.argmin(d));rest=[i for i in range(3) if i!=normal];up=max(rest,key=lambda i:abs(r[1,i]));width=next(i for i in rest if i!=up)
        y=r[:,up]*np.sign(r[1,up]);x=r[:,width];z=np.cross(x,y)
        return np.column_stack([x,y,z]).tolist(),d[[width,up,normal]].tolist()
    return r.tolist(),d.tolist()


def make_recipe(scene_path,video,output,plan=None):
    spec=json.loads(Path(scene_path).read_text(encoding='utf8'));output=Path(output)
    cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    cameras=spec['cameras']
    if not cameras or max(c['index'] for c in cameras)>=count:raise ValueError('Video frame IDs do not match the scene cameras')
    selected=[cameras[int(i)] for i in np.linspace(0,len(cameras)-1,min(7,len(cameras)),dtype=int)]
    types={'tv':'monitor','monitor':'monitor','keyboard':'keyboard','mouse':'mouse','desk_telephone':'telephone','telephone':'telephone','laptop':'laptop_assembly','game_controller':'gamepad','chair':'chair','bottle':'bottle','cup':'cup','backpack':'backpack'}
    assets=[]
    for o in spec['objects']:
        kind=types.get(o.get('label'),'proxy')
        r,d=canonical(o,kind)
        assumptions=list(o.get('assumptions',[]))
        if kind in ['laptop_assembly','chair','bottle','cup','backpack']:
            assumptions.append('Multipart category prior inside observed bounds; part ratios, front direction, materials and hidden shape are estimates, not instance measurements.')
        assets.append(dict(id=o['id'],template=kind,center=o['center'],dimensions=d,rotation=r,color=o.get('color',[150,150,150])[:3],primitive=o.get('primitive','box'),source=o.get('source','Estimated geometry'),evidence_frames=o.get('frame_indices',[]),assumptions=assumptions))
        if 'footprint' in o:assets[-1]['footprint']=o['footprint']
    surfaces=[e for e in assets if e.get('primitive')=='prism' and abs(np.array(e['rotation'])[1,1])>.98]
    for e in assets:
        if e['template'] not in ['keyboard','laptop_assembly','mouse','bottle','cup','monitor']:continue
        for surface in surfaces:
            local=(np.array(e['center'])-surface['center'])@np.array(surface['rotation'])
            if cv2.pointPolygonTest(np.array(surface['footprint'],np.float32),(float(local[0]),float(local[2])),False)<0:continue
            top=surface['center'][1]+surface['dimensions'][1]/2
            bottom=e['center'][1]-float(abs(np.array(e['rotation'])[1])@np.array(e['dimensions'])/2)
            if e['template']=='monitor' and -.04<bottom-top<.30:
                e['support']=top
            elif e['template']!='monitor' and abs(top-bottom)<.06:
                e['observed_center']=list(e['center']);e['center']=list(e['center']);e['center'][1]+=top-bottom
                e['support_adjustment_m']=top-bottom
                e['assumptions'].append('Bottom snapped to nearby observed planar support within 6 cm; monocular depth inconsistency remains.')
    if plan:
        value=json.loads(Path(plan).read_text(encoding='utf8'))
        if value.get('source_video_sha256')!=sha(video):raise ValueError('Asset plan is bound to a different processed video')
        overrides={e['id']:e for e in value['assets']}
        assets=[overrides.pop(e['id'],e) for e in assets]+list(overrides.values())
    recipe=dict(schema_version=2,sample='video_scene',source_video=str(Path(video).resolve()),source_video_sha256=sha(video),source_scene=str(Path(scene_path).resolve()),source_scene_sha256=sha(scene_path),output=str(output.resolve()),assets=assets,cameras=selected,all_cameras=cameras,review_frames=[c['index'] for c in selected],additions=[],scan=str(Path(scene_path).parent/'scan.glb'),limitations=['Automatic geometry follows detected masks and estimated monocular depth.','Fine instance shape and unseen structures are not guaranteed without a video-bound asset plan.','No live GPT API is invoked; supplied semantic decisions and asset plans are explicit inputs.'])
    for e in assets:
        if min(e['dimensions'])<=0 or not np.isfinite(e['dimensions']).all():raise ValueError('Invalid asset dimensions')
    for folder in ['evidence','renders','textures','comparisons']:(output/folder).mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    for c in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES,c['index']);ok,frame=cap.read()
        if not ok:raise ValueError('Missing evidence frame')
        width,height=c.get('resolution',[frame.shape[1],frame.shape[0]])
        if (frame.shape[1],frame.shape[0])!=(width,height):frame=cv2.resize(frame,(width,height))
        cv2.imwrite(str(output/'evidence'/f"frame_{c['index']:03d}.png"),frame)
    cap.release();write(output/'recipe.json',recipe);return recipe


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--blender',type=Path,required=True)
    p.add_argument('--scene-json',type=Path);p.add_argument('--vipe-dir',type=Path);p.add_argument('--sequence',default='input')
    p.add_argument('--refinement-config',type=Path);p.add_argument('--asset-plan',type=Path)
    p.add_argument('--segment',choices=['none','yolo','grounded_sam2'],default='yolo');p.add_argument('--inventory',type=Path);p.add_argument('--mask-cache',type=Path);p.add_argument('--review-decisions',type=Path)
    p.add_argument('--frames',type=int,default=160);p.add_argument('--samples',type=int,default=24);p.add_argument('--wsl-distro',default='Ubuntu-22.04')
    p.add_argument('--reviewer',choices=['none','codex','codex-direct'],default='none')
    p.add_argument('--review-model',default='gpt-6-astra');p.add_argument('--review-effort',default='max',choices=['low','medium','high','xhigh','max'])
    p.add_argument('--review-revisions',type=int,default=2,help='Maximum post-render correction rounds, in addition to pre-review build')
    a=p.parse_args(argv);runner=Runner(a.output);out=runner.output
    if not a.blender.is_file():p.error('Blender executable not found')
    if not 1<=a.review_revisions<=5:p.error('--review-revisions must be 1..5')
    if a.reviewer!='none' and a.refinement_config:p.error('Codex review currently uses the generic scene recipe path')
    builder=ROOT/'scripts/refine_blender.py'
    if a.refinement_config:
        config=json.loads(a.refinement_config.read_text(encoding='utf8'))
        if Path(config['output']).resolve()!=out:p.error('Refinement output must match config output; create a new config for another version')
        sig=dict(config=sha(a.refinement_config),builder=sha(builder),assets=sha(ROOT/'video2scene/blender_assets.py'),samples=a.samples)
        frames=','.join(map(str,config['review_frames']))
        for tag in ['before','after']:
            command=[a.blender,'--background','--factory-startup','--python-exit-code','1','--python',builder,'--','--config',a.refinement_config.resolve(),'--frames',frames,'--samples',a.samples,'--masks']
            if tag=='before':command+=['--baseline']
            required=[out/'renders'/f"{tag}_{i:03d}.png" for i in config['review_frames']]
            if tag=='after':required += [out/'model.glb',out/(config['sample']+'_refined.blend')]
            runner.stage('render_'+tag,sig,lambda cmd=command,t=tag:runner.command(t,cmd),required)
        recipe_path=a.refinement_config.resolve()
    else:
        if not a.video or not a.video.is_file():p.error('A readable --video is required')
        video=a.video.resolve();scene=a.scene_json.resolve() if a.scene_json else out/'geometry/scene.json'
        if not a.scene_json:
            if os.name=='nt':cmd=['wsl','-d',a.wsl_distro,'--exec','bash',linux(ROOT/'scripts/run_wsl.sh')];path=linux
            else:cmd=[sys.executable,'-m','video2scene'];path=lambda x:str(Path(x).resolve())
            worker=list(cmd)
            if a.vipe_dir:
                vipe=a.vipe_dir.resolve();sequence=a.sequence
            else:
                from .inference import prepare_video
                processed=out/'input.mp4';original=video
                runner.stage('prepare',dict(video=sha(original),frames=a.frames,code=sha(ROOT/'video2scene/inference.py')),lambda:prepare_video(original,processed,max_frames=a.frames),[processed,processed.with_suffix('.json')])
                vipe=out/'vipe';sequence='input'
                infer_command=worker+['infer','--video',path(processed),'--output',path(vipe),'--buffer',str(a.frames+16)]
                runner.stage('inference',dict(video=sha(processed),buffer=a.frames+16,code=sha(ROOT/'video2scene/inference.py')),lambda:runner.command('inference',infer_command),[vipe/'pose/input.npz',vipe/'intrinsics/input.npz',vipe/'depth/input.zip',vipe/'rgb/input.mp4'])
            video=vipe/'rgb'/f'{sequence}.mp4'
            cmd=worker+['reconstruct','--vipe-dir',path(vipe),'--sequence',sequence,'--output',path(out/'geometry')]
            cmd+=['--segment',a.segment]
            for flag,value in [('--inventory',a.inventory),('--mask-cache',a.mask_cache),('--review-decisions',a.review_decisions)]:
                if value:cmd += [flag,path(value)]
            if a.review_decisions:cmd += ['--semantic-review','file']
            sig=dict(video=sha(video),segment=a.segment,frames=a.frames,sequence=sequence,inventory=sha(a.inventory) if a.inventory else None,review=sha(a.review_decisions) if a.review_decisions else None,code=sha(ROOT/'video2scene/reconstruction.py'))
            sig['vipe']={str(x.relative_to(vipe)):sha(x) for directory in ['pose','intrinsics','depth'] for x in (vipe/directory).glob('*') if x.is_file()}
            runner.stage('geometry',sig,lambda:runner.command('geometry',cmd),[scene,scene.parent/'scan.glb'])
        recipe_path=out/'recipe.json'
        runner.stage('recipe',dict(scene=sha(scene),video=sha(video),plan=sha(a.asset_plan) if a.asset_plan else None,code=sha(Path(__file__))),lambda:make_recipe(scene,video,out,a.asset_plan),[recipe_path])
        if a.reviewer=='codex':
            from .review_loop import pre_review
            pre_review(runner,recipe_path,a.review_model,a.review_effort)
        generic=ROOT/'scripts/build_pipeline_blender.py'
        runner.stage('blender',dict(recipe=sha(recipe_path),builder=sha(generic),assets=sha(ROOT/'video2scene/blender_assets.py'),automatic_assets=sha(ROOT/'video2scene/automatic_assets.py'),samples=a.samples),lambda:runner.command('blender',[a.blender,'--background','--factory-startup','--python-exit-code','1','--python',generic,'--','--recipe',recipe_path,'--samples',str(a.samples)]),[out/'video_scene.blend',out/'model.glb',out/'scene.json'])
    presenter=ROOT/'scripts/present_refinement.py'
    runner.stage('presentation',dict(recipe=sha(recipe_path),model=sha(out/'model.glb'),code=sha(presenter),viewer=sha(ROOT/'video2scene/refinement_viewer.html')),lambda:runner.command('presentation',[sys.executable,presenter,'--config',recipe_path]),[out/'index.html',out/'comparison.jpg',out/'review_report.json'])
    validator=ROOT/'scripts/validate_refined_blender.py'
    native=out/(config['sample']+'_refined.blend') if a.refinement_config else out/'video_scene.blend'
    runner.stage('validation',dict(native=sha(native),glb=sha(out/'model.glb'),code=sha(validator)),lambda:runner.command('validation',[a.blender,'--background',native,'--python-exit-code','1','--python',validator]),[out/'native_validation.json'])
    if a.reviewer=='codex':
        from .review_loop import run_loop
        reviewed=run_loop(runner,recipe_path,a.blender,a.samples,a.review_model,a.review_effort,a.review_revisions)
        runner.state['visual_acceptance']=reviewed['visual_acceptance'];runner.state['reviewed_output']=reviewed['final_directory']
    elif a.reviewer=='codex-direct':
        from .direct_blender import run_direct
        reviewed=run_direct(runner,recipe_path,a.blender,a.samples,a.review_model,a.review_effort,a.review_revisions)
        runner.state['visual_acceptance']=reviewed['visual_acceptance'];runner.state['reviewed_output']=reviewed['final_directory']
    runner.state['status']='complete';runner.state['completed']=time.time();write(runner.path,runner.state)
    print('PIPELINE_COMPLETE',out/('reviewed/index.html' if a.reviewer!='none' else 'index.html'))


if __name__=='__main__':main()
