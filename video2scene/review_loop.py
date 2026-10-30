"""Source review -> constrained changes -> Blender -> visual review, resumably."""
import json
from pathlib import Path
import shutil
import sys
from .codex_review import digest
from .review_geometry import review,apply_review,write

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))


def signature(recipe,model,effort):
    return dict(recipe=digest(recipe),model=model,effort=effort,
                client=digest(ROOT/'video2scene/codex_review.py'),geometry=digest(ROOT/'video2scene/review_geometry.py'))


def pre_review(runner,recipe_path,model,effort):
    folder=runner.output/'reviews/pre';recipe=load(recipe_path)
    sig=signature(recipe_path,model,effort)
    sig['images']={str(i):digest(Path(recipe['output'])/f'evidence/frame_{i:03d}.png') for i in recipe['review_frames']}
    runner.stage('codex_pre',sig,lambda:review(recipe,folder,'pre',model,effort),
                 [folder/'response.json',folder/'request.json',folder/'events.jsonl'])
    return folder/'response.json'


def run_loop(runner,base_recipe,blender,samples,model,effort,max_revisions):
    out=runner.output;recipe_path=base_recipe;response_path=out/'reviews/pre/response.json';history=[];last_recipe=None
    summary_path=out/'review_pipeline.json'
    state=dict(status='running',model=model,reasoning_effort=effort,authentication='ChatGPT subscription',max_revisions=max_revisions,max_reviewed_models=max_revisions+1,
               source_video_sha256=load(base_recipe)['source_video_sha256'],history=history)
    write(summary_path,state)
    try:
        for iteration in range(max_revisions+1):
            target=out/'iterations'/f'{iteration:02d}';target.mkdir(parents=True,exist_ok=True)
            recipe=load(recipe_path);response=load(response_path);new_path=target/'recipe.json';audit_path=target/'applied_review.json'
            def apply():
                for name in ['evidence','textures']:
                    original=Path(recipe['output'])/name
                    if original.exists():shutil.copytree(original,target/name,dirs_exist_ok=True)
                updated,audit=apply_review(recipe,response,target)
                updated['baseline_dir']=str(out)
                (target/'renders').mkdir(exist_ok=True)
                for i in updated['review_frames']:shutil.copyfile(out/f'renders/after_{i:03d}.png',target/f'renders/before_{i:03d}.png')
                write(new_path,updated);write(audit_path,dict(source_response=str(response_path),source_response_sha256=digest(response_path),operations=audit))
            runner.stage(f'review_apply_{iteration:02d}',dict(recipe=digest(recipe_path),response=digest(response_path),code=digest(ROOT/'video2scene/review_geometry.py'),loop=digest(Path(__file__))),apply,[new_path,audit_path])
            builder=ROOT/'scripts/build_pipeline_blender.py'
            runner.stage(f'review_build_{iteration:02d}',dict(recipe=digest(new_path),builder=digest(builder),assets=digest(ROOT/'video2scene/automatic_assets.py'),base_assets=digest(ROOT/'video2scene/blender_assets.py'),samples=samples),
                         lambda:runner.command(f'review_build_{iteration:02d}',[blender,'--background','--factory-startup','--python-exit-code','1','--python',builder,'--','--recipe',new_path,'--samples',str(samples)]),
                         [target/'video_scene.blend',target/'model.glb',target/'scene.json']+[target/f'renders/after_{i:03d}.png' for i in recipe['review_frames']])
            validator=ROOT/'scripts/validate_refined_blender.py'
            runner.stage(f'review_validate_{iteration:02d}',dict(native=digest(target/'video_scene.blend'),code=digest(validator)),
                         lambda:runner.command(f'review_validate_{iteration:02d}',[blender,'--background',target/'video_scene.blend','--python-exit-code','1','--python',validator]),[target/'native_validation.json'])
            reviewed=load(new_path);folder=out/'reviews'/f'post_{iteration:02d}'
            sig=signature(new_path,model,effort);sig['renders']={str(i):digest(target/f'renders/after_{i:03d}.png') for i in reviewed['review_frames']}
            previous=dict(summary=response['summary'],issues=response['issues'],applied=load(audit_path)['operations'])
            runner.stage(f'codex_post_{iteration:02d}',sig,lambda:review(reviewed,folder,'post',model,effort,previous),
                         [folder/'response.json',folder/'request.json',folder/'events.jsonl'])
            post=load(folder/'response.json')
            accepted=post['status']=='accept' and not post['edits'] and not post['measurements'] and not any(x['priority']=='P0' for x in post['issues'])
            history.append(dict(iteration=iteration,recipe=str(new_path),response=str(folder/'response.json'),status=post['status'],accepted=accepted,
                                edits_applied=len(response['edits']),measurements_applied=sum(x.get('status')=='applied' for x in load(audit_path)['operations']),issues_remaining=len(post['issues'])))
            state.update(last_iteration=iteration,visual_acceptance='accepted' if accepted else 'needs_refinement');write(summary_path,state)
            last_recipe=new_path
            if accepted or (not post['edits'] and not post['measurements']):
                state['stop_reason']='accepted' if accepted else 'no_actionable_changes';break
            recipe_path=new_path;response_path=folder/'response.json'
        state.setdefault('stop_reason','revision_limit_reached')
        target=last_recipe.parent;presenter=ROOT/'scripts/present_refinement.py'
        runner.stage('review_presentation',dict(recipe=digest(last_recipe),code=digest(presenter),viewer=digest(ROOT/'video2scene/refinement_viewer.html')),
                     lambda:runner.command('review_presentation',[sys.executable,presenter,'--config',last_recipe]),[target/'index.html',target/'comparison.jpg',target/'review_report.json'])
        final=out/'reviewed'
        # Publish a separate copy; preserve every model and every review round.
        shutil.copytree(target,final,dirs_exist_ok=True)
        post=load(out/'reviews'/f'post_{history[-1]["iteration"]:02d}'/'response.json')
        notes=['# Codex subscription visual review','',f'Model: {model}; effort: {effort}. Actual CLI request/event/response records are in ../../reviews/.',
               '',f'Visual acceptance: {state["visual_acceptance"]}. Pipeline completion is not a claim of visual fidelity.',
               '',post['summary'],'','## Remaining issues','']
        notes += [f'- {x["priority"]} / {x["asset_id"]} / frames {x["frames"]}: {x["evidence"]} Uncertainty: {x["uncertainty"]}' for x in post['issues']]
        notes += ['','## Uncertainties','']+['- '+x for x in post['remaining_uncertainties']]
        (final/'refinement_notes.md').write_text('\n'.join(notes),encoding='utf8')
        # Presenter computes relative baseline URL for the iteration location;
        # update only this link when publishing at a different directory depth.
        data=load(final/'viewer_data.json');data['before']='../model.glb';write(final/'viewer_data.json',data)
        state.update(status='complete',final_directory=str(final),final_recipe=str(last_recipe),final_review=str(out/'reviews'/f'post_{history[-1]["iteration"]:02d}'/'response.json'))
        write(summary_path,state);return state
    except Exception as error:
        state.update(status='failed',error=str(error));write(summary_path,state);raise
