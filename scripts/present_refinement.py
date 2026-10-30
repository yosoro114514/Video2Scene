"""Build truthful before/after evidence, per-instance crops and an offline viewer."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import urllib.request

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def tile(image,label,width=640,height=480):
    h,w=image.shape[:2];scale=min(width/w,height/h)
    resized=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((height+28,width,3),245,np.uint8);y=28+(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    cv2.putText(canvas,label,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.49,(45,45,45),1,cv2.LINE_AA)
    return canvas


def corners(e,cam):
    from itertools import product
    vertices=np.array(list(product([-1,1],repeat=3)))*np.array(e['dimensions'])/2
    points=np.array(e['center'])+vertices@np.array(e['rotation']).T
    t=np.array(cam['c2w']);q=(points-t[:3,3])@t[:3,:3]
    if (q[:,2]<=.01).any():return None
    fx,fy,cx,cy=cam['intrinsics'];uv=q[:,:2]/q[:,2:3]*[fx,fy]+[cx,cy]
    lo=np.floor(uv.min(0)-14).astype(int);hi=np.ceil(uv.max(0)+14).astype(int)
    lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,cam.get('resolution',[640,480]))
    if np.any(hi-lo<15):return None
    return [*lo,*hi]


def mask_image(path,color):
    im=cv2.imread(str(path))
    if im is None:return None
    return np.max(abs(im[:,:,::-1].astype(int)-color),axis=2)<4


def raw_mask(path,ids):
    if not path.exists():return None
    with np.load(path) as data:
        shape=tuple(data['shape']);wanted=[i for i,k in enumerate(data['ids']) if int(k) in ids]
        if not wanted:return None
        return np.any(np.unpackbits(data['packed'][wanted],axis=1,count=int(np.prod(shape))).reshape(len(wanted),*shape),axis=0)


def vendor(out):
    cache=ROOT/'outputs/validation/three_0170'
    files=['build/three.module.js','examples/jsm/controls/OrbitControls.js','examples/jsm/loaders/GLTFLoader.js','examples/jsm/utils/BufferGeometryUtils.js']
    for relative in files:
        source=cache/relative
        if not source.exists():
            source.parent.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/three@0.170.0/'+relative,source)
        target=out/'vendor/three'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=read(a.config);out=Path(config['output']);comp=out/'comparisons';comp.mkdir(exist_ok=True)
    colors=read(out/'label_colors.json') if (out/'label_colors.json').exists() else {}
    cameras={c['index']:c for c in config['cameras']}
    frames=[];rows=[];images={}
    for i in config['review_frames']:
        source=cv2.imread(str(out/f'evidence/frame_{i:03d}.png'));after=cv2.imread(str(out/f'renders/after_{i:03d}.png'))
        before=cv2.imread(str(out/f'renders/before_{i:03d}.png')) if (out/f'renders/before_{i:03d}.png').exists() else None
        if source is None or after is None:continue
        after=cv2.resize(after,(source.shape[1],source.shape[0]))
        row=[tile(source,f'Video | frame {i}')]
        if before is not None:row.append(tile(before,'Before | same ViPE camera'))
        row.append(tile(after,'After | same ViPE camera'))
        row=np.hstack(row);cv2.imwrite(str(comp/f'frame_{i:03d}.jpg'),row)
        overlay=cv2.addWeighted(source,.5,after,.5,0);cv2.imwrite(str(comp/f'overlay_{i:03d}.jpg'),overlay)
        if i in config.get('fit_frames',[]) or len(rows)<3:rows.append(row)
        frames.append(i);images[i]=(source,before,after)
    cv2.imwrite(str(out/'comparison.jpg'),np.vstack(rows))
    assets=[];diagnostics=[]
    for e in config['assets']+config.get('additions',[])+config.get('clutter',[]):
        candidates=[]
        for i in frames:
            box=corners(e,cameras[i])
            mask=mask_image(out/f'renders/after_labels_{i:03d}.png',colors[e['id']]) if e['id'] in colors else None
            if box is None and mask is not None and mask.any():
                ys,xs=np.where(mask);h,w=mask.shape
                box=[max(0,int(xs.min())-8),max(0,int(ys.min())-8),min(w,int(xs.max())+9),min(h,int(ys.max())+9)]
            if box is None:continue
            visibility=int(mask.sum()) if mask is not None else (box[2]-box[0])*(box[3]-box[1])
            if visibility<35:continue
            if mask is not None and mask.any():
                h,w=mask.shape
                ys,xs=np.where(mask);box=[max(0,min(box[0],int(xs.min())-8)),max(0,min(box[1],int(ys.min())-8)),min(w,max(box[2],int(xs.max())+9)),min(h,max(box[3],int(ys.max())+9))]
            candidates.append((visibility,i,box))
        selected=[]
        for candidate in sorted(candidates,reverse=True):
            if all(abs(candidate[1]-v[1])>=20 for v in selected):selected.append(candidate)
            if len(selected)==3:break
        if len(selected)<3:
            for v in sorted(candidates,reverse=True):
                if v not in selected:selected.append(v)
                if len(selected)==3:break
        sheets=[]
        for _,i,(x0,y0,x1,y1) in sorted(selected,key=lambda v:v[1]):
            src,before,after=images[i]
            # Union crop uses an identical image rectangle; no per-image alignment.
            parts=[tile(src[y0:y1,x0:x1],f"{e['id']} | frame {i}",400,300)]
            if before is not None:parts.append(tile(before[y0:y1,x0:x1],'Before',400,300))
            parts.append(tile(after[y0:y1,x0:x1],'After',400,300))
            parts.append(tile(cv2.addWeighted(src[y0:y1,x0:x1],.5,after[y0:y1,x0:x1],.5,0),'Video / after overlay',400,300))
            sheets.append(np.hstack(parts))
        if sheets:
            file=f"comparisons/{e['id']}.jpg";cv2.imwrite(str(out/file),np.vstack(sheets))
            assets.append(dict(id=e['id'],file=file,frames=[v[1] for v in selected],coverage='three observed frames' if len(selected)>=3 else 'limited visibility',source=e.get('source')))
        if config['sample']=='fr2_xyz' and e.get('raw_ids') and e['id'] in colors:
            for i in frames:
                reference=raw_mask(ROOT/f'outputs/fr2_xyz_test/grounded/masks/{i:06d}.npz',e['raw_ids'])
                if reference is None:continue
                row=dict(id=e['id'],frame=i,reference='raw SAM mask; diagnostic only, may contain tracking or boundary errors')
                for tag in ['before','after']:
                    predicted=mask_image(out/f'renders/{tag}_labels_{i:03d}.png',colors[e['id']])
                    if predicted is not None:row[tag+'_iou']=float(np.logical_and(reference,predicted).sum()/max(1,np.logical_or(reference,predicted).sum()))
                diagnostics.append(row)
    report=dict(sample=config['sample'],frames=frames,assets=assets,fit_residuals={k:{'before_px':v['before_px'],'after_px':v['after_px'],'note':v['note']} for k,v in config.get('panel_updates',{}).items()},mask_diagnostics=diagnostics,limitations=config.get('limitations',config.get('assumptions',[])),review_status='Generated evidence; visual judgments and unresolved issues recorded separately in refinement_notes.md')
    (out/'review_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    notes=out/'refinement_notes.md'
    if not notes.exists():
        notes.write_text('# '+config['sample']+'\n\n'+('This is an automatic first-pass Blender scene. Fine reconstruction requires further instance review.\n\n' if config['sample']=='video_scene' else 'Visual refinement evidence is available in comparison.jpg and comparisons/. Detailed review follows.\n\n')+'\n'.join('- '+x for x in report['limitations'])+'\n',encoding='utf8')
    vendor(out)
    scene=read(out/'scene.json');native=scene['editable_source']
    before_url=os.path.relpath(Path(config['baseline_dir'])/'model.glb',out).replace('\\','/') if config.get('baseline_dir') else None
    data=dict(sample=config['sample'],frames=frames,cameras=config['cameras'],assets=assets,before=before_url,native=native)
    (out/'viewer_data.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    shutil.copyfile(ROOT/'video2scene/refinement_viewer.html',out/'index.html')
    print(json.dumps(dict(output=str(out),frames=len(frames),asset_sheets=len(assets)),indent=2))


if __name__=='__main__':main()
