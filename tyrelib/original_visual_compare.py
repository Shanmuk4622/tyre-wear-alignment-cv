"""Original paper study only: 17 classifiers, nine localisers, two geometry models."""
import gc
import hashlib
import html
import json
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from original_visual_support import sha, write, Publisher, boundaries, draw_lines

CLASSES=['Low','Medium','High']
ARCHS='resnet50 resnext50 densenet121 vgg16bn convnextv2_t effnetv2s regnety016 mobilenetv4 vit_s deit3_s swin_t swin_s coatnet0 maxvit_t dinov2_s dinov2_b clip_b16'.split()
REPO='Shanmuk4622/tyre-wear-study'


def select(root, assets, per_class=3):
    import pandas as pd
    assert 1<=per_class<=30
    if not root:
        candidates=[p.parent.parent for p in Path('/kaggle/input').rglob('clean_manifest.csv')
                    if p.parent.name=='manifests' and sha(p)==assets['clean_manifest_sha256']]
        if len(candidates)!=1: raise ValueError('Attach the ORIGINAL tire-dataset-prepared dataset, or set DATA_ROOT to its FINAL directory. Phase 2 is not accepted.')
        root=candidates[0]
    root=Path(root)
    if sha(root/'manifests/clean_manifest.csv')!=assets['clean_manifest_sha256']: raise ValueError('Not the frozen original final_v1 manifest.')
    frame=pd.read_csv(root/'manifests/clean_manifest.csv');assert len(frame)==418
    rows=[]
    for c,label in enumerate(['low_mileage_proxy','mid_mileage_proxy','high_mileage_proxy']):
        subset=frame[(frame.proxy_label==label)&(frame.fold_id==1)].sort_values('image_id')
        bags={k:v.to_dict('records') for k,v in subset.groupby('session_group',sort=True)}
        chosen=[]
        while bags and len(chosen)<per_class:
            for k in list(bags):
                chosen.append(bags[k].pop(0))
                if not bags[k]:del bags[k]
                if len(chosen)==per_class:break
        for r in chosen:
            p=root/r['relative_path']
            if sha(p)!=r['file_sha256']:raise ValueError('Original image hash mismatch: '+str(p))
            geometry_role=next((role for role,groups in assets['hrnet']['groups'].items() if r['session_group'] in groups),'unknown')
            rows.append(dict(key=f'{len(rows):03d}',image_id=r['image_id'],path=str(p),sha256=r['file_sha256'],
                class_index=c,session=r['session_group'],role='fold-1 validation',geometry_role=geometry_role))
    return rows


def source(row):
    with Image.open(row['path']) as im:return im.convert('RGB').copy()


def download(spec, work, publisher):
    from huggingface_hub import hf_hub_download
    # local_dir avoids retaining a second large global cache copy.
    folder=Path(work)/'active_checkpoint';folder.mkdir(exist_ok=True)
    if shutil.disk_usage(work).free<spec['bytes']+2*1024**3:raise RuntimeError('Insufficient space for the next checkpoint plus 2 GiB reserve.')
    path=Path(publisher.retry(lambda:hf_hub_download(REPO,spec['path'],repo_type='dataset',revision=spec['revision'],token=publisher.token,local_dir=str(folder))))
    if sha(path)!=spec['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    return path


def load(spec,path,assets,work):
    import torch
    import torch.nn.functional as F
    from original_visual_support import build_model, build_transforms, Geometry
    state=torch.load(path,map_location='cpu',weights_only=False)
    name=spec['model'];extra={}
    if spec['task']=='classification':
        cfg=state['config']
        assert state['arch']==name and cfg['arch']==name and cfg['fold']==1 and cfg['seed']==1
        assert state['classes']==['low_mileage_proxy','mid_mileage_proxy','high_mileage_proxy']
        assert state['dataset_version']=='final_v1'
        if spec['endpoint']=='last':assert state['epoch']==60
        model=build_model(name,pretrained=False,head=cfg['head_type'],img_size=cfg['input_resolution'],verify=False)
        model.load_state_dict(state['model'],strict=True)
        extra=dict(transform=build_transforms(cfg['input_resolution'],False,cfg.get('preprocessing','raw')),head=cfg['head_type'],epoch=state['epoch'],config=cfg)
    elif spec['task']=='geometry':
        assert state['protocol']==spec['protocol'] and state['seed']==1 and state['epoch']==60
        if name=='hrnet':model=Geometry(assets['hrnet']);weights=state['model']
        else:
            from transformers import SegformerConfig,SegformerForSemanticSegmentation
            cfg=SegformerConfig.from_dict(assets['configs']['nvidia/mit-b0']);cfg.num_labels=2
            model=SegformerForSemanticSegmentation(cfg);weights={k.removeprefix('net.'):v for k,v in state['model'].items()}
        model.load_state_dict(weights,strict=True)
    else:
        job=state['job'];assert job['model']==name and job['fold']==1 and job['seed']==1 and state['epoch']==60
        assert state['plan_hash']=='1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df'
        if spec['backend']=='yolo':
            from ultralytics import YOLO
            native=state['native'];net=native.get('ema') if native.get('ema') is not None else native['model']
            net.float().load_state_dict(state['ema_model'],strict=True)
            p=Path(work)/'active_yolo.pt';torch.save(dict(model=net.cpu().eval(),train_args=native.get('train_args',{}),epoch=60,version='8.4.20'),p)
            model=YOLO(str(p));extra['native_file']=str(p)
        elif name in ('unet_r34','deeplabv3plus_r34'):
            import segmentation_models_pytorch as smp
            model=(smp.Unet if name=='unet_r34' else smp.DeepLabV3Plus)(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=2,activation=None)
            model.load_state_dict(state['model'],strict=True)
        elif name.startswith('segformer'):
            from transformers import SegformerConfig,SegformerForSemanticSegmentation
            cfg=SegformerConfig.from_dict(assets['configs'][assets['plan']['models'][name][1]]);cfg.num_labels=2
            model=SegformerForSemanticSegmentation(cfg);model.load_state_dict(state['model'],strict=True)
        else:
            from transformers import RTDetrV2Config,RTDetrV2ForObjectDetection,RTDetrImageProcessor
            cfg=RTDetrV2Config.from_dict(assets['configs']['PekingU/rtdetr_v2_r18vd']);cfg.num_labels=2
            cfg.id2label={0:'tyre',1:'tread'};cfg.label2id={'tyre':0,'tread':1};cfg.disable_custom_kernels=True
            model=RTDetrV2ForObjectDetection(cfg);model.load_state_dict(state['model'],strict=True)
            extra['processor']=RTDetrImageProcessor.from_dict(assets['processor'],size={'height':512,'width':512})
    del state;gc.collect()
    model.to('cuda:0')
    if spec.get('backend')!='yolo':model.eval()
    return model,extra


def tensor(im,hw):
    import torch
    a=np.asarray(im.resize(tuple(hw[::-1]),Image.Resampling.BILINEAR),dtype=np.float32).copy()
    x=torch.from_numpy(a.transpose(2,0,1))[None].cuda()/255
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


def predict(model,extra,spec,im):
    import torch
    import torch.nn.functional as F
    w,h=im.size;name=spec['model'];result={}
    with torch.inference_mode():
        if spec['task']=='classification':
            z=model(extra['transform'](im)[None].cuda()).float()
            if extra['head']=='coral':
                q=z.sigmoid();p=torch.stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]],1).clamp_min(1e-8);p=p/p.sum(1,keepdim=True)
                pred=int((q>.5).sum(1).item());result['cumulative']=q[0].cpu().numpy()
            else:p=z.softmax(-1);pred=int(z.argmax(1).item())
            result.update(scores=p[0].cpu().numpy(),prediction=np.array(pred),epoch=np.array(extra['epoch']),head=np.array(extra['head']))
        elif spec['task']=='geometry':
            z=model(tensor(im,[512,384]))
            if name=='hrnet':
                from original_visual_support import positions
                result['positions']=positions(z)[0].cpu().numpy()
            else:result['masks']=(F.interpolate(z.logits.float(),(h,w),mode='bilinear',align_corners=False).sigmoid()[0]>=.5).cpu().numpy()
        elif spec['backend']=='semantic':
            z=model(tensor(im,[512,512]));z=z.logits if hasattr(z,'logits') else z
            z=F.interpolate(z,(512,512),mode='bilinear',align_corners=False)
            result['masks']=(F.interpolate(z.float(),(h,w),mode='bilinear',align_corners=False).sigmoid()[0]>=.5).cpu().numpy()
        elif spec['backend']=='yolo':
            r=model.predict(im,imgsz=512,conf=.001,device=0,verbose=False,retina_masks=True,max_det=100)[0]
            boxes=[];masks=np.zeros((2,h,w),bool)
            for i,(box,score,cls) in enumerate(zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.int().cpu().tolist())):
                if score>=.25:
                    boxes.append([*box,score,cls])
                    if r.masks is not None:masks[cls]|=F.interpolate(r.masks.data[i].float()[None,None],(h,w),mode='nearest')[0,0].cpu().numpy()>=.5
            result['boxes']=np.array(boxes).reshape(-1,6)
            if name.endswith('_seg'):result['masks']=masks
        else:
            proc=extra['processor'];inputs=proc(images=im,return_tensors='pt').to('cuda')
            r=proc.post_process_object_detection(model(**inputs),target_sizes=torch.tensor([[h,w]],device='cuda'),threshold=.25)[0]
            result['boxes']=np.array([[*b,s,c] for b,s,c in zip(r['boxes'].cpu().tolist(),r['scores'].cpu().tolist(),r['labels'].cpu().tolist())]).reshape(-1,6)
    return result


def run(rows,registry,assets,work,publisher,keys,local=None):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Enable a Kaggle GPU for the original all-model comparison.')
    try:
        for index,key in enumerate(keys):
            spec=registry[key];pending=[]
            for row in rows:
                p=publisher.out/(row['key']+'__'+key+'.npz')
                try:
                    with np.load(p,allow_pickle=False) as f:
                        assert str(f['image_sha256'])==row['sha256'] and str(f['checkpoint_sha256'])==spec['sha256']
                except (OSError,ValueError,KeyError,AssertionError,EOFError,zipfile.BadZipFile):pending.append(row)
            if not pending:print(key,'already complete');continue
            path=Path(local[key]) if local and key in local else download(spec,work,publisher)
            if sha(path)!=spec['sha256']:raise ValueError('Checkpoint SHA mismatch: '+key)
            model,extra=load(spec,path,assets,work)
            try:
                for row in pending:
                    result=predict(model,extra,spec,source(row))
                    result.update(image_sha256=np.array(row['sha256']),checkpoint_sha256=np.array(spec['sha256']))
                    p=publisher.out/(row['key']+'__'+key+'.npz');tmp=p.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,**result)
                    tmp.replace(p);print(key,row['image_id'],int(result['prediction']) if 'prediction' in result else 'overlay ready',flush=True)
                    if time.time()-publisher.last>=1800:publisher.publish('30-minute checkpoint')
            finally:
                del model;gc.collect();torch.cuda.empty_cache()
                if extra.get('native_file'):Path(extra['native_file']).unlink(missing_ok=True)
                # Delete only the checkpoint downloaded into this notebook's own storage.
                if path.resolve().is_relative_to((Path(work)/'active_checkpoint').resolve()):path.unlink(missing_ok=True)
            if (index+1)%4==0 or index==len(keys)-1:
                publisher.publish(f'{index+1}/{len(keys)} model pipelines complete')
    except BaseException:
        try:publisher.publish('interrupted; completed predictions retained')
        except Exception as e:print('Upload unavailable:',type(e).__name__,'Local results retained.')
        raise


def render(rows,keys,registry,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle
    out=Path(out);figures=[];table=[];plt.rcParams.update({'figure.facecolor':'#f5f8fa','font.size':9})
    classifiers=[k for k in keys if registry[k]['task']=='classification']
    localisers=[k for k in keys if registry[k]['task']=='localisation']
    def read(row,key):
        with np.load(out/(row['key']+'__'+key+'.npz'),allow_pickle=False) as f:return {k:f[k] for k in f.files}
    for row in rows:
        im=source(row);w,h=im.size
        if classifiers:
            fig,axs=plt.subplots(1,2,figsize=(13,max(7,len(classifiers)*.36)),gridspec_kw={'width_ratios':[1,2]})
            axs[0].imshow(im);axs[0].axis('off');axs[0].set_title('Original final_v1 photograph')
            scores=[];labels=[]
            for k in classifiers:
                v=read(row,k);p=int(v['prediction']);scores.append(v['scores']);labels.append(k+f' · e{int(v["epoch"])} → '+CLASSES[p]+(' ✓' if p==row['class_index'] else ' ×'))
                table.append(dict(image_id=row['image_id'],supplied=CLASSES[row['class_index']],model=k,epoch=int(v['epoch']),prediction=CLASSES[p],head=str(v['head']),low=float(v['scores'][0]),medium=float(v['scores'][1]),high=float(v['scores'][2])))
            axs[1].imshow(scores,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');axs[1].set_yticks(range(len(labels)),labels);axs[1].set_xticks(range(3),CLASSES)
            for i,values in enumerate(scores):
                for j,p in enumerate(values):axs[1].text(j,i,f'{p:.1%}',ha='center',va='center',color='white' if p>.6 else '#183847')
            axs[1].set_title('Class scores · actual decision shown at left\nCORAL uses threshold-count, not score argmax')
            fig.suptitle(f'{CLASSES[row["class_index"]].upper()} USE · {row["image_id"]}\nOriginal fold-1 validation · seed 1',fontsize=12)
            fig.tight_layout(rect=[0,.04,1,.93]);fig.text(.02,.015,'Mileage proxy labels. Selected-checkpoint validation illustrations are not an untouched test benchmark.',fontsize=9)
            p=out/(row['key']+'_classification.png');fig.savefig(p,dpi=130);plt.close(fig);figures.append(p)
        if localisers:
            fig,axes=plt.subplots(3,3,figsize=(13,15));axes=axes.ravel()
            for ax,key in zip(axes,localisers):
                v=read(row,key);ax.imshow(im);empty=True
                if 'masks' in v:
                    for mask,color in zip(v['masks'],[(0,.72,.8),(.98,.65,.1)]):
                        layer=np.zeros((*mask.shape,4));layer[mask]=[*color,.3];ax.imshow(layer);empty=empty and not mask.any()
                        if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors=[color],linewidths=.8)
                else:
                    for x0,y0,x1,y1,score,cls in v['boxes']:
                        color='#00b8cc' if cls==0 else '#faa619';ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=color,lw=2));ax.text(x0,y0,f'{["tyre","tread"][int(cls)]} {score:.2f}',color='white',bbox=dict(facecolor=color,alpha=.8));empty=False
                ax.set_title(key+('\nNO DETECTION' if empty else ''));ax.axis('off')
            for ax in axes[len(localisers):]:ax.axis('off')
            fig.suptitle(f'{CLASSES[row["class_index"]].upper()} · ORIGINAL LOCALISATION MODELS\nCyan: tyre · amber: tread · boxes remain boxes; no invented masks',fontsize=13)
            fig.tight_layout(rect=[0,0,1,.94]);p=out/(row['key']+'_regions.png');fig.savefig(p,dpi=120);plt.close(fig);figures.append(p)
        if all(k in keys for k in ['geometry-hrnet','geometry-matched']):
            hp=read(row,'geometry-hrnet')['positions'];mask=read(row,'geometry-matched')['masks'][1]
            ys=np.rint(np.repeat([384,768,1151],2)/1535*(h-1));hp=np.column_stack([hp*(w-1),ys]);sp=boundaries(mask)
            valid=np.isfinite(sp[:,0]);gap=float(abs(hp[valid,0]-sp[valid,0]).mean()/(w-1)*100) if valid.any() else None
            fig,axes=plt.subplots(1,3,figsize=(12,6))
            for ax,title in zip(axes,['HRNet W18','Matched SegFormer B0','Boundary disagreement']):ax.imshow(im);ax.set_title(title);ax.axis('off')
            draw_lines(axes[0],hp,'#d43091','HRNet');draw_lines(axes[1],sp,'#00bed1','SegFormer');draw_lines(axes[2],hp,'#d43091','HRNet');draw_lines(axes[2],sp,'#00bed1','SegFormer');axes[2].legend(fontsize=8)
            fig.suptitle(f'Original geometry models · geometry cohort: {row["geometry_role"]}\nCoverage {valid.sum()}/6 · gap '+(f'{gap:.2f}% width' if gap is not None else 'unavailable')+(' · CROSSED HRNet SIDES' if np.any(hp[::2,0]>=hp[1::2,0]) else ''),fontsize=12)
            fig.tight_layout(rect=[0,.04,1,.9]);fig.text(.03,.02,'Image-relative proposals; gap is disagreement, not ground-truth error or physical alignment.',fontsize=9)
            p=out/(row['key']+'_geometry.png');fig.savefig(p,dpi=130);plt.close(fig);figures.append(p)
    pd.DataFrame(table).to_csv(out/'classification_predictions.csv',index=False)
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Original study comparison</title><style>body{font:16px system-ui;background:#f5f8fa;margin:25px}img{display:block;max-width:100%;margin:20px auto}nav{position:sticky;top:0;background:white;padding:15px}a{margin:15px}</style><h1>Original paper study · model comparison</h1><p>Original final_v1 data only. No Phase 2 weights or data. Class scores are uncalibrated; failures are retained.</p><nav>'+''.join(f'<a href="#{r["key"]}">{CLASSES[r["class_index"]]} {r["key"]}</a>' for r in rows)+'</nav>'+''.join(f'<section id="{r["key"]}"><h2>{html.escape(r["image_id"])}</h2>'+''.join(f'<img loading="lazy" src="{p.name}">' for p in figures if p.name.startswith(r['key']+'_'))+'</section>' for r in rows),encoding='utf-8')
    return figures
