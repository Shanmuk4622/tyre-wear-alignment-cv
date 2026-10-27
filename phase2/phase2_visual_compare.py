"""NB07 inference gallery. Frozen exports, image/model checkpoints, no training."""
import gc
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

NAMES = ['mobilenetv4', 'resnet50', 'yolo26m', 'segformer', 'hrnet']
TITLES = ['MobileNet V4', 'ResNet50', 'YOLO26 Medium', 'SegFormer B0', 'HRNet W18']
CLASSES = ['Low', 'Medium', 'High']
REPO = 'Shanmuk4622/tyre-wear-study'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2), encoding='utf-8'); temp.replace(p)


def rgb(row):
    with Image.open(row['path']) as im:
        # Official prepared images have already been oriented. Custom photos use EXIF.
        if row['role'] == 'user supplied': im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert('RGB')).copy()


def select(data_root='', custom_root='', per_class=3):
    assert 1 <= per_class <= 30, 'Choose 1–30 images per category.'
    if custom_root:
        root = Path(custom_root); rows = []
        for c, folder in enumerate(['low', 'medium', 'high']):
            files = sorted(p for p in (root/folder).rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
            if not files: raise ValueError(f'No images in {root/folder}')
            for p in files[:per_class]:
                rows.append(dict(path=str(p), image_id=p.stem, class_index=c, role='user supplied', physical_tyre_id='unknown'))
    else:
        from phase2_training_data import locate, overlay
        root = locate(data_root); pool = overlay(root)['rows']; rows = []
        for c in range(3):
            # Selection is deterministic and independent of predictions. Diversify tyres first.
            subset = [r for r in pool if r['class_index']==c and r['domain']=='original_photo']
            role = 'test' if any(r['role']=='test' for r in subset) else 'train'
            bags = {}
            for r in sorted(subset, key=lambda r:r['image_id']):
                if r['role']==role: bags.setdefault(r['physical_tyre_id'], []).append(r)
            chosen = []
            while bags and len(chosen)<per_class:
                for group in list(sorted(bags)):
                    chosen.append(bags[group].pop(0))
                    if not bags[group]: del bags[group]
                    if len(chosen)==per_class: break
            if not chosen: raise ValueError('Missing category '+CLASSES[c])
            rows.extend(dict(r, path=str(root/r['image_path'])) for r in chosen)
    for i,r in enumerate(rows):
        r['image_sha256']=sha(r['path']); r['key']=f'{i:03d}_{r["image_sha256"][:12]}'
    return rows


class Publisher:
    """One batched output ZIP per checkpoint; conservative application-call budget."""
    def __init__(self, work, run_id, token, enabled=True):
        self.work=Path(work); self.out=self.work/'results'/run_id; self.out.mkdir(parents=True,exist_ok=True)
        self.prefix='phase2/visual-comparison-v1/'+run_id; self.enabled=enabled; self.last=time.time()
        self.calls=[]; self.token=token
        if enabled:
            from huggingface_hub import HfApi
            self.api=HfApi(token=token)
    def retry(self, fn):
        import email.utils
        for attempt in range(6):
            self.calls=[t for t in self.calls if t>time.time()-3600]
            while len(self.calls)>=60:
                time.sleep(2); self.calls=[t for t in self.calls if t>time.time()-3600]
            self.calls.append(time.time())
            try: return fn()
            except Exception as e:
                response=getattr(e,'response',None); status=getattr(response,'status_code',None)
                if status not in (None,429,500,502,503,504) or attempt==5: raise
                hint=getattr(response,'headers',{}).get('Retry-After','0')
                try: delay=float(hint)
                except ValueError:
                    try: delay=max(0,email.utils.parsedate_to_datetime(hint).timestamp()-time.time())
                    except Exception: delay=0
                delay=max(delay,min(300,10*2**attempt)); print(f'HF retry in {delay:.0f}s; local results preserved.')
                time.sleep(delay)
    def restore(self):
        if not self.enabled: return
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            p=self.retry(lambda:hf_hub_download(REPO,self.prefix+'/results.zip',repo_type='dataset',token=self.token,cache_dir=str(self.work/'hub_cache')))
        except EntryNotFoundError: return
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                target=(self.out/n).resolve()
                if not target.is_relative_to(self.out.resolve()): raise ValueError('Unsafe archive member')
            manifest=json.loads(z.read('SNAPSHOT.json'))
            for n,digest in manifest.items():
                if hashlib.sha256(z.read(n)).hexdigest()!=digest: raise ValueError('Remote result checksum mismatch')
                target=self.out/n
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(n))
        print('Recovered saved inference outputs from HF.')
    def publish(self, reason):
        if not self.enabled: return
        from huggingface_hub import CommitOperationAdd, hf_hub_download
        files=[p for p in self.out.rglob('*') if p.is_file() and p.suffix!='.tmp' and p.name!='SNAPSHOT.json']
        if not files: return
        write(self.out/'SNAPSHOT.json',{p.relative_to(self.out).as_posix():sha(p) for p in files})
        archive=self.work/'snapshot.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files+[self.out/'SNAPSHOT.json']: z.write(p,p.relative_to(self.out).as_posix())
        commit=self.retry(lambda:self.api.create_commit(REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=self.prefix+'/results.zip',path_or_fileobj=str(archive))],commit_message='NB07 '+reason))
        check=self.retry(lambda:hf_hub_download(REPO,self.prefix+'/results.zip',repo_type='dataset',revision=commit.oid,token=self.token,cache_dir=str(self.work/'hub_cache')))
        if sha(check)!=sha(archive): raise RuntimeError('HF snapshot verification failed')
        self.last=time.time(); print('HF verified:',commit.oid,reason)


def load_model(name, registry, runtime, device, work, publisher, local_weights=''):
    import torch
    import phase2_models as models
    meta=registry[name+'-combined']
    for f in ('phase2_models.py','phase2_training_data.py'):
        if sha(Path(runtime)/f)!=meta['contract']['runtime_sources'][f]: raise ValueError('Runtime contract mismatch: '+f)
    if local_weights: path=Path(local_weights)/name/'weights.pt'
    else:
        from huggingface_hub import hf_hub_download
        path=Path(publisher.retry(lambda:hf_hub_download(REPO,meta['weights'],repo_type='dataset',revision=meta['revision'],token=publisher.token,cache_dir=str(Path(work)/'weights_cache'))))
    if sha(path)!=meta['sha256']: raise ValueError('Checkpoint hash mismatch: '+name)
    if name=='segformer':
        from transformers import SegformerConfig,SegformerForSemanticSegmentation
        model=SegformerForSemanticSegmentation(SegformerConfig.from_dict(json.loads((Path(runtime)/'MODEL_CONFIG.json').read_text()))).to(device)
    else: model=models.build(name,Path(runtime),pretrained=False,device=device)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.eval()


def infer(model, a, name, meta, device):
    import torch
    import torch.nn.functional as F
    import phase2_models as models
    hw=meta['contract']['input_hw']
    x=np.asarray(Image.fromarray(a).resize(tuple(hw[::-1]),Image.Resampling.BILINEAR),dtype=np.float32).copy()
    x=torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)/255.
    if name!='yolo26m': x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    with torch.inference_mode():
        out=models.predict(model,{'img':x},name)
        if name in ('segformer','yolo26m'): out=F.interpolate(out.float(),a.shape[:2],mode='nearest').bool()
    return out[0].cpu().numpy()


def run(rows, registry, runtime, work, publisher, local_weights=''):
    import torch
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Inference device:',device,'| one model resident at a time | no training')
    try:
        for name in NAMES:
            pending=[]
            for r in rows:
                p=publisher.out/(r['key']+'_'+name+'.npz')
                valid=False
                if p.exists():
                    try:
                        with np.load(p,allow_pickle=False) as saved:
                            a=saved['output'];valid=bool(np.isfinite(a).all())
                            if name in ('mobilenetv4','resnet50'): valid=valid and a.shape==(3,) and abs(float(a.sum())-1)<1e-5
                            elif name=='hrnet': valid=valid and a.shape==(6,)
                            else: valid=valid and a.ndim==3 and a.shape[0]==2
                    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile): pass
                if not valid: pending.append(r)
            if not pending: print(name,'already complete'); continue
            if shutil.disk_usage(work).free<2*1024**3: raise RuntimeError('Less than 2 GiB free. Preserve results and free space before continuing.')
            model=load_model(name,registry,runtime,device,work,publisher,local_weights)
            try:
                for r in pending:
                    a=rgb(r)
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    start=time.perf_counter(); output=infer(model,a,name,registry[name+'-combined'],device)
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    p=publisher.out/(r['key']+'_'+name+'.npz'); temp=p.with_suffix('.tmp')
                    with temp.open('wb') as f: np.savez_compressed(f,output=output,seconds=elapsed)
                    temp.replace(p)
                    print(name,r['image_id'],f'{elapsed:.2f}s')
                    if time.time()-publisher.last>=1800: publisher.publish('30-minute checkpoint')
            finally:
                del model; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            publisher.publish(name+' complete')
    except BaseException:
        try: publisher.publish('interrupted; completed predictions saved')
        except Exception as e: print('Final upload unavailable:',type(e).__name__,'Keep local outputs.')
        raise


def boundaries(mask):
    h,w=mask.shape; ys=np.rint(np.array([384,768,1151])/1535*(h-1)).astype(int); pts=[]
    for y in ys:
        xs=np.flatnonzero(mask[y])
        pts.extend([(float(xs[j]),float(y)) if len(xs) and xs[j] not in (0,w-1) else (np.nan,float(y)) for j in (0,-1)])
    return np.array(pts)


def draw_lines(ax, p, color, label):
    ax.scatter(p[:,0],p[:,1],s=28,c=color,edgecolors='white',linewidths=.5,label=label)
    for indices in ([0,2,4],[1,3,5]):
        ax.plot(p[indices,0],p[indices,1],color=color,lw=1.7)


def render(rows, out, registry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Patch
    out=Path(out); reports=[]; images=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f4f7fa'})
    for r in rows:
        a=rgb(r);h,w=a.shape[:2]
        values={n:np.load(out/(r['key']+'_'+n+'.npz'))['output'] for n in NAMES}
        fig,axes=plt.subplots(2,3,figsize=(14,12),gridspec_kw={'height_ratios':[1,1]})
        axes[0,0].imshow(a); axes[0,0].set_title('SOURCE · supplied '+CLASSES[r['class_index']]); axes[0,0].axis('off')
        for ax,n,title in zip(axes[0,1:],NAMES[:2],TITLES[:2]):
            p=values[n];pred=int(p.argmax()); correct=pred==r['class_index']
            ax.barh(CLASSES,p,color=['#0a9396' if i==pred else '#d8e2e8' for i in range(3)])
            ax.set_xlim(0,1.12);ax.invert_yaxis();ax.set_xlabel('Softmax probability · uncalibrated')
            ax.set_title(title+' → '+CLASSES[pred]+('\nMatches supplied label' if correct else '\nDISAGREES with supplied label'),color='#087b71' if correct else '#b44f12')
            for i,v in enumerate(p): ax.text(float(v)+.02,i,f'{v:.1%}',va='center')
            reports.append(dict(image_id=r['image_id'],key=r['key'],supplied_class=CLASSES[r['class_index']],role=r['role'],model=n,prediction=CLASSES[pred],low=float(p[0]),medium=float(p[1]),high=float(p[2])))
        for ax,n,title in zip(axes[1,:2],NAMES[2:4],TITLES[2:4]):
            masks=values[n]; ax.imshow(a)
            for mask,color in zip(masks,[(0,.72,.8),(.98,.65,.1)]):
                rgba=np.zeros((*mask.shape,4));rgba[mask]=[*color,.32];ax.imshow(rgba)
                if mask.any() and not mask.all(): ax.contour(mask,levels=[.5],colors=[color],linewidths=.8)
            ax.set_title(title+' · tyre + tread'+('\nNO REGION DETECTED' if not masks.any() else ''));ax.axis('off')
            ax.legend(handles=[Patch(color='#00b8cc',label='Tyre'),Patch(color='#faa619',label='Tread')],loc='lower right',fontsize=8)
        ax=axes[1,2];ax.imshow(a)
        ys=np.repeat(np.array([384,768,1151])/1535*(h-1),2)
        hp=np.column_stack([values['hrnet']*(w-1),np.rint(ys)]);sp=boundaries(values['segformer'][1])
        draw_lines(ax,hp,'#d43091','HRNet');draw_lines(ax,sp,'#00bed1','SegFormer')
        valid=np.isfinite(sp[:,0]);delta=float(np.abs(hp[valid,0]-sp[valid,0]).mean()/(w-1)*100) if valid.any() else None
        crossed=bool(np.any(hp[::2,0]>=hp[1::2,0]));status='CROSSED HRNet sides' if crossed else 'Image-space proposals'
        ax.set_title('BOUNDARY AGREEMENT · '+status+'\n'+f'Paired coverage {valid.sum()}/6 · '+(f'gap {delta:.2f}% width' if delta is not None else 'gap unavailable'));ax.legend(loc='lower right',fontsize=8);ax.axis('off')
        for y in np.unique(ys): ax.axhline(y,color='white',alpha=.35,ls=':',lw=.7)
        disagreement=int(values['mobilenetv4'].argmax())!=int(values['resnet50'].argmax())
        fig.suptitle(f'{CLASSES[r["class_index"]].upper()} USE · {r["image_id"]}\n{r["role"].upper()} · '+('CLASSIFIERS DISAGREE' if disagreement else 'Classifiers agree'),fontsize=16,fontweight='bold')
        fig.text(.02,.015,'Supplied classes are mileage proxies. Masks show regions; boundary gap is model disagreement, not accuracy. No physical alignment or safety verdict.',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.93]);p=out/(r['key']+'_panel.png');fig.savefig(p,dpi=130);plt.close(fig);images.append(p)
        write(out/(r['key']+'_geometry.json'),dict(hrnet=hp.tolist(),segformer=[[None if not np.isfinite(x) else float(x),float(y)] for x,y in sp],coverage=int(valid.sum()),gap_percent_width=delta,crossed=crossed))
    pd.DataFrame(reports).to_csv(out/'classification_predictions.csv',index=False)
    # One compact strip per class, retaining every selected example without ranking.
    fig,axes=plt.subplots(len(rows),2,figsize=(10,max(3,len(rows)*.8)),squeeze=False)
    for i,r in enumerate(rows):
        for j,n in enumerate(NAMES[:2]):
            v=np.load(out/(r['key']+'_'+n+'.npz'))['output'];ax=axes[i,j]
            ax.imshow(v[None],vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            for k,p in enumerate(v): ax.text(k,0,f'{p:.1%}',ha='center',va='center',color='white' if p>.6 else '#102535')
            ax.set_yticks([0],[f'{CLASSES[r["class_index"]]} · {r["role"]}\n{r["image_id"][:18]}'] if j==0 else [''])
            ax.set_xticks(range(3),CLASSES if i==len(rows)-1 else ['']*3)
            if i==0:ax.set_title(TITLES[j])
    fig.suptitle('Prediction fingerprint · all selected examples · probabilities, not accuracy',y=1.01)
    fig.tight_layout();fig.savefig(out/'prediction_fingerprint.png',dpi=140,bbox_inches='tight');plt.close(fig)
    import html
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Tyre model comparison</title><style>body{background:#f4f7fa;font:16px system-ui;margin:25px}img{max-width:100%;display:block;margin:20px auto}nav{position:sticky;top:0;background:white;padding:15px}a{margin-right:20px}</style><h1>Tyre model comparison</h1><p>Five frozen Phase 2 models. Illustrative examples; no accuracy estimate. Medium examples in the prepared dataset are training-cohort images.</p><nav>'+''.join(f'<a href="#{c}">{c} use</a>' for c in CLASSES)+'</nav><img src="prediction_fingerprint.png">'+''.join(f'<section id="{c}"><h2>{c} use</h2>'+''.join(f'<img loading="lazy" src="{html.escape(p.name)}">' for r,p in zip(rows,images) if CLASSES[r['class_index']]==c)+'</section>' for c in CLASSES),encoding='utf-8')
    return images
