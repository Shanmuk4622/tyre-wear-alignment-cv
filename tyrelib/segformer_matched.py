"""Matched-split SegFormer experiment. Reuses frozen, single-writer resume engine.

Runtime bindings are process-local; no HRNet source or artifact is changed.
"""
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import hrnet_protocol as p
import hrnet_runtime as h
import hrnet_amp_repair as amp

VERSION = 'segformer-matched-2026-09-15-r1'
HR_KEY = '351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712'
HR_PREFIX = 's9/hrnet-geometry-2026-09-15-r1/' + HR_KEY
HR_REV = 'a92c0f9c5c1b78c6a06e13d51e18722195230658'
S5_KEY = '1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df'
S5_PATH = 's5/s5-manual-2026-09-10-r1/' + S5_KEY + '/protocol.json'
S5_REV = '05bf37c0f2067b0119ef057a2442d7759cf9ac51'
ORIGINAL_BATCH = h.load_batch
ORIGINAL_PUBLISH = p.publish
MASKS = None

def fetch(path, revision, token=None):
    import requests
    def download():
        # Public immutable JSON, bounded while streaming; no global-cache permissions needed.
        with requests.get(f'https://huggingface.co/datasets/{p.REPO}/resolve/{revision}/{path}',
                stream=True,timeout=60) as response:
            response.raise_for_status(); chunks=[]; size=0
            for chunk in response.iter_content(65536):
                size+=len(chunk)
                assert size < 5*1024**2, 'Unexpected metadata size'
                chunks.append(chunk)
        return json.loads(b''.join(chunks))
    return p.retry(download)

def contract(work=None):
    old = fetch(HR_PREFIX + '/report/CONTRACT.json', HR_REV)
    assert p.sha(p.canonical(old)) == HR_KEY
    s5 = fetch(S5_PATH, S5_REV)
    assert p.sha(p.canonical(s5)) == S5_KEY
    cfg = {k: copy.deepcopy(old[k]) for k in ['rows', 'groups', 'identity', 'input_hw', 'batch_size',
        'epochs', 'seeds', 'lr', 'weight_decay', 'point_names', 'annotations_sha256', 'package_id']}
    assert [sum(r['role'] == role for r in cfg['rows']) for role in ['train','validation','test']] == [72,24,24]
    records = {r['image_id']: r for r in s5['data']['records']}
    cfg['mask_sha256'] = {}
    for r in cfg['rows']:
        assert records[r['image_id']]['image_sha256'] == r['image_sha256']
        if r['role'] == 'train': cfg['mask_sha256'][r['image_id']] = records[r['image_id']]['mask_sha256']
    cfg.update(version=VERSION, hrnet_protocol=HR_KEY, hrnet_revision=HR_REV, s5_protocol=S5_KEY,
        s5_revision=S5_REV, model='nvidia/mit-b0', pretrained_revision=s5['model_revisions']['nvidia/mit-b0'],
        packages={'transformers':'4.51.3','tokenizers':'0.21.4','huggingface_hub':'0.36.0','safetensors':'0.5.3'}, optimizer='AdamW', scheduler='cosine_epoch',
        augmentation='none', batchnorm='frozen running statistics', endpoint='fixed_epoch_60',
        loss='mean binary cross entropy plus mean soft Dice; overlapping tyre (>0) and tread (2 or 3)',
        extraction='native-resolution bilinear logits; tread sigmoid >= 0.5; guide-row extrema; reject frame edges/empty rows',
        fallback='train-only point-coordinate mean for each missing boundary; report raw coverage and conditional error separately',
        limitations=['matched data split/budget/endpoint, NOT equal supervision: dense masks versus six points',
            'different pretrained backbones; not a pure architecture causal ablation',
            'only two test tyres; already inspected cohort; exploratory, no test tuning',
            'not physical angles, safety, depth, or full S9 completion'])
    cfg['scripts'] = {n: p.file_sha(Path(__file__).parent/n) for n in
        ['segformer_matched.py','hrnet_runtime.py','hrnet_protocol.py','hrnet_amp_repair.py']}
    return cfg

def prefix(cfg): return 's9/' + VERSION + '/' + p.sha(p.canonical(cfg))

def locate_masks(root, cfg):
    candidates = [Path(root)/'annotations/clean/masks', Path(root).parent/'annotations/clean/masks']
    if Path('/kaggle/input').exists(): candidates += list(Path('/kaggle/input').glob('**/annotations/clean/masks'))
    matches = []
    for folder in dict.fromkeys(candidates):
        if all((folder/(key+'.png')).is_file() for key in cfg['mask_sha256']): matches.append(folder)
    assert matches, 'Attach the SAME prepared dataset with annotations/clean/masks used by NB14. No new labels needed.'
    folder = matches[0]
    for key, digest in cfg['mask_sha256'].items():
        file = folder/(key+'.png')
        assert p.file_sha(file) == digest, 'Training mask bytes differ: '+key
        r = next(r for r in cfg['rows'] if r['image_id']==key)
        with Image.open(file) as im:
            assert im.size == (r['width'],r['height'])
            assert set(np.unique(im)).issubset({0,1,2,3,4})
    return folder

def preflight(work, root, token):
    cfg = contract(); p.validate_images(cfg,root); locate_masks(root,cfg)
    out = Path(work)/'preflight'/p.sha(p.canonical(cfg)); out.mkdir(parents=True,exist_ok=True)
    p.write(out/'CONTRACT.json',cfg)
    p.write(out/'STATUS.json',dict(status='preflight_passed',protocol=p.sha(p.canonical(cfg)),
        images=120,training_masks=72,split=[72,24,24],tyres=[8,2,2],new_annotations_needed=False))
    ORIGINAL_PUBLISH(out,prefix(cfg)+'/preflight',token)
    print('PREFLIGHT PASSED. Same 72/24/24 images, 8/2/2 tyres; 72 frozen training masks verified. Run NB31.',flush=True)

class Segmentation(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        import transformers
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        assert transformers.__version__ == cfg['packages']['transformers']
        config=SegformerConfig.from_pretrained(cfg['model'],revision=cfg['pretrained_revision'])
        config.num_labels=2
        self.net=SegformerForSemanticSegmentation(config)
        assert sum(v.numel() for v in self.parameters())==3714658, 'Unexpected SegFormer-B0 two-channel identity'
    def forward(self,x): return self.net(pixel_values=x).logits
    def training_mode(self):
        self.train()
        for module in self.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()

def pretrained(model,cfg,work,token):
    from transformers import SegformerModel
    encoder=SegformerModel.from_pretrained(cfg['model'],revision=cfg['pretrained_revision'],token=token,
        use_safetensors=False,weights_only=True)
    model.net.segformer.load_state_dict(encoder.state_dict(),strict=True)
    del encoder
    return dict(model=cfg['model'],revision=cfg['pretrained_revision'],head='new random two-channel decoder',
        parameters=sum(v.numel() for v in model.parameters()),
        tensor_signature=p.sha(p.canonical({k:list(v.shape) for k,v in model.state_dict().items()})))

def load_batch(rows,root,cfg,device='cuda'):
    x,_=ORIGINAL_BATCH(rows,root,cfg,device)
    assert all(r['role']=='train' for r in rows), 'Dense-mask loader must never see held-out labels'
    yy=[]
    for r in rows:
        with Image.open(MASKS/(r['image_id']+'.png')) as im:
            m=np.asarray(im.resize(tuple(reversed(cfg['input_hw'])),Image.Resampling.NEAREST))
        yy.append(np.stack([m>0,(m==2)|(m==3)]).astype(np.float32))
    return x,torch.from_numpy(np.stack(yy)).to(device)

def loss(logits,target):
    z=F.interpolate(logits.float(),target.shape[-2:],mode='bilinear',align_corners=False)
    prob=z.sigmoid(); dims=(0,2,3)
    dice=(2*(prob*target).sum(dims)+1)/(prob.sum(dims)+target.sum(dims)+1)
    return F.binary_cross_entropy_with_logits(z,target)+1-dice.mean()

def boundaries(mask,guides):
    result=[]
    for y in guides:
        xs=np.flatnonzero(mask[y])
        for side in (0,-1):
            v=int(xs[side]) if len(xs) else None
            result.append(None if v is None or v in (0,mask.shape[1]-1) else v/(mask.shape[1]-1))
    return result

def evaluate(model,rows,root,cfg):
    model.eval(); records=[]; times=[]
    mean=np.mean([r['x'] for r in cfg['rows'] if r['role']=='train'],axis=0)
    with torch.inference_mode():
        for r in rows:
            x,_=ORIGINAL_BATCH([r],root,cfg)
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            native=F.interpolate(logits[:,1:2].float(),(r['height'],r['width']),mode='bilinear',align_corners=False)
            mask=(native[0,0]>=0).cpu().numpy()
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            pred=boundaries(mask,r['guide_y'])
            for i,v in enumerate(pred):
                effective=float(mean[i]) if v is None else v
                error=abs(effective-r['x'][i])
                records.append(dict(pilot_id=r['pilot_id'],tyre=r['session'],point=p.POINTS[i],
                    raw_prediction=v,prediction=effective,label=r['x'][i],fallback_used=v is None,
                    error_width_fraction=error,error_px=error*(r['width']-1)))
    covered=[r for r in records if not r['fallback_used']]
    return dict(n_images=len(rows),n_points=len(records),coverage=len(covered)/len(records),
        mean_width_error=float(np.mean([r['error_width_fraction'] for r in records])),
        conditional_mean_width_error=float(np.mean([r['error_width_fraction'] for r in covered])) if covered else None,
        note='mean_width_error includes predeclared train-mean fallback; inspect raw coverage',
        inference_seconds=times,latency_scope='forward + native interpolation + device transfer; excludes image loading; first sample warmup included',
        records=records)

def bind(cfg,root):
    global MASKS
    MASKS=locate_masks(root,cfg)
    p.contract=lambda work:cfg; p.prefix=prefix
    h.Geometry=Segmentation; h.pretrained=pretrained; h.load_batch=load_batch
    h.coordinate_loss=loss; h.step=amp.step; h.evaluate=evaluate
    def publish(folder,path,token):
        p.write(Path(folder)/'NUMERICS.json',dict(events_this_process=amp.EVENTS,policy='same batch retry; bounded AMP backoff then FP32'))
        for name in cfg['scripts']:
            (Path(folder)/name).write_bytes((Path(__file__).parent/name).read_bytes())
        return ORIGINAL_PUBLISH(folder,path,token)
    p.publish=publish

def compare(work,token):
    from huggingface_hub import HfApi
    cfg=contract(); rev=p.retry(lambda:HfApi(token=token).repo_info(p.REPO,repo_type='dataset').sha)
    key=p.sha(p.canonical(cfg)); out=Path(work)/'comparison'/key; out.mkdir(parents=True,exist_ok=True)
    test={r['pilot_id']:r for r in cfg['rows'] if r['role']=='test'}
    expected={(pid,n) for pid in test for n in p.POINTS}; results=[]; paired=[]
    fallback=np.mean([r['x'] for r in cfg['rows'] if r['role']=='train'],axis=0)
    for seed in cfg['seeds']:
        sets=[]
        for base,revision,protocol in [(HR_PREFIX,HR_REV,HR_KEY),(prefix(cfg),rev,key)]:
            remote=base+f'/runs/seed{seed}'
            st=fetch(remote+'/STATUS.json',revision,token)
            assert st['status']=='completed' and st['completed_epochs']==60 and st['protocol']==protocol and st['seed']==seed, 'Training not complete; rerun NB31'
            metric=fetch(remote+'/TEST_FINAL.json',revision,token); records=metric['records']
            assert len(records)==144 and {(r['pilot_id'],r['point']) for r in records}==expected
            for r in records:
                label=test[r['pilot_id']]['x'][p.POINTS.index(r['point'])]
                assert r['label']==label and r['tyre']==test[r['pilot_id']]['session']
                assert np.isfinite(r['prediction']) and 0<=r['prediction']<=1
                assert abs(abs(r['prediction']-label)-r['error_width_fraction'])<1e-9
            sets.append({(r['pilot_id'],r['point']):r for r in records})
        hr,sg=sets; covered=[k for k in expected if not sg[k]['fallback_used']]
        for k,r in sg.items():
            assert r['fallback_used'] == (r['raw_prediction'] is None)
            expected_prediction=float(fallback[p.POINTS.index(k[1])]) if r['fallback_used'] else r['raw_prediction']
            assert r['prediction']==expected_prediction, 'Unregistered boundary fallback or changed prediction'
        result=dict(seed=seed,hrnet_mean=float(np.mean([r['error_width_fraction'] for r in hr.values()])),
            segformer_fallback_mean=float(np.mean([r['error_width_fraction'] for r in sg.values()])),
            segformer_raw_coverage=len(covered)/144,
            segformer_conditional_mean=float(np.mean([sg[k]['error_width_fraction'] for k in covered])) if covered else None,
            hrnet_on_same_covered_points=float(np.mean([hr[k]['error_width_fraction'] for k in covered])) if covered else None,
            per_tyre={t:{name:float(np.mean([r['error_width_fraction'] for r in data.values() if r['tyre']==t]))
                for name,data in [('hrnet',hr),('segformer_with_fallback',sg)]} for t in cfg['groups']['test']})
        results.append(result)
        paired.extend(dict(seed=seed,pilot_id=k[0],point=k[1],tyre=hr[k]['tyre'],
            hrnet_error=hr[k]['error_width_fraction'],segformer_error=sg[k]['error_width_fraction'],
            fallback_used=sg[k]['fallback_used'],difference_segformer_minus_hrnet=sg[k]['error_width_fraction']-hr[k]['error_width_fraction']) for k in sorted(expected))
    report=dict(protocol=key,source_revision=rev,hrnet_revision=HR_REV,results=results,
        limitations=cfg['limitations'],decision='Review all seeds, raw coverage and both tyres before integration. No automatic winner or significance claim.')
    p.write(out/'REPORT.json',report);p.write(out/'PAIRED_POINTS.json',paired);p.write(out/'CONTRACT.json',cfg)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));xx=np.arange(3)
    ax.bar(xx-.18,[r['hrnet_mean']*100 for r in results],.36,label='HRNet')
    ax.bar(xx+.18,[r['segformer_fallback_mean']*100 for r in results],.36,label='SegFormer + fixed fallback')
    ax.set(xticks=xx,xticklabels=['Seed 1','Seed 2','Seed 3'],ylabel='Mean point error (% image width)',title='Same 24 images / two test tyres — exploratory')
    ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    ORIGINAL_PUBLISH(out,prefix(cfg)+'/comparison',token)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','train','compare'])
    parser.add_argument('--work',required=True);parser.add_argument('--root',default='');args=parser.parse_args()
    token=os.environ['HF_TOKEN'];Path(args.work).mkdir(parents=True,exist_ok=True)
    if args.mode=='compare': compare(args.work,token)
    else:
        root=p.root_data(args.root)
        if args.mode=='preflight':preflight(args.work,root,token)
        else:
            cfg=contract();bind(cfg,root)
            # Each GPU session verifies actual model continuation before any training.
            h.smoke(args.work,root,token)
            remaining=[]
            for seed in cfg['seeds']:
                st,_=h.pull_status(prefix(cfg)+f'/runs/seed{seed}/STATUS.json',token)
                if st and st['status']=='completed':
                    assert st['protocol']==p.sha(p.canonical(cfg)) and st['completed_epochs']==60 and st['seed']==seed
                    print(f'Seed {seed} already complete on HF; no checkpoint download or retraining.',flush=True)
                else: remaining.append(seed)
            h.train(args.work,root,token,seeds=remaining)
