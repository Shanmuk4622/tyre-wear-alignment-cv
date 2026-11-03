"""Bounded saved-table readers and exact existing mask/annotation adapter."""
import csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from common import HERE,read,sha

PIN=read(HERE/'FIT_INPUT.json')

class Saved:
    def __init__(self,root,manifest):
        self.root=Path(root);self.manifest=manifest;self.contract=read(self.root/'CONTRACT.json')
        self.schema=self.contract['schema']['families']
        assert self.contract['mode']=='measured_single_candidate'
    def paths(self,family):
        return [self.root/n for n in sorted(self.manifest['files']) if n.startswith('csv/'+family+'/') or n.startswith('active/csv_'+family+'_')]
    def table(self,family,usecols=None):
        frames=[]
        # Text columns remain text (including session-scoped HTTP event IDs).
        types={c['name']:str for c in self.schema[family]['columns'] if c['dtype']=='utf8'}
        for path in self.paths(family):
            df=pd.read_csv(path,usecols=usecols,dtype={k:v for k,v in types.items() if usecols is None or k in usecols},keep_default_na=True)
            if len(df):frames.append(df)
        columns=usecols or [c['name'] for c in self.schema[family]['columns']]
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
    def vector(self,family,observations,dimensions):
        result=np.full((PIN['expected_samples'],2,dimensions),np.nan,dtype='float32');count=0
        for path in self.paths(family):
            frame=pd.read_csv(path,usecols=['sample','view','dimension','value'])
            if not len(frame):continue
            ids=frame['sample'].to_numpy(dtype='int64');views=frame['view'].to_numpy(dtype='int64');dims=frame['dimension'].to_numpy(dtype='int64')
            assert np.isnan(result[ids,views,dims]).all(),'Duplicate vector coordinate'
            result[ids,views,dims]=frame['value'].to_numpy(dtype='float32');count+=len(frame)
        assert np.isfinite(result).all() and count==PIN['expected_samples']*2*dimensions
        return result

def locate(value=''):
    if value:roots=[Path(value)]
    elif Path('/kaggle/input').exists():
        roots=[p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
    else:roots=[HERE.parents[2]/'phase2/data/phase2_dataset_v1']
    assert len(roots)==1,'Attach exactly one existing expanded shanmuk4622/tire-dataset-prepared-phase2 dataset'
    root=roots[0].resolve();assert sha(root/'SHA256SUMS.txt')==PIN['source_checksum'],'Wrong prepared data release'
    return root

def ground_truth(root,source):
    """No image/model forward: load the existing masks once and cache coarse counts."""
    checksum={name:expected for expected,name in (line.split('  ',1) for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines())}
    registry={r['image_id']:r for r in read(root/'manifests/images.json')};result={}
    for row in source.itertuples():
        original=registry[row.image_id];assert original['image_sha256']==row.image_sha256
        values=[]
        for field in ('tyre_mask','tread_mask','ignore_mask'):
            name=original[field];path=(root/name).resolve();assert path.is_relative_to(root) and sha(path)==checksum[name],name
            with Image.open(path) as image:values.append(np.asarray(image.resize((384,512),Image.Resampling.NEAREST))>0)
        masks=np.stack(values);valid=~masks[2]
        coarse=masks[:2].reshape(2,128,4,96,4).mean((2,4)).astype('float32')
        coarse_valid=valid.reshape(128,4,96,4).all((1,3))
        foreground=(masks[:2]&valid).reshape(2,128,4,96,4).sum((2,4)).astype('uint8')
        available=valid.reshape(128,4,96,4).sum((1,3)).astype('uint8')
        result[row.image_id]=dict(target=coarse,valid=coarse_valid,foreground=foreground,available=available)
    assert len(result)==570
    return result

def sigmoid(x):
    z=np.exp(-np.abs(x));return np.where(x>=0,1/(1+z),z/(1+z))

def latent(z):
    z=np.asarray(z,dtype='float64')
    if len(z)<2:return dict(images=len(z),effective_rank=0.,std_median=None,std_min=None,leading_fraction=None)
    cov=np.cov(z,rowvar=False,ddof=1);eig=np.linalg.eigvalsh(cov).clip(0);total=eig.sum();p=eig[eig>0]/total if total else []
    return dict(images=len(z),effective_rank=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,std_median=float(np.median(np.sqrt(np.maximum(0,np.diag(cov))))),std_min=float(np.sqrt(np.maximum(0,np.diag(cov))).min()),leading_fraction=float(eig[-1]/total) if total else 0.)
