"""Exact read-only Kaggle input, original role/component binding and source schedule."""
import csv,math
from pathlib import Path
import numpy as np
from common import HERE,read,sha,digest
from legacy_data import plan
from phase2_training_data import sample

def locate(root=''):
    roots=[Path(root)] if root else [p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
    if len(roots)!=1:raise ValueError('Attach exactly one expanded shanmuk4622/tire-dataset-prepared-phase2 input, or set DATA_ROOT.')
    root=roots[0].resolve();assert sha(root/'SHA256SUMS.txt')==read(HERE/'DATA_PIN.json')['checksum_file_sha256']
    return root

def dataset(root,verify=True):
    root=Path(root).resolve();overlay=plan(root)
    assert digest(overlay)==read(HERE/'DATA_PIN.json')['overlay_sha256'],'Original role/annotation overlay differs'
    if verify:
        for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected,name=line.split('  ',1);path=(root/name).resolve()
            assert path.is_relative_to(root) and sha(path)==expected,name
    with (HERE/'SOURCE_COMPONENTS.csv').open(encoding='utf-8',newline='') as stream:registry={r['image_id']:r for r in csv.DictReader(stream)}
    rows=sorted(overlay['rows'],key=lambda r:r['image_id'])
    for i,row in enumerate(rows):
        pin=registry[row['image_id']];assert row['image_sha256']==pin['image_sha256'] and row['role']==pin['historical_study_split']
        row['component']=pin['leakage_component'];row['slot']=i
    assert len(rows)==570
    return rows

def schedule(rows,seed,epoch,fixture=False):
    train=[r['slot'] for r in rows if r['role']=='train']
    if fixture:train=train[:8]
    gen=np.random.default_rng(seed*1000003+epoch);ordered=list(gen.permutation(train).astype(int))
    primary=[True]*len(ordered)
    missing=(-len(ordered))%8
    if missing:
        last=set(ordered[len(ordered)//8*8:]);pool=[i for i in train if i not in last]
        ordered+=list(gen.choice(pool,missing,replace=False).astype(int));primary+=[False]*missing
    batches=[(ordered[i:i+8],primary[i:i+8]) for i in range(0,len(ordered),8)]
    assert all(len(set(indices))==8 for indices,_ in batches)
    assert sorted(i for indices,flags in batches for i,p in zip(indices,flags) if p)==sorted(train)
    return [batches[0],(list(reversed(batches[0][0])),[False]*8)] if fixture else batches

def load_original(root,row):return sample(root,row,(512,384),seed=None,model='segformer')
