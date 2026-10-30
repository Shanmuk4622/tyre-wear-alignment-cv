"""Real frozen-model image-unit recovery and train-only proposal acceptance."""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
GATE=ROOT/'tread_two/appearance_gate';PILOT=ROOT/'tread_two/pilot'
sys.path[:0]=[str(GATE),str(PILOT),str(ROOT/'analysis')]
from common import read,sha,atomic
from gate_features import contrast,choose_bank
from gate_worker import SCHEMAS
from gate_controller import prepare
from store import Store
from check_tread_two_training import Service,fake_publisher
checks=[]
for path in GATE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'))
rules=read(GATE/'RULES.json')
gen=torch.Generator().manual_seed(811)
rgb=.2+.4*torch.rand((4,3,64,64),generator=gen)
valid=torch.ones((4,1,64,64))
base=contrast(rgb,valid,rules)
for factor in [.8,1.2]:
    value=contrast(rgb*factor,valid,rules)
    assert float((value-base).norm()/base.norm())<2e-4
gray=rgb.mean(1,keepdim=True).expand_as(rgb)
assert torch.allclose(contrast(gray,valid,rules),base,atol=5e-6)
assert torch.isfinite(contrast(torch.zeros_like(rgb),torch.zeros_like(valid),rules)).all()
checks.append('Masked four-channel local contrast is finite for empty support and scale/gray-consistent on unclipped textured synthetic input')
records=[];rng=np.random.default_rng(711)
for cls,counts in [(0,[7]),(1,[3,3,3]),(2,[3,2,5,10])]:
    for g,count in enumerate(counts):
        for i in range(count):
            for patch_index in [0,1]:
                records.append(dict(role='train',class_index=cls,domain='original_photo',physical_tyre_id=f'{cls}-{g}',image_id=f'{cls}-{g}-{i}',patch=patch_index,valid_fraction=1.,glare_fraction=0.,gray_std=.1,descriptor=rng.normal(size=256)))
chosen=choose_bank(records,rules)
assert len(chosen)==18 and len({r['image_id'] for r in chosen})==18
for cls,expected in [(0,1),(1,3),(2,4)]:
    group=[r for r in chosen if r['class_index']==cls]
    assert len(group)==6 and len({r['physical_tyre_id'] for r in group})==expected
for tyre in {r['physical_tyre_id'] for r in records}:
    all_ids={r['image_id'] for r in records if r['physical_tyre_id']==tyre}
    selected={r['image_id'] for r in chosen if r['physical_tyre_id']==tyre}
    assert len(all_ids-selected)>=1
bad=[dict(records[0],role='test')]
try: choose_bank(bad,rules)
except AssertionError: pass
else: raise AssertionError('Held-out role was accepted')
checks.append('Candidate bank uses distinct photos, balances available training tyres, leaves query photos and rejects held-out roles')
OUT=ROOT/'tread_two/gate_validation';OUT.mkdir(exist_ok=True)
fixture=Path(tempfile.mkdtemp(prefix='fixture_',dir=OUT))
cache=fixture/'assets/input_cache/artifacts/epochs/epoch_0018.pt';cache.parent.mkdir(parents=True)
known=ROOT/'tread_two/gate_validation/engineering_v2/assets/input_cache/artifacts/epochs/epoch_0018.pt'
shutil.copy2(known,cache)
args=argparse.Namespace(work=str(fixture),source=str(PILOT),review_source=str(ROOT/'tread_two/review'),manifest=str(ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit_MANIFEST.json'),anchor=str(ROOT/'completion_20261028/hf/report/weights.pt'),data_root=None)
with patch('requests.Session.get',side_effect=AssertionError('Validation must be offline')):
    cfg=prepare(args)
def worker(folder,limit):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    with (fixture/(folder.name+'.log')).open('ab') as stream:
        result=subprocess.run([sys.executable,'-B',str(GATE/'gate_worker.py'),'--config',str(fixture/'CONFIG.json'),'--folder',str(folder),'--limit-images',str(limit)],env=environment,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==91,'Inspect gate fixture log: '+str(folder)
expected=fixture/'expected';resumed=fixture/'resumed'
worker(expected,2);worker(resumed,1)
partial=Store(resumed,cfg['contract'],SCHEMAS)
assert partial.state['next_image']==1 and partial.state['updates']==0
snapshot=partial.export(fixture/'snapshot',GATE,'paused')
service=Service();hub=fake_publisher(service,fixture)
hub.destination='phase3/tread-two/atlas-v1/appearance-gate/'+cfg['protocol']
hub.claim();service.lost=True
receipt=hub.publish(snapshot,release=True)
downloaded=hub.download(receipt['revision'],fixture/'remote',cfg['protocol'])
restored=Store(fixture/'restored',cfg['contract'],SCHEMAS);restored.install(downloaded)
worker(fixture/'restored',1)
a=Store(expected,cfg['contract'],SCHEMAS);b=Store(fixture/'restored',cfg['contract'],SCHEMAS)
assert a.state['next_image']==b.state['next_image']==2 and a.state['model_sha256']==b.state['model_sha256']
omit={'compute_seconds','cuda_peak_allocated_dense_bytes','cuda_peak_allocated_head_bytes'}
counts={}
for family in SCHEMAS:
    x,y=list(a.rows(family)),list(b.rows(family));counts[family]=len(x)
    if family=='execution': x,y=([{k:v for k,v in row.items() if k not in omit} for row in group] for group in [x,y])
    assert x==y,family
def same(x,y):
    if isinstance(x,np.ndarray): np.testing.assert_array_equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x: same(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for v,w in zip(x,y): same(v,w)
    else: assert x==y
for i in [0,1]: same(torch.load(expected/f'artifacts/train/{i:04d}.pt',weights_only=False),torch.load(fixture/f'restored/artifacts/train/{i:04d}.pt',weights_only=False))
checks.append('Actual selected-model two-image uninterrupted and pause/native/fake-HF/fresh-process resume match every scientific CSV and native value')
path=next((fixture/'restored/csv').rglob('predictions.csv'));length=path.stat().st_size
with path.open('ab') as stream: stream.write(b'uncommitted interruption suffix')
Store(fixture/'restored',cfg['contract'],SCHEMAS)
assert path.stat().st_size==length
checks.append('Uncommitted CSV suffix is rolled back against the native image cursor; zero optimizer state is invented')
assert cfg['contract']['training_image_ids'] and len(cfg['contract']['training_image_ids'])==386
assert len(a.state['artifacts'])==2 and counts['predictions']==2*7 and counts['affinities']==2*7*32*18
assert all(sha(PILOT/name)==meta['sha256'] for name,meta in read(PILOT/'FILES.json').items())
baseline=read(OUT/'PRESERVATION.json')['notebooks']
assert all(sha(Path(item['path']))==item['sha256'] for item in baseline.values())
result=dict(status='passed',checks=checks,count=len(checks),fixture=str(fixture),actual_training_images_probed_in_recovery_fixture=2,scientific_rows=counts,
    zero_training_updates=True,validation_or_test_forwards=0,fake_HF_recovery_passed=True,network_requests=0,remote_writes=0,
    actual_dual_T4_or_live_gate_HF_tested=False,source_hashes={p.name:sha(p) for p in GATE.iterdir() if p.suffix=='.py' or p.name in ('RULES.json','UPSTREAM.json')})
atomic(OUT/'VALIDATION.json',result)
print(json.dumps(result,indent=2))
