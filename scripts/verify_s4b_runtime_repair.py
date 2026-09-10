"""Read-only HF audit: preserve four existing checkpoint hashes during runtime repair."""
import sys,io
from pathlib import Path
from types import SimpleNamespace
import requests
import pandas as pd
from huggingface_hub import HfApi,hf_hub_url
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tyrelib'))
import tyrelib as tl
from s4b_confirmation import FACTORS
repo='Shanmuk4622/tyre-wear-study'
rev='2def8d1b1f00f7c08fd029e8f339c5ab1004b6c8'
files=set(HfApi().list_repo_files(repo,repo_type='dataset',revision=rev))
jobs=[('sampler_classweighted',s) for s in (1,2,3)]+[('sampler_uniform',1)]
for factor,seed in jobs:
    cfg=tl.Session.config(SimpleNamespace(stage='c'),'convnextv2_t',1,seed,stage='c',
        technique='s4b_r1_'+factor,_strict_resume=True,_single_gpu=True,**FACTORS[factor])
    prefix='runs/'+cfg['run_id']
    response=requests.get(hf_hub_url(repo,prefix+'/STATUS.json',repo_type='dataset',revision=rev),timeout=60)
    response.raise_for_status();status=response.json()
    # Trainer.__init__ computes this hash before loading its checkpoint.
    assert tl.config_hash(cfg)==status['config_hash']
    for name in ('ckpt_last.pt','ckpt_best.pt'):
        assert prefix+'/checkpoints/'+name in files
    response=requests.get(hf_hub_url(repo,prefix+'/metrics/epochs.csv',repo_type='dataset',revision=rev),timeout=60)
    response.raise_for_status();history=pd.read_csv(io.BytesIO(response.content))
    assert list(history.epoch)==list(range(1,status['epoch']+1))
    print(cfg['run_id'],status['epoch'],'epochs; preserved hash',status['config_hash'],
          'median epoch seconds',round(history.epoch_seconds.median()))
print('PASS: all four published config hashes preserved; complete epoch histories and both checkpoint paths present')
