"""Local S5 regression checks; no HF writes, no experiment training."""
import ast
import base64
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tyrelib'))
import s5_data as d
import s5_runtime as r


def test_geometry():
    mask = np.array([[0,1,2], [0,3,4]], dtype='uint8')
    regions = d.regions(mask)
    assert regions.shape==(2,2,3)
    assert regions[0].sum()==4 and regions[1].sum()==2
    assert d.box(regions[0])==[1,0,3,2]
    assert d.box(np.zeros((5,5),bool)) is None
    assert d.padded_box(None,100,80)==[0,0,100,80]
    assert d.padded_box([-20,-20,5,5],100,80)==[0,0,7,7]
    for region in regions:
        scores=d.mask_scores(region,region)
        assert scores['iou']==scores['dice']==scores['boundary_f1_2px']==1
    assert d.mask_scores(np.zeros((2,2)),np.zeros((2,2)))['iou'] is None
    try:
        d.regions(np.array([[5]]))
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown mask value accepted')
    target=r.coco_target(7,regions)
    assert [x['category_id'] for x in target['annotations']]==[0,1]
    assert target['annotations'][0]['bbox']==[1,0,2,2]


def test_plan():
    plan=d.protocol({'records':[], 'splits':{}})
    assert len(d.jobs(plan))==81
    assert [len(d.jobs(plan,k)) for k in ('semantic','yolo','rtdetr')]==[36,36,9]
    for family in ('semantic','yolo','rtdetr'):
        shards=[d.assigned(plan,family,w,4) for w in range(4)]
        flattened=[j['run_id'] for shard in shards for j in shard]
        assert len(flattened)==len(set(flattened))==len(d.jobs(plan,family))
        assert set(flattened)=={j['run_id'] for j in d.assigned(plan,family,0,1)}
    other=dict(plan, epochs=59)
    assert d.signature(plan)!=d.signature(other)
    return plan


def test_resume(plan):
    # Real optimizer/scheduler state roundtrip on CPU, no dataset training.
    net=torch.nn.Linear(2,2); optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60)
    loss=net(torch.ones(2,2)).square().sum();loss.backward();optimizer.step();scheduler.step()
    job=d.jobs(plan)[0]
    state=dict(plan_hash=d.signature(plan),job=job,epoch=1,history=[{'epoch':1}],runtime={'test':'cpu'},
               model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
    with tempfile.TemporaryDirectory() as temp, patch.object(r,'runtime_versions',return_value={'test':'cpu'}):
        out=Path(temp)
        r.publish_local(out,state)
        restored=torch.load(out/'state.pt',weights_only=False)
        r.check_checkpoint(restored,plan,job)
        again=torch.nn.Linear(2,2); opt2=torch.optim.AdamW(again.parameters(),lr=.99)
        again.load_state_dict(restored['model']);opt2.load_state_dict(restored['optimizer'])
        assert opt2.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
        assert all(torch.equal(a,b) for a,b in zip(net.parameters(),again.parameters()))
        status=json.loads((out/'STATUS.json').read_text())
        assert status['checkpoint_sha256']==d.digest(out/'state.pt') and not status['evaluated']
        for changed in [dict(plan,epochs=59)]:
            try:r.check_checkpoint(restored,changed,job)
            except AssertionError:pass
            else:raise AssertionError('Wrong protocol resumed')
        bad=dict(restored,history=[{'epoch':2}])
        try:r.check_checkpoint(bad,plan,job)
        except AssertionError:pass
        else:raise AssertionError('Incomplete epoch history accepted')


def test_notebooks():
    names=['NB13_S5_Prepare.ipynb','NB14_S5_Semantic.ipynb','NB15_S5_YOLO.ipynb',
           'NB16_S5_RTDETRv2.ipynb','NB17_S5_Report.ipynb']
    for name in names:
        nb=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
        sources=[]
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                source=''.join(cell['source']);ast.parse(source,filename=f'{name}:{i}');sources.append(source)
                assert cell['execution_count'] is None and not cell['outputs']
        joined='\n'.join(sources)
        for helper in ['s5_data.py','s5_runtime.py','s5_notebook.py']:
            encoded=base64.b64encode((ROOT/'tyrelib'/helper).read_bytes()).decode()
            assert encoded in joined
        assert 'HF_TOKEN' in json.dumps(nb)
        if name.startswith(('NB14','NB15','NB16')):
            mode = 'AUTO' if name.startswith('NB15') else ('TRAIN' if name.startswith('NB16') else 'PILOT')
            assert f"MODE = '{mode}'" in joined and "PREFIX = ''" in joined
            assert "'torch','torchvision','numpy'" in joined


if __name__=='__main__':
    test_geometry(); plan=test_plan(); test_resume(plan); test_notebooks()
    print('PASS S5 geometry, overlapping regions, 81 jobs, one/four-worker ownership,')
    print('checkpoint/optimizer roundtrip, mismatch rejection, notebook syntax and embedded sources.')
    print('GPU backend execution remains unverified: run each family PILOT on Kaggle first.')
