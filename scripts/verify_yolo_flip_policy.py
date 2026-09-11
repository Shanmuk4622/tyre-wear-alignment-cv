"""No-training regressions for the YOLO correction and run-all AUTO workflow."""
import ast
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import s5_data as d
import s5_runtime as r
import s5_notebook as sn


def check():
    Albumentations=type('Albumentations',(),{})
    alb=Albumentations();alb.transform=NS(transforms=[])
    trainer=NS(args=NS(augmentations=[]),train_loader=NS(dataset=NS(transforms=NS(transforms=[alb]))))
    r.verify_yolo_augmentations(trainer)
    alb.transform.transforms=[NS(p=.01)]
    try:r.verify_yolo_augmentations(trainer)
    except AssertionError:pass
    else:raise AssertionError('Hidden augmentation was accepted')
    tree=ast.parse((ROOT/'tyrelib/s5_runtime.py').read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert any(k.arg=='augmentations' and isinstance(k.value,ast.List) and not k.value.elts
               for call in calls for k in call.keywords)
    plan=d.protocol({'records':[],'splits':{}})
    old=dict(plan,implementation_sha256=dict(plan['implementation_sha256'],s5_runtime_py='wrong'))
    assert not sn.source_compatible(old)
    legacy=dict(plan,implementation_sha256=dict(plan['implementation_sha256']))
    legacy['implementation_sha256']['s5_runtime.py']=sn.PRE_REPAIR_RUNTIME
    assert sn.source_compatible(legacy)
    legacy['implementation_sha256']['s5_runtime.py']='unknown'
    assert not sn.source_compatible(legacy)
    with tempfile.TemporaryDirectory() as temp:
        remote=Path(temp)/'fake_hf';remote.mkdir()
        calls=[]
        def enqueue(path,rel,*args,**kwargs):
            target=remote/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        sess=NS(stage_dir=Path(temp),worker_id=0,num_workers=1,
                uploader=NS(enqueue=enqueue),guard=NS(near_limit=lambda **kwargs:False))
        def pull(sess,rel,*args,**kwargs):
            p=remote/rel
            return p if p.exists() else None
        def launch(sess,plan,job,root,annotations,out,prefix,action):
            calls.append(action)
            (out/'SMOKE.json').write_text(json.dumps(dict(status='passed',runtime={'test':'cpu'},resumed=action=='resume_test')))
        with patch.object(sn,'revision',return_value='pinned'),patch.object(sn,'pull',side_effect=pull),\
             patch.object(sn,'push'),patch.object(sn,'launch',side_effect=launch),\
             patch.object(r,'runtime_versions',return_value={'test':'cpu'}),\
             patch.object(d,'assigned',return_value=[]) as assigned:
            sn.run_family(sess,plan,sn.plan_prefix(plan),None,None,'yolo','AUTO')
            assert calls==['smoke','resume_test']*4
            assert assigned.call_count==1, 'AUTO did not proceed into training assignment'
            calls.clear()
            sn.run_family(sess,plan,sn.plan_prefix(plan),None,None,'yolo','AUTO')
            assert not calls, 'AUTO reran already verified corrected pilots'
            assert assigned.call_count==2
    print('PASS flip-only loader gate, explicit empty override, narrow source compatibility, AUTO pilot-to-training and rerun skip')


if __name__=='__main__':check()
