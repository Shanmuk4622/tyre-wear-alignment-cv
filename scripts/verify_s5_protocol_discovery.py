"""Regression for blank PREFIX and four-worker PILOT; no HF writes/training."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import s5_data as d
import s5_notebook as sn


def check():
    with tempfile.TemporaryDirectory() as temp:
        sess=SimpleNamespace(stage_dir=Path(temp),worker_id=1,num_workers=4)
        plan=d.protocol({'records':[], 'splits':{}})
        p=Path(temp)/'protocol.json';p.write_text(json.dumps(plan))
        prefix=sn.plan_prefix(plan)
        entries=[SimpleNamespace(path=prefix+'/protocol.json')]
        api=SimpleNamespace(list_repo_tree=lambda *args,**kwargs:entries)
        with patch.object(sn,'token',return_value=False),patch.object(sn,'HfApi',return_value=api),patch.object(sn,'pull',return_value=p):
            assert sn.resolve_prefix(sess,'','pinned')==prefix
            assert sn.resolve_prefix(sess,' '+prefix+'/ ','pinned')==prefix
            with patch.object(sn,'revision',return_value='pinned'),patch.object(d,'inspect_data',return_value=plan['data']):
                assert sn.load_plan(sess,'',None,None)==plan
            entries.clear()
            try:sn.resolve_prefix(sess,'','pinned')
            except RuntimeError as exc:assert 'No compatible' in str(exc)
            else:raise AssertionError('Missing protocol accepted')
            # Two valid same-code protocols must not silently choose latest.
            plan2=dict(plan,source_revision='other');p2=Path(temp)/'other.json';p2.write_text(json.dumps(plan2))
            prefix2=sn.plan_prefix(plan2)
            entries.extend([SimpleNamespace(path=prefix+'/protocol.json'),SimpleNamespace(path=prefix2+'/protocol.json')])
            with patch.object(sn,'pull',side_effect=lambda sess,rel,*args:p2 if rel.startswith(prefix2+'/') else p):
                try:sn.resolve_prefix(sess,'','pinned')
                except RuntimeError as exc:assert 'Multiple compatible' in str(exc)
                else:raise AssertionError('Ambiguous protocol selected')
        with patch.object(sn,'launch',side_effect=AssertionError('Non-owner launched pilot')):
            sn.run_family(sess,plan,prefix,None,None,'semantic','PILOT')
    print('PASS blank/explicit/missing/ambiguous protocol handling and non-owner PILOT skip')


if __name__=='__main__':
    check()
