"""Fault injection: exact per-checkpoint NumPy restoration without CUDA replacement."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import s5_notebook as sn
import s5_runtime as rt


def check():
    current=dict(torch='2.10.0+cu128',torchvision='0.25.0+cu128',numpy='2.0.2',transformers='4.51.3')
    wanted=dict(current,numpy='2.4.6')
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'job';out.mkdir();state=out/'state.pt';state.write_bytes(b'checkpoint-evidence')
        calls=[]
        def run(args,**kwargs):
            calls.append(args)
            if '-m' in args:
                assert '--no-deps' in args and 'numpy==2.4.6' in args
                assert not any(a.startswith('torch==') for a in args)
                return NS(returncode=0,stdout='',stderr='')
            if "s=torch.load" in args[2]:
                return NS(returncode=0,stdout=json.dumps(wanted),stderr='')
            restored='numpy_2.4.6' in kwargs['env'].get('PYTHONPATH','')
            versions=wanted if restored else current
            return NS(returncode=0,stdout=json.dumps(dict(versions=versions,loaded_numpy=versions['numpy'])),stderr='')
        with patch.object(sn.subprocess,'run',side_effect=run),patch.object(rt,'runtime_versions',return_value=current):
            env=sn.worker_environment(out,dict(os.environ))
        assert 'numpy_2.4.6' in env['PYTHONPATH']
        assert state.read_bytes()==b'checkpoint-evidence'
        assert json.loads((out/'worker_environment.json').read_text())['expected']==wanted
        assert any('-m' in x for x in calls)
        state.unlink()  # only this test-created temporary fixture
        bad=dict(current,torch='different-CUDA-build')
        result=NS(returncode=0,stdout=json.dumps(dict(versions=bad,loaded_numpy=bad['numpy'])),stderr='')
        with patch.object(sn.subprocess,'run',return_value=result) as command,patch.object(rt,'runtime_versions',return_value=current):
            try:sn.worker_environment(out,dict(os.environ))
            except RuntimeError as exc:assert 'Non-NumPy' in str(exc)
            else:raise AssertionError('CUDA mismatch was silently accepted')
            assert command.call_count==1
    print('PASS saved NumPy wins; isolated no-deps install; checkpoint preserved; CUDA mismatch rejected')


if __name__=='__main__':check()
