"""Inject interruptions into real checkpoint publication; no network or training."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import s5_notebook as sn
import s5_runtime as rt
import s5_data as d
import tyrelib as tl


def check():
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)
        state=dict(epoch=1,history=[{'epoch':1}],plan_hash='fixture',job={'backend':'semantic'},model={})
        rt.publish_local(out,state)
        original=rt.os.replace
        def interrupt_after_checkpoint(src,dst):
            original(src,dst)
            if Path(dst).name=='state.pt':
                raise KeyboardInterrupt('injected stop between checkpoint and sidecars')
        newer=dict(state,epoch=2,history=[{'epoch':1},{'epoch':2}])
        with patch.object(rt.os,'replace',side_effect=interrupt_after_checkpoint):
            try:rt.publish_local(out,newer)
            except KeyboardInterrupt:pass
            else:raise AssertionError('No fault injected')
        assert json.loads((out/'STATUS.json').read_text())['epoch']==1
        digest=d.digest(out/'state.pt')
        sn.repair_local_metadata(out)
        status=json.loads((out/'STATUS.json').read_text())
        assert status['epoch']==2 and status['checkpoint_sha256']==digest
        assert d.digest(out/'state.pt')==digest
        assert len(sn.pd.read_csv(out/'epochs.csv'))==2
        # New journal but checkpoint replacement never happened: keep old epoch.
        def interrupt_before_checkpoint(src,dst):
            if Path(dst).name=='state.pt':raise KeyboardInterrupt('before swap')
            original(src,dst)
        with patch.object(rt.os,'replace',side_effect=interrupt_before_checkpoint):
            try:rt.publish_local(out,dict(newer,epoch=3,history=[{'epoch':i} for i in range(1,4)]))
            except KeyboardInterrupt:pass
        sn.repair_local_metadata(out)
        assert json.loads((out/'STATUS.json').read_text())['epoch']==2
        # First-ever checkpoint interrupted before STATUS exists.
        first=out/'first'
        with patch.object(rt.os,'replace',side_effect=interrupt_after_checkpoint):
            try:rt.publish_local(first,state)
            except KeyboardInterrupt:pass
        sn.repair_local_metadata(first)
        assert json.loads((first/'STATUS.json').read_text())['epoch']==1
    with patch.object(tl,'container_memory',return_value=(95,100,'cgroup:v2')):
        with patch.object(Path,'read_text',return_value='inactive_file 20\nfile_dirty 2\nfile_writeback 3\n'):
            assert sn.memory_pressure()['working']==80
        with patch.object(Path,'read_text',side_effect=OSError('missing')):
            assert sn.memory_pressure()['working']==95
    print('PASS: stop before/after checkpoint replacement, first-save interruption, intact weights, conservative cache accounting')


if __name__=='__main__':check()
