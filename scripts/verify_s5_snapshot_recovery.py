"""Offline fault injection for S5 snapshot publication and sidecar recovery."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tyrelib'))
os.environ['PYTHONPATH'] = str(ROOT/'tyrelib') + os.pathsep + os.environ.get('PYTHONPATH', '')
import torch
import tyrelib as tl
import s5_notebook as sn
import s5_data as d


def check():
    # Background commit drains the queue. Foreground flush must still wait.
    uploader = tl.Uploader('fixture', '', enabled=False)
    uploader.enabled = True
    entered, release, flushed = threading.Event(), threading.Event(), threading.Event()
    def serial(*args):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return True
    with patch.object(uploader, '_push_batch_serial', side_effect=serial):
        first = threading.Thread(target=lambda: uploader._push_batch(False))
        first.start(); assert entered.wait(5)
        second = threading.Thread(target=lambda: (uploader.flush(), flushed.set()))
        second.start(); assert not flushed.wait(.1)
        release.set(); first.join(); second.join(); assert flushed.is_set()
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)/'job'; out.mkdir()
        plan = {'fixture': True}; job = dict(model='segformer_b0', backend='semantic', fold=1, seed=2, run_id='fixture')
        state = dict(plan_hash=d.signature(plan), job=job, epoch=45,
                     history=[{'epoch':i} for i in range(1,46)], model={}, optimizer={}, scheduler={}, scaler={}, rng={}, runtime={})
        torch.save(state, out/'state.pt')
        original = dict(plan_hash=d.signature(plan), job=job, epoch=44, checkpoint_sha256='stale', status='resumable', evaluated=False)
        digest = d.digest(out/'state.pt')
        repaired = sn.recover_sidecars(out/'state.pt', original, plan, job, 'pinned-fixture')
        assert repaired['epoch']==45 and repaired['checkpoint_sha256']==digest
        assert d.digest(out/'state.pt')==digest
        assert json.loads((out/'resume_recovery.json').read_text())['original_status']==original
        try: sn.recover_sidecars(out/'state.pt', dict(original,evaluated=True), plan, job, 'fixture')
        except RuntimeError: pass
        else: raise AssertionError('Completed evidence silently repaired')
        try: sn.recover_sidecars(out/'state.pt', original, {'wrong':True}, job, 'fixture')
        except RuntimeError: pass
        else: raise AssertionError('Wrong protocol accepted')
        batches = []
        class Queue:
            def enqueue_batch(self, pairs):
                pairs = list(pairs)
                assert len(pairs)>=3
                assert d.digest(pairs[0][0])==digest
                batches.append(pairs)
        sess = NS(uploader=Queue(), account='test', session_id='test')
        with patch.object(sn,'push'):
            sn.snapshot(sess,out,'remote','test')
            sn.snapshot(sess,out,'remote','test')
        assert batches[0][0][0] != batches[1][0][0], 'Snapshot files reused'
        # Batch enqueue does not leave a partial generation if a file is missing.
        uploader._buffer.clear()
        try: uploader.enqueue_batch([(out/'state.pt','state'),(out/'missing','status')])
        except FileNotFoundError: pass
        else: raise AssertionError('Missing file accepted')
        assert not uploader._buffer
    print('PASS: serialized flush, atomic queue, immutable snapshots, epoch45 recovery, identity/completed rejection')


if __name__=='__main__':
    check()
