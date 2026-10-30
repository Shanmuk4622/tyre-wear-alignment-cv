"""Actual short training lifecycle: pause after a durable update, restore, finish, compare."""
from PIL import Image
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import threading
from pathlib import Path
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
import numpy as np
import torch
import worker
from common import atomic, delta, deterministic, digest, read
from data import plan
from evaluation import gallery


def run():
    root = ROOT.parent/'phase2/data/phase2_dataset_v1'
    p = plan(root)
    # Explicit tiny lifecycle fixture, never relabelled as the 60-epoch experiment.
    train = [r for r in p['rows'] if r['role'] == 'train'][:8]
    validation = [r for r in p['rows'] if r['role'] == 'validation' and r['point_kind'] == 'human'][:2]
    test = [r for r in p['rows'] if r['role'] == 'test' and r['point_kind'] == 'human'][:2]
    tiny = dict(p, rows=train+validation+test)
    cfg = dict(root=str(root), plan=tiny, seed=13, protocol=digest({'fixture': 'two-epoch-lifecycle-r1'}))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    target = ROOT/'validation/lifecycle'
    baseline, resumed = target/'uninterrupted', target/'resumed'
    for folder in (baseline, resumed):
        folder.mkdir(parents=True, exist_ok=True)
        # This test owns only its two exact generated checkpoint paths.
        for name in ('state.pt', 'state.pt.tmp'):
            if (folder/name).exists():
                (folder/name).unlink()
    real_save = worker.save
    def pause_after_one(folder, state):
        real_save(folder, state)
        if state['updates'] == 1:
            worker.STOP = True
    with patch.object(worker, 'EPOCHS', 2), patch.object(worker, 'schedule', return_value=(train, [list(range(8))])), \
         patch.object(worker, 'smoke', return_value={'status': 'already-tested-real-model-seam'}):
        worker.STOP = False
        worker.train(cfg, baseline, device)
        worker.STOP = False
        with patch.object(worker, 'save', side_effect=pause_after_one):
            worker.train(cfg, resumed, device)
        partial = torch.load(resumed/'state.pt', map_location='cpu', weights_only=False)
        assert partial['updates'] == 1 and partial['epoch'] == 0 and partial['cursor'] == 1
        worker.STOP = False
        worker.train(cfg, resumed, device)
    a = torch.load(baseline/'state.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed/'state.pt', map_location='cpu', weights_only=False)
    assert a['completed'] and b['completed'] and b['updates'] == 2
    differences = {k: delta(a[k], b[k]) for k in ('model', 'optimizer', 'scaler', 'scheduler', 'rng', 'best', 'best_score')}
    assert max(differences.values()) == 0
    assert [r['loss'] for r in a['history']] == [r['loss'] for r in b['history']]
    for endpoint in ('FINAL', 'SELECTED'):
        arrays = np.load(resumed/f'TEST_{endpoint}.npz')
        assert arrays['masks'].shape == (2, 2, 128, 96)
        assert np.max(np.abs(arrays['contributions'].sum((-2, -1))+arrays['bias']-arrays['logits'])) < 1e-5
    from model import TreadOne
    model = TreadOne().to(device)
    model.load_state_dict(b['best'], strict=True)
    gallery(model, p, root, device, target/'gallery')
    atomic(ROOT/'validation/LIFECYCLE_TESTS.json', dict(status='passed', differences=differences,
           durable_pause={'epoch': partial['epoch'], 'cursor': partial['cursor'], 'updates': partial['updates']},
           resumed_updates=b['updates'], predictions_npz_checked=True, gallery_written=True,
           scope='Two-epoch/two-update fixture on actual architecture/images. Separate GPU seam is a fresh-process test. Not a long training result.'))
    print('REAL-MODEL PAUSE/RESUME/FINAL EVALUATION/EXPORT LIFECYCLE PASSED')


if __name__ == '__main__':
    threading.stack_size(32*1024**2)
    errors = []
    def target():
        try:
            run()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
