"""Focused contract tests with real data and a real-model process restore."""
from PIL import Image
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import threading
import faulthandler
faulthandler.enable(all_threads=True)
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda', 'Run through phase3/run_cv.cmd'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
import numpy as np
import torch
from common import atomic, deterministic, digest, read
from data import batch, plan
from model import TreadOne, loss_terms
from worker import smoke


def run():
    p = plan(ROOT.parent/'phase2/data/phase2_dataset_v1')
    root = str(ROOT.parent/'phase2/data/phase2_dataset_v1')
    folder = ROOT/'validation'
    folder.mkdir(exist_ok=True)
    deterministic(7)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rows = [next(r for r in p['rows'] if r['role'] == 'train' and r['point_kind'] == 'human'),
            next(r for r in p['rows'] if r['domain'] == 'new_video_frame')]
    b = batch(root, rows, device, seed=7)
    print('Checking real-model forward/backward and explanation contracts', flush=True)
    model = TreadOne().to(device)
    out = model(b['img'], b['guides'])
    residual = float((out['contributions'].sum((-2, -1))+out['bias']-out['logits']).abs().max())
    assert residual == 0
    points = out['points'].reshape(2, 3, 2)
    assert (points[..., 0] < points[..., 1]).all() and (points >= 0).all() and (points <= 1).all()
    value = loss_terms(out, b)
    value['total'].backward()
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
    # Segmentation-logit bias changes must not receive gradients from classification through its detached gate.
    model.zero_grad(set_to_none=True)
    out = model(b['img'], b['guides'])
    out['logits'].sum().backward()
    assert model.seg.weight.grad is None and model.seg.bias.grad is None
    # Explanation removal must exactly account for the class-logit change at the evidence layer.
    values = out['contributions'][0, 0].flatten()
    removed = values.topk(max(1, len(values)//10)).indices
    altered = values.clone()
    altered[removed] = 0
    assert torch.allclose(out['logits'][0, 0]-(values[removed].sum()), altered.sum()+out['bias'][0], atol=1e-6)
    identity = dict(parameters=sum(t.numel() for t in model.parameters()), evidence_residual=residual,
                    foreground_gate_detached=True, ordered_points=True, finite_backward=True,
                    source_and_weak_labels=True, device=str(device))
    atomic(folder/'model_contracts.json', identity)
    print('Model contracts passed; running batch-eight independent-process seam', flush=True)
    del model, out, value, b
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    cfg = dict(root=root, plan=p, seed=1, protocol=digest({'purpose': 'phase3-local-resume-test-r1'}), smoke_batch=8)
    report = smoke(cfg, folder/'fresh_process', device)
    atomic(folder/'LOCAL_TESTS.json', dict(status='passed', contracts=identity, fresh_process=report,
           caveat='Local GTX1650 batch 8; Kaggle dual-T4 preflight remains required. No long training or HF live publication.'))
    print('MODEL CONTRACTS AND REAL-MODEL FRESH-PROCESS RESUME PASSED', flush=True)


if __name__ == '__main__':
    threading.stack_size(32*1024**2)
    errors = []
    def target():
        try:
            run()
        except BaseException as exc:
            errors.append(exc)
    t = threading.Thread(target=target)
    t.start()
    t.join()
    if errors:
        raise errors[0]
