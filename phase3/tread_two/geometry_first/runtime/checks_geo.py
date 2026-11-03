"""Independent numerical and transaction checks used by NB16 and offline validation."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import cpu, read, seed_all
from model_geo import Geo, ordered_points
from objective_geo import coarse_truth, full_geometry, geometry_loss, quality_loss, texture_loss, variance_covariance
from store_geo import Store


def numeric_checks(model, batch, base, seeds):
    report = {}
    out = model(base, batch, seeds)
    a, b = out['views']
    _, terms = texture_loss(out['views'], [r['component'] for r in batch['rows']], 1)
    assert terms['texture_step']
    eligible = a['eligible'] & b['eligible']
    z = a['projected'][eligible]
    centered = z-z.mean(0)
    # An independent loop covariance agrees with the matrix implementation.
    manual = sum(torch.outer(v, v) for v in centered)/(len(z)-1)
    variance, covariance = variance_covariance(z)
    assert torch.allclose(variance, (1-(manual.diag()+1e-4).sqrt()).clamp_min(0).mean(), atol=1e-6)
    offdiagonal = sum(manual[i,j].square() for i in range(64) for j in range(64) if i!=j)/64
    assert torch.allclose(covariance, offdiagonal, atol=1e-6, rtol=1e-5)
    numerator, denominator = 0., 0.
    for view in (a, b):
        pred, target, weight = [cpu(view[k]).numpy().astype(np.float64) for k in ('prediction', 'target', 'weight')]
        mask = cpu(view['selected']).numpy()[...,None,None,None]*cpu(eligible).numpy()[:,None,None,None,None]
        delta = np.abs(pred-target)
        element = np.where(delta<.1, delta**2/.2, delta-.05)
        numerator += float((element*weight*mask).sum())
        denominator += float((weight*mask).sum()*4)
    assert np.isclose(float(terms['masked_numerator']), numerator, rtol=1e-5)
    assert denominator == float(terms['masked_denominator'])
    reconstructed = 25*float(terms['invariance'])+25*(float(terms['variance_a'])+float(terms['variance_b']))+float(terms['covariance_a'])+float(terms['covariance_b'])+.5*numerator/denominator
    assert np.isclose(float(terms['texture']), reconstructed, rtol=1e-5)
    report['VIC_covariance_and_masked_Huber_independent_reductions'] = True
    # Real hidden RGB changes leave every visible LC tile and masked-context
    # output unchanged. Metadata/support are held fixed to test the texture path.
    altered_atlas = dict(out['atlas'])
    altered_atlas['rgb'] = out['atlas']['rgb'].clone()
    for i in range(len(batch['rows'])):
        for patch in torch.where(a['selected'][i])[0].cpu().tolist():
            row, col = divmod(patch, 4)
            altered_atlas['rgb'][i,:,row*64:(row+1)*64,col*64:(col+1)*64] = .37
    with torch.no_grad():
        altered = model.texture_view(altered_atlas, [int(s)*2 for s in seeds])
        assert torch.equal(a['lc'].detach()[~a['selected']], altered['lc'][~a['selected']])
        assert torch.equal(a['prediction'].detach(), altered['prediction'])
    report['hidden_RGB_does_not_leak_to_visible_LC_or_masked_prediction'] = True
    _, quality = quality_loss(out['quality'], out['refined'], batch)
    assert torch.equal(quality['available'][:,2], batch['human'])
    assert not quality['targets'].requires_grad
    empty = dict(batch)
    empty['point_weight'] = torch.zeros_like(batch['point_weight'])
    empty['masks'] = batch['masks'].clone()
    empty['masks'][:,2] = 1
    loss, g = geometry_loss(out['refined'], base, empty)
    assert float(g['bce_numerator'])==float(g['bce_denominator'])==float(g['point_numerator'])==float(g['point_denominator'])==0
    assert torch.isfinite(loss)
    assert (out['refined']['points'].reshape(-1,3,2)[...,1] > out['refined']['points'].reshape(-1,3,2)[...,0]).all()
    report['ignore_missing_targets_point_order_human_only_quality'] = True
    # Verify the proposed 4x4 sufficient counts reproduce original-pixel TP/FP/FN.
    pred = out['refined']['masks'].detach()>=.5
    target = batch['masks'][:,:2]
    valid = 1-batch['masks'][:,2:3]
    n = len(pred)
    foreground = (target*valid).reshape(n,2,128,4,96,4).sum((3,5))
    available = valid.reshape(n,1,128,4,96,4).sum((3,5))
    metrics = full_geometry(out['refined'], batch)
    assert torch.equal((pred*foreground).sum((-2,-1)).long(), metrics['tp'])
    assert torch.equal((pred*(available-foreground)).sum((-2,-1)).long(), metrics['fp'])
    assert torch.equal(((~pred)*foreground).sum((-2,-1)).long(), metrics['fn'])
    report['dense_block_counts_reconstruct_full_grid_confusion'] = True
    # A skipped texture group must preserve parameters AND existing Adam moments.
    from engineering_geo import optimizer_for, equal
    clone = Geo().to(batch['rgb'].device)
    clone.load_state_dict(model.state_dict())
    opt = optimizer_for(clone)
    for p in clone.parameters():
        p.grad = torch.ones_like(p)*.001
    opt.step()
    old_params = [p.detach().clone() for p in clone.groups()['texture']]
    old_moments = [cpu(opt.state[p]) for p in clone.groups()['texture']]
    opt.zero_grad(set_to_none=True)
    for name in ('geometry','quality'):
        for p in clone.groups()[name]:
            p.grad = torch.ones_like(p)*.001
    opt.step()
    assert all(torch.equal(a,p) for a,p in zip(old_params,clone.groups()['texture']))
    assert all(equal(a,cpu(opt.state[p])) for a,p in zip(old_moments,clone.groups()['texture']))
    fake_views = [{**v,'eligible':torch.zeros_like(v['eligible'])} for v in out['views']]
    _, skipped = texture_loss(fake_views, [r['component'] for r in batch['rows']], 1)
    assert not skipped['texture_step']
    report['skipped_texture_has_no_decay_or_moment_update'] = True
    return report


def transaction_checks(folder):
    from engineering_geo import equal
    folder = Path(folder)
    result = {}
    for crash in ('after_evidence','after_state','after_pointer'):
        seed_all(123)
        model = torch.nn.Linear(2,2)
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        store = Store(folder/crash, 'engineering_crash_fixture')
        x = torch.tensor([[1.,2.]])
        opt.zero_grad(set_to_none=True)
        model(x).square().sum().backward()
        opt.step()
        store.commit(model,opt,1,{'geometry':1}, {'loss':[{'update':1,'value':np.float32(.1)}]}, {'raw':np.ones((2,3),dtype=np.float32)})
        old = cpu(model.state_dict())
        old_opt = cpu(opt.state_dict())
        opt.zero_grad(set_to_none=True)
        model(x).square().sum().backward()
        opt.step()
        new = cpu(model.state_dict())
        try:
            store.commit(model,opt,2,{'geometry':2}, {'loss':[{'update':2,'value':np.float32(.2)}]}, {'raw':np.zeros((2,3),dtype=np.float32)}, crash=crash)
            raise AssertionError('Crash was not injected')
        except InterruptedError:
            pass
        state = store.restore(model,opt)
        assert state['update']==(2 if crash=='after_pointer' else 1)
        assert equal(cpu(model.state_dict()),new if crash=='after_pointer' else old)
        if crash!='after_pointer':
            assert equal(cpu(opt.state_dict()),old_opt)
        result[crash] = 'complete_prior_or_new_unit_restored'
    values = np.random.default_rng(19).standard_normal(10000).astype('float32')
    recovered = np.array([Store.format(x) for x in values], dtype='float32')
    assert np.array_equal(values.view('uint32'),recovered.view('uint32'))
    result['FP32_CSV_roundtrip_10000'] = 'exact_bits'
    return result
