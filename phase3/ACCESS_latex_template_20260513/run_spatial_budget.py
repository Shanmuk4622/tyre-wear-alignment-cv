"""Fit algebraic spatial bases on training records and audit the frozen decoder."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PAPER = Path(__file__).resolve().parent
PHASE = PAPER.parent
OLD = PHASE / 'texture_aggregation_20261102'
OUT = PAPER / 'evidence'
DETAIL = PAPER / '.build' / 'spatial_budget'
DETAIL.mkdir(parents=True, exist_ok=True)
assert Path(sys.prefix).name == 'cv_conda'
sys.path.insert(0, str(PHASE / 'analysis'))
sys.path.insert(0, str(PHASE / 'tread_two' / 'trace_batch'))
from cache import Cache
from review import Saved
from texture_aggregation_math import bases, huber, mlp

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

protocol = OUT / 'spatial_budget_protocol.json'
config = read(protocol)
fit = Path(read(PHASE/'tread_two/nb24_completion/LATEST.json')['path'])/'remote/fit'
pins = pd.read_csv(OLD/'NATIVE_INPUTS.csv')
for entry in pins.itertuples():
    path = fit/entry.file
    assert path.stat().st_size == entry.bytes and sha(path) == entry.sha256, entry.file
old_seal = read(OLD/'ARTIFACTS.json')['files']
def sealed(name):
    path = OLD/name
    assert sha(path) == old_seal[name]['sha256'], name
    return path

alignment = pd.read_csv(sealed('IMAGE_ALIGNMENT.csv'))
assert len(alignment) == 570
role = alignment.role.to_numpy()
group = alignment.component.to_numpy()
ids = alignment.observation.to_numpy()
saved = Saved(fit)
mass = saved.get('mass', ids).astype('float64')
hidden = saved.get('hidden', ids).astype(bool)
training = role == 'train'
full = (mass[:,0] > 0).all(axis=1)
assert training.sum() == 386 and (training & full).sum() == 384 and full[~training].all()
groups = sorted(set(group[training]))
r = np.load(sealed('native/fixed_train_mass.npy'))
r = r / r.sum()
sqrt_r = np.sqrt(r)
assert np.all(r > 0)
dc = sqrt_r
# An orthonormal complement to the mandatory DC mode.
comp = np.linalg.qr(dc[:, None], mode='complete')[0][:, 1:]

def qr_basis(modes):
    q, _ = np.linalg.qr(comp.T @ (sqrt_r[:,None]*modes))
    return np.column_stack([dc, comp@q])

y, x = np.meshgrid(np.arange(8), np.arange(4), indexing='ij')
freq = sorted(((i,j) for i in range(8) for j in range(4) if i+j), key=lambda v:((v[0]/8)**2+(v[1]/4)**2,v))
dct = qr_basis(np.stack([np.cos(np.pi*(y.ravel()+.5)*i/8)*np.cos(np.pi*(x.ravel()+.5)*j/4)
                         for i,j in freq], axis=1))
moment = qr_basis(bases()['moments5'][:,1:])
random = {}
for seed in config['random_seeds']:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(31,31)))
    random[f'random_{seed}'] = np.column_stack([dc,comp@q])

ckpt = torch.load(fit/'exports/final.pt',map_location='cpu',weights_only=False)
assert ckpt['trace']['decoder.0.weight'].shape == (96,48)
weights = {k:v.numpy().astype('float64') for k,v in ckpt['trace'].items() if k.startswith('decoder.')}
def decode(tokens):
    return mlp(tokens,weights,'decoder')

def jacobian(anchor):
    pre = weights['decoder.0.weight']@anchor + weights['decoder.0.bias']
    sigmoid = np.exp(-np.logaddexp(0.,-pre))
    return (weights['decoder.2.weight']*(sigmoid+pre*sigmoid*(1-sigmoid))) @ weights['decoder.0.weight']

def eigenspace(snapshots, sample_weights):
    # Position covariance after applying the fixed mass and removing the DC mode.
    projected = np.einsum('ts,itd->isd',comp,snapshots*sqrt_r[None,:,None],optimize=True)
    covariance = np.einsum('i,isd,itd->st',sample_weights,projected,projected,optimize=True)
    eig, q = np.linalg.eigh(covariance)
    directions = q[:,::-1]
    q = np.column_stack([dc,comp@directions])
    assert np.allclose(q.T@q,np.eye(32),atol=1e-12)
    for k in (1,2,4,5,8,16,32):
        kept = directions[:,:k-1]
        residual = projected-(kept@kept.T)@projected
        direct = np.einsum('i,isd,isd->',sample_weights,residual,residual,optimize=True)
        assert np.isclose(direct,eig[:32-k].sum(),atol=1e-12,rtol=1e-10)
    return q,eig[::-1]

def fit_bases(tokens, template, population, sample_weights, decoded, output_template):
    use = population & full
    residual = tokens[use,0]-template
    jac = jacobian(r @ template)
    epsilon = 1e-5
    delta = np.eye(48)*epsilon
    finite_diff = (decode(r@template+delta)-decode(r@template-delta)).T/(2*epsilon)
    assert np.allclose(jac,finite_diff,atol=2e-8)
    score_space = {'token_PCA':residual,
                   'Jacobian_PCA':residual@jac.T,
                   'decoded_PCA':decoded[use,0]-output_template}
    return {k:eigenspace(v,sample_weights[use]) for k,v in score_space.items()}

cache = Cache(fit/'inputs/cache')
stats = cache.statistics()
mu, sd = stats['mean'].numpy(), stats['std'].numpy()
target = ((cache.array('target',alignment.slot)-mu[None,None,:,None,None]) /
          sd[None,None,:,None,None]).astype('float64').reshape(570,1,32,96)
pixel_weight = cache.array('weight',alignment.slot).astype('float64')
position = ((stats['position_mean'].numpy()-mu[None,:,None,None]) /
            sd[None,:,None,None]).astype('float64').reshape(1,1,32,96)
available = (stats['position_count'].numpy()>0).reshape(1,1,32,1,4,4)

summary, component, descriptor, basis_rows, fitting = [], [], [], [], []
details = DETAIL/'IMAGE_RESULTS.csv'
first = True
readouts = 0
bound_error = 0.0
endpoint_error = 0.0
max_mode_identity = 0.0
for layer, branch in [('encoded','local'),('masked_context','masked')]:
    tokens = saved.get(layer,ids).astype('float64')
    template = np.load(sealed('native/template_'+layer+'.npy'))
    decoded = decode(tokens)
    readouts += 570*2*32
    saved_original = np.load(sealed('native/prediction_'+layer+'_identity.npy')).reshape(decoded.shape)
    endpoint_error = max(endpoint_error,float(np.max(abs(decoded-saved_original))))
    assert np.array_equal(decoded,saved_original) or np.allclose(decoded,saved_original,atol=1e-12)
    use = (mass>0) if branch == 'local' else hidden
    weight6 = pixel_weight[:,None]*use[...,None,None,None]*available
    weight = np.broadcast_to(weight6,(570,2,32,6,4,4)).reshape(570,2,32,96)
    counts = weight.sum((2,3))
    channel_counts = weight.reshape(570,2,32,6,16).sum((2,4))
    assert np.all(counts>0) and np.all(channel_counts>0)
    def mean(value):
        return (value*weight).sum((2,3))/counts
    baseline = mean(huber(decoded,target))
    control = mean(huber(position,target))
    old = pd.read_csv(sealed(f'IMAGE_LOSSES_{layer}.csv'))
    old = old[old.variant=='identity'].sort_values(['image_id','view'])
    ordering = np.argsort(alignment.image_id.to_numpy())
    assert np.allclose(baseline[ordering].ravel(),old.loss,atol=1e-12)
    assert np.allclose(control[ordering].ravel(),old.position_control_loss,atol=1e-12)

    for fit_id in ['main','component_balanced',*[f'omit_{i}' for i in range(len(groups))]]:
        population = training.copy()
        excluded = None
        if fit_id.startswith('omit_'):
            excluded = groups[int(fit_id.split('_')[1])]
            population &= group!=excluded
        sample = population.astype('float64')
        if fit_id == 'component_balanced':
            for name in groups:
                inside = population & (group==name)
                sample[inside] /= (inside&full).sum()
        sample /= sample[full].sum()
        if fit_id == 'main':
            anchor = template
        else:
            support = (mass[:,0]>0)*sample[:,None]
            anchor = (tokens[:,0]*support[...,None]).sum(0)/support.sum(0)[:,None]
        assert np.all(sample[~training]==0)
        output_anchor = decode(anchor)
        fitted = fit_bases(tokens,anchor,population,sample,decoded,output_anchor)
        residual = tokens - anchor
        readouts += 32 + 96 # template readout and 96 Jacobian finite-difference evaluations
        fitting.append(dict(branch=branch,fit=fit_id,excluded_component=excluded,
                            template_images=int(population.sum()),basis_images=int((population&full).sum())))
        candidates = {k:q for k,(q,_) in fitted.items()}
        if fit_id == 'main':
            candidates.update({'DCT':dct,'moments':moment,**random})
        for method,q in candidates.items():
            if method in fitted:
                for mode in range(32):
                    for tile in range(32):
                        basis_rows.append(dict(branch=branch,fit=fit_id,method=method,mode=mode,
                            tile=tile,coefficient=q[tile,mode],eigenvalue=None if mode==0 else fitted[method][1][mode-1]))
            budgets = config['mode_counts_including_constant'] if fit_id=='main' else [5]
            if method == 'moments':
                budgets = [5]
            for k in budgets:
                if k in (1,32) and method!='DCT':continue
                qk = q[:,:k]
                proj = (qk@qk.T)*sqrt_r[None,:]/sqrt_r[:,None]
                compressed = anchor+proj@residual
                prediction = decode(compressed)
                readouts += 570*2*32
                output_first = output_anchor+proj@(decoded-output_anchor)
                task_loss = mean(huber(prediction,target))
                after_loss = mean(huber(output_first,target))
                drift = mean(np.abs(prediction-decoded))
                order_gap = mean(np.abs(prediction-output_first))
                bound_error = max(bound_error,float(np.max(np.abs(task_loss-baseline)-drift)),
                                  float(np.max(np.abs(task_loss-after_loss)-order_gap)))
                assert bound_error < 1e-12
                feature_mse = ((compressed-tokens)**2*r[None,None,:,None]).sum((2,3))/48
                channel = ((huber(prediction,target)*weight).reshape(570,2,32,6,16)
                           .sum((2,4))/channel_counts)
                if k == 32:
                    max_mode_identity = max(max_mode_identity,float(np.max(abs(prediction-decoded))))
                    assert max_mode_identity < 1e-10
                fields = dict(branch=branch,fit=fit_id,method=method,k=k,coefficients=48*k)
                for view in (0,1):
                    frame = alignment[['image_id','role','component']].assign(**fields,view=view,
                        native_loss=baseline[:,view], position_loss=control[:,view], loss=task_loss[:,view],
                        output_first_loss=after_loss[:,view], feature_mse=feature_mse[:,view],
                        decoder_drift=drift[:,view], order_gap=order_gap[:,view],
                        **{f'channel_{c}_loss':channel[:,view,c] for c in range(6)})
                    frame.to_csv(details,mode='w' if first else 'a',header=first,index=False)
                    first=False
                    def reduced(indices):
                        base, pos, loss, after = (float(a[indices,view].mean()) for a in (baseline,control,task_loss,after_loss))
                        return dict(images=int(indices.sum()),loss=loss,native_loss=base,position_loss=pos,
                            retained=(pos-loss)/(pos-base),output_first_retained=(pos-after)/(pos-base),
                            delta_loss=loss-base,output_space_delta=after-base,order_delta=loss-after,
                            decoder_drift=float(drift[indices,view].mean()),order_gap=float(order_gap[indices,view].mean()),
                            feature_mse=float(feature_mse[indices,view].mean()))
                    for key in ('train','validation','test'):
                        indices = role==key
                        summary.append(dict(**fields,role=key,view=view,**reduced(indices)))
                        if fit_id=='main' and k==5 and not method.startswith('random'):
                            for component_id in sorted(set(group[indices])):
                                component.append(dict(**fields,role=key,view=view,component=component_id,
                                                      **reduced(indices&(group==component_id))))
                            for c in range(6):
                                descriptor.append(dict(**fields,role=key,view=view,channel=c,
                                    loss=float(channel[indices,view,c].mean())))
    print(f'{branch}: {len(summary)} cumulative summary rows',flush=True)

for name, records in [('summary',summary),('components',component),('descriptors',descriptor),('fits',fitting)]:
    pd.DataFrame(records).to_csv(OUT/f'spatial_budget_{name}.csv',index=False)
pd.DataFrame(basis_rows).to_csv(DETAIL/'BASIS_VECTORS.csv',index=False)
receipt = dict(status='passed',protocol_sha256=sha(protocol),script_sha256=sha(Path(__file__)),
    pinned_input_files=len(pins),basis_images_primary=384,template_images_primary=386,
    decoder_vector_readouts=readouts,bound_max_violation=bound_error,
    max_full_budget_prediction_difference=max_mode_identity,identity_bridge_max_difference=endpoint_error,
    eigentail_cost_verified=True, analytic_jacobian_finite_difference_verified=True,
    image_results_rows=570*2*(len(summary)//6), summary_rows=len(summary),
    optimizer_updates=0,encoder_or_geometry_forwards=0,remote_writes=0,
    outputs={path.relative_to(PAPER).as_posix():sha(path) for path in
        [*sorted(OUT.glob('spatial_budget_*.csv')),details,DETAIL/'BASIS_VECTORS.csv']})
(PAPER/'audit/SPATIAL_BUDGET.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
