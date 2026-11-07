"""Verify TRACE topology and analyze frozen tokens for the reviewer response."""
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import pandas as pd
import torch

PAPER = Path(__file__).resolve().parent
PHASE = PAPER.parent
EVIDENCE = PAPER / 'evidence'
DETAIL = PAPER / '.build' / 'reviewer'
DETAIL.mkdir(exist_ok=True)
assert Path(sys.prefix).name == 'cv_conda'
sys.path[:0] = [str(PHASE/'tread_two/trace_batch'), str(PHASE/'analysis')]
from cache import Cache
from model import Trace
from review import Saved
from texture_aggregation_math import bases, huber, mlp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_architecture():
    torch.manual_seed(4622)
    model = Trace().eval()
    modules = {
        'encoder1': model.encoder[:3], 'encoder2': model.encoder[3:6],
        'encoder3': model.encoder[6:9], 'encoder4': model.encoder[9:12],
        'metadata': model.metadata, 'axial1': model.context[0], 'axial2': model.context[1],
        'decoder': model.decoder, 'projector': model.projector,
    }
    counts = {name: sum(p.numel() for p in layer.parameters()) for name, layer in modules.items()}
    counts['mask_token'] = model.mask_token.numel()
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters()) == 71022
    calls = {}
    hooks = []
    for name in ('encoder','metadata','context','decoder','projector'):
        calls[name] = []
        hooks.append(getattr(model, name).register_forward_hook(
            lambda _, args, out, name=name: calls[name].append(
                {'input':list(args[0].shape), 'output':list(out.shape)})))
    image = torch.rand(1,32,3,64,64)
    valid = torch.ones(1,32,1,64,64)
    hidden = torch.zeros(1,32,dtype=torch.bool)
    hidden[:,::4] = True
    with torch.inference_mode():
        output = model(image,valid,hidden)
        changed = image.clone()
        changed[:,::4] = 1-image[:,::4]
        for hook in hooks:
            hook.remove()
        counterfactual = model(changed,valid,hidden)
    assert {k:len(v) for k,v in calls.items()} == {
        'encoder':2,'metadata':1,'context':2,'decoder':2,'projector':1}
    assert torch.equal(output['masked_prediction'],counterfactual['masked_prediction'])
    assert not torch.equal(output['local_prediction'],counterfactual['local_prediction'])
    return {'source_sha256':sha(PHASE/'tread_two/trace_batch/model.py'),
            'parameters':counts,'calls':calls,'outputs':{k:list(v.shape) for k,v in output.items()},
            'hidden_rgb_invariance_fixed_support':True}


arch = check_architecture()
(PAPER/'audit/TRACE_ARCHITECTURE.json').write_text(json.dumps(arch,indent=2)+'\n')
print('Verified topology and shared modules:',arch['parameters'],flush=True)

old = PHASE/'texture_aggregation_20261102'
fit = Path(read(PHASE/'tread_two/nb24_completion/LATEST.json')['path'])/'remote/fit'
pins = pd.read_csv(old/'NATIVE_INPUTS.csv')
for item in pins.itertuples():
    assert sha(fit/item.file) == item.sha256,item.file
prior = read(PAPER/'audit/SPATIAL_BUDGET.json')
basis_path = PAPER/'.build/spatial_budget/BASIS_VECTORS.csv'
assert sha(basis_path) == prior['outputs']['.build/spatial_budget/BASIS_VECTORS.csv']
vectors = pd.read_csv(basis_path, float_precision='round_trip')
sealed = read(old/'ARTIFACTS.json')['files']


def pinned(name):
    path = old/name
    assert sha(path) == sealed[name]['sha256']
    return path


align = pd.read_csv(pinned('IMAGE_ALIGNMENT.csv'))
ids = align.observation.to_numpy()
roles = align.role.to_numpy()
training = roles=='train'
saved = Saved(fit)
mass = saved.get('mass',ids).astype('float64')
hidden = saved.get('hidden',ids).astype(bool)
assert len(ids)==570 and training.sum()==386
full_train = training & (mass[:,0]>0).all(axis=1)
assert full_train.sum()==384
r = np.load(pinned('native/fixed_train_mass.npy'))
root_mass = np.sqrt(r/r.sum())
complement = np.linalg.qr(root_mass[:,None],mode='complete')[0][:,1:]
cache = Cache(fit/'inputs/cache')
stats = cache.statistics()
mu, sd = stats['mean'].numpy(), stats['std'].numpy()
truth = ((cache.array('target',align.slot)-mu[None,None,:,None,None]) /
         sd[None,None,:,None,None]).astype('float64').reshape(570,32,96)
position = ((stats['position_mean'].numpy()-mu[None,:,None,None]) /
            sd[None,:,None,None]).astype('float64').reshape(32,96)
pixel_weights = cache.array('weight',align.slot).astype('float64')
available = (stats['position_count'].numpy()>0).reshape(32,1,4,4)
state = torch.load(fit/'exports/final.pt',map_location='cpu',weights_only=False)
weights = {k:v.numpy().astype('float64') for k,v in state['trace'].items() if k.startswith('decoder.')}
first, last = weights['decoder.0.weight'], weights['decoder.2.weight']


def jacobian(tokens):
    pre = tokens@first.T+weights['decoder.0.bias']
    s = np.exp(-np.logaddexp(0.,-pre))
    return (last*(s+pre*s*(1-s))[...,None,:]) @ first


def decoder(tokens):
    return mlp(tokens,weights,'decoder')


def fixed_basis(modes):
    q, _ = np.linalg.qr(complement.T @ (root_mass[:,None]*modes))
    return np.column_stack((root_mass,complement@q))[:,:5]


def get_bases(branch):
    result = {}
    for method in ('token_PCA','Jacobian_PCA','decoded_PCA'):
        use = vectors[(vectors.branch==branch)&(vectors.fit=='main')&(vectors.method==method)]
        result[method] = use.pivot(index='tile',columns='mode',values='coefficient').to_numpy()[:,:5]
    result['moments'] = fixed_basis(bases()['moments5'][:,1:])
    y,x = np.meshgrid(np.arange(8),np.arange(4),indexing='ij')
    pairs = sorted(((i,j) for i in range(8) for j in range(4) if i+j),
                   key=lambda a:((a[0]/8)**2+(a[1]/4)**2,a))[:4]
    result['DCT'] = fixed_basis(np.stack([np.cos(np.pi*(y.ravel()+.5)*i/8)*
                                          np.cos(np.pi*(x.ravel()+.5)*j/4) for i,j in pairs],axis=1))
    for seed in read(EVIDENCE/'spatial_budget_protocol.json')['random_seeds']:
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(31,31)))
        result[f'random_{seed}'] = np.column_stack((root_mass,complement@q))[:,:5]
    return result


def spatial_project(q, tokens, template):
    p = (q@q.T)*root_mass[None,:]/root_mass[:,None]
    return template+p@(tokens-template)


def linear_readout(tokens, target, weight):
    """Fit a shared affine decoder with separate weights per valid output cell."""
    x = np.column_stack((tokens.reshape(-1,48),np.ones(tokens.shape[0]*32)))
    y = target.reshape(-1,6,16)
    w = weight.reshape(-1,16)
    penalty = np.diag([1e-3]*48+[0.])
    result = np.empty((49,6,16))
    for cell in range(16):
        unit = w[:,cell]/w[:,cell].sum()
        weighted = x.T*unit
        lhs = weighted@x+penalty
        rhs = weighted@y[:,:,cell]
        result[:,:,cell] = np.linalg.solve(lhs,rhs)
        assert np.max(abs(lhs@result[:,:,cell]-rhs))<1e-10
    return result.reshape(49,96)


angle_rows, jac_rows, nonlinear_rows, probe_rows, probe_coefficients, paired_rows, examples = [],[],[],[],[],[],[]
for branch,layer in [('local','encoded'),('masked','masked_context')]:
    token = saved.get(layer,ids)[:,0].astype('float64')
    template = np.load(pinned(f'native/template_{layer}.npy'))
    spatial = get_bases(branch)
    named = [name for name in spatial if not name.startswith('random')]
    for i,left in enumerate(named):
        for right in named[i+1:]:
            q1,q2 = spatial[left],spatial[right]
            assert np.allclose(q1.T@q1,np.eye(5),atol=1e-12)
            cosines = np.linalg.svd(q1[:,1:].T@q2[:,1:],compute_uv=False).clip(0,1)
            degrees = np.degrees(np.arccos(cosines))
            angle_rows.append(dict(branch=branch,left=left,right=right,
                                   residual_overlap=np.mean(cosines**2),
                                   full_overlap=(1+np.sum(cosines**2))/5,
                                   **{f'angle_{j+1}_degrees':d for j,d in enumerate(degrees)}))
    anchor = (root_mass**2)@template
    shared_j = jacobian(anchor)
    local_j = jacobian(template)
    # All full-support training tokens, in bounded batches.
    rows = token[full_train].reshape(-1,48)
    rel = np.concatenate([np.linalg.norm(jacobian(batch)-shared_j,axis=(-1,-2))/np.linalg.norm(shared_j)
                          for batch in np.array_split(rows,48)])
    templ_rel = np.linalg.norm(local_j-shared_j,axis=(-1,-2))/np.linalg.norm(shared_j)
    for scope,values in [('template_positions',templ_rel),('train_tokens',rel)]:
        jac_rows.append(dict(branch=branch,scope=scope,samples=len(values),
                             q10=np.quantile(values,.1),median=np.median(values),
                             q90=np.quantile(values,.9),maximum=values.max()))
    residual = token[full_train]-template
    out = decoder(token[full_train])-decoder(template)
    linear_approximations = {'shared_anchor':residual@shared_j.T,
                             'position_anchor':np.einsum('itc,toc->ito',residual,local_j)}
    for scope,linear in linear_approximations.items():
        nonlinear_rows.append(dict(branch=branch,scope=scope,
            relative_output_rmse=np.linalg.norm((out-linear)*root_mass[None,:,None])/
                                 np.linalg.norm(out*root_mass[None,:,None])))

    support = (mass[:,0]>0) if branch=='local' else hidden[:,0]
    valid = pixel_weights*support[...,None,None,None]*available
    w = np.broadcast_to(valid,(570,32,6,4,4)).reshape(570,32,96)
    assert np.all(w.sum((1,2))>0)
    coeff = linear_readout(token[training],truth[training],valid[training])
    for c in range(49):
        for o in range(96):
            probe_coefficients.append(dict(branch=branch,input=c,output=o,coefficient=coeff[c,o]))
    def score(prediction):
        return (huber(prediction,truth)*w).sum((1,2))/w.sum((1,2))
    def linear_predict(x):
        return x@coeff[:-1]+coeff[-1]
    base = score(linear_predict(token))
    control = score(position)
    for method in ['identity',*spatial]:
        transformed = token if method=='identity' else spatial_project(spatial[method],token,template)
        loss = score(linear_predict(transformed))
        for record,loss_i,base_i,pos_i in zip(align.itertuples(),loss,base,control):
            paired_rows.append(dict(branch=branch,method=method,image_id=record.image_id,
                role=record.role,component=record.component,loss=loss_i,native_loss=base_i,position_loss=pos_i))
        for role in ('train','validation','test'):
            use = roles==role
            model_loss,base_loss,pos_loss = (float(x[use].mean()) for x in (loss,base,control))
            probe_rows.append(dict(branch=branch,method=method,role=role,images=int(use.sum()),
                loss=model_loss,native_loss=base_loss,position_loss=pos_loss,
                retained=(pos_loss-model_loss)/(pos_loss-base_loss) if pos_loss>base_loss else None))
    if branch=='local':
        for case in read(EVIDENCE/'qualitative/EXAMPLES.json'):
            idx = np.flatnonzero(align.image_id.to_numpy()==case['image_id']).item()
            pred = decoder(spatial_project(spatial['token_PCA'],token[idx],template)).reshape(32,6,4,4)
            raw_axis = pred[:,2]*float(sd[2])+float(mu[2])
            for t,u,v in np.ndindex(32,4,4):
                examples.append(dict(case=case['case'],tile=t,cell_y=u,cell_x=v,orientation_axis_pca=raw_axis[t,u,v]))
    print('Completed',branch,'subspace, Jacobian and affine readout analysis',flush=True)

for name, rows in [('subspace_angles',angle_rows),('jacobian_variation',jac_rows),
                   ('linearization_error',nonlinear_rows),('affine_readout',probe_rows),
                   ('qualitative_pca',examples)]:
    pd.DataFrame(rows).to_csv(EVIDENCE/f'review_{name}.csv',index=False)
pd.DataFrame(paired_rows).to_csv(DETAIL/'AFFINE_IMAGE_LOSSES.csv',index=False)
pd.DataFrame(probe_coefficients).to_csv(DETAIL/'AFFINE_COEFFICIENTS.csv',index=False)
receipt = {
    'status':'passed','input_count':len(pins),'checkpoint_epoch':60,'basis_fit_view':0,
    'image_rows':len(paired_rows),'probe_fits':2,'ridge':.001,
    'weights':'valid target counts, normalized per cell, intercept unpenalized',
    'network_optimizer_updates':0,'image_or_geometry_forwards':0,'synthetic_trace_forwards':2,
    'output_sha256':{p.relative_to(PAPER).as_posix():sha(p) for p in
                     [*EVIDENCE.glob('review_*.csv'),*DETAIL.glob('*.csv')]},
}
(PAPER/'audit/REVIEWER_ANALYSIS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
