"""Train-only engineering checks. No measured GEO experiment is implemented here."""
import copy
import csv
import math
import time
from pathlib import Path
import numpy as np
import torch
from common import HERE, atomic, cpu, csv_write, digest, environment, file_meta, native, read, rng_state, seed_all, sha
from anchor_geo import Anchor
from atlas_geo import canonicalize, grid_from_rows
from data_geo import dataset, load_batch, locate
from geometry import canonicalize as original_atlas
from model_geo import Geo
from objective_geo import coarse_truth, full_geometry, losses, texture_loss, variance_covariance
from schedule_geo import schedule
from store_geo import Store


def optimizer_for(model):
    return torch.optim.AdamW([dict(params=group, lr=lr, name=name, initial_lr=lr)
                             for (name, group), lr in zip(model.groups().items(), (1e-4, 1e-3, 3e-4))],
                            weight_decay=1e-4, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def source_contract(device, anchor_device):
    inventory = read(HERE/'FILES.json')
    assert all(file_meta(HERE/name) == meta for name, meta in inventory.items()), 'Source drift; use unchanged notebook'
    return dict(mode='engineering_fixture', revision='geo-nb16-r1', sources=inventory,
                environment=environment(), train_device=device, frozen_device=anchor_device,
                plan=read(HERE/'PLAN.json'), measured_updates=0)


def prepare(output, weights, data_root='', device='cuda:0', anchor_device='cuda:1', live=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if live:
        assert torch.cuda.device_count() == 2, 'Select Kaggle GPU T4 x2'
        assert all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'Actual dual T4 gate required'
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
    seed_all(1)
    torch.set_num_threads(2)
    root = locate(data_root)
    print('Verifying every prepared data checksum and historical annotation role...', flush=True)
    overlay, rows = dataset(root, verify=True)
    plans, plan_hashes = [], []
    for epoch in range(1, 61):
        order, hashed = schedule(rows, 1, epoch)
        plans.extend(d for batch in order for d in batch)
        plan_hashes.append(hashed)
    csv_write(output/'source_images.csv', [dict(image_id=r['image_id'], role=r['role'], component=r['component'],
              image_sha256=r['image_sha256'], point_kind=r['point_kind'], point_weight=r['point_weight']) for r in overlay['rows']],
              ['image_id', 'role', 'component', 'image_sha256', 'point_kind', 'point_weight'])
    csv_write(output/'schedule.csv', plans, list(plans[0]))
    contract = source_contract(device, anchor_device)
    contract['schedule_hashes'] = plan_hashes
    protocol = digest(contract)
    if (output/'CONTRACT.json').exists():
        assert digest(read(output/'CONTRACT.json')) == protocol, 'Incompatible source/environment: preserve this output and use a separate engineering directory'
    atomic(output/'CONTRACT.json', contract)
    atomic(output/'PREPARED.json', dict(protocol=protocol, root=str(root), weights=str(weights),
           live=live, device=device, anchor_device=anchor_device))
    anchor = Anchor(weights, anchor_device)
    counts = dict(frozen=sum(p.numel() for p in anchor.model.parameters()))
    model = Geo().to(device)
    counts.update({k:sum(p.numel() for p in v) for k,v in model.groups().items()})
    counts['trainable'] = sum(counts[k] for k in ('geometry', 'texture', 'quality'))
    counts['total'] = counts['frozen']+counts['trainable']
    assert counts['frozen'] == 1372159 and counts['trainable'] <= 1000000 and counts['total'] <= 3000000
    atomic(output/'PARAMETERS.json', counts)
    return root, overlay, rows, contract, protocol, anchor, counts


def fetch(root, rows, anchor, device, update):
    order, _ = schedule(rows, 1, 1)
    draws = order[update-1]
    lookup = {r['image_id']: r for r in rows}
    chosen = [lookup[d['image_id']] for d in draws]
    batch = load_batch(root, chosen, device, seeds=[d['seed'] for d in draws])
    base = anchor(batch['rgb'], batch['guides'], device)
    return batch, base, [d['seed'] for d in draws]


def evidence(out, parts, batch, update, norms, counters):
    rows = {}
    def scalar(value):
        return float(value.detach()) if torch.is_tensor(value) else value
    objective = {'update': update}
    for key in ('geometry', 'texture'):
        objective.update({key+'_'+k: scalar(v) for k,v in parts[key].items()})
    objective.update(quality_numerator=scalar(parts['quality']['numerator']),
                     quality_denominator=scalar(parts['quality']['denominator']))
    rows['objectives'] = [objective]
    rows['gradients'] = [dict(update=update, group=k, preclip_norm=float(v), clip_limit=1., completed_steps=counters[k]) for k,v in norms.items()]
    rows['samples'] = [dict(update=update, slot=i, image_id=r['image_id'], role=r['role'], component=r['component'], point_kind=r['point_kind']) for i,r in enumerate(batch['rows'])]
    rows['geometry_points'] = [dict(update=update, slot=i, coordinate=j, predicted=np.float32(x),
                                    target=np.float32(batch['points'][i,j].item()) if batch['point_weight'][i]>0 else None,
                                    point_weight=float(batch['point_weight'][i]))
                               for i, point in enumerate(out['refined']['points'].detach().cpu().numpy()) for j,x in enumerate(point)]
    metrics = full_geometry(out['refined'], batch)
    rows['region_metrics'] = [dict(update=update, slot=i, region=j, tp=int(metrics['tp'][i,j]), fp=int(metrics['fp'][i,j]),
                                   fn=int(metrics['fn'][i,j]), dice=float(metrics['dice'][i,j])) for i in range(len(batch['rows'])) for j in (0,1)]
    rows['quality_errors'] = [dict(update=update, slot=i, output=j, prediction=float(out['quality'][i,j]),
                                   target=float(parts['quality']['targets'][i,j]) if parts['quality']['available'][i,j] else None,
                                   available=bool(parts['quality']['available'][i,j])) for i in range(len(batch['rows'])) for j in range(3)]
    arrays = {'geometry_logits': cpu(out['refined']['seg_logits']).numpy(),
              'row_parameters': cpu(out['atlas']['row_parameters']).numpy(),
              'row_valid': cpu(out['atlas']['row_valid']).numpy()}
    rows['vectors'], rows['tile_support'], rows['masked_tiles'], rows['view_plans'] = [], [], [], []
    for vi, view in enumerate(out['views']):
        for field in ('embedding', 'projected'):
            for i, vector in enumerate(cpu(view[field]).numpy()):
                rows['vectors'].extend(dict(update=update, slot=i, view=vi, kind=field, dimension=j, value=np.float32(x)) for j,x in enumerate(vector))
        selected = view['selected'].detach().cpu().numpy()
        for i in range(len(selected)):
            factors = cpu(view['factors'][i]).numpy()
            rows['view_plans'].append(dict(update=update, slot=i, view=vi, brightness=np.float32(factors[0]),
                    contrast=np.float32(factors[1]), gamma=np.float32(factors[2]),
                    eligible=bool(view['eligible'][i]), masked_tiles=int(selected[i].sum()),
                    valid_loss_weight=float((view['weight'][i]*view['selected'][i,:,None,None,None]).sum()*4)))
            for j in range(32):
                rows['tile_support'].append(dict(update=update, slot=i, view=vi, patch=j,
                         coverage=float(out['atlas']['coverage'][i,j]), mass=float(out['atlas']['mass'][i,j]),
                         selected=bool(selected[i,j]), eligible=bool(view['eligible'][i])))
        for name in ('encoded', 'full'):
            arrays[f'v{vi}_{name}'] = cpu(view[name]).numpy()
        # Only up to eight masked tiles have prediction targets; pad explicitly.
        for field, channels in (('prediction', 4), ('target', 4), ('weight', 1)):
            value = cpu(view[field]).numpy()
            packed = np.zeros((len(selected), 8, channels, 16, 16), dtype=np.float32)
            for i in range(len(selected)):
                ids = np.where(selected[i])[0]
                packed[i,:len(ids)] = value[i,ids]
                if field == 'prediction':
                    rows['masked_tiles'].extend(dict(update=update, slot=i, view=vi, packed_slot=k, patch=int(p)) for k,p in enumerate(ids))
            arrays[f'v{vi}_{field}'] = packed
    return rows, arrays


def update_once(model, optimizer, batch, base, seeds, update, counters):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr']*.5*(1+math.cos(math.pi*(update-1)/2940))
    out = model(base, batch, seeds)
    total, parts = losses(out, base, batch, seeds, epoch=1)
    assert torch.isfinite(total), 'Nonfinite objective; previous complete state remains authoritative'
    total.backward()
    if not parts['texture']['texture_step']:
        for param in model.groups()['texture']:
            param.grad = None
    norms = {}
    parameter_rows = []
    owner = {id(p):name for name,group in model.groups().items() for p in group}
    for name,p in model.named_parameters():
        grad = p.grad
        parameter_rows.append(dict(update=update, group=owner[id(p)], parameter=name, available=grad is not None,
                numel=p.numel(), preclip_l2=None if grad is None else float(grad.norm()),
                preclip_max_abs=None if grad is None else float(grad.abs().max()),
                preclip_mean_abs=None if grad is None else float(grad.abs().mean()),
                zero_fraction=None if grad is None else float((grad==0).float().mean()),
                finite=None if grad is None else bool(torch.isfinite(grad).all())))
    for name, group in model.groups().items():
        norms[name] = torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True).detach().cpu()
        if any(p.grad is not None for p in group):
            counters[name] += 1
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    rows, arrays = evidence(out, parts, batch, update, norms, counters)
    rows['parameter_gradients'] = parameter_rows
    return rows, arrays


def isolation_checks(model, batch, base, seeds, anchor):
    groups = model.groups()
    report = {}
    for branch in ('geometry', 'texture', 'quality'):
        model.zero_grad(set_to_none=True)
        out = model(base, batch, seeds)
        _, parts = losses(out, base, batch, seeds, 1)
        if branch == 'geometry':
            loss = parts['geometry']['geometry']
        elif branch == 'texture':
            assert parts['texture']['texture_step'], 'Fixture needs nonempty predicted texture support'
            loss = parts['texture']['texture']
        else:
            loss = parts['quality']['numerator']/parts['quality']['denominator'].clamp_min(1)
        loss.backward()
        counts = {name:sum(int(p.grad is not None and torch.count_nonzero(p.grad)>0) for p in params) for name,params in groups.items()}
        assert counts[branch] > 0 and all(n == 0 for name,n in counts.items() if name != branch), (branch, counts)
        assert all(p.grad is None for p in anchor.model.parameters())
        report[branch] = counts
    model.zero_grad(set_to_none=True)
    # Hidden pixels cannot enter masked contextualization through their encoder token.
    out = model(base, batch, seeds)
    view = out['views'][0]
    encoded = view['encoded'].detach().clone()
    altered = encoded.clone()
    altered[view['selected']] += 17
    with torch.no_grad():
        assert torch.equal(model.contextualize(encoded, view['metadata'], view['selected']),
                           model.contextualize(altered, view['metadata'], view['selected']))
    projected = torch.ones(8, 64, device=batch['rgb'].device)
    variance, covariance = variance_covariance(projected)
    assert abs(float(variance)-.99)<1e-6 and float(covariance)==0
    return dict(isolation=report, masked_token_leakage_passed=True, collapse_fixture_passed=True)


def run_fixture(output, weights, data_root='', device='cuda:0', anchor_device='cuda:1', live=True):
    output = Path(output)
    root, overlay, rows, contract, protocol, anchor, counts = prepare(output, weights, data_root, device, anchor_device, live)
    seed_all(1)
    initial = Geo().to(device)
    init = cpu(initial.state_dict())
    batch, base, seeds = fetch(root, rows, anchor, device, 1)
    refined = initial.refine(base)
    assert torch.equal(refined['seg_logits'], base['anchor_logits'])
    assert torch.equal(refined['points'], base['anchor_points'])
    old = original_atlas(batch['rgb'], refined['masks'])
    atlas = canonicalize(batch['rgb'], refined['masks'], batch['masks'][:,2:3])
    assert torch.equal(old['rgb'], atlas['rgb']) and torch.equal(old['valid'], atlas['valid'])
    assert torch.equal(grid_from_rows(atlas['row_parameters'], atlas['row_valid']), atlas['grid'])
    checks = isolation_checks(initial, batch, base, seeds, anchor)
    from checks_geo import numeric_checks, transaction_checks
    checks['numerical_tests'] = numeric_checks(initial, batch, base, seeds)
    # Each validation invocation owns a fresh fixture subdirectory. Resume may
    # repeat these disposable checks, never measured training.
    checks['transaction_tests'] = transaction_checks(output/'crash_fixtures'/str(time.time_ns()))
    assert not parts_any_grad(anchor.model)
    # Direct and interrupted/restored paths use the same actual full 512x384 model and batch8.
    states, run_folders = {}, {}
    start = time.monotonic()
    fixture_contract = digest({'protocol': protocol, 'fixture': 'full_model_four_updates'})
    for name in ('direct', 'resumed'):
        seed_all(1)
        model = Geo().to(device)
        model.load_state_dict(init)
        optimizer = optimizer_for(model)
        counters = dict(geometry=0, texture=0, quality=0)
        store = Store(output/name, fixture_contract)
        run_folders[name] = store.root
        completed = 0
        if (store.root/'POINTER.json').exists():
            restored = store.restore(model, optimizer)
            counters, completed = dict(restored['counters']), restored['update']
        for update in range(completed+1, 5):
            if (output/'STOP_REQUESTED').exists():
                raise InterruptedError('Catchable stop requested at complete-update boundary')
            import shutil
            assert shutil.disk_usage(output).free > 2000000000, 'Free-space reserve reached; preserve committed units and publish'
            print(f'Engineering {name}: update {update}/4, frozen features then backward / CSV / native commit', flush=True)
            batch, base, seeds = fetch(root, rows, anchor, device, update)
            records, arrays = update_once(model, optimizer, batch, base, seeds, update, counters)
            store.commit(model, optimizer, update, dict(counters), records, arrays)
            checkpoint = output/'checkpoints'/f'{name}_{update:04d}'
            store.snapshot(checkpoint)
            atomic(output/'LATEST_COMPLETE.json', dict(stage=name, update=update, fixture_contract=fixture_contract,
                   immutable_snapshot=checkpoint.relative_to(output).as_posix()))
            if name == 'direct' and update == 1:
                store.snapshot(output/'seam_checkpoint')
            if name == 'resumed' and update == 1:
                # Construct fresh modules and optimizer; never reuse their in-memory state.
                model = Geo().to(device)
                optimizer = optimizer_for(model)
                restored = store.restore(model, optimizer)
                counters = dict(restored['counters'])
        states[name] = dict(model=cpu(model.state_dict()), optimizer=cpu(optimizer.state_dict()), counters=counters, rng=rng_state())
        store.verify()
        del model, optimizer
    assert equal(states['direct'], states['resumed']), 'Restore changed model/moments/counters'
    a, b = run_folders['direct'], run_folders['resumed']
    files = [p.relative_to(a) for p in (a/'units').rglob('*') if p.suffix in ('.csv', '.npy')]
    assert files and all(sha(a/p) == sha(b/p) for p in files), 'Restore changed retained scientific observations'
    assert not parts_any_grad(anchor.model)
    # The fixture itself has no measured-fit counters and cannot become an NB17 checkpoint.
    proof = dict(status='passed', scope='engineering_fixture_only', protocol=protocol,
                 fixture_contract=fixture_contract, environment=contract['environment'], devices=[device, anchor_device],
                 actual_dual_T4=live, measured_fit_ready=False, measured_fit_updates=0,
                 warm_start_exact=True, original_atlas_exact=True, row_grid_reconstruction_exact=True,
                 resume_model_optimizer_counters_exact=True, resume_scientific_files_exact=len(files),
                 resume_RNG_exact=True,
                 parameters=counts, seconds=time.monotonic()-start, **checks)
    resources = [dict(device=i, name=torch.cuda.get_device_name(i),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())]
    # Local CPU validation may see an installed unused GPU; allocated=0 is retained honestly.
    if resources:
        csv_write(output/'RESOURCES.csv', resources, list(resources[0]))
    proof['GPU_memory'] = resources
    atomic(output/'ENGINEERING_PROOF.json', proof)
    return proof


def continue_remote(output, restored_folder):
    """A fresh worker restores the actual downloaded unit1 and proves updates2–4."""
    output, restored_folder = Path(output), Path(restored_folder)
    cfg = read(output/'PREPARED.json')
    proof = read(output/'ENGINEERING_PROOF.json')
    device, anchor_device = cfg['device'], cfg['anchor_device']
    assert digest(read(output/'CONTRACT.json')) == proof['protocol']
    assert source_contract(device, anchor_device)['sources'] == read(output/'CONTRACT.json')['sources']
    assert environment() == proof['environment'], 'Numerical environment changed'
    seed_all(1)
    torch.set_num_threads(2)
    root = locate(cfg['root'])
    _, rows = dataset(root, verify=False)
    anchor = Anchor(cfg['weights'], anchor_device)
    model = Geo().to(device)
    optimizer = optimizer_for(model)
    store = Store(restored_folder/'resume', proof['fixture_contract'])
    state = store.restore(model, optimizer)
    assert state['update'] == 1, 'Live seam must restore the actual published unit1'
    counters = dict(state['counters'])
    for update in range(2, 5):
        if (output/'STOP_REQUESTED').exists():
            raise InterruptedError('Stop at complete-update boundary')
        print(f'Actual HF-restored continuation update {update}/4', flush=True)
        batch, base, seeds = fetch(root, rows, anchor, device, update)
        records, arrays = update_once(model, optimizer, batch, base, seeds, update, counters)
        store.commit(model, optimizer, update, dict(counters), records, arrays)
    reference_store = Store(output/'direct', proof['fixture_contract'])
    reference_pointer = reference_store.verify()
    reference = torch.load(reference_store.root/reference_pointer['state'], map_location='cpu', weights_only=False)
    assert equal(cpu(model.state_dict()), reference['model']) and equal(cpu(optimizer.state_dict()), reference['optimizer'])
    assert counters == reference['counters'] and equal(rng_state(),reference['rng'])
    names = [p.relative_to(reference_store.root) for p in (reference_store.root/'units').rglob('*') if p.suffix in ('.csv', '.npy')]
    assert all(sha(reference_store.root/p) == sha(store.root/p) for p in names)
    result = dict(status='passed', actual_remote_download_restore_continue=True, scientific_files_identical=len(names),
                  model_optimizer_counters_exact=True, protocol=proof['protocol'], measured_fit_updates=0)
    atomic(output/'HF_CONTINUATION_PROOF.json', result)
    return result


def parts_any_grad(model):
    return any(p.grad is not None for p in model.parameters())
