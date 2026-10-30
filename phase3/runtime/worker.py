"""One independent seed per GPU, completed-update checkpoints and fresh-process seam."""
from PIL import Image  # Native imaging DLL order on the existing Windows host.
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from filelock import FileLock
from torch.nn import functional as F

from common import atomic, cpu, delta, deterministic, environment, memory, read, rng, set_rng, sha, validate_completed
from data import BATCH, EPOCHS, batch, counterfactual, schedule
from evaluation import evaluate
from model import TreadOne, loss_terms

STOP = False


def stop(*_):
    global STOP
    STOP = True


def components(device):
    model = TreadOne().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda', init_scale=512.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    return model, opt, scaler, scheduler


def update(model, opt, scaler, b, cursor):
    before = rng()
    model.train()
    events = []
    for attempt in range(9):
        if STOP:
            raise InterruptedError('Stop before retry/update; prior durable state retained')
        set_rng(before)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(b['img'].device.type, dtype=torch.float16, enabled=b['img'].is_cuda and attempt < 8):
            out = model(b['img'], b['guides'])
            terms = loss_terms(out, b)
            consistency = terms['total'].new_zeros(())
            if cursor % 4 == 0:
                altered = model(counterfactual(b), b['guides'])
                consistency = F.kl_div(altered['probabilities'].clamp_min(1e-7).log(), out['probabilities'].detach(), reduction='batchmean')
            value = terms['total'] + .2*consistency
        finite = bool(torch.isfinite(value))
        if finite:
            scaler.scale(value).backward()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = bool(torch.isfinite(norm))
        if finite:
            scaler.step(opt)
            scaler.update()
            return dict(loss=float(value.detach()), components={k: float(v.detach()) for k, v in terms.items()},
                        background_consistency=float(consistency.detach()), grad_norm=float(norm), retries=events)
        opt.zero_grad(set_to_none=True)
        if attempt == 8 or not scaler.is_enabled():
            set_rng(before)
            raise FloatingPointError('Nonfinite update; no sample skipped. Prior completed-update checkpoint is recoverable.')
        scale = float(scaler.get_scale())
        scaler.update(new_scale=max(scale*.5, 1e-8))
        events.append(dict(attempt=attempt+1, previous_scale=scale, action='same_batch_rng_retry'))
    raise AssertionError('unreachable')


def make_state(cfg, device, model, opt, scaler, scheduler):
    return dict(protocol=cfg['protocol'], seed=cfg['seed'], environment=environment(device),
                model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng(), epoch=0, cursor=0, updates=0,
                history=[], validation=[], best=None, best_score=None, best_epoch=None,
                elapsed_seconds=0., exposures={}, completed=False)


def capture(state, model, opt, scaler, scheduler):
    state.update(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                 scheduler=scheduler.state_dict(), rng=rng())
    return state


def restore(state, cfg, device, model, opt, scaler, scheduler):
    assert state['protocol'] == cfg['protocol'] and state['seed'] == cfg['seed'], 'Protocol/seed mismatch; no silent restart'
    assert state['environment'] == environment(device), 'Runtime/GPU changed; explicit compatibility review required'
    model.load_state_dict(state['model'], strict=True)
    opt.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    scheduler.load_state_dict(state['scheduler'])
    # Torch's first-step warning flag is process-local and is not in optimizer.state_dict.
    # At an epoch boundary after restore, earlier durable updates already happened.
    if state['updates'] > 0:
        opt._opt_called = True
    set_rng(state['rng'])


def save(folder, state):
    with FileLock(str(Path(folder)/'checkpoint.lock')):
        atomic(Path(folder)/'state.pt', state, binary=True)


def smoke(cfg, folder, device, continuation=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    deterministic(cfg['seed'])
    model, opt, scaler, scheduler = components(device)
    rows, order = schedule(cfg['plan'], cfg['seed'], 0)
    test_batch = int(cfg.get('smoke_batch', BATCH))
    times = []

    def steps(start, count):
        losses = []
        for cursor in range(start, start+count):
            b = batch(cfg['root'], [rows[i] for i in order[cursor][:test_batch]], device,
                      cfg['seed'], 0, cursor)
            begin = time.monotonic()
            value = update(model, opt, scaler, b, cursor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.monotonic()-begin)
            losses.append(value['loss'])
        return losses

    if continuation:
        state = torch.load(folder/'seam.pt', map_location='cpu', weights_only=False)
        restore(state, cfg, device, model, opt, scaler, scheduler)
        losses = steps(2, 3)
        atomic(folder/'actual.pt', dict(model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()),
               scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), rng=rng(), losses=losses), binary=True)
        return
    steps(0, 2)
    scheduler.step()  # Non-initial scheduler state must survive the seam too.
    state = make_state(cfg, device, model, opt, scaler, scheduler)
    state.update(cursor=2, updates=2)
    atomic(folder/'seam.pt', state, binary=True)
    losses = steps(2, 3)  # Includes the counterfactual branch at cursor 4.
    expected = dict(model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()), scaler=scaler.state_dict(),
                    scheduler=scheduler.state_dict(), rng=rng(), losses=losses)
    parameters = sum(p.numel() for p in model.parameters())
    del model, opt, scaler, scheduler, state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    atomic(folder/'config.json', cfg)
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'continue-smoke', '--config',
                    str(folder/'config.json'), '--folder', str(folder), '--device', str(device)], check=True)
    actual = torch.load(folder/'actual.pt', map_location='cpu', weights_only=False)
    differences = {k: delta(expected[k], actual[k]) for k in expected}
    passed = max(differences.values()) == 0
    report = dict(status='passed' if passed else 'failed', protocol=cfg['protocol'],
                  runtime_sources={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                  differences=differences, parameters=parameters,
                  device=environment(device), batch=test_batch, input_hw=[512, 384],
                  scope='Real model and real transformed images, fresh-process restore, noninitial scheduler, counterfactual branch',
                  step_seconds=times, peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                  estimated_compute_hours_per_seed=sum(times[-3:])/3*len(order)*EPOCHS/3600,
                  estimate_excludes='Per-update durable saves, validation, image decoding and uploads')
    atomic(folder/'SMOKE.json', report)
    for name in ('seam.pt', 'actual.pt'):
        (folder/name).unlink()
    if not passed:
        raise RuntimeError('Fresh-process resume differs: long training blocked; see SMOKE.json')
    print(json.dumps(report), flush=True)
    return report


def train(cfg, folder, device):
    folder = Path(folder)
    report = smoke(cfg, folder/'smoke', device)
    atomic(folder/'SMOKE.json', report)
    if STOP:
        return
    deterministic(cfg['seed'])
    model, opt, scaler, scheduler = components(device)
    atomic(folder/'IDENTITY.json', dict(model='TREAD-ONE-r1', initialization='random, no pretrained weights',
           parameters=sum(p.numel() for p in model.parameters()), shapes={k: list(v.shape) for k, v in model.state_dict().items()}))
    if (folder/'state.pt').exists():
        state = torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
        restore(state, cfg, device, model, opt, scaler, scheduler)
    else:
        state = make_state(cfg, device, model, opt, scaler, scheduler)
        save(folder, state)
    if state['completed']:
        validate_completed(state, folder)
        return
    try:
        while state['epoch'] < EPOCHS:
            rows, order = schedule(cfg['plan'], cfg['seed'], state['epoch'])
            while state['cursor'] < len(order):
                if STOP:
                    return
                if shutil.disk_usage(folder).free < 2*1024**3:
                    raise RuntimeError('Scratch below 2 GiB; durable checkpoint retained')
                mem = memory()
                if mem['source'] == 'cgroup' and mem['fraction'] > .90:
                    raise RuntimeError('Live cgroup RAM exceeds 90%; durable checkpoint retained')
                chosen = [rows[i] for i in order[state['cursor']]]
                b = batch(cfg['root'], chosen, device, cfg['seed'], state['epoch'], state['cursor'])
                started = time.monotonic()
                value = update(model, opt, scaler, b, state['cursor'])
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                duration = time.monotonic()-started
                state['history'].append(dict(epoch=state['epoch']+1, cursor=state['cursor'], update=state['updates']+1,
                    image_ids=[r['image_id'] for r in chosen], lr=opt.param_groups[0]['lr'], seconds=duration,
                    amp_scale=float(scaler.get_scale()), memory=mem,
                    allocated_gpu_bytes=torch.cuda.memory_allocated() if device.type == 'cuda' else None, **value))
                for r in chosen:
                    key = str(r['class_index'])+'/'+r['domain']+'/'+str(r['physical_tyre_id'] or r['source_video'])
                    state['exposures'][key] = state['exposures'].get(key, 0)+1
                state['elapsed_seconds'] += duration
                state['cursor'] += 1
                state['updates'] += 1
                save(folder, capture(state, model, opt, scaler, scheduler))
                del b
            result = evaluate(model, [r for r in cfg['plan']['rows'] if r['role'] == 'validation'],
                              cfg['root'], device, stop=lambda: STOP)
            state['validation'].append(dict(epoch=state['epoch']+1, result=result))
            score = result['joint_validation_utility']
            assert score is not None, 'Joint validation requires human point references'
            if state['best_score'] is None or score > state['best_score']:
                state['best_score'], state['best_epoch'] = score, state['epoch']+1
                state['best'] = cpu(model.state_dict())
            scheduler.step()
            state['epoch'] += 1
            state['cursor'] = 0
            save(folder, capture(state, model, opt, scaler, scheduler))
            print(f"seed{cfg['seed']} epoch {state['epoch']}/{EPOCHS} | joint {score:.5f} | balanced {result['balanced_accuracy_observed']:.4f} | Dice {result['mean_two_region_dice']:.4f} | point width error {result['human_point_mae_width_fraction']:.5f}", flush=True)
        test_rows = [r for r in cfg['plan']['rows'] if r['role'] == 'test']
        final = evaluate(model, test_rows, cfg['root'], device, stop=lambda: STOP, interventions=True,
                         artifact_path=folder/'TEST_FINAL.npz')
        state['model'] = cpu(model.state_dict())
        model.load_state_dict(state['best'], strict=True)
        selected = evaluate(model, test_rows, cfg['root'], device, stop=lambda: STOP, interventions=True,
                            artifact_path=folder/'TEST_SELECTED.npz')
        state['evaluation'] = dict(selected=selected, fixed_final=final,
                                   selected_epoch=state['best_epoch'], selection='Frozen joint validation utility; earliest epoch tie')
        state['artifacts'] = {name: sha(folder/name) for name in ('TEST_FINAL.npz', 'TEST_SELECTED.npz')}
        state['completed'] = True
        # Keep model/optimizer/RNG at final training state; do not relabel selected as last.
        save(folder, state)
        atomic(folder/'EVALUATION.json', state['evaluation'])
    except InterruptedError:
        return


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'smoke', 'continue-smoke'])
    p.add_argument('--config', required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = read(args.config)
    device = torch.device(args.device)
    try:
        if args.mode == 'train':
            train(cfg, folder, device)
        else:
            smoke(cfg, folder, device, args.mode == 'continue-smoke')
    except BaseException:
        if (folder/'smoke/SMOKE.json').exists():
            atomic(folder/'SMOKE.json', read(folder/'smoke/SMOKE.json'))
        atomic(folder/'ERROR.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    if os.name == 'nt':
        import threading
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        threading.stack_size(32*1024**2)
        errors = []
        def windows_main():
            try:
                main()
            except BaseException as e:
                errors.append(e)
        thread = threading.Thread(target=windows_main)
        thread.start()
        thread.join()
        if errors:
            raise errors[0]
    else:
        main()
