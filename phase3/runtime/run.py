"""One Kaggle session, two independent GPU slots, three seeds, one publisher."""
from PIL import Image
import argparse
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from filelock import FileLock

from common import atomic, cpu, digest, environment, memory, read, sha, validate_completed
from data import BATCH, EPOCHS, HW, SEEDS, plan
from evaluation import gallery
from hub import Publisher
from model import MODEL_VERSION, TreadOne
from phase2_training_data import locate
from phase2_dataset_verify import verify

STOP = False
CHILDREN = []


def stop(*_):
    global STOP
    if STOP:
        return
    STOP = True
    for process in list(CHILDREN):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    print('Stop requested: finishing the active update, then attempting an immediate HF flush.', flush=True)


def preflight(root, work):
    root = locate(root)
    result = verify(root)
    p = plan(root)
    # Source verifier describes source-v1, not this verified runnable overlay.
    report = dict(source=result, overlay_counts=p['counts'], groups=p['groups'],
                  geometry_counts=p['geometry_counts'], overlay_sha256=digest(p),
                  phase3_training_overlay_ready=True,
                  limitations=p['decisions'], source_root=str(root))
    atomic(Path(work)/'DATA_PREFLIGHT.json', report)
    return root, p, report


def contract(p):
    sources = {q.name: sha(q) for q in sorted(Path(__file__).parent.glob('*.py'))}
    versions = {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'Pillow', 'huggingface_hub', 'filelock')}
    bundle = read(Path(__file__).parent/'BUNDLE.json') if (Path(__file__).parent/'BUNDLE.json').exists() else None
    return dict(model=MODEL_VERSION, initialization='random; no pretrained model or teacher', bundle=bundle,
                overlay_sha256=digest(p), sources=sources, packages=versions,
                epochs=EPOCHS, seeds=list(SEEDS), input_hw=list(HW), batch=BATCH,
                optimizer=dict(type='AdamW', lr=.0001, weight_decay=.01, clip_norm=1.),
                scheduler='CosineAnnealingLR, 60 epoch steps, eta_min=1e-6',
                precision='FP16 AMP; initial scale512; eight same-batch scale retries then one FP32 attempt',
                loss=dict(classification=1., ordinal_cdf=.2, segmentation_bce_dice=1., geometry_smoothl1=2.,
                          containment=.1, row_interval_agreement=.1, background_kl=.2),
                background_intervention='Every fourth optimizer update; roll batch outside human tyre/tread/ignore union; detached reference KL',
                selection='0.50*observed-class balanced accuracy +0.35*Dice +0.15*max(0,1-human-point-MAE/0.05); earliest epoch/seed ties',
                sampler='Stateless class, then domain, then tyre/clip balanced sampling with replacement; 49 batches/epoch',
                checkpoint='Every completed update; full state including history, partial cursor, RNG, best weights and selection state',
                augmentation='Frozen Phase2 stateless horizontal flip with left/right swap; brightness/contrast and mild blur',
                explanation='Exact signed class-logit evidence decomposition at 1/4 input resolution; not semantic concepts or a proof of pixel-level causality',
                scope=p['decisions'])


def prepare(args):
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = locate(args.root)
    p = plan(root)
    # Full integrity report is reused only when tied to current checksum list and source paths.
    # Training always verifies bytes again in its own session.
    root, p, report = preflight(str(root), work)
    c = contract(p)
    atomic(work/'CONTRACT.json', c)
    atomic(work/'OVERLAY.json', p)
    atomic(work/'ENVIRONMENT.json', dict(runtime=environment('cuda:0' if torch.cuda.is_available() else 'cpu'),
           memory=memory(), output_free=shutil.disk_usage(work).free, scratch_free=shutil.disk_usage('/tmp' if os.name != 'nt' else work).free,
           gpus=[dict(name=torch.cuda.get_device_name(i), total_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]))
    return root, p, c, work


def export_report(root, p, c, runroot, work):
    report_dir = work/'report'
    report_dir.mkdir(exist_ok=True)
    summaries = []
    winner = None
    for seed in SEEDS:
        path = runroot/f'seed{seed}'/'state.pt'
        if not path.exists():
            return {}
        s = torch.load(path, map_location='cpu', weights_only=False)
        if not s['completed']:
            return {}
        assert s['protocol'] == digest(c)
        validate_completed(s, runroot/f'seed{seed}')
        summaries.append(dict(seed=seed, best_epoch=s['best_epoch'], validation_utility=s['best_score'], evaluation=s['evaluation']))
        if winner is None or s['best_score'] > winner['validation_utility']:
            winner = dict(seed=seed, epoch=s['best_epoch'], validation_utility=s['best_score'])
        atomic(report_dir/f'EVALUATION_seed{seed}.json', s['evaluation'])
        del s
    endpoints = {}
    import numpy as np
    for endpoint in ('selected', 'fixed_final'):
        endpoints[endpoint] = {metric: dict(mean=float(np.mean([s['evaluation'][endpoint][metric] for s in summaries])),
                               population_sd=float(np.std([s['evaluation'][endpoint][metric] for s in summaries])))
                              for metric in ('balanced_accuracy_observed', 'mean_two_region_dice', 'mean_two_region_iou',
                                             'human_point_mae_width_fraction', 'mask_derived_box_iou')}
    atomic(report_dir/'REPORT.json', dict(status='completed', protocol=digest(c), runs=summaries, summary=endpoints,
           export_selection=winner, limits=c['scope'], interpretation='One architecture, three seeds. Seeds/images are not independent tyres. No performance superiority is pre-claimed.'))
    state = torch.load(runroot/f'seed{winner["seed"]}'/'state.pt', map_location='cpu', weights_only=False)
    atomic(report_dir/'weights.pt', state['best'], binary=True)
    atomic(report_dir/'EXPORT.json', dict(model=MODEL_VERSION, **winner, protocol=digest(c),
           weights_sha256=sha(report_dir/'weights.pt'), contract=c,
           input='RGB PIL bilinear direct resize 384wide x512high; ImageNet normalization',
           outputs='Softmax mileage proxy; independent tyre/tread masks at quarter resolution; ordered six guide points; exact signed local logit contributions',
           app_installed=False))
    # CPU final gallery uses the declared validation winner, never the best test seed.
    model = TreadOne()
    model.load_state_dict(state['best'], strict=True)
    gallery(model, p, root, torch.device('cpu'), report_dir/'gallery')
    return {'report/'+str(q.relative_to(report_dir)).replace('\\', '/'): q for q in report_dir.rglob('*') if q.is_file()}


def run(args):
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.tag), 'RUN_TAG must be a short alphanumeric identifier'
    root, p, c, work = prepare(args)
    publisher = Publisher(work, c, args.tag+('-preflight' if args.mode == 'preflight' else ''), args.take_over)
    if args.mode == 'preflight':
        publisher.claim()
        try:
            publisher.publish(work/'no_runs', {'preflight/DATA_PREFLIGHT.json': work/'DATA_PREFLIGHT.json',
                'CONTRACT.json': work/'CONTRACT.json', 'OVERLAY.json': work/'OVERLAY.json',
                'ENVIRONMENT.json': work/'ENVIRONMENT.json'}, release=True)
        except BaseException:
            print('Preflight publication failed; local report retained at', work, flush=True)
            raise
        return
    assert torch.cuda.is_available(), 'Select GPU T4 x2 for training'
    slots = min(2, torch.cuda.device_count())
    assert slots == 2 or args.allow_single_gpu, 'This run expects dual T4. Set ALLOW_SINGLE_GPU explicitly only after a resource review.'
    assert shutil.disk_usage(work).free > 3*1024**3 and shutil.disk_usage('/tmp').free > 5*1024**3
    publisher.claim()
    runroot = Path('/tmp')/('treadone_'+publisher.protocol)/'runs'
    runroot.mkdir(parents=True, exist_ok=True)
    extra = {name: work/name for name in ('CONTRACT.json', 'OVERLAY.json', 'ENVIRONMENT.json', 'DATA_PREFLIGHT.json')}
    extra.update({'sources/'+q.name: q for q in Path(__file__).parent.glob('*.py')})
    active, queue, logs = {}, [], {}
    started = last_push = time.monotonic()
    failure = None
    try:
        print('RUN_PREFIX =', publisher.prefix, '| GPUs:', slots, '| three seeds; one model', flush=True)
        for seed in SEEDS:
            folder = runroot/f'seed{seed}'
            folder.mkdir(exist_ok=True)
            publisher.restore(seed, folder)
            if (folder/'state.pt').exists():
                saved = torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
                assert saved['protocol'] == publisher.protocol
                if saved['completed']:
                    validate_completed(saved, folder)
                    print('Verified restored completed seed; skip', seed, flush=True)
                    continue
            queue.append(seed)
        publisher.publish(runroot, extra)
        last_status = 0
        while queue or active:
            if time.monotonic()-started > 8.25*3600:
                stop()
            for slot in range(slots):
                if STOP or slot in active or not queue:
                    continue
                seed = queue.pop(0)
                folder = runroot/f'seed{seed}'
                cfg = dict(root=str(root), plan=p, protocol=publisher.protocol, seed=seed)
                atomic(folder/'config.json', cfg)
                env = os.environ.copy()
                env.pop('HF_TOKEN', None)
                env.update(CUDA_VISIBLE_DEVICES=str(slot), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                           HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
                log = open(folder/'console.log', 'a', encoding='utf-8')
                logs[slot] = log
                proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).with_name('worker.py')), 'train',
                       '--config', str(folder/'config.json'), '--folder', str(folder), '--device', 'cuda:0'], env=env,
                       stdout=log, stderr=subprocess.STDOUT)
                CHILDREN.append(proc)
                active[slot] = (proc, seed)
                print('GPU', slot, 'seed', seed, 'started; real-model resume gate first', flush=True)
            finished = False
            for slot, (proc, seed) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                logs.pop(slot).close()
                del active[slot]
                CHILDREN.remove(proc)
                finished = True
                path = runroot/f'seed{seed}'/'console.log'
                print('\n'.join(path.read_text(errors='replace').splitlines()[-5:]), flush=True)
                if code and not STOP:
                    raise RuntimeError(f'seed{seed} failed; checkpoint and full console.log retained')
            if finished or time.monotonic()-last_push >= 1800:
                publisher.publish(runroot, extra)
                last_push = time.monotonic()
            if time.monotonic()-last_status > 60:
                for _, seed in active.values():
                    path = runroot/f'seed{seed}'/'console.log'
                    lines = path.read_text(errors='replace').splitlines() if path.exists() else []
                    print('seed', seed, lines[-1] if lines else 'smoke/resume preflight running', flush=True)
                last_status = time.monotonic()
            if STOP and not active:
                break
            time.sleep(1)
        if not STOP:
            extra.update(export_report(root, p, c, runroot, work))
    except BaseException as exc:
        failure = exc
        atomic(work/'SESSION_ERROR.json', dict(type=type(exc).__name__, traceback=traceback.format_exc(), time=time.time()))
        extra['SESSION_ERROR.json'] = work/'SESSION_ERROR.json'
    finally:
        for proc, _ in active.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc, _ in active.values():
            proc.wait()
        for log in logs.values():
            log.close()
        try:
            publisher.publish(runroot, extra, release=True)
        except BaseException:
            print('Final HF flush failed. Last verified remote:', publisher.last.get('revision'),
                  '| live local checkpoints:', runroot, '| output reports:', work, flush=True)
            raise
    if failure:
        raise failure
    print('Paused safely.' if STOP else 'All available work complete; reports and export published.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'train'])
    parser.add_argument('--root', default='')
    parser.add_argument('--work', required=True)
    parser.add_argument('--tag', default='treadone-r1')
    parser.add_argument('--take-over', action='store_true')
    parser.add_argument('--allow-single-gpu', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with FileLock(str(Path(args.work).parent/'treadone_session.lock'), timeout=1):
        run(args)


if __name__ == '__main__':
    main()
