"""Kaggle-only publication owner, live engineering gate and safe worker lifecycle."""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from filelock import FileLock
import acceptance
from common import HERE, atomic, capacity, digest, event, file_meta, read, sha
from data_atlas import dataset, locate
from evidence_hub import delay_for, wait
from model_atlas import WEIGHTS_SHA
from publisher import Publisher
from report import build
from schemas import SCHEMAS
from spec import configuration
from store import Store

STOP = False
CHILD = None
ACTIVE_WORK = None


def stop(*_):
    global STOP
    STOP = True
    if ACTIVE_WORK is not None:
        atomic(ACTIVE_WORK / 'STOP_REQUEST.json', dict(reason='catchable_signal'))


def claim(hub):
    # Never steal or delete a live lease. The earlier NB05 contention has a
    # bounded, stop-aware wait here instead of failing immediately.
    for attempt in range(19):
        if STOP:
            raise InterruptedError('Stop during lease wait')
        try:
            hub.claim()
            return
        except RuntimeError as error:
            if 'Another live gate owns' not in str(error) or attempt == 18:
                raise
            print('Another session owns this run. Waiting safely for its release (up to 90 minutes).', flush=True)
            for _ in range(10):
                if STOP:
                    raise InterruptedError('Stop during lease wait')
                time.sleep(30)


def heartbeat(hub):
    return hub.commit([], 'LEASES/' + uuid.uuid4().hex + '.json', dict(writer=hub.owner), 'ATLAS writer lease heartbeat')


def release(hub):
    return hub.commit([], 'RELEASES/' + uuid.uuid4().hex + '.json', dict(writer=hub.owner), 'ATLAS clean owner release', release=True)


def snapshot(store, work, cfg, status, extra=None):
    base = Path(work) / 'snapshots'
    base.mkdir(exist_ok=True)
    for previous in base.glob('snapshot-*'):
        marker=previous/'VERIFIED_PUBLICATION.json'
        if marker.exists():
            resolved=previous.resolve();assert resolved.parent==base.resolve()
            shutil.rmtree(resolved)
    target = store.export(base / ('snapshot-' + uuid.uuid4().hex), HERE, status)
    atomic(target / 'CREATED_BY_ATLAS.json', dict(protocol=cfg['protocol'], purpose='immutable derived publication staging'))
    if (Path(work) / 'PREPARED.json').exists():
        atomic(target / 'PREPARATION.json', read(Path(work) / 'PREPARED.json'))
    if (Path(work) / 'PREFLIGHT_ACCEPTED.json').exists():
        atomic(target / 'LIVE_GATE_RECEIPT.json', read(Path(work) / 'PREFLIGHT_ACCEPTED.json'))
    atomic(target / 'source/FILES.json', read(HERE / 'FILES.json'))
    for scope in ('assets','preflight'):
        for name in ('HTTP_REQUESTS.jsonl','REMOTE_HISTORY.jsonl'):
            origin = Path(work) / scope / name
            if origin.exists():
                dest = target / 'control_history' / scope / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin, dest)
    for name, value in (extra or {}).items():
        assert Path(name).name == name
        atomic(target / name, value)
    build(target, cfg)
    manifest = read(target / 'MANIFEST.json')
    manifest['files'] = {p.relative_to(target).as_posix(): file_meta(p) for p in target.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
    atomic(target / 'MANIFEST.json', manifest)
    return target


def prune_snapshots(work, keep):
    base = (Path(work) / 'snapshots').resolve()
    for path in base.iterdir():
        if path == keep:
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(base) and path.name.startswith('snapshot-') and (path / 'CREATED_BY_ATLAS.json').exists():
            shutil.rmtree(resolved)  # Verified-published derived staging only; run CSV/native state remains.


def prepare(work, token, data_root=''):
    root = locate(data_root)
    assert torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'Select Kaggle GPU T4 x2'
    capacity(work, reserve=12 * 1024**3)
    print('Checking the attached prepared dataset; it stays read-only.', flush=True)
    dataset(root, verify=True)
    assets = work / 'assets'
    assets.mkdir(exist_ok=True)
    weights = assets / 'tread_one_seed3_epoch20.pt'
    if not weights.exists() or sha(weights) != WEIGHTS_SHA:
        hub = Publisher('phase3/tread-two/asset-read', token, assets / 'REMOTE_HISTORY.jsonl')
        url = 'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a0296f18e02b12d28f7e51377783d68538032209/phase3/tread-one-r1/371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c/report/weights.pt'
        def transfer():
            part = weights.with_suffix('.part')
            with hub.session.get(url, timeout=(20,150), stream=True) as response:
                response.raise_for_status()
                with part.open('wb') as stream:
                    for block in response.iter_content(4 * 1024**2):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            assert sha(part) == WEIGHTS_SHA, 'Frozen anchor download hash mismatch'
            os.replace(part, weights)
        hub.call(transfer, 'pinned_anchor_download')
    cfg = configuration(root, weights)
    if (work / 'CONFIG.json').exists():
        assert read(work / 'CONFIG.json')['contract'] == cfg['contract'], 'Source/environment changed; do not overwrite this run'
    atomic(work / 'CONFIG.json', cfg)
    atomic(work / 'PREPARED.json', dict(status='passed', protocol=cfg['protocol'], data_verified=True,
        train_device='cuda:0', eval_device='cuda:1', capacity=capacity(work), optimizer_updates=0,
        note='Prepared only. No measured classifier fit has started.'))
    print('Prepared existing data and exact anchor. Next: automatic live recovery preflight.', flush=True)


def live_preflight(work, token):
    measured = read(work / 'CONFIG.json')
    cfg = configuration(measured['data_root'], measured['weights'], engineering=True)
    cfg['control_root'] = str(work)
    folder = work / 'preflight'
    folder.mkdir(exist_ok=True)
    hub = Publisher(cfg['contract']['HF_prefix'] + '/engineering/' + cfg['protocol'], token, folder / 'REMOTE_HISTORY.jsonl')
    claim(hub)
    try:
        (work / 'STOP_REQUEST.json').unlink(missing_ok=True)
        parent = hub.head()
        manifest = hub.get_json('MANIFEST.json', parent)
        if manifest and 'PREFLIGHT_PROOF.json' in manifest['files']:
            assert manifest['contract'] == cfg['protocol']
            info = manifest['files']['PREFLIGHT_PROOF.json']
            proof = hub.get_json(info['object'], parent)
            assert digest(proof) == info['sha256'] and proof['status'] == 'passed' and proof['actual_live_HF_restore']
            release(hub)
            atomic(work / 'PREFLIGHT_ACCEPTED.json', dict(protocol=measured['protocol'], engineering_protocol=cfg['protocol'], proof=proof, revision=parent))
            print('Verified completed live preflight reused; no repeated fit fixture.', flush=True)
            return
        cases = folder / ('cases-' + cfg['protocol'][:12])
        if (cases / 'SOURCE_SNAPSHOT.json').exists():
            source = Path(read(cases / 'SOURCE_SNAPSHOT.json')['path'])
            if not (cases / 'ABRUPT_PROOF.json').exists():
                source = acceptance.prepare(cfg, cases)
        else:
            source = acceptance.prepare(cfg, cases)
        receipt = hub.publish(source)
        pinned = hub.download(receipt['revision'], folder / ('pinned-' + receipt['revision'][:12]), cfg['protocol'])
        restored, proof = acceptance.finish(cfg, cases, pinned, receipt['revision'])
        proof['devices'] = cfg['contract']['environment']['devices']
        proof['engineering_protocol'] = cfg['protocol']
        target = snapshot(restored, work, cfg, 'live_preflight_passed', {'PREFLIGHT_PROOF.json': proof})
        final = hub.publish(target, release=True)
        atomic(work / 'PREFLIGHT_ACCEPTED.json', dict(protocol=measured['protocol'], engineering_protocol=cfg['protocol'], proof=proof, revision=final['revision']))
        prune_snapshots(work, target)
        print('Live pinned-state restore, full-batch equivalence and storage gates passed.', flush=True)
    except BaseException:
        try:
            if hub.get_json('OWNER.json', hub.head())['owner'] == hub.owner:
                preserved = False
                if 'cases' in locals():
                    for case in ('restored','paused','expected','abrupt'):
                        local = Store(cases / case, cfg['contract'], SCHEMAS)
                        if local.state:
                            target = snapshot(local, work, cfg, 'incomplete_engineering',
                                {'PREFLIGHT_INCOMPLETE.json': dict(case=case, gate_passed=False, measured_fit_started=False)})
                            hub.publish(target, release=True)
                            preserved = True
                            break
                if not preserved:
                    release(hub)
        except BaseException:
            pass
        raise


def restore(hub, work, cfg):
    store = Store(work / 'run', cfg['contract'], SCHEMAS)
    parent = hub.head()
    manifest = hub.get_json('MANIFEST.json', parent)
    if manifest:
        assert manifest['contract'] == cfg['protocol'], 'Remote contract mismatch'
        last = [time.monotonic()]
        def tick():
            if time.monotonic() - last[0] >= 900:
                heartbeat(hub)
                last[0] = time.monotonic()
        hub.transfer_tick = tick
        downloaded = hub.download(parent, work / 'remote_resume', cfg['protocol'])
        remote_units = read(downloaded / 'HISTORY.json')
        if store.state is None or len(remote_units) > len(store.units):
            store.install(downloaded)
        else:
            assert store.units[:len(remote_units)] == remote_units, 'Local/remote history diverged; preserve both for review'
        atomic(work / 'LAST_REMOTE.json', dict(revision=parent, protocol=cfg['protocol'], restored_or_checked=True))
        # The run now has a verified complete copy; remove only derived transfer
        # staging after checking its exact resolved directory.
        path = (work / 'remote_resume').resolve()
        assert path.parent == work.resolve() and path.name == 'remote_resume'
        shutil.rmtree(path)
    return store


def start_worker(cfg, work):
    environment = dict(os.environ)
    environment.pop('HF_TOKEN', None)
    log = (work / 'run/console.log').open('ab')
    child = subprocess.Popen([sys.executable,'-B','-u',str(HERE / 'worker.py'),'--config',str(work / 'CONFIG.json'),
                              '--folder',str(work / 'run')], stdout=log, stderr=subprocess.STDOUT, env=environment)
    return child, log


def publish_run(hub, work, cfg, status, release_owner=False):
    store = Store(work / 'run', cfg['contract'], SCHEMAS)
    if store.state is None:
        if release_owner:
            release(hub)
        return None
    target = snapshot(store, work, cfg, status, {'CAPACITY.json': capacity(work, reserve=0)})
    receipt = hub.publish(target, release=release_owner)
    atomic(target/'VERIFIED_PUBLICATION.json',receipt)
    atomic(work / 'LAST_REMOTE.json', dict(**receipt, protocol=cfg['protocol'], updates=store.state['updates'], phase=store.state['phase']))
    atomic(work / 'LATEST_SNAPSHOT.json', dict(path=str(target), status=status))
    prune_snapshots(work, target)
    return receipt


def train(work, token):
    global CHILD
    cfg = read(work / 'CONFIG.json')
    fresh = configuration(cfg['data_root'], cfg['weights'])
    assert fresh['contract'] == cfg['contract'], 'Runtime/environment drift; no silent compatibility bypass'
    accepted = read(work / 'PREFLIGHT_ACCEPTED.json')
    assert accepted['protocol'] == cfg['protocol'] and accepted['proof']['actual_live_HF_restore']
    assert accepted['proof']['storage_projection']['projection_passed']
    hub = Publisher(cfg['contract']['HF_prefix'] + '/fit/' + cfg['protocol'], token, work / 'run/REMOTE_HISTORY.jsonl')
    claim(hub)
    child, log = None, None
    try:
        store = restore(hub, work, cfg)  # Always check remote before deciding resume/skip.
        if store.state and store.state['phase'] == 'completed':
            publish_run(hub, work, cfg, 'completed_already_verified', True)
            print('Completed fit restored/verified; no training repeated.', flush=True)
            return
        for name in ('STOP_REQUEST.json','FLUSH_REQUEST.json'):
            (work / name).unlink(missing_ok=True)
        heartbeat(hub)
        child, log = start_worker(cfg, work)
        CHILD = child
        last, lease, progress = time.monotonic(), time.monotonic(), 0.
        while True:
            if STOP:
                raise InterruptedError('Catchable stop; commit current unit and flush')
            code = child.poll()
            if code == 0:
                final = Store(work / 'run', cfg['contract'], SCHEMAS)
                assert final.state['phase'] == 'completed' and final.state['updates'] == 2940
                publish_run(hub, work, cfg, 'completed', True)
                print('One ATLAS-LC r2 fit complete. Preserve this executed notebook for independent audit.', flush=True)
                return
            if code is not None:
                raise RuntimeError('Training worker failed; inspect run/console.log; last committed state is retained')
            if time.monotonic() - progress >= 20 and (work / 'run/POINTER.json').exists():
                pointer = read(work / 'run/POINTER.json')
                unit = read(work / 'run' / pointer['unit'])
                print('Saved progress:', unit['updates'], '/ 2940 updates; stage:', unit['phase'], flush=True)
                progress = time.monotonic()
            if time.monotonic() - lease >= 900:
                heartbeat(hub)
                lease = time.monotonic()
            if time.monotonic() - last >= 1800:
                atomic(work / 'FLUSH_REQUEST.json', dict(reason='30_minute_publication'))
                while child.poll() is None:
                    if STOP:
                        raise InterruptedError('Stop while awaiting safe publication boundary')
                    time.sleep(1)
                assert child.returncode == 91
                publish_run(hub, work, cfg, 'periodic')
                log.close()
                (work / 'FLUSH_REQUEST.json').unlink()
                child, log = start_worker(cfg, work)
                CHILD = child
                last, lease = time.monotonic(), time.monotonic()
            time.sleep(2)
    except BaseException as error:
        event(work / 'run/LIFECYCLE.jsonl', 'coordinator_stop_or_error', error_type=type(error).__name__)
        atomic(work / 'STOP_REQUEST.json', dict(reason=type(error).__name__))
        if child and child.poll() is None:
            deadline = time.monotonic() + 180
            while child.poll() is None and time.monotonic() < deadline:
                time.sleep(1)
            if child.poll() is None:
                # Forced termination is an owned worker only; byte-cursor rollback
                # verifies the last complete state before attempting publication.
                child.terminate()
                child.wait(timeout=30)
        try:
            publish_run(hub, work, cfg, 'paused_or_error', True)
            print('Last completed native/CSV state flushed to HF.', flush=True)
        except BaseException as flush_error:
            print('HF flush pending:', type(flush_error).__name__, 'Local state and unsent evidence retained.', flush=True)
        raise
    finally:
        if log:
            log.close()
        CHILD = None


def main():
    global ACTIVE_WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','preflight','train'])
    parser.add_argument('--work', required=True)
    parser.add_argument('--data-root', default='')
    args = parser.parse_args()
    work = Path(args.work).resolve()
    assert Path('/kaggle/working').exists() and work.is_relative_to(Path('/kaggle/working').resolve()), 'Measured publisher is designed for the Kaggle notebook'
    work.mkdir(parents=True, exist_ok=True)
    ACTIVE_WORK = work
    token = os.environ.get('HF_TOKEN')
    assert token, 'Enable Kaggle HF_TOKEN Secret; no credential fallback is used'
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    with FileLock(str(work / 'CONTROLLER.lock'), timeout=1):
        if args.stage == 'prepare':
            prepare(work, token, args.data_root)
        elif args.stage == 'preflight':
            live_preflight(work, token)
        else:
            train(work, token)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('ATLAS controller stopped:', type(error).__name__, 'Inspect the retained safe logs and rerun unchanged after resolving the cause.', flush=True)
        sys.exit(1)
