"""NB16 coordinator: sole HF owner, visible worker, verified remote recovery seam.

This finite engineering job does not start the 60-epoch measured candidate.
"""
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from filelock import FileLock
from common import HERE, atomic, digest, file_meta, read, sha


def status_worker(arguments, output, tick=None):
    environment = dict(os.environ)
    environment.pop('HF_TOKEN', None)
    # Separate process group prevents a notebook interrupt from killing the
    # worker halfway through its durable update before it sees STOP_REQUESTED.
    process = subprocess.Popen([sys.executable, '-B', str(HERE/'worker_geo.py'), *arguments],
                cwd=HERE, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=os.name!='nt',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    messages = queue.Queue()
    def pump():
        for line in process.stdout:
            messages.put(line)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    heartbeat = 0.
    stopped = False
    while process.poll() is None or reader.is_alive() or not messages.empty():
        try:
            try:
                line = messages.get(timeout=.5)
                print(line, end='', flush=True)
            except queue.Empty:
                pass
            if time.monotonic()-heartbeat >= 30:
                cursor = read(output/'LATEST_COMPLETE.json') if (output/'LATEST_COMPLETE.json').exists() else {'stage':'source verification / initialization'}
                print('NB16 alive:', json.dumps(cursor), '— next complete boundary is protected.', flush=True)
                heartbeat = time.monotonic()
                if tick is not None:
                    tick()
        except KeyboardInterrupt:
            (output/'STOP_REQUESTED').write_text('catchable_notebook_stop\n', encoding='utf-8')
            stopped = True
            print('Stop requested. Waiting for the current complete update; then attempting HF publication.', flush=True)
    if stopped:
        raise KeyboardInterrupt
    if process.returncode:
        raise RuntimeError('Engineering worker stopped. Local complete units retained; see WORKER_STOP.json and output above.')


def make_snapshot(output, target, protocol, resume=None):
    """Publish explicit evidence files; exclude secrets, caches and stale native states."""
    import shutil
    from store_geo import Store
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    export_transport_csv(output)
    if resume is not None:
        fixture = read(resume/'POINTER.json')['contract']
        Store(resume, fixture).snapshot(target/'resume')
    if (output/'LATEST_COMPLETE.json').exists():
        cursor = read(output/'LATEST_COMPLETE.json')
        checkpoint = (output/cursor['immutable_snapshot']).resolve()
        assert checkpoint.is_relative_to(output.resolve())
        Store(checkpoint, cursor['fixture_contract']).snapshot(target/'progress')
        atomic(target/'LATEST_COMPLETE.json', cursor)
    for name in ('CONTRACT.json', 'PREPARED_PUBLIC.json', 'PARAMETERS.json', 'source_images.csv', 'schedule.csv',
                 'ENGINEERING_PROOF.json', 'HF_CONTINUATION_PROOF.json', 'STORAGE.json', 'ENGINEERING_SCHEMA.json',
                 'RESOURCES.csv', 'Engineering_Passport.html', 'GALLERY.json', 'FINAL_GATE.json', 'WORKER_STOP.json'):
        if (output/name).exists():
            shutil.copyfile(output/name, target/name)
    for path in output.glob('HF_*.csv'):
        shutil.copyfile(path, target/path.name)
    for name in read(HERE/'FILES.json'):
        dest = target/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/name, dest)
    shutil.copyfile(HERE/'FILES.json', target/'source/FILES.json')
    entries = {p.relative_to(target).as_posix():file_meta(p) for p in target.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    # Nested native snapshot manifest is retained as evidence too.
    for subfolder in ('resume', 'progress'):
        if (target/subfolder/'MANIFEST.json').exists():
            entries[subfolder+'/MANIFEST.json'] = file_meta(target/subfolder/'MANIFEST.json')
    atomic(target/'MANIFEST.json', dict(contract=protocol, files=entries))
    return target


def export_transport_csv(output):
    """Keep safe transport events directly browsable, not JSONL-only records."""
    from common import csv_write
    events = []
    for relative in ('DOWNLOAD_REQUESTS.jsonl','hub/REQUESTS.jsonl','hub/HTTP_REQUESTS.jsonl','RATE_EVENTS.jsonl'):
        path = output/relative
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # Current unflushed tail; the next export includes it.
            events.append(dict(source=relative, **item))
    if events:
        fields = sorted({k for row in events for k in row})
        records = [{k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in events]
        csv_write(output/'HF_transport_events.csv', records, fields)


def storage_report(output):
    """Actual engineering bytes, conservative full-fit lower bound and explicit gaps."""
    import csv
    from collections import defaultdict
    sizes, rows = defaultdict(int), defaultdict(int)
    for path in (output/'direct/units').rglob('*.csv'):
        with path.open(encoding='utf-8', newline='') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        sizes[path.stem] += path.stat().st_size
        rows[path.stem] += count
    schemas = {}
    for path in sorted((output/'direct/units/0004').glob('*.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            schemas[path.stem] = dict(columns=reader.fieldnames, cadence='every completed disposable update',
                                     fixture_mode_only=True, missing='empty field with availability/point_weight context')
    atomic(output/'ENGINEERING_SCHEMA.json', dict(status='actual_engineering_schema', unit_CSV_families=schemas,
                additional_tables=['source_images.csv','schedule.csv','HF_transport_events.csv'],
                measured_40_family_schema='source/SCHEMAS.json is the approved design, not an implemented fit claim'))
    native_bytes = sum(p.stat().st_size for p in (output/'direct/units').rglob('*.npy'))
    plan = read(HERE/'BUDGET.json')
    sample_bound = plan['sample_upper_bound']
    # All fixture batches have eight sources. Includes NPY file header overhead.
    projected_observation_native = math_ceil(native_bytes*sample_bound/32)
    state_bytes = max(p.stat().st_size for p in (output/'direct').glob('state_*.pt'))
    vectors = sizes['vectors']*sample_bound/32
    # Exact endpoint dense CSV row widths measured using the approved 8-column schema.
    from store_geo import Store
    import io
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    values = np_load(output/'direct/units/0004/geometry_logits.npy')
    for sample in range(8):
        for region in range(2):
            for y in range(128):
                for x in range(96):
                    prob = float(1/(1+__import__('numpy').exp(-values[sample,region,y,x])))
                    writer.writerow((100000+sample, region, y, x, format(prob,'.9g'), int(prob>=.5), 16, 16))
    dense_per_row = len(stream.getvalue().encode())/(8*2*128*96)
    projected_dense = math_ceil(dense_per_row*2*570*2*128*96)
    estimate = projected_observation_native+math_ceil(vectors)+projected_dense+600000000+1150000000
    report = dict(scope='engineering_observation_measurements_not_complete_fit_budget',
                  CSV_rows=dict(rows), CSV_bytes=dict(sizes), native_fixture_bytes=native_bytes,
                  max_rolling_state_bytes=state_bytes, native_projected_bytes=projected_observation_native,
                  vector_CSV_projected_bytes=math_ceil(vectors), dense_measured_bytes_per_row=dense_per_row,
                  endpoint_dense_CSV_projected_bytes=projected_dense,
                  provisional_total_with_other_allowances_bytes=math_ceil(estimate*1.2),
                  free_output_bytes=shutil.disk_usage(output).free,
                  complete_40_family_measurement=False, measured_fit_storage_gate_passed=False,
                  reason='NB17 shard writer, all 40 executable CSV families and staging peak require implementation/measurement before measured fit')
    atomic(output/'STORAGE.json', report)
    return report


def math_ceil(value):
    import math
    return math.ceil(value)


def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=False)


def run(token, output='/kaggle/working/tread_two_geo_nb16', data_root=''):
    # HF SDK imports only in this coordinator; numerical workers never see token.
    from publisher import Publisher
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    assert str(output.resolve()).startswith('/kaggle/working/'), 'Run this coordinator in Kaggle; local tests use the offline fixture'
    assert token, 'Enable Kaggle Secret HF_TOKEN for this notebook'
    with FileLock(str(output/'COORDINATOR.lock'), timeout=1):
        return _owned_run(token, output, data_root, Publisher)


def _owned_run(token, output, data_root, Publisher):
    from report_geo import render
    from rate_budget import Budget, BudgetSession
    if (output/'RUN_RESULT.json').exists():
        from engineering_geo import source_contract
        previous = read(output/'CONTRACT.json')
        current = source_contract('cuda:0', 'cuda:1')
        assert previous['sources']==current['sources'] and previous['environment']==current['environment']
        result = read(output/'RUN_RESULT.json')
        assert result['status']=='engineering_gate_passed'
        print('Compatible completed NB16: displaying its verified receipt; no optimizer step or new HF write.', flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return result
    assert shutil.disk_usage(output).free >= 4000000000, 'Start NB16 with at least 4 GB free working output'
    # A new invocation explicitly acknowledges an earlier safe stop.
    if (output/'STOP_REQUESTED').exists():
        (output/'STOP_REQUESTED').unlink()
    spec = read(HERE/'PLAN.json')['anchor']
    weights = output/'anchor.pt'
    if not weights.exists() or sha(weights)!=spec['weights_sha256']:
        session = BudgetSession(Budget(output.parent/'geo_hf_budget'), output/'DOWNLOAD_REQUESTS.jsonl')
        # Pinned public anchor; no dataset archive and no alternate credential lookup.
        print('Downloading the pinned 5.5 MB anchor; photos remain on Kaggle input.', flush=True)
        with session.get(spec['url'], timeout=(20, 150), stream=True) as response:
            response.raise_for_status()
            with weights.with_suffix('.part').open('wb') as stream:
                for block in response.iter_content(1024**2):
                    stream.write(block)
        weights.with_suffix('.part').replace(weights)
    assert sha(weights)==spec['weights_sha256']
    publisher = None
    last_push = 0.
    def get_publisher():
        nonlocal publisher
        if publisher is None and (output/'CONTRACT.json').exists():
            protocol = digest(read(output/'CONTRACT.json'))
            publisher = Publisher('phase3/tread-two/geo-r1/engineering/'+protocol, token, output/'hub/REQUESTS.jsonl')
            publisher.claim()
        return publisher
    def publish_checkpoint(release=False, forced=False):
        nonlocal last_push
        pub = get_publisher()
        if pub is None:
            return None
        if not forced and time.monotonic()-last_push<1800:
            return None
        protocol = digest(read(output/'CONTRACT.json'))
        resume = output/'seam_checkpoint' if (output/'seam_checkpoint/POINTER.json').exists() else None
        snapshot = make_snapshot(output, output/'staging'/str(time.time_ns()), protocol, resume)
        print('HF: publishing direct engineering CSV/native files; server waits remain visible.', flush=True)
        receipt = pub.publish(snapshot, release=release)
        atomic(output/'LAST_HF_PUBLICATION.json', receipt)
        last_push = time.monotonic()
        return receipt
    old_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        status_worker(['fixture', '--output', str(output), '--weights', str(weights), '--data-root', data_root], output,
                      tick=lambda:publish_checkpoint())
        cfg = read(output/'PREPARED.json')
        atomic(output/'PREPARED_PUBLIC.json', {k:v for k,v in cfg.items() if k not in ('root','weights')})
        storage_report(output)
        render(output)
        # Publish the original unit1 state, download it from the verified revision,
        # then continue in a fresh numeric worker with no access to the token.
        receipt = publish_checkpoint(forced=True)
        protocol = cfg['protocol']
        restored = output/'remote_restore'/receipt['revision']
        get_publisher().download(receipt['revision'], restored, protocol)
        # A rerun downloads to a fresh attempt so a prior continuation's update4
        # cannot be mistaken for the remote unit1 seam.
        work = output/'continuations'/str(time.time_ns())
        shutil.copytree(restored, work)
        status_worker(['continue', '--output', str(output), '--restored', str(work)], output, tick=lambda:publish_checkpoint())
        result = dict(status='engineering_gate_passed', actual_dual_T4=True, actual_HF_restore_continue=True,
                      protocol=protocol, measured_fit_started=False, measured_fit_ready=False,
                      next='Independent NB16 receipt review, finish 40-family fit/shard integration and freeze NB17',
                      scope='Preparation only; no 60-epoch fit or scientific improvement claim')
        atomic(output/'FINAL_GATE.json', result)
        result['publication'] = publish_checkpoint(release=True, forced=True)
        atomic(output/'RUN_RESULT.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException as exc:
        try:
            publish_checkpoint(release=True, forced=True)
        except BaseException as failure:
            atomic(output/'HF_PENDING.json', dict(status='pending', error_type=type(failure).__name__,
                        stop_type=type(exc).__name__, last_verified=None if publisher is None else publisher.last_verified))
            print('HF publication pending; preserve Kaggle output and rerun unchanged. See HF_PENDING.json.', flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
