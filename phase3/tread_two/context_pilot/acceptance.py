"""Fresh-process actual-batch-eight acceptance; engineering fixtures only."""
import itertools
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from common import HERE, atomic, file_meta, read
from schemas import SCHEMAS
from store import Store


def launch(cfg, folder, limit=None, fault=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / 'CONFIG.json'
    atomic(config, cfg)
    command = [sys.executable, '-B', '-u', str(HERE / 'worker.py'), '--config', str(config), '--folder', str(folder)]
    if limit is not None:
        command += ['--limit-updates', str(limit)]
    if fault:
        command += ['--fault-stage', fault]
    environment = dict(os.environ)
    environment.pop('HF_TOKEN', None)
    with (folder / 'console.log').open('ab') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False)
    assert result.returncode in (0, 91, 86), 'Engineering worker failed; inspect its console.log'
    if fault:
        assert result.returncode == 86
    elif limit is not None:
        assert result.returncode == 91
    else:
        assert result.returncode == 0
    return Store(folder, cfg['contract'], SCHEMAS)


def same(a, b, path='root'):
    if torch.is_tensor(a):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            same(a[key], b[key], path + '.' + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + '.' + str(i))
    else:
        assert a == b, (path, a, b)


def compare(expected, restored):
    assert expected.state['phase'] == restored.state['phase'] == 'completed'
    assert expected.state['updates'] == restored.state['updates'] == 4
    for key in ('model','optimizer','scheduler','scaler','rng','best','sample_cursor','epoch_orders','dense_seen','row_cursors'):
        same(expected.state[key], restored.state[key], key)
    omit = {'compute_seconds','record_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','host_peak_rss_bytes','free_disk_bytes'}
    counts = {}
    marker = object()
    for family in SCHEMAS:
        count = 0
        for a, b in itertools.zip_longest(expected.rows(family), restored.rows(family), fillvalue=marker):
            assert a is not marker and b is not marker, family
            if family == 'train_updates':
                a = {k: v for k, v in a.items() if k not in omit}
                b = {k: v for k, v in b.items() if k not in omit}
            assert a == b, (family, count)
            count += 1
        counts[family] = count
    artifacts = {name for name in expected.state['artifacts'] if name.endswith('.pt')}
    assert artifacts == {name for name in restored.state['artifacts'] if name.endswith('.pt')}
    for name in artifacts:
        same(torch.load(expected.root / name, map_location='cpu', weights_only=False),
             torch.load(restored.root / name, map_location='cpu', weights_only=False), name)
    return dict(status='passed', actual_batch=8, optimizer_updates=4, scientific_CSV_families_equal=len(counts),
                scientific_rows_by_family=counts, native_model_optimizer_scheduler_scaler_RNG_equal=True,
                native_artifacts_equal=len(artifacts), scope='Engineering fixture, validation examples only; no measured accuracy claim')


def prepare(cfg, folder):
    assert cfg['contract']['mode'] == 'engineering_fixture'
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print('Engineering: uninterrupted full-batch reference', flush=True)
    expected = launch(cfg, folder / 'expected')
    print('Engineering: stop after completed update 1', flush=True)
    paused = launch(cfg, folder / 'paused', limit=1)
    assert paused.state['updates'] == 1 and paused.state['cursor'] == 1
    snapshot = paused.export(folder / ('source_snapshot-' + uuid.uuid4().hex[:10]), HERE, 'engineering_source_after_update_1')
    atomic(folder / 'SOURCE_SNAPSHOT.json', dict(path=str(snapshot)))
    print('Engineering: abrupt termination during new native save', flush=True)
    broken = launch(cfg, folder / 'abrupt', fault='after_native')
    assert broken.state['updates'] == 0
    abrupt = launch(cfg, folder / 'abrupt')
    abrupt_proof = compare(expected, abrupt)
    atomic(folder / 'ABRUPT_PROOF.json', abrupt_proof)
    return snapshot


def finish(cfg, folder, source, remote_revision=None):
    folder = Path(folder)
    restored = Store(folder / 'restored', cfg['contract'], SCHEMAS)
    if restored.state is None:
        restored.install(source)
    restored = launch(cfg, folder / 'restored')
    expected = Store(folder / 'expected', cfg['contract'], SCHEMAS)
    proof = compare(expected, restored)
    proof.update(actual_live_HF_restore=remote_revision is not None, pinned_source_revision=remote_revision,
                 uninterrupted_vs_abrupt=read(folder / 'ABRUPT_PROOF.json')['status'], full_training_started=False)
    # Conservative resource projection uses measured raw CSV bytes/rows, not
    # a guessed universal CSV row size. Scientific cache bounds use all 570 images.
    sizes, rows = {}, expected.state['row_cursors']
    for path in (expected.root / 'csv').rglob('*.csv'):
        sizes[path.stem] = sizes.get(path.stem, 0) + path.stat().st_size
    observed_dense = len(expected.state['dense_seen'])
    total_queries = 2 * 60 * 49 * 8 + 60 * 81 + 2 * 103 * 9
    prediction_count = rows['predictions']
    estimate = 0
    per_family = {}
    for family, size in sizes.items():
        if family in ('source_images','bank_cases','bank_pixels'):
            planned = size
        elif family in ('dense_pixels','dense_regions','dense_points','atlas_rows'):
            planned = size * 570 / max(observed_dense, 1)
        elif family in ('train_updates','parameter_gradients','coefficients','sample_exposures','view_pairs','view_patch_consistency'):
            planned = size * 2940 / 4
        elif family in ('affinities','patch_support','patch_evidence','predictions','case_rules','context_evidence','context_summary'):
            planned = size * total_queries / prediction_count
        elif family=='endpoint_attention':
            planned=size*103/2
        else:
            planned = size * 60 / 2 * 16  # Generous group/checkpoint/endpoint bound.
        per_family[family] = int(planned)
        estimate += planned
    # Extrapolate each native artifact category by its actual observation count.
    # Do not multiply epoch weights by the endpoint-image ratio.
    categories={kind:sum(v['bytes'] for n,v in expected.state['artifacts'].items() if n.startswith('artifacts/'+kind+'/') and n.endswith('.pt')) for kind in ('epochs','endpoint_maps','dense')}
    assert all(value>0 for value in categories.values()),'Missing measured native category'
    projected_native=categories['epochs']*60/2+categories['endpoint_maps']*103/2+categories['dense']*570/max(observed_dense,1)+100_000_000
    # Working copy + immutable staging + temporary restore/history reserve.
    projected_peak = int(2 * (estimate + projected_native) + 2_000_000_000)
    proof['storage_projection'] = dict(measured_fixture_CSV_bytes=sizes, projected_CSV_bytes=per_family,
        projected_native_bytes=int(projected_native),native_fixture_bytes_by_category=categories, conservative_peak_output_bytes=projected_peak,
        output_guard_bytes=16_000_000_000, projection_passed=projected_peak < 16_000_000_000,
        note='Planning extrapolation, not a quota guarantee; runtime capacity guards remain active')
    assert proof['storage_projection']['projection_passed'], 'Projected output exceeds the guard; revise storage before measured training'
    atomic(folder / 'PREFLIGHT_PROOF.json', proof)
    return restored, proof
