"""Audit immutable completed artifacts and recover readable CSV without training.

Only the user's hash-verified experiment checkpoints are deserialized. No optimizer
step, architecture change, remote mutation or test-based model selection occurs.
"""
from PIL import Image
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda' or Path('/kaggle/input').exists()
import numpy as np
import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
CSV = OUT/'csv'
CSV.mkdir(parents=True, exist_ok=True)
REPO = 'Shanmuk4622/tyre-wear-study'
REVISION = 'a0296f18e02b12d28f7e51377783d68538032209'
PROTOCOL = '371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write(name, rows, columns=None):
    rows = list(rows)
    fields = list(columns or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    assert fields, name
    with open(CSV/name, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def close(a, b, tol=1e-6):
    assert a is not None and b is not None and abs(float(a)-float(b)) <= tol, (a, b)


def classification(records):
    matrix = np.zeros((3, 3), dtype=int)
    p = np.asarray([r['probabilities'] for r in records], dtype=np.float64)
    y = np.array([r['label'] for r in records], dtype=int)
    pred = p.argmax(1)
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=1e-6)
    assert all(int(v) == r['prediction'] for v, r in zip(pred, records))
    for a, b in zip(y, pred):
        matrix[a, b] += 1
    support = matrix.sum(1)
    observed = support > 0
    per_class = []
    for c in range(3):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum()-tp)
        fn = int(matrix[c].sum()-tp)
        per_class.append(dict(class_index=c, support=int(support[c]), predicted=int(matrix[:, c].sum()),
                        tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                        recall=tp/(tp+fn) if tp+fn else None,
                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
    weights = (np.arange(3)[:, None]-np.arange(3)[None, :])**2/4
    expected = np.outer(matrix.sum(1), matrix.sum(0))/len(records)
    denominator = float((weights*expected).sum())
    confidence = p.max(1)
    correct = (pred == y)
    ece = 0.
    bins = []
    for i in range(10):
        members = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        if members.any():
            avg_conf = float(confidence[members].mean())
            accuracy = float(correct[members].mean())
            ece += float(members.mean())*abs(avg_conf-accuracy)
            bins.append(dict(bin_low=i/10, bin_high=(i+1)/10, images=int(members.sum()), confidence=avg_conf, accuracy=accuracy))
    point = [r['human_point_mae'] for r in records if r['human_point_mae'] is not None]
    metrics = dict(images=len(records), balanced_accuracy_observed=float(np.mean(np.diag(matrix)[observed]/support[observed])),
                   accuracy=float(correct.mean()), macro_f1_three_classes_zero_for_missing=float(np.mean([r['f1'] for r in per_class])),
                   ordinal_mae=float(np.abs(y-pred).mean()), quadratic_weighted_kappa=1-float((weights*matrix).sum())/denominator if denominator else None,
                   nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                   brier_sum_over_classes=float(np.square(p-np.eye(3)[y]).sum(1).mean()),
                   entropy_nats=float(-(p*np.log(np.clip(p, 1e-12, 1))).sum(1).mean()), ece_10_equal_width=ece,
                   low_recall=per_class[0]['recall'], mid_recall=per_class[1]['recall'], high_recall=per_class[2]['recall'],
                   mean_two_region_dice=float(np.mean([r['dice'] for r in records])),
                   mean_two_region_iou=float(np.mean([r['iou'] for r in records])),
                   mask_derived_box_iou=float(np.mean([r['box_iou'] for r in records])),
                   human_point_mae_width_fraction=float(np.mean(point)) if point else None,
                   human_point_images=len(point), evidence_logit_max_residual=max(r['evidence_residual'] for r in records))
    metrics['joint_validation_utility'] = .5*metrics['balanced_accuracy_observed']+.35*metrics['mean_two_region_dice']+.15*max(0, 1-metrics['human_point_mae_width_fraction']/.05) if point else None
    return metrics, matrix, per_class, bins


def flat_prediction(record, **identity):
    result = dict(identity, image_id=record['image_id'], tyre=record['tyre'], domain=record['domain'],
                  label=record['label'], prediction=record['prediction'], human_point_mae=record['human_point_mae'],
                  evidence_residual=record['evidence_residual'], review_flag_count=len(record['review_flags']))
    for key in ('probabilities', 'logits', 'concept_mass'):
        result.update({key+'_'+str(i): v for i, v in enumerate(record[key])})
    for c, region in enumerate(('tyre', 'tread')):
        for key in ('dice', 'iou', 'box_iou'):
            result[region+'_'+key] = record[key][c]
        for kind in ('boxes', 'target_boxes'):
            box = record[kind][c]
            for i, coordinate in enumerate(('x0', 'y0', 'x1_exclusive', 'y1_exclusive')):
                result[region+'_'+kind+'_'+coordinate] = box[i] if box else None
    return result


def main():
    tree = read(OUT/'remote_tree.json')
    entries = {r['path']: r for r in tree['files']}
    report, contract, overlay = [read(OUT/'hf'/p) for p in ('report/REPORT.json', 'CONTRACT.json', 'OVERLAY.json')]
    assert hashlib.sha256(canonical(contract)).hexdigest() == PROTOCOL
    assert hashlib.sha256(canonical(overlay)).hexdigest() == contract['overlay_sha256']
    assert report['status'] == 'completed' and report['protocol'] == PROTOCOL
    rows = {r['image_id']: r for r in overlay['rows']}
    assert len(rows) == 570 and Counter(r['role'] for r in rows.values()) == {'train': 386, 'validation': 81, 'test': 103}
    relevant = ['INDEX.json', 'CONTRACT.json', 'OVERLAY.json', 'ENVIRONMENT.json', 'DATA_PREFLIGHT.json', 'report/REPORT.json', 'report/EXPORT.json']
    for name in relevant+[f'report/EVALUATION_seed{s}.json' for s in (1, 2, 3)]:
        path = OUT/'hf'/name
        body = path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest() == entries[name]['blob_id'], name
    source_root = os.environ.get('PHASE3_SOURCE_ROOT')
    if not source_root:
        if Path('/kaggle/input').exists():
            options = [p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title') == 'Tire Dataset Prepared phase2']
            assert len(options) == 1
            source_root = options[0]
        else:
            source_root = ROOT.parent/'phase2/data/phase2_dataset_v1'
    source_root = Path(source_root)
    assert sha(source_root/'SHA256SUMS.txt') == overlay['source_checksum']
    source_hashes = dict(line.split('  ', 1)[::-1] for line in (source_root/'SHA256SUMS.txt').read_text().splitlines())
    mask_cache = {}
    verified_source_masks = set()
    for r in rows.values():
        if r['role'] != 'test':
            continue
        arrays = []
        for key in ('tyre_mask', 'tread_mask', 'ignore_mask'):
            path = source_root/r[key]
            assert sha(path) == source_hashes[r[key]]
            verified_source_masks.add(r[key])
            with Image.open(path) as im:
                arrays.append(np.asarray(im.resize((384, 512), Image.Resampling.NEAREST)) > 0)
        mask_cache[r['image_id']] = np.stack(arrays)
    train_steps, samples, numeric_events, epochs, val_pred, test_pred = [], [], [], [], [], []
    val_conf, class_rows, calibration, points, flags, test_groups, test_summary, region_stats, exposures, resource_smoke = [], [], [], [], [], [], [], [], [], []
    validations = []
    for seed in (1, 2, 3):
        name = f'runs/seed{seed}/latest.zip'
        archive = OUT/'hf'/name
        assert archive.stat().st_size == entries[name]['bytes'] and sha(archive) == entries[name]['lfs_sha256']
        extracted = OUT/'extracted'/f'seed{seed}'
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('MANIFEST.json'))
            assert set(z.namelist()) == set(manifest['files']) | {'MANIFEST.json'}
            assert all(Path(n).name == n for n in z.namelist())
            assert manifest['protocol'] == PROTOCOL and manifest['seed'] == seed
            for name, meta in manifest['files'].items():
                body = z.read(name)
                assert len(body) == meta['bytes'] and hashlib.sha256(body).hexdigest() == meta['sha256'], name
                if not name.endswith('.npz'):
                    (extracted/name).write_bytes(body)
            state = torch.load(extracted/'state.pt', map_location='cpu', weights_only=False)
            assert state['completed'] and state['protocol'] == PROTOCOL and state['seed'] == seed
            assert state['epoch'] == 60 and state['cursor'] == 0 and state['updates'] == 2940
            assert [v['update'] for v in state['history']] == list(range(1, 2941))
            assert [(v['epoch'], v['cursor']) for v in state['history']] == [(e, b) for e in range(1, 61) for b in range(49)]
            assert [v['epoch'] for v in state['validation']] == list(range(1, 61))
            assert all(torch.isfinite(t).all() for t in state['model'].values())
            assert all(torch.isfinite(t).all() for t in state['best'].values())
            assert sum(t.numel() for t in state['model'].values()) == 1372159
            smoke = read(extracted/'SMOKE.json')
            assert smoke['status'] == 'passed' and max(smoke['differences'].values()) == 0
            assert smoke['device']['device'] == 'Tesla T4'
            assert smoke['runtime_sources'] == contract['sources']
            resource_smoke.append(dict(seed=seed, input_height=512, input_width=384, batch=smoke['batch'], parameters=smoke['parameters'], gpu=smoke['device']['device'],
                peak_allocated_gpu_bytes=smoke['peak_allocated_gpu_bytes'], steps_seconds_mean=float(np.mean(smoke['step_seconds'][1:])),
                state_difference_max=max(smoke['differences'].values())))
            counted = Counter()
            for item in state['history']:
                assert len(item['image_ids']) == 8
                assert math.isfinite(item['loss']) and math.isfinite(item['grad_norm'])
                record = dict(seed=seed, epoch=item['epoch'], cursor=item['cursor'], update=item['update'],
                              loss=item['loss'], background_consistency=item['background_consistency'], lr=item['lr'],
                              update_compute_seconds=item['seconds'], grad_norm_pre_clip=item['grad_norm'], amp_scale=item['amp_scale'],
                              gpu_allocated_bytes=item['allocated_gpu_bytes'], numerical_retry_count=len(item['retries']))
                record.update({'loss_'+k: v for k, v in item['components'].items()})
                record.update({'memory_'+k: v for k, v in item['memory'].items()})
                train_steps.append(record)
                for slot, image_id in enumerate(item['image_ids']):
                    r = rows[image_id]
                    assert r['role'] == 'train'
                    counted[str(r['class_index'])+'/'+r['domain']+'/'+str(r['physical_tyre_id'] or r['source_video'])] += 1
                    samples.append(dict(seed=seed, epoch=item['epoch'], update=item['update'], batch_slot=slot, image_id=image_id,
                                   class_index=r['class_index'], domain=r['domain'], tyre=r['physical_tyre_id'], source_video=r['source_video']))
                for event in item['retries']:
                    numeric_events.append(dict(seed=seed, update=item['update'], **event))
            assert dict(counted) == state['exposures']
            for identity, count in counted.items():
                cls, domain, group = identity.split('/', 2)
                exposures.append(dict(seed=seed, class_index=cls, domain=domain, tyre_or_clip=group, sample_exposures=count))
            for entry in state['validation']:
                epoch, result = entry['epoch'], entry['result']
                assert {r['image_id'] for r in result['records']} == {k for k, v in rows.items() if v['role'] == 'validation'}
                derived, confusion, per_class, bins = classification(result['records'])
                for key in ('balanced_accuracy_observed', 'mean_two_region_dice', 'mean_two_region_iou', 'mask_derived_box_iou', 'human_point_mae_width_fraction', 'joint_validation_utility'):
                    close(derived[key], result[key])
                steps = state['history'][(epoch-1)*49:epoch*49]
                ep = dict(seed=seed, epoch=epoch, selected=(epoch == state['best_epoch']), **derived)
                ep.update({'train_'+k: float(np.mean([r[k] for r in train_steps if r['seed'] == seed and r['epoch'] == epoch]))
                           for k in ('loss', 'loss_classification', 'loss_ordinal', 'loss_segmentation', 'loss_geometry', 'loss_containment', 'loss_alignment', 'background_consistency', 'grad_norm_pre_clip')})
                epochs.append(ep)
                for y in range(3):
                    for pred in range(3):
                        val_conf.append(dict(seed=seed, epoch=epoch, target_class=y, predicted_class=pred, images=int(confusion[y, pred])))
                class_rows += [dict(seed=seed, split='validation', epoch=epoch, endpoint='', **v) for v in per_class]
                calibration += [dict(seed=seed, split='validation', epoch=epoch, endpoint='', **v) for v in bins]
                for rec in result['records']:
                    val_pred.append(flat_prediction(rec, seed=seed, epoch=epoch, split='validation'))
                    for i, value in enumerate(rec['points']):
                        target = rec['point_target'][i] if rec['point_target'] is not None else None
                        points.append(dict(seed=seed, split='validation', epoch=epoch, endpoint='', image_id=rec['image_id'], tyre=rec['tyre'],
                                      point_index=i, guide_index=i//2, side='left' if i%2 == 0 else 'right', prediction=value, target=target,
                                      abs_error_width_fraction=abs(value-target) if target is not None else None,
                                      abs_error_native_pixels=abs(value-target)*(rows[rec['image_id']]['width']-1) if target is not None else None))
                    flags += [dict(seed=seed, split='validation', epoch=epoch, endpoint='', image_id=rec['image_id'], flag=f) for f in rec['review_flags']]
            best = max(state['validation'], key=lambda v: v['result']['joint_validation_utility'])
            assert best['epoch'] == state['best_epoch'] and best['result']['joint_validation_utility'] == state['best_score']
            record = report['runs'][seed-1]
            assert record['seed'] == seed and record['best_epoch'] == state['best_epoch']
            close(record['validation_utility'], state['best_score'])
            assert state['evaluation'] == record['evaluation'] == read(OUT/'hf'/f'report/EVALUATION_seed{seed}.json')
            for endpoint, member in [('selected', 'TEST_SELECTED.npz'), ('fixed_final', 'TEST_FINAL.npz')]:
                result = state['evaluation'][endpoint]
                assert {r['image_id'] for r in result['records']} == {k for k, v in rows.items() if v['role'] == 'test'}
                derived, confusion, per_class, bins = classification(result['records'])
                for key in ('balanced_accuracy_observed', 'mean_two_region_dice', 'mean_two_region_iou', 'mask_derived_box_iou', 'human_point_mae_width_fraction'):
                    close(derived[key], result[key])
                test_summary.append(dict(seed=seed, endpoint=endpoint, selected_epoch=state['best_epoch'], validation_utility=state['best_score'], **derived))
                class_rows += [dict(seed=seed, split='test', epoch=state['best_epoch'] if endpoint == 'selected' else 60, endpoint=endpoint, **v) for v in per_class]
                calibration += [dict(seed=seed, split='test', epoch=state['best_epoch'] if endpoint == 'selected' else 60, endpoint=endpoint, **v) for v in bins]
                with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as raw:
                    assert raw['image_ids'].tolist() == [r['image_id'] for r in result['records']]
                    for i, rec in enumerate(result['records']):
                        image_id = rec['image_id']
                        prob = np.exp(raw['logits'][i]-raw['logits'][i].max())
                        prob /= prob.sum()
                        assert np.allclose(prob, rec['probabilities'], atol=1e-6)
                        assert np.allclose(raw['points'][i], rec['points'], atol=1e-7)
                        assert np.max(np.abs(raw['contributions'][i].sum((-2, -1))+raw['bias']-raw['logits'][i])) < 1e-5
                        masks = np.repeat(np.repeat(raw['masks'][i], 4, -2), 4, -1)
                        target_masks = mask_cache[image_id]
                        valid = ~target_masks[2]
                        for c, name in enumerate(('tyre', 'tread')):
                            pred_mask, truth = (masks[c] >= .5)&valid, target_masks[c]&valid
                            tp, fp, fn, tn = int((pred_mask&truth).sum()), int((pred_mask&~truth&valid).sum()), int((~pred_mask&truth).sum()), int((~pred_mask&~truth&valid).sum())
                            dice, iou = (2*tp+1)/(2*tp+fp+fn+1), (tp+1)/(tp+fp+fn+1)
                            close(dice, rec['dice'][c], 1e-8)
                            close(iou, rec['iou'][c], 1e-8)
                            region_stats.append(dict(seed=seed, endpoint=endpoint, image_id=image_id, region=name, tp=tp, fp=fp, fn=fn, tn=tn,
                               valid_pixels=int(valid.sum()), ignored_pixels=int((~valid).sum()), dice=dice, iou=iou,
                               precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None))
                        flat = flat_prediction(rec, seed=seed, endpoint=endpoint, epoch=state['best_epoch'] if endpoint == 'selected' else 60, split='test')
                        for key in ('background_neutralized_probability_l1', 'remove_top10_feature_evidence_logit_drop', 'occlude_top10_image_probability_change'):
                            flat[key] = rec[key]
                        test_pred.append(flat)
                        for j, value in enumerate(rec['points']):
                            target = rec['point_target'][j] if rec['point_target'] is not None else None
                            points.append(dict(seed=seed, split='test', epoch=flat['epoch'], endpoint=endpoint, image_id=image_id, tyre=rec['tyre'],
                                          point_index=j, guide_index=j//2, side='left' if j%2 == 0 else 'right', prediction=value, target=target,
                                          abs_error_width_fraction=abs(value-target) if target is not None else None,
                                          abs_error_native_pixels=abs(value-target)*(rows[image_id]['width']-1) if target is not None else None))
                        flags += [dict(seed=seed, split='test', epoch=flat['epoch'], endpoint=endpoint, image_id=image_id, flag=f) for f in rec['review_flags']]
                for tyre in sorted({r['tyre'] for r in result['records']}):
                    metrics, _, _, _ = classification([r for r in result['records'] if r['tyre'] == tyre])
                    test_groups.append(dict(seed=seed, endpoint=endpoint, tyre=tyre, **metrics))
            validations.append(dict(seed=seed, epoch=state['epoch'], updates=state['updates'], history_rows=len(state['history']), validation_epochs=len(state['validation']),
                   best_epoch=state['best_epoch'], validation_utility=state['best_score'], finite_losses=True,
                   nonfinite_retry_events=sum(len(r['retries']) for r in state['history']), compute_seconds=state['elapsed_seconds'], bundle_sha256=entries[f'runs/seed{seed}/latest.zip']['lfs_sha256']))
            del state
        print('Seed', seed, 'checkpoint, 60 validation epochs, both raw test endpoints and scalar histories verified', flush=True)
    expected_winner = max(report['runs'], key=lambda r: r['validation_utility'])
    assert report['export_selection']['seed'] == expected_winner['seed'] == 3
    assert report['export_selection']['epoch'] == expected_winner['best_epoch'] == 20
    totals = {}
    for name, data in [('training_steps.csv', train_steps), ('training_samples.csv', samples), ('validation_epochs.csv', epochs),
                       ('validation_predictions.csv', val_pred), ('validation_confusion.csv', val_conf), ('class_metrics.csv', class_rows),
                       ('calibration_bins.csv', calibration), ('point_predictions.csv', points), ('review_flags.csv', flags),
                       ('test_predictions.csv', test_pred), ('test_region_stats.csv', region_stats), ('test_group_metrics.csv', test_groups),
                       ('run_summary.csv', test_summary), ('sample_exposures.csv', exposures), ('gpu_resume_checks.csv', resource_smoke), ('completion_checks.csv', validations)]:
        totals[name] = write(name, data)
    totals['numerical_events.csv'] = write('numerical_events.csv', numeric_events, ['seed', 'update', 'attempt', 'previous_scale', 'action'])
    assert totals['training_steps.csv'] == 8820 and totals['training_samples.csv'] == 70560
    assert totals['validation_predictions.csv'] == 14580 and totals['test_predictions.csv'] == 618
    assert totals['validation_epochs.csv'] == 180 and totals['test_region_stats.csv'] == 1236
    for endpoint in ('selected', 'fixed_final'):
        selected = [r for r in test_summary if r['endpoint'] == endpoint]
        for metric, saved in report['summary'][endpoint].items():
            close(np.mean([r[metric] for r in selected]), saved['mean'])
            close(np.std([r[metric] for r in selected]), saved['population_sd'])
    validation = dict(status='passed', source_repo=REPO, source_revision=REVISION, protocol=PROTOCOL,
        seeds=validations, csv_rows=totals, source_mask_files_verified=len(verified_source_masks),
        claims=dict(completed_runs=3, epochs_per_run=60, updates_per_run=2940, test_images_per_endpoint=103, human_geometry_test_images=25),
        all_bundle_and_member_hashes_verified=True, raw_test_metrics_recalculated=True, validation_metrics_recalculated=True,
        checkpoint_or_model_modified=False, training_performed=False,
        derived_metrics='NLL, Brier, entropy, ECE, QWK, ordinal MAE, class precision/recall/F1 and raw-test region TP/FP/FN/TN derived post hoc from existing saved predictions.',
        limits='No missing training-image predictions, continuous resource traces, or per-epoch full validation mask arrays can be retroactively invented.')
    (OUT/'AUDIT.json').write_text(json.dumps(validation, indent=2)+'\n')
    write('artifact_manifest.csv', [dict(path=str(p.relative_to(OUT)).replace('\\', '/'), bytes=p.stat().st_size, sha256=sha(p), source_revision=REVISION) for p in sorted(CSV.glob('*.csv')) if p.name != 'artifact_manifest.csv'])
    print(json.dumps(dict(status='passed', rows=totals, summaries=test_summary, checks=validations), indent=2), flush=True)


if __name__ == '__main__':
    main()
