"""Complete feasible prediction/evidence rows and explicit-denominator metrics."""
import math
from collections import defaultdict

import numpy as np
import torch
from common import digest
from data_atlas import truth_masks
from frozen_anchor import boxes


def learner_sha(model):
    import hashlib
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def coefficients(model, run, update):
    rows = []
    weights = model.coefficients().detach().cpu().numpy()
    for cls in range(3):
        for index in range(18):
            rows.append(dict(run=run, parameter_update=update, kind='signed_weight', class_index=cls,
                             prototype=index, value=float(weights[cls, index])))
        rows.append(dict(run=run, parameter_update=update, kind='bias', class_index=cls, prototype=None,
                         value=float(model.bias[cls].detach())))
    rows.append(dict(run=run, parameter_update=update, kind='temperature', class_index=None,
                     prototype=None, value=float((.05 + .95 * model.raw_temperature.sigmoid()).detach())))
    return rows


def observations(out, rows, registry, cfg, stage, endpoint, epoch, update, start_sample,
                 parameter_update, fingerprint, variant='identity', occluded=None):
    values = {key: out[key].detach().float().cpu().numpy() for key in
              ('logits', 'probabilities', 'patch_evidence', 'affinity', 'cosine', 'patch_weights', 'patch_coverage')}
    biases = out['bias'].detach().float().cpu().numpy()
    labels = torch.tensor([row['class_index'] for row in rows], device=out['logits'].device)
    nll = torch.nn.functional.cross_entropy(out['logits'], labels, reduction='none').detach().cpu().numpy()
    replacements = out['replacement'].detach().cpu().numpy() if out['replacement'] is not None else None
    anchor_logits = out['tread_one_logits'].detach().cpu().numpy()
    anchor_probabilities = out['tread_one_probabilities'].detach().cpu().numpy()
    tables = {name: [] for name in ('predictions', 'affinities', 'patch_support', 'patch_evidence')}
    internal = []
    for slot, row in enumerate(rows):
        sample = start_sample + slot
        probability, logit = values['probabilities'][slot], values['logits'][slot]
        label = int(row['class_index'])
        alpha = values['patch_weights'][slot]
        residual = float(np.max(np.abs(values['patch_evidence'][slot].sum(0, dtype=np.float64) + biases - logit)))
        assert residual < 2e-6 and np.isfinite(probability).all() and abs(float(probability.sum()) - 1) < 1e-6
        pred = dict(run=cfg['run'], sample=sample, stage=stage, endpoint=endpoint, epoch=epoch,
                    update=update, parameter_update=parameter_update, image_index=registry[row['image_id']], image_id=row['image_id'],
                    label=label, prediction=int(probability.argmax()), nll=float(nll[slot]),
                    brier=float(((probability - np.eye(3)[label])**2).sum()),
                    entropy=float(-(probability * np.log(probability.clip(1e-12))).sum()),
                    correct=int(probability.argmax() == label), evidence_residual=residual,
                    supported_patches=int((alpha > 0).sum()), supported_bands=int((alpha.reshape(8, 4).sum(1) > 0).sum()),
                    clipped_fraction=out['atlas']['diagnostics'][slot]['clipped_row_fraction'],
                    empty_support=int(alpha.sum() == 0), variant=variant,
                    occluded_patch=int(occluded[slot]) if occluded is not None else None,
                    replacement_r=float(replacements[slot, 0]) if replacements is not None else None,
                    replacement_g=float(replacements[slot, 1]) if replacements is not None else None,
                    replacement_b=float(replacements[slot, 2]) if replacements is not None else None,
                    learner_sha256=fingerprint)
        pred.update(source_sha256=row['image_sha256'],**dict(zip(('allowed_low','allowed_mid','allowed_high'),out['allowed_counts'][slot].detach().cpu().tolist())))
        pred.update({f'logit_{c}': float(logit[c]) for c in range(3)})
        pred.update({f'probability_{c}': float(probability[c]) for c in range(3)})
        pred.update({f'anchor_logit_{c}': float(anchor_logits[slot,c]) for c in range(3)})
        pred.update({f'anchor_probability_{c}': float(anchor_probabilities[slot,c]) for c in range(3)})
        pred['anchor_prediction'] = int(anchor_probabilities[slot].argmax())
        tables['predictions'].append(pred)
        internal.append(dict(pred, _domain=row['domain'], _tyre=row['physical_tyre_id'],
                             _source_group=row['physical_tyre_id'] or row['source_video']))
        for tile in range(32):
            tables['patch_support'].append(dict(run=cfg['run'], sample=sample, patch=tile,
                                                valid_fraction=float(values['patch_coverage'][slot, tile]), pooling_mass=float(alpha[tile])))
            for prototype in range(18):
                tables['affinities'].append(dict(run=cfg['run'], sample=sample, patch=tile, prototype=prototype,
                                                cosine=float(values['cosine'][slot, tile, prototype]), affinity=float(values['affinity'][slot, tile, prototype])))
            for cls in range(3):
                tables['patch_evidence'].append(dict(run=cfg['run'], sample=sample, patch=tile,
                                                    class_index=cls, contribution=float(values['patch_evidence'][slot, tile, cls])))
    allowed=out['reference_allowed'].detach().cpu().numpy();scales=out['reference_scale'].detach().cpu().numpy()
    tables['case_rules']=[dict(run=cfg['run'],sample=start_sample+i,prototype=j,reference_class=j//6,allowed=int(allowed[i,j]),same_source_sha256=int(not allowed[i,j]),normalization_scale=float(scales[i,j])) for i in range(len(rows)) for j in range(18)]
    return tables, internal


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.


def dense_once(out, rows, registry, cfg, seen):
    tables = {key: [] for key in ('dense_pixels', 'atlas_rows', 'dense_regions', 'dense_points')}
    packed = []
    newly_seen = []
    for slot, row in enumerate(rows):
        image_id = row['image_id']
        if image_id in seen or image_id in newly_seen:
            continue
        index = registry[image_id]
        masks = out['masks'][slot].detach().cpu().numpy()
        points = out['points'][slot].detach().cpu().numpy()
        guides = out['guides'][slot].detach().cpu().numpy()
        valid = out['atlas']['valid'][slot, 0].cpu().numpy()
        grid = out['atlas']['grid'][slot].cpu().numpy()
        truth = truth_masks(cfg['data_root'], row)
        predicted = np.repeat(np.repeat(masks, 4, -2), 4, -1) >= .5
        unignored = ~truth[2]
        pb = boxes(torch.tensor(predicted & unignored))
        tb = boxes(torch.tensor(truth[:2] & unignored))
        for channel in range(2):
            for y in range(128):
                tables['dense_pixels'].extend(dict(run=cfg['run'], image_index=index, channel=channel,
                                                   row=y, column=x, probability=float(masks[channel, y, x])) for x in range(96))
            p, t = predicted[channel] & unignored, truth[channel] & unignored
            intersection, union = int((p & t).sum()), int((p | t).sum())
            a, b = pb[channel], tb[channel]
            record = dict(run=cfg['run'], image_index=index, region=channel, predicted_pixels=int(p.sum()), target_pixels=int(t.sum()),
                          intersection=intersection, union=union, valid_pixels=int(unignored.sum()),
                          dice=(2 * intersection + 1) / (int(p.sum() + t.sum()) + 1), iou=(intersection + 1) / (union + 1), box_iou=box_iou(a, b))
            for key, value in zip(('pred_left','pred_top','pred_right_exclusive','pred_bottom_exclusive'), a or [None] * 4):
                record[key] = value
            for key, value in zip(('truth_left','truth_top','truth_right_exclusive','truth_bottom_exclusive'), b or [None] * 4):
                record[key] = value
            tables['dense_regions'].append(record)
        for point in range(6):
            target = row['points'][point] if row['points'] is not None else None
            target_y = row['guide_fractions'][point // 2]
            comparable = target is not None and abs(float(guides[point // 2]) - target_y) <= 1e-5
            tables['dense_points'].append(dict(run=cfg['run'], image_index=index, point=point,
                                               predicted_x_fraction=float(points[point]), target_x_fraction=target, point_kind=row['point_kind'],
                                               prediction_guide_y_fraction=float(guides[point // 2]), target_guide_y_fraction=target_y,
                                               absolute_error_width_fraction=abs(float(points[point]) - target) if comparable else None,
                                               metric_available=int(comparable), unavailable_reason='' if comparable else 'no_target_or_different_guide_row'))
        for y in range(512):
            usable = bool(valid[y].any())
            tables['atlas_rows'].append(dict(run=cfg['run'], image_index=index, atlas_row=y,
                                             source_y=float((grid[y, 0, 1] + 1) * 511 / 2) if usable else None,
                                             source_left=float((grid[y, 0, 0] + 1) * 383 / 2) if usable else None,
                                             source_right=float((grid[y, -1, 0] + 1) * 383 / 2) if usable else None,
                                             valid_fraction=float(valid[y].mean())))
        packed.append(dict(image_id=image_id, image_index=index, masks=masks, points=points, guides=guides))
        newly_seen.append(image_id)
    return tables, packed, newly_seen


def summarize(records, cfg, stage, endpoint, epoch, update):
    assert records
    groups = {('cohort', 'all'): records}
    for field, level in (('_domain', 'domain'), ('_tyre', 'physical_tyre'), ('_source_group', 'source_group')):
        for group in sorted({r[field] for r in records if r[field]}):
            groups[(level, group)] = [r for r in records if r[field] == group]
    tables = {key: [] for key in ('metrics', 'confusion', 'calibration_bins')}
    cohort = None
    for (level, group), subset in groups.items():
        base = dict(run=cfg['run'], stage=stage, endpoint=endpoint, epoch=epoch, update=update, level=level, group=group)
        cm = np.zeros((3, 3), dtype=np.int64)
        for row in subset:
            cm[row['label'], row['prediction']] += 1
        recalls, f1s = [], []
        for cls in range(3):
            n = int(cm[cls].sum())
            positives = int(cm[:, cls].sum())
            recall = float(cm[cls, cls] / n) if n else None
            precision = float(cm[cls, cls] / positives) if positives else None
            f1 = float(2 * cm[cls, cls] / (n + positives)) if n + positives else None
            if n:
                recalls.append(recall)
                f1s.append(f1)
            for metric, value, denominator in (('recall', recall, n), ('precision', precision, positives), ('f1', f1, n + positives), ('target_count', n, n)):
                tables['metrics'].append(dict(base, class_index=cls, metric=metric, value=value, n=denominator,
                                              unavailable_reason='zero_denominator' if value is None else ''))
            for pred in range(3):
                tables['confusion'].append(dict(base, target_class=cls, predicted_class=pred, count=int(cm[cls, pred])))
        confidence = np.array([max(r[f'probability_{c}'] for c in range(3)) for r in subset])
        correct = np.array([r['correct'] for r in subset])
        ece = 0.
        for index in range(10):
            selected = np.clip((confidence * 10).astype(int), 0, 9) == index
            count = int(selected.sum())
            accuracy, mean = (float(correct[selected].mean()), float(confidence[selected].mean())) if count else (None, None)
            if count:
                ece += count / len(subset) * abs(accuracy - mean)
            tables['calibration_bins'].append(dict(base, bin=index, lower=index / 10, upper=(index + 1) / 10,
                                                   count=count, correct=int(correct[selected].sum()), confidence_sum=float(confidence[selected].sum()),
                                                   accuracy=accuracy, mean_confidence=mean))
        summary = dict(accuracy=float(np.trace(cm) / len(subset)), balanced_accuracy_observed=float(np.mean(recalls)),
                       macro_f1_observed=float(np.mean(f1s)), nll=float(np.mean([r['nll'] for r in subset])),
                       brier=float(np.mean([r['brier'] for r in subset])), ece_10_equal_width=ece,
                       unsupported_images=sum(r['empty_support'] for r in subset), images=len(subset),
                       evidence_max_residual=max(r['evidence_residual'] for r in subset))
        anchor_recalls = [float(np.mean([r['anchor_prediction'] == r['label'] for r in subset if r['label'] == c]))
                          for c in range(3) if any(r['label'] == c for r in subset)]
        summary['frozen_anchor_on_original_accuracy'] = float(np.mean([r['anchor_prediction'] == r['label'] for r in subset]))
        summary['frozen_anchor_on_original_balanced_accuracy_observed'] = float(np.mean(anchor_recalls))
        for metric, value in summary.items():
            tables['metrics'].append(dict(base, class_index=None, metric=metric, value=value, n=len(subset), unavailable_reason=''))
        if level == 'cohort':
            cohort = summary
    return tables, cohort


def splat(atlas, patch_evidence):
    """Deterministic CPU bilinear scatter, conserving signed logit mass."""
    valid = atlas['valid'][0].astype(bool)
    grid = atlas['grid'].astype(np.float64)
    evidence = patch_evidence.astype(np.float64)
    mass = np.zeros((3, 512, 256), dtype=np.float64)
    unsupported = np.zeros(3)
    for patch in range(32):
        y, x = (patch // 4) * 64, (patch % 4) * 64
        support = valid[y:y + 64, x:x + 64]
        count = int(support.sum())
        if count:
            mass[:, y:y + 64, x:x + 64] = evidence[patch, :, None, None] * support[None] / count
        else:
            unsupported += evidence[patch]
    yy = np.clip((grid[..., 1] + 1) * 511 / 2, 0, 511).reshape(-1)
    xx = np.clip((grid[..., 0] + 1) * 383 / 2, 0, 383).reshape(-1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    dy, dx = yy - y0, xx - x0
    projected = np.zeros((3, 512 * 384), dtype=np.float64)
    for y, x, weight in ((y0, x0, (1 - dy) * (1 - dx)), (y0, np.minimum(x0 + 1, 383), (1 - dy) * dx),
                         (np.minimum(y0 + 1, 511), x0, dy * (1 - dx)), (np.minimum(y0 + 1, 511), np.minimum(x0 + 1, 383), dy * dx)):
        for cls in range(3):
            np.add.at(projected[cls], y * 384 + x, mass[cls].reshape(-1) * weight)
    assert np.max(np.abs(projected.sum(1) + unsupported - evidence.sum(0))) < 1e-9
    return projected.reshape(3, 512, 384), unsupported
