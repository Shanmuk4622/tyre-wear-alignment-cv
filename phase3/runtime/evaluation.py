"""Task-specific metrics, faithful logit accounting and explicit scope labels."""
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import atomic
import os
from data import HW, batch, counterfactual
from model import boxes


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.0


def summarise(records):
    assert records
    confusion = [[sum(r['label'] == y and r['prediction'] == p for r in records) for p in range(3)] for y in range(3)]
    observed = [c for c in range(3) if sum(confusion[c])]
    recalls = {str(c): confusion[c][c]/sum(confusion[c]) for c in observed}
    f1 = []
    for c in range(3):
        denominator = sum(confusion[c])+sum(row[c] for row in confusion)
        f1.append(2*confusion[c][c]/denominator if denominator else 0.0)
    human = [r['human_point_mae'] for r in records if r['human_point_mae'] is not None]
    dice = float(np.mean([r['dice'] for r in records]))
    balanced = float(np.mean(list(recalls.values())))
    point_error = float(np.mean(human)) if human else None
    # Frozen engineering utility, not a common task accuracy or a calibrated score.
    utility = .5*balanced + .35*dice + .15*max(0., 1-point_error/.05) if point_error is not None else None
    return dict(balanced_accuracy_observed=balanced, observed_classes=observed, recalls=recalls,
                macro_f1_three_classes_zero_for_missing=float(np.mean(f1)), confusion=confusion,
                mean_two_region_dice=dice, mean_two_region_iou=float(np.mean([r['iou'] for r in records])),
                mask_derived_box_iou=float(np.mean([r['box_iou'] for r in records])),
                human_point_mae_width_fraction=point_error, human_point_images=len(human),
                evidence_logit_max_residual=max(r['evidence_residual'] for r in records),
                joint_validation_utility=utility, images=len(records))


def evaluate(model, rows, root, device, stop=lambda: False, interventions=False, artifact_path=None):
    model.eval()
    records = []
    raw_outputs = []
    with torch.inference_mode():
        for row in rows:
            if stop():
                raise InterruptedError('Evaluation interrupted; last completed-update checkpoint retained')
            b = batch(root, [row], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(b['img'], b['guides'])
            mask = (out['masks'][0].repeat_interleave(4, -2).repeat_interleave(4, -1) >= .5)
            valid = b['masks'][0, 2] < .5
            target = b['masks'][0, :2] > .5
            dices, ious = [], []
            for c in range(2):
                pred, truth = mask[c] & valid, target[c] & valid
                intersection = int((pred & truth).sum())
                dices.append((2*intersection+1)/(int(pred.sum()+truth.sum())+1))
                ious.append((intersection+1)/(int((pred | truth).sum())+1))
            predicted_boxes, target_boxes = boxes(mask & valid), boxes(target & valid)
            residual = (out['contributions'].sum((-2, -1))+out['bias']-out['logits']).abs().max()
            rec = dict(image_id=row['image_id'], tyre=row['physical_tyre_id'], domain=row['domain'],
                       label=row['class_index'], prediction=int(out['probabilities'][0].argmax()),
                       probabilities=out['probabilities'][0].cpu().tolist(), logits=out['logits'][0].cpu().tolist(),
                       dice=dices, iou=ious, boxes=predicted_boxes, target_boxes=target_boxes,
                       box_iou=[box_iou(a, b) for a, b in zip(predicted_boxes, target_boxes)],
                       points=out['points'][0].cpu().tolist(), point_target=row['points'],
                       human_point_mae=float((out['points'][0]-b['points'][0]).abs().mean()) if row['point_kind'] == 'human' else None,
                       evidence_residual=float(residual),
                       concept_mass=(out['concepts']*out['gate']).sum((-2, -1))[0].cpu().tolist())
            flags = []
            small_mask = out['masks'][0, 1] >= .5
            for j, fraction in enumerate(row['guide_fractions']):
                line = small_mask[round(fraction*(small_mask.shape[0]-1))]
                xx = torch.where(line)[0]
                if not len(xx):
                    flags.append(f'guide{j}:empty_tread_row')
                elif int(xx.min()) == 0 or int(xx.max()) == len(line)-1:
                    flags.append(f'guide{j}:clipped_tread_row')
                else:
                    reference = out['points'].new_tensor([int(xx.min()), int(xx.max())])/(len(line)-1)
                    if float((out['points'][0, j*2:j*2+2]-reference).abs().max()) > .05:
                        flags.append(f'guide{j}:point_mask_disagreement_over_5pct_width')
            rec['review_flags'] = flags
            rec['flags_are'] = 'Uncalibrated review heuristics, not visibility/accuracy probabilities'
            if artifact_path:
                raw_outputs.append({k: out[k][0].cpu().numpy() for k in ('masks', 'points', 'contributions', 'concepts', 'gate', 'logits')})
            if interventions:
                # Truth is used to define diagnostic interventions only, never normal inference.
                foreground = (b['masks'][:, :1] > .5) | (b['masks'][:, 1:2] > .5) | (b['masks'][:, 2:3] > .5)
                replaced = torch.where(foreground, b['img'], torch.zeros_like(b['img']))
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    altered = model(replaced, b['guides'])
                rec['background_neutralized_probability_l1'] = float((out['probabilities']-altered['probabilities']).abs().sum())
                cls = rec['prediction']
                score = out['contributions'][0, cls]
                count = max(1, score.numel()//10)
                # Exact feature-evidence intervention: gate and features are held fixed.
                positive = score.flatten().topk(count).values.sum()
                rec['remove_top10_feature_evidence_logit_drop'] = float(positive)
                # A separate input intervention is not expected to equal that algebraic drop.
                selected = torch.zeros_like(score, dtype=torch.bool).flatten()
                selected[score.flatten().topk(count).indices] = True
                selected = selected.reshape(score.shape).repeat_interleave(4, -2).repeat_interleave(4, -1)
                occluded = torch.where(selected[None, None], torch.zeros_like(b['img']), b['img'])
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    other = model(occluded, b['guides'])
                rec['occlude_top10_image_probability_change'] = float(out['probabilities'][0, cls]-other['probabilities'][0, cls])
            records.append(rec)
    result = summarise(records)
    result['records'] = records
    result['per_tyre'] = {g: summarise([r for r in records if r['tyre'] == g])
                          for g in sorted({r['tyre'] for r in records})}
    result['scope'] = 'Frozen Phase 2 low/high old-tyre holdout only. No held-out mid, unseen-video, depth or safety claim.'
    if artifact_path:
        path = Path(artifact_path)
        with open(path.with_suffix('.tmp'), 'wb') as stream:
            np.savez_compressed(stream, image_ids=np.array([r['image_id'] for r in records]),
                                bias=model.class_bias.detach().cpu().numpy(),
                                **{k: np.stack([r[k] for r in raw_outputs]) for k in raw_outputs[0]})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path.with_suffix('.tmp'), path)
    return result


def gallery(model, plan, root, device, folder):
    """First fixed example/class; pictures are illustrations, not a new benchmark."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = [sorted([r for r in plan['rows'] if r['class_index'] == c], key=lambda r: r['image_id'])[0] for c in range(3)]
    metadata = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            b = batch(root, [row], device)
            out = model(b['img'], b['guides'])
            with Image.open(Path(root)/row['image_path']) as im:
                original = im.convert('RGB').resize((HW[1], HW[0]), Image.Resampling.BILINEAR)
            mask = out['masks'][0].cpu().numpy().repeat(4, 1).repeat(4, 2)
            colors = np.zeros((HW[0], HW[1], 3), np.uint8)
            colors[mask[0] > .5] = (245, 180, 45)
            colors[mask[1] > .5] = (35, 205, 150)
            region = Image.blend(original, Image.fromarray(colors), .35)
            draw = ImageDraw.Draw(region)
            points = out['points'][0].reshape(3, 2).cpu().numpy()
            for (left, right), y in zip(points, row['guide_fractions']):
                y = y*(HW[0]-1)
                draw.line((left*(HW[1]-1), y, right*(HW[1]-1), y), fill='magenta', width=2)
            cls = int(out['logits'][0].argmax())
            score = out['contributions'][0, cls].cpu().numpy().repeat(4, 0).repeat(4, 1)
            scale = max(float(np.abs(score).max()), 1e-8)
            heat = np.zeros_like(colors)
            heat[..., 0] = (np.maximum(score, 0)/scale*255).astype(np.uint8)
            heat[..., 2] = (np.maximum(-score, 0)/scale*255).astype(np.uint8)
            evidence = Image.blend(original, Image.fromarray(heat), .5)
            canvas = Image.new('RGB', (HW[1]*3, HW[0]+70), 'white')
            for i, panel in enumerate((original, region, evidence)):
                canvas.paste(panel, (i*HW[1], 0))
            ImageDraw.Draw(canvas).text((8, HW[0]+5), f"{row['role']} | label {row['class_index']} -> prediction {cls} | probabilities {np.round(out['probabilities'][0].cpu().numpy(), 3)}\noriginal | tyre/tread + points | signed predicted-class logit evidence (red positive, blue negative)", fill='black')
            name = f'example_class{row["class_index"]}.png'
            canvas.save(folder/name)
            np.savez_compressed(folder/f'example_class{row["class_index"]}.npz', masks=mask,
                                points=points, contributions=out['contributions'][0].cpu().numpy(),
                                concepts=out['concepts'][0].cpu().numpy(), gate=out['gate'][0].cpu().numpy(),
                                logits=out['logits'][0].cpu().numpy(), bias=out['bias'].cpu().numpy())
            metadata.append(dict(image_id=row['image_id'], cohort=row['role'], figure=name))
    atomic(folder/'GALLERY.json', dict(examples=metadata, scope='Fixed illustrative examples; mid is training-cohort data.'))
