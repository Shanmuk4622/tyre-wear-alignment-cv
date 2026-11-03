"""Reuse the frozen safe Phase 2 overlay, never its trained weights."""
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import phase2_training_data as legacy
from common import digest, read, sha

HW = (512, 384)
BATCH = 8
EPOCHS = 60
SEEDS = (1, 2, 3)


def plan(root):
    result = legacy.overlay(root)
    human = {r['image_id']: r for r in read(Path(root)/'geometry/original_human_points.json')}
    weak = {r['image_id']: r for r in read(Path(root)/'geometry/new_point_proposals.json')}
    for row in result['rows']:
        record = human.get(row['image_id']) or weak.get(row['image_id'])
        row['guide_fractions'] = ([record['points'][i]['y']/(row['height']-1) for i in (0, 2, 4)] if record else [384/1535, 768/1535, 1151/1535])
    assert result['counts'] == {'train': 386, 'validation': 81, 'test': 103}
    assert len(set(result['groups']['train']) & set(result['groups']['test'])) == 0
    return result


def schedule(p, seed, epoch):
    rows = [r for r in p['rows'] if r['role'] == 'train']
    pools = {}
    for i, r in enumerate(rows):
        group = r['physical_tyre_id'] or r['source_video']
        pools.setdefault(r['class_index'], {}).setdefault(r['domain'], {}).setdefault(group, []).append(i)
    rs = np.random.default_rng(seed*1000003+epoch)
    result = []
    for _ in range(math.ceil(len(rows)/BATCH)*BATCH):
        cls = int(rs.choice(sorted(pools)))
        domain = str(rs.choice(sorted(pools[cls])))
        group = str(rs.choice(sorted(pools[cls][domain])))
        result.append(int(rs.choice(pools[cls][domain][group])))
    return rows, [result[i:i+BATCH] for i in range(0, len(result), BATCH)]


def batch(root, rows, device='cpu', seed=None, epoch=0, cursor=0, hw=HW):
    arrays = [legacy.sample(root, r, hw, None if seed is None else seed*10000019+epoch*100003+cursor*97+i, 'segformer') for i, r in enumerate(rows)]
    image = torch.tensor(np.stack([x[0] for x in arrays]), device=device)
    return dict(img=(image-image.new_tensor([.485, .456, .406])[None, :, None, None])/image.new_tensor([.229, .224, .225])[None, :, None, None],
                masks=torch.tensor(np.stack([x[1] for x in arrays]), device=device, dtype=torch.float32),
                points=torch.tensor(np.stack([x[2] for x in arrays]), device=device),
                labels=torch.tensor([r['class_index'] for r in rows], device=device),
                point_weight=image.new_tensor([r['point_weight'] for r in rows]),
                guides=image.new_tensor([r['guide_fractions'] for r in rows]))


def counterfactual(b):
    """Swap only labelled outside-tyre pixels during training; no test labels enter inference."""
    foreground = (b['masks'][:, :1] > .5) | (b['masks'][:, 1:2] > .5) | (b['masks'][:, 2:3] > .5)
    return torch.where(foreground, b['img'], b['img'].roll(1, 0))
