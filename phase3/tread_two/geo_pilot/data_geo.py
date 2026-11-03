"""Existing expanded Kaggle data; exact old overlay, no flips or new photos."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from common import HERE, digest, read, sha
from legacy_data import plan
from phase2_training_data import sample


def locate(value=''):
    if value:
        roots = [Path(value)]
    elif Path('/kaggle/input').is_dir():
        roots = [p.parent for p in Path('/kaggle/input').rglob('VERSION.json')
                 if read(p).get('dataset_title') == 'Tire Dataset Prepared phase2']
    else:
        roots = [HERE.parents[3] / 'phase2/data/phase2_dataset_v1']
    if len(roots) != 1:
        raise ValueError('Attach exactly one expanded shanmuk4622/tire-dataset-prepared-phase2 input.')
    root = roots[0].resolve()
    assert sha(root / 'SHA256SUMS.txt') == read(HERE / 'PLAN.json')['data']['checksum_file_sha256']
    return root


def dataset(root, verify=True):
    design = read(HERE / 'PLAN.json')
    overlay = plan(root)
    assert digest(overlay) == design['data']['overlay_sha256'], 'Historical overlay drift'
    if verify:
        for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected, name = line.split('  ', 1)
            path = (root / name).resolve()
            assert path.is_relative_to(root) and sha(path) == expected, name
    with (HERE / 'SOURCE_COMPONENTS.csv').open(encoding='utf-8', newline='') as stream:
        registry = {r['image_id']: r for r in csv.DictReader(stream)}
    for row in overlay['rows']:
        reference = registry[row['image_id']]
        assert row['image_sha256'] == reference['image_sha256']
        assert row['role'] == reference['historical_study_split']
        row['component'] = reference['leakage_component']
    train = sorted((r for r in overlay['rows'] if r['role'] == 'train'), key=lambda r: r['image_id'])
    assert Counter(r['point_kind'] for r in train) == {'human': 70, 'polygon_weak': 152, 'none': 164}
    assert sorted(Counter(r['component'] for r in train).values()) == [2, 3, 5, 60, 67, 249]
    return overlay, train


def photometric(rgb, factors):
    """Brightness, contrast about image spatial mean per channel, gamma; clamp each stage."""
    b, c, g = factors.unbind(-1)
    shape = (-1, 1, 1, 1)
    x = (rgb * b.reshape(shape)).clamp(0, 1)
    mean = x.mean((-2, -1), keepdim=True)
    x = ((x - mean) * c.reshape(shape) + mean).clamp(0, 1)
    return x.clamp_min(1e-8).pow(g.reshape(shape)).clamp(0, 1)


def load_batch(root, rows, device, seeds=None):
    arrays = [sample(root, r, (512, 384), seed=None, model='segformer') for r in rows]
    rgb = torch.from_numpy(np.stack([a[0] for a in arrays])).to(device)
    factors = np.ones((len(rows), 3), dtype=np.float32)
    if seeds is not None:
        factors = np.stack([np.random.default_rng(s).uniform(.9, 1.1, 3) for s in seeds]).astype('float32')
        rgb = photometric(rgb, torch.from_numpy(factors).to(device))
    return {'rgb': rgb, 'masks': torch.from_numpy(np.stack([a[1] for a in arrays])).float().to(device),
            'points': torch.from_numpy(np.stack([a[2] for a in arrays])).to(device),
            'point_weight': rgb.new_tensor([r['point_weight'] for r in rows]),
            'human': torch.tensor([r['point_kind'] == 'human' for r in rows], device=device),
            'guides': rgb.new_tensor([r['guide_fractions'] for r in rows]),
            'factors': factors, 'rows': rows}


def normalize(rgb):
    return (rgb - rgb.new_tensor([.485, .456, .406])[None, :, None, None]) / rgb.new_tensor([.229, .224, .225])[None, :, None, None]
