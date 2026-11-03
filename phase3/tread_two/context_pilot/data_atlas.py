"""Old roles; all386 queries; full source SHA; sqrt-size group sampling."""
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from common import HERE, digest, read, sha
from legacy_data import plan

DATA_CHECKSUM = '06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
OVERLAY_SHA = 'f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039'


def locate(value=''):
    if value:
        roots = [Path(value)]
    elif Path('/kaggle/input').exists():
        roots = [p.parent for p in Path('/kaggle/input').rglob('VERSION.json')
                 if read(p).get('dataset_title') == 'Tire Dataset Prepared phase2']
    else:
        roots = [HERE.parents[2] / 'phase2/data/phase2_dataset_v1']
    assert len(roots) == 1, 'Attach exactly one expanded shanmuk4622/tire-dataset-prepared-phase2 input'
    root = roots[0].resolve()
    assert sha(root / 'SHA256SUMS.txt') == DATA_CHECKSUM, 'Wrong prepared dataset release'
    return root


def bank_cases():
    with (HERE / 'assets/prototype_bank.csv').open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load_bank():
    rgb = np.load(HERE / 'assets/bank_rgb.npy', allow_pickle=False)
    labels = np.load(HERE / 'assets/bank_labels.npy', allow_pickle=False)
    assert rgb.dtype == np.float32 and rgb.shape == (18, 3, 64, 64)
    return torch.from_numpy(rgb), torch.from_numpy(labels)


def dataset(root, verify=False):
    result = plan(root)
    assert digest(result) == OVERLAY_SHA, 'Historical role/point overlay drift'
    if verify:
        for line in (Path(root) / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected, name = line.split('  ', 1)
            path = (Path(root) / name).resolve()
            assert path.is_relative_to(Path(root).resolve()) and sha(path) == expected, name
    cases = bank_cases()
    bank_ids = {case['source_image_id'] for case in cases}
    lookup = {row['image_id']: row for row in result['rows']}
    for case in cases:
        row = lookup[case['source_image_id']]
        assert row['role'] == 'train' and row['domain'] == 'original_photo'
        assert row['image_sha256'] == case['source_image_sha256']
        assert row['physical_tyre_id'] == case['source_tyre_id']
    queries = sorted([r for r in result['rows'] if r['role'] == 'train' ], key=lambda r: r['image_id'])
    assert len(queries) == 386
    return result, queries


def schedule(rows, seed, epoch, batches=49, batch=8):
    pools = {}
    for index, row in enumerate(rows):
        group = row['physical_tyre_id'] or row['source_video']
        pools.setdefault(row['class_index'], {}).setdefault(row['domain'], {}).setdefault(group, []).append(index)
    generator = np.random.default_rng(seed * 1000003 + epoch)
    draws = []
    for occurrence in range(batches * batch):
        cls = int(generator.choice(sorted(pools)))
        domain = str(generator.choice(sorted(pools[cls])))
        groups = sorted(pools[cls][domain])
        weights = np.sqrt([len(pools[cls][domain][g]) for g in groups]); weights /= weights.sum()
        group = str(generator.choice(groups, p=weights))
        index = int(generator.choice(pools[cls][domain][group]))
        draws.append(dict(index=index, occurrence=occurrence, class_draw=cls, domain_draw=domain,
                          source_group=group, augmentation_seed=seed * 10000019 + epoch * 100003 + occurrence,
                          probability=1 / len(pools) / len(pools[cls]) * float(weights[groups.index(group)]) / len(pools[cls][domain][group])))
    return [draws[start:start + batch] for start in range(0, len(draws), batch)]


def input_rgb(root, row):
    path = Path(root) / row['image_path']
    with Image.open(path) as image:
        value = np.asarray(image.convert('RGB').resize((384, 512), Image.Resampling.BILINEAR), dtype=np.float32)
    return value.transpose(2, 0, 1).copy() / 255.


def batch(root, rows, device, draws=None):
    images = torch.tensor(np.stack([input_rgb(root, row) for row in rows]), device=device)
    labels = torch.tensor([row['class_index'] for row in rows], device=device)
    factors = None
    if draws is not None:
        factors = torch.tensor(np.stack([np.random.default_rng(draw['augmentation_seed']).uniform(.8, 1.2, (2,3))
                                         for draw in draws]), dtype=torch.float32, device=device)
    return images, labels, factors


def truth_masks(root, row):
    values = []
    for field in ('tyre_mask', 'tread_mask', 'ignore_mask'):
        with Image.open(Path(root) / row[field]) as image:
            values.append(np.asarray(image.resize((384, 512), Image.Resampling.NEAREST)) > 0)
    return np.stack(values)


def source_hashes(rows,device):
    return torch.tensor([list(bytes.fromhex(r['image_sha256'])) for r in rows],dtype=torch.uint8,device=device)
