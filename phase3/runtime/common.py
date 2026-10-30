"""Small, explicit persistence primitives. No credentials in artifacts."""
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic(path, value, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as stream:
        if binary:
            torch.save(value, stream)
        else:
            stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)


def environment(device):
    return dict(python=sys.version, packages={k: importlib.metadata.version(k) for k in
                ('torch', 'numpy', 'Pillow', 'huggingface_hub', 'filelock')},
                device=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU')


def memory():
    """Report container limits when present; host RAM is only a fallback."""
    for base, current, maximum in [('/sys/fs/cgroup', 'memory.current', 'memory.max'),
                                   ('/sys/fs/cgroup/memory', 'memory.usage_in_bytes', 'memory.limit_in_bytes')]:
        try:
            used = int((Path(base)/current).read_text())
            limit = int((Path(base)/maximum).read_text())
            if 0 < limit < 2**60:
                return dict(source='cgroup', used=used, limit=limit, fraction=used/limit)
        except (OSError, ValueError):
            pass
    try:
        import psutil
        m = psutil.virtual_memory()
        return dict(source='host_fallback', used=m.used, limit=m.total, fraction=m.percent/100)
    except ImportError:
        return dict(source='unavailable', fraction=None)


def delta(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.0
    if isinstance(a, np.ndarray):
        return float(np.abs(a.astype(float)-b.astype(float)).max()) if a.size else 0.0
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([delta(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return max([delta(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)):
        return abs(a-b)
    assert a == b
    return 0.0


def validate_completed(state, folder=None):
    assert state['completed'] and state['epoch'] == 60 and state['cursor'] == 0
    assert state['updates'] == 2940 and len(state['history']) == 2940
    assert [r['update'] for r in state['history']] == list(range(1, 2941))
    assert [r['epoch'] for r in state['validation']] == list(range(1, 61))
    assert state['best'] is not None and state['best_epoch'] in range(1, 61)
    best = max(state['validation'], key=lambda r: r['result']['joint_validation_utility'])
    assert state['best_epoch'] == best['epoch'] and state['best_score'] == best['result']['joint_validation_utility']
    for endpoint in ('selected', 'fixed_final'):
        evaluation = state['evaluation'][endpoint]
        assert evaluation['images'] == 103 and evaluation['human_point_images'] == 25
        assert evaluation['observed_classes'] == [0, 2]
        assert len({r['image_id'] for r in evaluation['records']}) == 103
    assert set(state['artifacts']) == {'TEST_FINAL.npz', 'TEST_SELECTED.npz'}
    if folder is not None:
        for name, expected in state['artifacts'].items():
            assert sha(Path(folder)/name) == expected, 'Completed raw test artifact mismatch: '+name
