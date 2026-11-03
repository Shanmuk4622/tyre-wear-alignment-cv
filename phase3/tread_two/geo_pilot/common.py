"""Strict local persistence, deterministic state and portable run identities."""
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def sync_directory(path):
    if os.name != 'nt':
        fd = os.open(Path(path), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def native(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        if path.suffix == '.npy':
            np.save(stream, value, allow_pickle=False)
        else:
            assert path.suffix == '.pt'
            torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def csv_write(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def file_meta(path):
    return dict(sha256=sha(path), bytes=Path(path).stat().st_size)


def inside(root, relative):
    root = Path(root).resolve()
    assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
    path = (root / relative).resolve()
    assert path.is_relative_to(root) and path != root
    return path


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu(item) for item in value)
    return value


def seed_all(seed):
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


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu'])
    if value['cuda']:
        assert len(value['cuda']) == torch.cuda.device_count(), 'GPU topology differs from saved state'
        torch.cuda.set_rng_state_all(value['cuda'])


def environment():
    import importlib.metadata as metadata
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                pillow=metadata.version('Pillow'), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def capacity(root, reserve=2 * 1024**3):
    root = Path(root)
    scope = Path('/kaggle/working') if Path('/kaggle/working').exists() else root
    logical = sum(path.stat().st_size for path in scope.rglob('*') if path.is_file())
    free = shutil.disk_usage(root).free
    result = dict(scope=str(scope), logical_bytes=logical, free_bytes=free, budget_bytes=16_000_000_000)
    assert logical < 16_000_000_000 and free > reserve, 'Storage guard: preserve state, flush and inspect free space'
    return result


def event(path, kind, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(canonical(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), kind=kind, **values)) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
