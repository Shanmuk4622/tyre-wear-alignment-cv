"""Download only four pinned study checkpoints; compact for offline inference."""
import bootstrap
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from registry import MODELS, REPO, REVISION, S5, checkpoint
from bootstrap import ROOT


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(repo, filename, revision, destination, repo_type='dataset'):
    """Short Windows paths, HTTP resume, bounded retry and immutable revisions."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo, filename, revision=revision, repo_type=repo_type)
    token = os.environ.get('HF_TOKEN')
    meta = get_hf_file_metadata(url, token=token)
    if meta.commit_hash != revision:
        raise RuntimeError('Remote revision did not match the pinned commit')
    expected = meta.etag.strip('"')
    def valid(p):
        if not p.exists() or p.stat().st_size != meta.size:
            return False
        # LFS ETags are SHA256; ordinary Git blobs have a Git-object SHA1.
        if len(expected) == 64:
            return digest(p) == expected
        data = p.read_bytes()
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == expected
    if valid(destination):
        return destination
    partial = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(6):
        if partial.exists() and partial.stat().st_size >= meta.size:
            if valid(partial):
                os.replace(partial, destination)
                return destination
            partial.unlink()
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    delay = min(300, max(2 ** (attempt + 1), int(r.headers.get('Retry-After', '0'))))
                    print(f'Hub busy; retry in {delay}s', flush=True)
                    time.sleep(delay)
                    continue
                if r.status_code == 416 and valid(partial):
                    os.replace(partial, destination)
                    return destination
                r.raise_for_status()
                if r.status_code == 206 and not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected range response')
                with partial.open('ab' if r.status_code == 206 else 'wb') as f:
                    for block in r.iter_content(4 * 1024 * 1024):
                        f.write(block)
            if not valid(partial):
                partial.unlink(missing_ok=True)
                raise RuntimeError('Downloaded file failed size/hash verification')
            os.replace(partial, destination)
            return destination
        except (requests.RequestException, RuntimeError):
            if attempt == 5:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise RuntimeError('Download retry limit reached; rerun to resume')


def prepare(name):
    import torch
    spec = MODELS[name]
    target = checkpoint(name)
    provenance = target.with_suffix('.json')
    if target.exists() and provenance.exists():
        p = json.loads(provenance.read_text())
        if p.get('revision') == spec['revision'] and p.get('path') == spec['path'] and digest(target) == p['inference_sha256']:
            print(f'{name}: verified, ready offline', flush=True)
            return
    print(f'{name}: fetching final-epoch weights', flush=True)
    raw = download(REPO, spec['path'], spec['revision'], ROOT / '.cache' / f'{name}.pt')
    # Only the user's own fixed, hash-verified study artifacts are unpickled.
    state = torch.load(raw, map_location='cpu', weights_only=False)
    print(f'{name}: epoch={state.get("epoch")} keys={list(state)[:15]}', flush=True)
    if spec['task'] == 'classifier':
        cfg = state['config']
        if (state['arch'] != name or cfg['arch'] != name or cfg['fold'] != 1
                or cfg['seed'] != 1 or state['epoch'] != 60
                or state['classes'] != ['low_mileage_proxy', 'mid_mileage_proxy', 'high_mileage_proxy']):
            raise RuntimeError('Classifier architecture, endpoint, fold/seed or label order mismatch')
        compact = dict(model=state['model'], config=cfg, epoch=state['epoch'])
    else:
        job = state['job']
        if job['model'] != name or job['fold'] != 1 or job['seed'] != 1 or state['epoch'] != 60:
            raise RuntimeError('Unexpected S5 identity or incomplete training')
        if name == 'yolo26n_seg':
            native = state['native']
            model = native.get('ema') if native.get('ema') is not None else native['model']
            model.float().load_state_dict(state['ema_model'], strict=True)
            compact = dict(model=model.cpu().eval(), train_args=native.get('train_args', {}),
                           epoch=60, version='8.4.20')
        else:
            cfg_path = download('nvidia/mit-b0', 'config.json', '80983a413c30d36a39c20203974ae7807835e2b4',
                                ROOT / '.cache' / 'segformer_config.json', repo_type='model')
            compact = dict(model=state['model'], config=json.loads(cfg_path.read_text()), epoch=60)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    torch.save(compact, tmp)
    os.replace(tmp, target)
    p = dict(repo=REPO, **spec, fold=1, seed=1, epoch=state['epoch'],
             source_sha256=digest(raw), inference_sha256=digest(target),
             endpoint='final_epoch', local_bytes=target.stat().st_size)
    provenance.write_text(json.dumps(p, indent=2), encoding='utf-8')
    print(f'{name}: ready ({target.stat().st_size / 1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    for name in args.models:
        prepare(name)
