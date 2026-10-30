"""Bounded, revision-pinned read of Phase 3 HF evidence. Never uploads."""
import email.utils
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda' or Path('/kaggle/input').exists()
from huggingface_hub import HfApi, get_token
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
REPO = 'Shanmuk4622/tyre-wear-study'
REVISION = 'a0296f18e02b12d28f7e51377783d68538032209'
PREFIX = 'phase3/tread-one-r1/371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c'
TOKEN = get_token()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            if status not in (None, 429, 500, 502, 503, 504) or attempt == 5:
                raise
            if status is None and not isinstance(exc, requests.exceptions.RequestException):
                raise
            headers = getattr(response, 'headers', {}) or {}
            try:
                delay = float(headers.get('Retry-After', '0'))
            except ValueError:
                delay = 0
            match = re.search(r'\bt=(\d+)', headers.get('RateLimit', ''))
            delay = max(delay, float(match.group(1)) if match else 0, min(120, 5*2**attempt))
            print('Read retry', type(exc).__name__, 'status', status, 'delay', delay, flush=True)
            time.sleep(delay)


def fetch(relative):
    target = OUT/'hf'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    # Explicit short local paths avoid Windows long-path cache failures.
    def download():
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PREFIX}/{relative}'
        headers = {'Authorization': 'Bearer '+TOKEN} if TOKEN else {}
        with requests.get(url, headers=headers, timeout=(20, 120), stream=True) as response:
            response.raise_for_status()
            with open(target.with_suffix(target.suffix+'.part'), 'wb') as stream:
                total = 0
                reported = 0
                for block in response.iter_content(4*1024**2):
                    stream.write(block)
                    total += len(block)
                    if total-reported >= 32*1024**2:
                        print(relative, round(total/1024**2), 'MiB downloaded', flush=True)
                        reported = total
        target.with_suffix(target.suffix+'.part').replace(target)
    retry(download)
    return target


def main():
    # Fresh sessions have no audit directory until it is explicitly created.
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=TOKEN or False)
    if (OUT/'remote_tree.json').exists():
        tree = json.loads((OUT/'remote_tree.json').read_text())
        assert tree['revision'] == REVISION and tree['prefix'] == PREFIX
        metadata = tree['files']
    else:
        entries = retry(lambda: list(api.list_repo_tree(REPO, path_in_repo=PREFIX, revision=REVISION,
                         repo_type='dataset', recursive=True)))
        metadata = []
        for item in entries:
            if not hasattr(item, 'size'):
                continue
            lfs = getattr(item, 'lfs', None)
            metadata.append(dict(path=item.path[len(PREFIX)+1:], bytes=item.size,
                            lfs_sha256=getattr(lfs, 'sha256', None), blob_id=getattr(item, 'blob_id', None)))
        (OUT/'remote_tree.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION, prefix=PREFIX, files=metadata), indent=2)+'\n')
    names = ['INDEX.json', 'LEASE.json', 'CONTRACT.json', 'OVERLAY.json', 'ENVIRONMENT.json', 'DATA_PREFLIGHT.json',
             'report/REPORT.json', 'report/EXPORT.json']+[f'report/EVALUATION_seed{s}.json' for s in (1, 2, 3)]
    for name in names:
        path = fetch(name)
        print('Fetched', name, path.stat().st_size, 'bytes', flush=True)
    if '--bundles' in sys.argv:
        bundles = [v for v in metadata if v['path'].endswith('.zip')]
        assert sum(v['bytes'] for v in bundles) < 500*1024**2, 'Download scope exceeds this audit budget'
        for item in bundles:
            path = fetch(item['path'])
            assert path.stat().st_size == item['bytes'] and sha(path) == item['lfs_sha256'], item['path']
            print('SHA256 verified bundle', item['path'], flush=True)
    report = json.loads((OUT/'hf/report/REPORT.json').read_text())
    print('REPORT', json.dumps(dict(status=report['status'], export=report['export_selection'], summary=report['summary'])), flush=True)
    print('BUNDLES', json.dumps([v for v in metadata if v['path'].endswith('.zip')]), flush=True)
    print('AUTHENTICATED_READS', bool(TOKEN), flush=True)
    records = []
    for item in metadata:
        path = OUT/'hf'/item['path']
        if path.exists():
            records.append(dict(**item, downloaded_sha256=sha(path)))
    (OUT/'fetched_metadata.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
