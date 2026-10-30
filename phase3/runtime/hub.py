"""Single notebook publisher: CAS lease fencing, verified ZIP snapshots, bounded retries."""
import email.utils
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
from filelock import FileLock

from common import atomic, canonical, digest, read, sha

REPO = 'Shanmuk4622/tyre-wear-study'
ALLOWED = {'state.pt', 'SMOKE.json', 'IDENTITY.json', 'ERROR.json', 'EVALUATION.json', 'console.log',
           'TEST_FINAL.npz', 'TEST_SELECTED.npz'}


def retry_delay(response, attempt):
    headers = getattr(response, 'headers', {}) or {}
    hint = headers.get('Retry-After', headers.get('retry-after', '0'))
    try:
        delay = float(hint)
    except (ValueError, TypeError):
        try:
            delay = max(0., (email.utils.parsedate_to_datetime(hint)-datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError):
            delay = 0.
    # Current Hub headers use t=seconds-until-reset. Honour both kinds of hint.
    for name in ('RateLimit', 'ratelimit', 'RateLimit-Reset', 'ratelimit-reset'):
        value = str(headers.get(name, ''))
        match = re.search(r'\bt\s*=\s*(\d+)', value)
        if match:
            delay = max(delay, float(match.group(1)))
        elif name.lower() == 'ratelimit-reset' and value.isdigit():
            delay = max(delay, float(value))
    return max(delay, min(300., 2.**attempt*5)+.3*(attempt+1))


def extract_verified(archive, folder, expected_protocol, expected_seed):
    """Validate everything before replacing any recoverable local checkpoint."""
    folder = Path(folder)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP member'
        assert set(names) <= ALLOWED | {'MANIFEST.json'}, 'Unexpected snapshot member'
        assert all(PurePosixPath(n).name == n for n in names), 'Unsafe ZIP path'
        assert sum(v.file_size for v in z.infolist()) < 2*1024**3, 'Snapshot exceeds decompressed budget'
        manifest = json.loads(z.read('MANIFEST.json'))
        assert manifest['protocol'] == expected_protocol and manifest['seed'] == expected_seed
        assert set(names) == set(manifest['files']) | {'MANIFEST.json'}
        for name, meta in manifest['files'].items():
            payload = z.read(name)
            assert len(payload) == meta['bytes'] and digest_bytes(payload) == meta['sha256'], 'Snapshot member checksum mismatch'
        temporary = folder/'restore_pending'
        temporary.mkdir(parents=True, exist_ok=True)
        for name in manifest['files']:
            (temporary/name).write_bytes(z.read(name))
        if 'state.pt' in manifest['files']:
            saved = torch.load(temporary/'state.pt', map_location='cpu', weights_only=False)
            assert saved['protocol'] == expected_protocol and saved['seed'] == expected_seed
            assert saved['updates'] == manifest['updates'] and saved['completed'] == manifest['completed']
        else:
            assert manifest['updates'] == 0 and not manifest['completed'], 'Missing required measured state'
        for name in manifest['files']:
            os.replace(temporary/name, folder/name)
        temporary.rmdir()
    return manifest


def digest_bytes(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


class Publisher:
    def __init__(self, work, contract, tag, take_over=False):
        from huggingface_hub import HfApi
        self.work = Path(work)
        self.contract = contract
        self.protocol = digest(contract)
        self.prefix = 'phase3/tread-one-r1/'+self.protocol
        self.owner = uuid.uuid4().hex
        self.api = HfApi(token=os.environ['HF_TOKEN'])
        self.token = os.environ['HF_TOKEN']
        self.calls = deque(read(self.work/'request_budget.json') if (self.work/'request_budget.json').exists() else [])
        self.last = read(self.work/'LAST_REMOTE.json') if (self.work/'LAST_REMOTE.json').exists() else dict(files={})
        self.index = {}
        self.tag = tag
        self.take_over = take_over

    def request(self, function):
        for attempt in range(6):
            while self.calls and self.calls[0] < time.time()-3600:
                self.calls.popleft()
            while len(self.calls) >= 48:
                time.sleep(min(2., max(.05, self.calls[0]+3600-time.time())))
                while self.calls and self.calls[0] < time.time()-3600:
                    self.calls.popleft()
            self.calls.append(time.time())
            atomic(self.work/'request_budget.json', list(self.calls))
            try:
                return function()
            except Exception as exc:
                response = getattr(exc, 'response', None)
                status = getattr(response, 'status_code', None)
                if status not in (None, 429, 500, 502, 503, 504) or attempt == 5:
                    raise
                if status is None and isinstance(exc, (ValueError, AssertionError, TypeError, FileNotFoundError)):
                    raise
                delay = retry_delay(response, attempt)
                print(f'HF transient status={status}; retaining local state; retry after {delay:.1f}s', flush=True)
                end = time.monotonic()+delay
                while time.monotonic() < end:
                    time.sleep(min(2., end-time.monotonic()))

    def revision(self):
        return self.request(lambda: self.api.repo_info(REPO, repo_type='dataset').sha)

    def fetch(self, path, revision, optional=False):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return Path(self.request(lambda: hf_hub_download(REPO, path, repo_type='dataset', revision=revision,
                         token=self.token, cache_dir=str(self.work/'read_cache'))))
        except EntryNotFoundError:
            if optional:
                return None
            raise

    def claim(self):
        from huggingface_hub import CommitOperationAdd
        anchor = 'phase3/tread-one-r1/anchors/'+self.tag+'.json'
        for attempt in range(8):
            rev = self.revision()
            anchored = self.fetch(anchor, rev, True)
            if anchored:
                assert read(anchored)['protocol'] == self.protocol, 'RUN_TAG is bound to a different runtime/protocol. Do not silently start over: restore original sources/environment or review compatibility.'
            lease = self.fetch(self.prefix+'/LEASE.json', rev, True)
            old = read(lease) if lease else {}
            if old.get('owner') not in (None, self.owner) and old.get('expires', 0) > time.time() and not self.take_over:
                raise RuntimeError('Study already leased. Stop the old notebook and wait for its flush before setting TAKE_OVER=True once.')
            operations = [CommitOperationAdd(path_in_repo=anchor, path_or_fileobj=canonical(dict(protocol=self.protocol))),
                          CommitOperationAdd(path_in_repo=self.prefix+'/LEASE.json', path_or_fileobj=canonical(dict(owner=self.owner, expires=time.time()+7200)))]
            try:
                commit = self.request(lambda: self.api.create_commit(REPO, repo_type='dataset', operations=operations,
                                      parent_commit=rev, commit_message='TREAD-ONE claim'))
                index = self.fetch(self.prefix+'/INDEX.json', commit.oid, True)
                self.index = read(index)['runs'] if index else {}
                for meta in self.index.values():
                    meta.setdefault('revision', commit.oid)
                return
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) not in (409, 412) or attempt == 7:
                    raise

    def restore(self, seed, folder):
        key = 'seed'+str(seed)
        meta = self.index.get(key)
        if not meta:
            return
        folder = Path(folder)
        if (folder/'state.pt').exists():
            state = torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
            assert state['protocol'] == self.protocol and state['seed'] == seed
            if state['updates'] >= meta['updates'] and (state['completed'] or not meta['completed']):
                return
        archive = self.fetch(meta['path'], meta['revision'])
        assert sha(archive) == meta['sha256'], 'Remote snapshot byte checksum mismatch'
        extract_verified(archive, folder, self.protocol, seed)

    def publish(self, runroot, extra=None, release=False):
        from huggingface_hub import CommitOperationAdd
        stage = self.work/'immutable_upload'
        stage.mkdir(exist_ok=True)
        files = {}
        updates = dict(self.index)
        for seed in (1, 2, 3):
            folder = Path(runroot)/('seed'+str(seed))
            if not any((folder/name).exists() for name in ALLOWED):
                continue
            archive = stage/f'seed{seed}.zip'
            with FileLock(str(folder/'checkpoint.lock')):
                state = (torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
                         if (folder/'state.pt').exists() else dict(updates=0, epoch=0, cursor=0, completed=False))
                manifest = dict(protocol=self.protocol, seed=seed, updates=state['updates'], epoch=state['epoch'],
                                cursor=state['cursor'], completed=state['completed'], has_state=(folder/'state.pt').exists(), files={})
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
                    for name in sorted(ALLOWED):
                        p = folder/name
                        if p.exists():
                            manifest['files'][name] = dict(sha256=sha(p), bytes=p.stat().st_size)
                            z.writestr(zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)), p.read_bytes())
                    z.writestr(zipfile.ZipInfo('MANIFEST.json', (2020, 1, 1, 0, 0, 0)), canonical(manifest))
            remote = self.prefix+f'/runs/seed{seed}/latest.zip'
            payload_hash = sha(archive)
            if self.last['files'].get(remote) != payload_hash:
                files[remote] = archive
                updates['seed'+str(seed)] = dict(path=remote, sha256=payload_hash, updates=manifest['updates'],
                                                epoch=manifest['epoch'], cursor=manifest['cursor'], completed=manifest['completed'])
            del state
        for remote_name, path in (extra or {}).items():
            full = self.prefix+'/'+remote_name
            if self.last['files'].get(full) != sha(path):
                # Caller passes immutable reports or source files, never live checkpoints.
                target = stage/(digest(full)[:12]+'_'+Path(path).name)
                shutil.copy2(path, target)
                files[full] = target
        sizes = sum(p.stat().st_size for p in stage.iterdir() if p.is_file())
        if sizes > 4*1024**3 or shutil.disk_usage(stage).free < 1024**3:
            raise RuntimeError('Upload staging storage guard; live checkpoints retained')
        # An upload revision cannot name itself inside INDEX.json. Each changed bundle
        # is resolved by the containing index commit on restore; unchanged ones retain theirs.
        index = dict(protocol=self.protocol, runs=updates, time=time.time())
        atomic(stage/'INDEX.json', index)
        files[self.prefix+'/INDEX.json'] = stage/'INDEX.json'
        for attempt in range(8):
            rev = self.revision()
            lease = self.fetch(self.prefix+'/LEASE.json', rev)
            if read(lease).get('owner') != self.owner:
                raise RuntimeError('Lease fencing rejected this stale writer; all local state retained')
            operations = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in files.items()]
            operations.append(CommitOperationAdd(path_in_repo=self.prefix+'/LEASE.json', path_or_fileobj=canonical(dict(
                owner=None if release else self.owner, expires=0 if release else time.time()+7200))))
            try:
                commit = self.request(lambda: self.api.create_commit(REPO, repo_type='dataset', parent_commit=rev,
                                      operations=operations, commit_message='TREAD-ONE progress'+(' and release' if release else '')))
                downloaded = self.fetch(self.prefix+'/INDEX.json', commit.oid)
                assert sha(downloaded) == sha(stage/'INDEX.json'), 'Committed index mismatch'
                binary = {k: p for k, p in files.items() if p.suffix in ('.zip', '.pt')}
                if binary:
                    infos = self.request(lambda: self.api.get_paths_info(REPO, list(binary), repo_type='dataset', revision=commit.oid))
                    assert {v.path for v in infos} == set(binary)
                    for info in infos:
                        if info.lfs:
                            assert info.lfs.sha256 == sha(binary[info.path]), 'Remote binary SHA256 mismatch'
                        else:
                            assert sha(self.fetch(info.path, commit.oid)) == sha(binary[info.path])
                for key, meta in updates.items():
                    if 'revision' not in meta:
                        meta['revision'] = commit.oid
                self.index = updates
                self.last['files'].update({k: sha(p) for k, p in files.items()})
                self.last.update(revision=commit.oid, time=time.time(), runs=updates)
                atomic(self.work/'LAST_REMOTE.json', self.last)
                print('HF snapshot verified:', commit.oid, {k: v['updates'] for k, v in updates.items()}, flush=True)
                for path in stage.iterdir():
                    if path.is_file():
                        path.unlink()
                return commit.oid
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) not in (409, 412) or attempt == 7:
                    raise
