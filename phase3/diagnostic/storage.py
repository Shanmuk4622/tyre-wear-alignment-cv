"""TREAD-ONE diagnostic persistence: atomic units, readable snapshots and HF CAS.

All metrics are direct CSV. Local complete-unit manifests provide transaction
boundaries. Remote CSV plus cursor JSON can reconstruct those units on restart.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import requests
from huggingface_hub import HfApi, CommitOperationAdd


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name+'.tmp')
    with open(part, 'wb') as stream:
        stream.write(canonical(obj)); stream.flush(); os.fsync(stream.fileno())
    os.replace(part, path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name+'.tmp')
    with open(part, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(part, path)


class Ledger:
    def __init__(self, folder, contract, schemas):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.contract, self.schemas = contract, schemas
        p = self.folder/'CONTRACT.json'
        if p.exists():
            assert read(p) == contract, 'Diagnostic contract mismatch; no silent mixed-run resume'
        else:
            atomic(p, contract)

    def units(self):
        result = {}
        for p in sorted((self.folder/'units').glob('*/UNIT.json')):
            rec = read(p)
            assert rec['contract'] == digest(self.contract)
            for name, meta in rec['files'].items():
                path = p.parent/name
                assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'], str(path)
            result[rec['unit']] = rec
        return result

    def commit(self, unit, job, rows, gallery=None):
        destination = self.folder/'units'/unit
        destination.mkdir(parents=True, exist_ok=True)
        assert not (destination/'UNIT.json').exists(), 'Completed diagnostic unit is immutable'
        files = {}
        for table, fields in self.schemas.items():
            path = destination/(table+'.csv')
            records = [dict(unit=unit, **r) for r in rows.get(table, [])]
            write_csv(path, records, fields)
            files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size, rows=len(records))
        if gallery:
            # Direct NumPy files are native arrays; no external ZIP/NPZ archive.
            import numpy as np
            for image_id, record in gallery.items():
                path = destination/(hashlib.sha256(image_id.encode()).hexdigest()[:16]+'.npy')
                with open(path, 'wb') as stream:
                    np.save(stream, record['array'], allow_pickle=False)
                    stream.flush(); os.fsync(stream.fileno())
                files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
            atomic(destination/'GALLERY.json', {k: dict({n: v for n, v in r.items() if n != 'array'},array_file=hashlib.sha256(k.encode()).hexdigest()[:16]+'.npy') for k, r in gallery.items()})
            files['GALLERY.json'] = dict(sha256=sha(destination/'GALLERY.json'), bytes=(destination/'GALLERY.json').stat().st_size)
        atomic(destination/'UNIT.json', dict(unit=unit, job=job, contract=digest(self.contract), files=files))

    def snapshot(self, report_builder, status):
        units = self.units()
        # Every snapshot is immutable after assembly; upload reads only this generation.
        generation = f'{len(units):04d}-{status}-{uuid.uuid4().hex[:8]}'
        folder = self.folder/'snapshots'/generation
        folder.mkdir(parents=True)
        atomic(folder/'CONTRACT.json', self.contract)
        schema = {}
        for table, fields in self.schemas.items():
            def records():
                for name in units:
                    with open(self.folder/'units'/name/(table+'.csv'), encoding='utf-8', newline='') as stream:
                        yield from csv.DictReader(stream)
            path = folder/'csv'/(table+'.csv')
            write_csv(path, records(), fields)
            with open(path, encoding='utf-8', newline='') as stream:
                count = sum(1 for _ in csv.DictReader(stream))
            schema[table] = dict(columns=fields, rows=count, sha256=sha(path))
        atomic(folder/'SCHEMA.json', dict(tables=schema, missing='Blank means unavailable/not applicable; zero is an observed zero.',
               numeric='FP32 diagnostic inference; CSV decimal scalars retain Python float precision.',
               units='Probabilities/Dice/IoU and normalized point errors are fractions; durations seconds, memory bytes.',
               provenance='All measurements are post-hoc. Nominal training-set inference is not historical augmented-batch training accuracy.'))
        gallery = []
        arrays = []
        import numpy as np
        for unit in units:
            source = self.folder/'units'/unit
            if not (source/'GALLERY.json').exists():
                continue
            for image_id, meta in read(source/'GALLERY.json').items():
                arrays.append(np.load(source/meta['array_file'], allow_pickle=False))
                gallery.append(dict(unit=unit, image_id=image_id, **meta))
        if arrays:
            (folder/'gallery').mkdir()
            with open(folder/'gallery/arrays.npy', 'wb') as stream:
                np.save(stream, np.stack(arrays), allow_pickle=False)
            atomic(folder/'gallery/METADATA.json', dict(images=gallery, axes=['image','input_variant','channel','height','width'],
                   input_variants=['normal','background_neutralized','top10_image_occluded'],
                   channels=['tyre_mask','tread_mask']+[f'latent_{i}' for i in range(12)]+[f'class_evidence_{i}' for i in range(3)]+['gate','oracle_tread_gate'],
                   grid_hw=[128,96], dtype='float32', scope='Fixed first image per training/validation tyre-or-clip and checkpoint; illustrative diagnostic cohort.'))
        atomic(folder/'CURSOR.json', dict(status=status, contract=digest(self.contract), units={k: {'job': v['job']} for k,v in units.items()}))
        if (self.folder/'EVENTS.jsonl').exists(): shutil.copy2(self.folder/'EVENTS.jsonl',folder/'EVENTS.jsonl')
        if (self.folder/'PUBLICATION.json').exists(): shutil.copy2(self.folder/'PUBLICATION.json',folder/'PREVIOUS_PUBLICATION.json')
        report_builder(folder)
        files = {str(p.relative_to(folder)).replace('\\','/'): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in sorted(folder.rglob('*')) if p.is_file()}
        assert not any(name.endswith(('.zip','.npz')) for name in files)
        atomic(folder/'MANIFEST.json', dict(contract=digest(self.contract), files=files))
        atomic(self.folder/'LATEST.json', dict(snapshot=str(folder.relative_to(self.folder)).replace('\\','/'), status=status))
        return folder

    def restore(self, folder):
        """Rebuild local complete units from a verified direct-CSV remote snapshot."""
        folder = Path(folder)
        assert read(folder/'CONTRACT.json') == self.contract
        cursor = read(folder/'CURSOR.json')
        grouped = {u: {t: [] for t in self.schemas} for u in cursor['units']}
        for table in self.schemas:
            with open(folder/'csv'/(table+'.csv'), encoding='utf-8', newline='') as stream:
                reader=csv.DictReader(stream)
                assert reader.fieldnames == self.schemas[table], 'Remote CSV schema mismatch'
                for row in reader:
                    unit = row.pop('unit')
                    grouped[unit][table].append(row)
        galleries = {}
        if (folder/'gallery/METADATA.json').exists():
            import numpy as np
            info = read(folder/'gallery/METADATA.json')['images']
            arrays = np.load(folder/'gallery/arrays.npy', mmap_mode='r', allow_pickle=False)
            for index, rec in enumerate(info):
                meta = {k:v for k,v in rec.items() if k not in ('unit','image_id')}
                galleries.setdefault(rec['unit'], {})[rec['image_id']] = dict(meta, array=arrays[index])
        existing = self.units()
        for unit, records in grouped.items():
            if unit not in existing:
                self.commit(unit, cursor['units'][unit]['job'], records, galleries.get(unit))
            else:
                for table,rows in records.items():
                    with open(self.folder/'units'/unit/(table+'.csv'),encoding='utf-8',newline='') as stream:
                        actual=list(csv.DictReader(stream))
                    assert actual == [dict(unit=unit,**r) for r in rows], 'Conflicting local and remote complete unit: '+unit
        if (folder/'EVENTS.jsonl').exists():
            local=self.folder/'EVENTS.jsonl'
            records={}
            for path in (folder/'EVENTS.jsonl',local):
                if path.exists():
                    for line in path.read_text(encoding='utf-8').splitlines():
                        try:
                            record=json.loads(line);records[record['event_id']]=record
                        except json.JSONDecodeError: pass  # Incomplete final line was never a committed event.
            part=local.with_name(local.name+'.tmp')
            part.write_bytes(b''.join(canonical(v)+b'\n' for v in sorted(records.values(),key=lambda r:r['time_utc'])))
            part.replace(local)


def retry_delay(exc, attempt):
    response = getattr(exc, 'response', None)
    code = getattr(response, 'status_code', None)
    transient = code in (429, 500, 502, 503, 504) or (code is None and isinstance(exc, requests.RequestException))
    if not transient:
        return None
    headers = getattr(response, 'headers', {}) or {}
    try:
        delay = float(headers.get('Retry-After', '0'))
    except ValueError:
        from email.utils import parsedate_to_datetime
        delay = max(0., parsedate_to_datetime(headers['Retry-After']).timestamp()-time.time())
    match = re.search(r'\bt=(\d+)', headers.get('RateLimit',''))
    return max(delay, float(match.group(1)) if match else 0., min(60, 5*2**attempt))


def pause(delay):
    print('HF server/backoff delay:', delay, 'seconds', flush=True)
    while delay > 0:
        step = min(delay, 30.)
        time.sleep(step); delay -= step


class Publisher:
    REPO = 'Shanmuk4622/tyre-wear-study'
    def __init__(self, destination, token):
        assert token, 'Enable the Kaggle HF_TOKEN secret for publication.'
        self.destination, self.token = destination, token
        self.owner = uuid.uuid4().hex
        self.api = HfApi(token=token)
        self.session = requests.Session(); self.session.trust_env = False
        self.session.headers['Authorization'] = 'Bearer '+token
        self.last_verified = None

    def call(self, fn):
        for attempt in range(6):
            try:
                return fn()
            except Exception as exc:
                delay = retry_delay(exc, attempt)
                if delay is None or attempt == 5:
                    raise
                pause(delay)

    def json(self, name, revision):
        url = f'https://huggingface.co/datasets/{self.REPO}/resolve/{revision}/{self.destination}/{name}'
        def get():
            with self.session.get(url, timeout=(20,120)) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status(); return response.json()
        return self.call(get)

    def claim(self):
        who = self.call(lambda: self.api.whoami())
        assert who['name'].lower() == 'shanmuk4622', 'Use the intended HF account.'
        # Refresh parent/ownership after every ambiguous response or CAS conflict.
        for attempt in range(8):
            parent = self.call(lambda: self.api.repo_info(self.REPO, repo_type='dataset').sha)
            owner = self.json('OWNER.json', parent)
            if owner and owner['owner'] not in (None, self.owner) and owner['expires'] > time.time():
                raise RuntimeError('Another notebook owns this diagnostic. Stop it cleanly or wait for the 45-minute lease expiry; do not run duplicate writers.')
            value = canonical(dict(owner=self.owner, expires=time.time()+45*60))
            try:
                self.api.create_commit(self.REPO, repo_type='dataset', parent_commit=parent,
                     operations=[CommitOperationAdd(path_in_repo=self.destination+'/OWNER.json', path_or_fileobj=value)],
                     commit_message='TREAD-ONE diagnostic writer claim')
                return
            except Exception as exc:
                if getattr(getattr(exc,'response',None),'status_code',None) in (409,412):
                    continue
                delay = retry_delay(exc, attempt)
                if delay is None or attempt == 7:
                    raise
                pause(delay)
        raise RuntimeError('Could not establish diagnostic writer ownership.')

    def restore(self, ledger):
        revision = self.call(lambda: self.api.repo_info(self.REPO, repo_type='dataset').sha)
        manifest = self.json('MANIFEST.json', revision)
        if manifest is None:
            return False
        assert manifest['contract'] == digest(ledger.contract)
        folder = ledger.folder/'remote_restore'/revision
        for name, meta in manifest['files'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe manifest path'
            target = folder/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) == meta['sha256']:
                continue
            def transfer():
                url = f'https://huggingface.co/datasets/{self.REPO}/resolve/{revision}/{self.destination}/{name}'
                part = target.with_name(target.name+'.part')
                with self.session.get(url, timeout=(20,120), stream=True) as response:
                    response.raise_for_status()
                    with open(part,'wb') as stream:
                        for block in response.iter_content(4*1024**2): stream.write(block)
                part.replace(target)
            self.call(transfer)
            assert target.stat().st_size == meta['bytes'] and sha(target) == meta['sha256'], name
        ledger.restore(folder)
        self.last_verified = revision
        return True

    def publish(self, snapshot, release=False):
        snapshot = Path(snapshot)
        manifest = read(snapshot/'MANIFEST.json')
        files = dict(manifest['files'])
        files['MANIFEST.json'] = dict(sha256=sha(snapshot/'MANIFEST.json'), bytes=(snapshot/'MANIFEST.json').stat().st_size)
        for attempt in range(8):
            parent = self.call(lambda: self.api.repo_info(self.REPO, repo_type='dataset').sha)
            owner = self.json('OWNER.json', parent)
            assert owner and owner['owner'] == self.owner, 'Writer ownership lost; no stale publication.'
            previous = self.json('MANIFEST.json', parent)
            changed = {n:m for n,m in files.items() if n == 'MANIFEST.json' or not previous or previous['files'].get(n) != m}
            operations = [CommitOperationAdd(path_in_repo=self.destination+'/'+n, path_or_fileobj=str(snapshot/n)) for n in changed]
            operations.append(CommitOperationAdd(path_in_repo=self.destination+'/OWNER.json',
                       path_or_fileobj=canonical(dict(owner=None if release else self.owner, expires=0 if release else time.time()+45*60))))
            try:
                commit = self.api.create_commit(self.REPO, repo_type='dataset', parent_commit=parent, operations=operations,
                                commit_message='TREAD-ONE diagnostic direct CSV, cursor and visual snapshot')
                revision = commit.oid
            except Exception as exc:
                # A lost successful response must be recognized before retrying.
                current = self.call(lambda: self.api.repo_info(self.REPO, repo_type='dataset').sha)
                remote = self.json('MANIFEST.json', current)
                current_owner = self.json('OWNER.json', current)
                if remote == manifest and current_owner and current_owner['owner'] == (None if release else self.owner):
                    revision = current
                else:
                    code = getattr(getattr(exc,'response',None),'status_code',None)
                    delay = retry_delay(exc, attempt)
                    if code not in (409,412) and (delay is None or attempt == 7): raise
                    if delay is not None: pause(delay)
                    continue
            infos = self.call(lambda: list(self.api.list_repo_tree(self.REPO, path_in_repo=self.destination,
                          revision=revision, repo_type='dataset', recursive=True)))
            lookup = {i.path[len(self.destination)+1:]: i for i in infos if hasattr(i,'size')}
            for name, meta in files.items():
                info = lookup[name]
                assert info.size == meta['bytes'], name
                if info.lfs:
                    assert info.lfs.sha256 == meta['sha256'], name
                else:
                    body = (snapshot/name).read_bytes()
                    assert info.blob_id == hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest(), name
            self.last_verified = revision
            atomic(snapshot.parent.parent/'PUBLICATION.json', dict(status='published_and_verified', revision=revision,
                  destination=self.destination, files=len(files), zip_uploads=0, owner_released=release))
            print('Verified HF diagnostic snapshot:', revision, 'files:', len(files), flush=True)
            return revision
        raise RuntimeError('Diagnostic publication retries exhausted; local complete units retained.')
