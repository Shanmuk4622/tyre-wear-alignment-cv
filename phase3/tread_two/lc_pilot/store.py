"""Append-only direct CSV with native-anchored byte cursors and atomic pointers.

An interrupted unit's uncommitted CSV suffix is truncated on recovery. Historical
chunks have hashes; only the latest two local native states are retained.
Remote snapshots consolidate the small unit ledger as JSON, never a database.
"""
import csv
import hashlib
import io
import os
import shutil
import time
import uuid
from pathlib import Path

import torch
from common import atomic, canonical, csv_write, digest, file_meta, inside, native, read, sha


class Store:
    def __init__(self, root, contract, schemas, verify=True):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract, self.schemas = contract, schemas
        self.protocol = digest(contract)
        path = self.root / 'CONTRACT.json'
        if path.exists():
            assert read(path) == contract, 'Run contract differs; do not mix or silently restart'
        else:
            atomic(path, contract)
        self.units = []
        self.state = None
        if (self.root / 'POINTER.json').exists():
            pointer = read(self.root / 'POINTER.json')
            chain = []
            name, expected = pointer['unit'], pointer['sha256']
            while name:
                path = inside(self.root, name)
                assert sha(path) == expected, 'Unit ledger changed'
                unit = read(path)
                assert unit['protocol'] == self.protocol
                chain.append(unit)
                name, expected = unit['parent'], unit['parent_sha256']
            self.units = list(reversed(chain))
            latest = self.units[-1]
            path = inside(self.root, latest['state_path'])
            assert file_meta(path) == latest['state_meta'], 'Latest native state is corrupt'
            self.state = torch.load(path, map_location='cpu', weights_only=False)
            assert self.state['protocol'] == self.protocol and self.state['seq'] == latest['seq']
            assert self.state['row_cursors'] == latest['row_cursors']
            assert self.state['file_cursors'] == latest['file_cursors']
            if verify:
                self.verify_and_rollback()
        elif verify:
            self.verify_and_rollback()

    def verify_and_rollback(self):
        counts = {name: 0 for name in self.schemas}
        offsets = {}
        for seq, unit in enumerate(self.units):
            assert unit['seq'] == seq
            for chunk in unit['chunks']:
                path = inside(self.root, chunk['path'])
                assert chunk['start'] == offsets.get(chunk['path'], 0)
                with path.open('rb') as stream:
                    stream.seek(chunk['start'])
                    body = stream.read(chunk['end'] - chunk['start'])
                assert hashlib.sha256(body).hexdigest() == chunk['sha256'], 'Committed CSV chunk changed'
                offsets[chunk['path']] = chunk['end']
                counts[chunk['family']] += chunk['rows']
            assert counts == unit['row_cursors']
        if self.state:
            assert counts == self.state['row_cursors'] and offsets == self.state['file_cursors']
            for name, meta in self.state['artifacts'].items():
                assert file_meta(inside(self.root, name)) == meta
        for path in (self.root / 'csv').rglob('*.csv') if (self.root / 'csv').exists() else []:
            relative = path.relative_to(self.root).as_posix()
            length = offsets.get(relative, 0)
            assert path.stat().st_size >= length
            if length:
                if path.stat().st_size > length:
                    with path.open('r+b') as stream:
                        stream.truncate(length)
                        stream.flush()
                        os.fsync(stream.fileno())
            else:
                path.unlink()  # Only uncommitted generated CSV under this run's own csv/.
        self.rotate_states()

    def commit(self, state, tables=None, partition=0, artifacts=None, fault=None):
        begun = time.monotonic()
        tables, artifacts = tables or {}, artifacts or {}
        assert state['protocol'] == self.protocol and set(tables) <= set(self.schemas)
        seq = len(self.units)
        previous = self.units[-1] if self.units else None
        counts = dict(previous['row_cursors']) if previous else {name: 0 for name in self.schemas}
        offsets = dict(previous['file_cursors']) if previous else {}
        known = dict(self.state['artifacts']) if self.state else {}
        chunks = []
        for family, rows in tables.items():
            if not rows:
                continue
            relative = f'csv/epoch_{partition:04d}/{family}.csv'
            path = inside(self.root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            start = offsets.get(relative, 0)
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=self.schemas[family], extrasaction='raise', lineterminator='\n')
            if not start:
                writer.writeheader()
            writer.writerows(rows)
            body = stream.getvalue().encode('utf-8')
            with path.open('r+b' if path.exists() else 'w+b') as target:
                assert target.seek(0, 2) == start, 'Recover uncommitted suffix before writing'
                target.write(body)
                target.flush()
                os.fsync(target.fileno())
            offsets[relative] = start + len(body)
            counts[family] += len(rows)
            chunks.append(dict(path=relative, family=family, start=start, end=offsets[relative],
                               rows=len(rows), sha256=hashlib.sha256(body).hexdigest()))
        if fault == 'after_csv':
            os._exit(86)
        after_csv = time.monotonic()
        for relative, value in artifacts.items():
            name = 'artifacts/' + relative
            path = inside(self.root, name)
            assert name not in known, 'Scientific artifacts are immutable; use a new identity'
            if path.suffix in ('.pt', '.npy'):
                native(path, value)
            else:
                atomic(path, value)
            known[name] = file_meta(path)
        name = f'{seq:08d}-{uuid.uuid4().hex[:10]}'
        state = dict(state, seq=seq, row_cursors=counts, file_cursors=offsets, artifacts=known)
        relative = f'states/{name}.pt'
        native(inside(self.root, relative), state)
        after_native = time.monotonic()
        if fault == 'after_native':
            os._exit(86)
        parent_name = read(self.root / 'POINTER.json')['unit'] if previous else None
        unit = dict(seq=seq, protocol=self.protocol, parent=parent_name,
                    parent_sha256=sha(self.root / parent_name) if parent_name else None,
                    state_path=relative, state_meta=file_meta(self.root / relative),
                    chunks=chunks, row_cursors=counts, file_cursors=offsets,
                    updates=state['updates'], phase=state['phase'])
        unit_name = f'units/{name}.json'
        atomic(self.root / unit_name, unit)
        if fault == 'before_pointer':
            os._exit(86)
        atomic(self.root / 'POINTER.json', dict(unit=unit_name, sha256=sha(self.root / unit_name)))
        self.units.append(unit)
        self.state = state
        if fault == 'after_pointer':
            os._exit(86)
        timing = self.root / 'COMMIT_TIMINGS.csv'
        with timing.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['unit','update','phase','CSV_seconds','artifacts_native_seconds','pointer_seconds','total_seconds'], lineterminator='\n')
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(dict(unit=seq, update=state['updates'], phase=state['phase'], CSV_seconds=after_csv-begun,
                artifacts_native_seconds=after_native-after_csv, pointer_seconds=time.monotonic()-after_native, total_seconds=time.monotonic()-begun))
            stream.flush()
            os.fsync(stream.fileno())
        self.rotate_states()
        return unit['state_meta']['sha256']

    def rotate_states(self):
        keep = {unit['state_path'] for unit in self.units[-2:]}
        for path in (self.root / 'states').glob('*.pt'):
            if path.relative_to(self.root).as_posix() not in keep:
                path.unlink()  # Own generated recovery states only; all CSV/history remains.

    def rows(self, family):
        for path in sorted((self.root / 'csv').glob(f'epoch_*/{family}.csv')):
            with path.open(encoding='utf-8', newline='') as stream:
                yield from csv.DictReader(stream)

    def export(self, target, source_root=None, status='snapshot'):
        assert self.state is not None
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        selected = ['CONTRACT.json', 'POINTER.json'] + list(self.state['file_cursors']) + list(self.state['artifacts'])
        for name in selected:
            source, dest = inside(self.root, name), inside(target, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)  # CSVs must never hard-link to mutable append files.
        shutil.copy2(self.root / self.units[-1]['state_path'], target / 'state.pt')
        atomic(target / 'HISTORY.json', self.units)
        if (Path(source_root or '') / 'assets/COLUMN_DICTIONARY.csv').exists():
            shutil.copy2(Path(source_root) / 'assets/COLUMN_DICTIONARY.csv', target / 'COLUMN_DICTIONARY.csv')
        csv_write(target / 'STATE_VERSIONS.csv',
            (dict(unit=u['seq'], update=u['updates'], phase=u['phase'], state_sha256=u['state_meta']['sha256'],
                  local_native_present=int((self.root / u['state_path']).exists())) for u in self.units),
            ['unit','update','phase','state_sha256','local_native_present'])
        atomic(target / 'SCHEMA.json', dict(families=self.schemas, row_cursors=self.state['row_cursors'],
                                            missing='Blank means unavailable/not applicable; numeric zeros are retained'))
        atomic(target / 'CURSOR.json', dict(status=status, updates=self.state['updates'], phase=self.state['phase'],
                                            state_sha256=sha(target / 'state.pt'), row_cursors=self.state['row_cursors']))
        if source_root:
            for name, expected in self.contract['sources'].items():
                source = inside(source_root, name)
                assert sha(source) == expected
                dest = inside(target, 'source/' + name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        for name in ('LIFECYCLE.jsonl', 'REMOTE_HISTORY.jsonl', 'HTTP_REQUESTS.jsonl', 'NUMERICAL_EVENTS.jsonl', 'RESOURCE_SAMPLES.csv', 'COMMIT_TIMINGS.csv'):
            if (self.root / name).exists():
                shutil.copy2(self.root / name, target / name)
        for source in (self.root / 'session_history').glob('*'):
            dest = target / 'session_history' / source.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)
        files = {p.relative_to(target).as_posix(): file_meta(p) for p in target.rglob('*') if p.is_file()}
        assert not any(name.endswith(('.zip', '.npz', '.sqlite')) for name in files)
        atomic(target / 'MANIFEST.json', dict(contract=self.protocol, files=files))
        return target

    def install(self, snapshot):
        snapshot = Path(snapshot)
        manifest = read(snapshot / 'MANIFEST.json')
        assert manifest['contract'] == self.protocol and read(snapshot / 'CONTRACT.json') == self.contract
        for name, meta in manifest['files'].items():
            assert file_meta(inside(snapshot, name)) == meta
        state = torch.load(snapshot / 'state.pt', map_location='cpu', weights_only=False)
        assert state['protocol'] == self.protocol
        assert sha(snapshot / 'state.pt') == read(snapshot / 'CURSOR.json')['state_sha256']
        units = read(snapshot / 'HISTORY.json')
        if self.units:
            assert len(units) >= len(self.units) and units[:len(self.units)] == self.units, 'Divergent remote history; preserve both states for review'
        for name in list(state['file_cursors']) + list(state['artifacts']):
            source, dest = inside(snapshot, name), inside(self.root, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.install.tmp')
            shutil.copy2(source, tmp)
            os.replace(tmp, dest)
        for name in manifest['files']:
            if name.startswith(('session_history/','control_history/')) or name in ('LIFECYCLE.jsonl','REMOTE_HISTORY.jsonl','HTTP_REQUESTS.jsonl','NUMERICAL_EVENTS.jsonl','RESOURCE_SAMPLES.csv','COMMIT_TIMINGS.csv'):
                source = inside(snapshot, name)
                filename = Path(name).name if name.startswith('session_history/') else sha(source)[:20] + '-' + Path(name).name
                dest = self.root / 'session_history' / filename
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(source, dest)
        for index, unit in enumerate(units):
            name = units[index + 1]['parent'] if index + 1 < len(units) else read(snapshot / 'POINTER.json')['unit']
            atomic(inside(self.root, name), unit)
        path = inside(self.root, units[-1]['state_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / 'state.pt', path)
        atomic(self.root / 'POINTER.json', read(snapshot / 'POINTER.json'))
        restored = Store(self.root, self.contract, self.schemas)
        self.units, self.state = restored.units, restored.state
