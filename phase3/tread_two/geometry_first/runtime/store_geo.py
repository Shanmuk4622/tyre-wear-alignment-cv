"""Engineering complete-update transactions, direct CSV and uncompressed NPY evidence.

Each fixture unit is immutable. Two rolling state files rotate only after a new
hash-verified pointer is durable. No measured-fit shard/capacity claim is made.
"""
import csv
import io
import os
from pathlib import Path
import numpy as np
import torch
from common import atomic, canonical, cpu, digest, file_meta, inside, native, read, restore_rng, rng_state, sha, sync_directory


class Store:
    def __init__(self, root, contract):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract

    def commit(self, model, optimizer, update, counters, rows, arrays, crash=None):
        pointer_path = self.root/'POINTER.json'
        previous = read(pointer_path) if pointer_path.exists() else None
        assert previous is None or update == previous['update']+1
        relative = f'units/{update:04d}'
        folder = self.root/relative
        folder.mkdir(parents=True, exist_ok=True)
        inventory = {}
        for family, records in rows.items():
            assert records and len({tuple(r.keys()) for r in records}) == 1
            path = folder/(family+'.csv')
            # A single explicit CSV schema per family/unit; fail on extra/missing fields.
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
                writer.writeheader()
                for record in records:
                    writer.writerow({k: self.format(v) for k, v in record.items()})
                stream.flush()
                os.fsync(stream.fileno())
            inventory[path.relative_to(self.root).as_posix()] = file_meta(path)
        for name, array in arrays.items():
            path = folder/(name+'.npy')
            native(path, np.asarray(array))
            inventory[path.relative_to(self.root).as_posix()] = file_meta(path)
        if crash == 'after_evidence':
            raise InterruptedError('Injected interruption after uncommitted evidence')
        state_path = self.root/f'state_{update%2}.pt'
        state = dict(contract=self.contract, update=update, counters=counters, model=cpu(model.state_dict()),
                     optimizer=cpu(optimizer.state_dict()), rng=rng_state(), precision='FP32_no_scaler',
                     scheduler=dict(completed_updates=update, total_updates=2940, convention='cosine_at_start_using_previous_complete_cursor'),
                     prior_units=[] if previous is None else previous['units'])
        native(state_path, state)
        if crash == 'after_state':
            raise InterruptedError('Injected interruption before pointer swap')
        record = dict(update=update, files=inventory, previous=None if previous is None else previous['chain'])
        record['chain'] = digest(record)
        atomic(folder/'UNIT.json', record)
        pointer = dict(contract=self.contract, update=update, chain=record['chain'], state=state_path.name,
                       state_meta=file_meta(state_path), units=state['prior_units']+[relative+'/UNIT.json'])
        atomic(pointer_path, pointer)
        if crash == 'after_pointer':
            raise InterruptedError('Injected interruption after durable pointer')
        return pointer

    @staticmethod
    def format(value):
        if value is None:
            return ''
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            assert np.isfinite(value)
            return format(float(value), '.9g' if isinstance(value, np.float32) else '.17g')
        return value

    def verify(self):
        pointer = read(self.root/'POINTER.json')
        assert pointer['contract'] == self.contract
        assert file_meta(inside(self.root, pointer['state'])) == pointer['state_meta']
        prior = None
        for name in pointer['units']:
            record = read(inside(self.root, name))
            assert record['previous'] == prior
            assert digest({k:v for k,v in record.items() if k!='chain'}) == record['chain']
            for relative, meta in record['files'].items():
                assert file_meta(inside(self.root, relative)) == meta, relative
            prior = record['chain']
        assert prior == pointer['chain'] and len(pointer['units']) == pointer['update']
        return pointer

    def restore(self, model, optimizer):
        pointer = self.verify()
        state = torch.load(self.root/pointer['state'], map_location='cpu', weights_only=False)
        assert state['contract'] == self.contract and state['update'] == pointer['update']
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        return state

    def snapshot(self, destination):
        """Frozen copies of selected files only: never include uncommitted tails."""
        import shutil
        pointer = self.verify()
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        names = ['POINTER.json', pointer['state']]
        for unit in pointer['units']:
            names.append(unit)
            names.extend(read(self.root/unit)['files'])
        for name in names:
            path = target/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root/name, path)
        atomic(target/'MANIFEST.json', dict(contract=self.contract, files={n:file_meta(target/n) for n in names}))
        return target
