"""Actual ATLAS training/recovery and offline storage/publication acceptance."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/lc_pilot'
sys.path.insert(0, str(PILOT))
import numpy as np
import torch
import requests
from huggingface_hub import CommitOperationAdd
from common import atomic, canonical, digest, file_meta, read, sha
from data_atlas import locate
from evidence_hub import delay_for
from publisher import Publisher
from rate_budget import Budget
from schemas import SCHEMAS
from spec import configuration, sources
from store import Store
import acceptance

TOY = dict(sources={}, engineering_fixture=True, revision='ATLAS append-cursor fault test')
FIELDS = {'toy': ['update','value']}


def toy_commit(store, update, fault=None):
    store.commit(dict(protocol=digest(TOY), updates=update, phase='fixture', value=torch.tensor([float(update)])),
                 {'toy': [dict(update=update,value=update / 4)]}, partition=1, fault=fault)


def rejected(call):
    try:
        call()
    except (AssertionError, FileNotFoundError, RuntimeError, TimeoutError):
        return
    raise AssertionError('Expected rejection')


class Service:
    """In-memory immutable Git/LFS fixture; no network operations."""
    def __init__(self):
        self.files, self.history, self.head, self.commits, self.lost = {}, {'r0': {}}, 'r0', 0, False
    def whoami(self):
        return {'name': 'Shanmuk4622'}
    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha=self.head)
    def create_commit(self, *args, parent_commit, operations, **kwargs):
        if parent_commit != self.head:
            raise requests.HTTPError('fixture CAS', response=SimpleNamespace(status_code=409,headers={}))
        files = dict(self.files)
        for op in operations:
            with op.as_file() as stream:
                files[op.path_in_repo] = stream.read()
        self.commits += 1
        self.head = f'r{self.commits}'
        self.files = files
        self.history[self.head] = dict(files)
        if self.lost:
            self.lost = False
            raise requests.ConnectionError('Successful response lost')
        return SimpleNamespace(oid=self.head)
    def list_repo_tree(self, *args, path_in_repo, revision, **kwargs):
        import hashlib
        return [SimpleNamespace(path=n,size=len(value), lfs=SimpleNamespace(sha256=__import__('hashlib').sha256(value).hexdigest()) if n.endswith(('.pt','.npy')) else None,
            blob_id=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest())
            for n,value in self.history[revision].items() if n.startswith(path_in_repo + '/')]
    def get(self, url, **kwargs):
        revision, name = url.split('/resolve/',1)[1].split('/',1)
        body = self.history[revision].get(name)
        class Response:
            status_code = 200 if body is not None else 404
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.HTTPError('fixture missing', response=self)
            def json(self): return json.loads(body)
            def iter_content(self, size):
                for offset in range(0, len(body), size): yield body[offset:offset+size]
        return Response()


def fake_publisher(service, folder, owner='owner'):
    hub = Publisher.__new__(Publisher)
    hub.destination, hub.owner, hub.api, hub.session = 'offline-atlas', owner, service, service
    hub.calls, hub.last_verified, hub.log = 0, None, folder / (owner + '.jsonl')
    return hub


def offline(folder):
    checks = []
    for fault in ('after_csv','after_native','before_pointer','after_pointer'):
        path = folder / fault
        store = Store(path, TOY, FIELDS)
        toy_commit(store, 0)
        result = subprocess.run([sys.executable,'-B',str(Path(__file__)),'--fault-child',str(path),fault], capture_output=True, text=True)
        assert result.returncode == 86, result.stderr
        restored = Store(path, TOY, FIELDS)
        assert restored.state['updates'] == int(fault == 'after_pointer')
        assert len(list(restored.rows('toy'))) == restored.state['updates'] + 1
        checks.append('abrupt_' + fault + '_consistent_native_and_CSV_cursor')
    store = Store(folder / 'rotation', TOY, FIELDS)
    for update in range(4):
        toy_commit(store, update)
    assert len(list((store.root / 'states').glob('*.pt'))) == 2
    checks.append('latest_two_local_native_states_all_CSV_history_retained')
    snapshot = store.export(folder / 'snapshot')
    restored = Store(folder / 'install', TOY, FIELDS)
    restored.install(snapshot)
    assert restored.state['updates'] == 3
    checks.append('direct_CSV_native_snapshot_restore_without_database_or_archive')
    before = sha(next((snapshot / 'csv').rglob('*.csv')))
    toy_commit(store, 4)
    assert sha(next((snapshot / 'csv').rglob('*.csv'))) == before
    checks.append('immutable_snapshot_does_not_hardlink_mutable_CSV')
    newer = store.export(folder / 'newer')
    restored.install(newer)
    assert restored.state['updates'] == 4
    checks.append('verified_remote_history_extension_fast_forward')
    path = next((restored.root / 'csv').rglob('*.csv'))
    body = path.read_bytes().replace(b'0.25',b'0.95')
    path.write_bytes(body)
    rejected(lambda: Store(restored.root, TOY, FIELDS))
    checks.append('committed_CSV_value_tamper_rejected')
    rejected(lambda: Store(store.root, {'changed': True}, FIELDS))
    checks.append('contract_drift_rejected')
    # Multiple immutable batches, ambiguous responses, CAS fencing and no archives.
    for i in range(165):
        p = snapshot / 'fixture' / f'table_{i:03d}.csv'
        p.parent.mkdir(exist_ok=True)
        p.write_text(f'index,value\n{i},{i/4}\n',encoding='utf-8')
    manifest = read(snapshot / 'MANIFEST.json')
    manifest['files'] = {p.relative_to(snapshot).as_posix():file_meta(p) for p in snapshot.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
    atomic(snapshot / 'MANIFEST.json', manifest)
    service = Service()
    hub = fake_publisher(service, folder)
    service.lost = True
    hub.claim()
    operations = []
    original = service.create_commit
    def record(*args, **kwargs):
        operations.append(len(kwargs['operations']))
        return original(*args, **kwargs)
    service.create_commit = record
    service.lost = True
    receipt = hub.publish(snapshot)
    assert max(operations) <= 82
    checks.append('bounded_80_object_batches_and_ambiguous_success_reconciliation')
    downloaded = hub.download(receipt['revision'], folder / 'download', digest(TOY))
    remote = Store(folder / 'remote', TOY, FIELDS)
    remote.install(downloaded)
    assert remote.state['updates'] == 3
    checks.append('pinned_Git_LFS_hash_verified_remote_restore_fixture')
    before = service.commits
    hub.publish(snapshot, release=True)
    assert service.commits == before + 1
    checks.append('unchanged_objects_deduplicated_and_owner_released')
    rejected(lambda: hub.publish(snapshot))
    checks.append('stale_writer_fenced')
    new = fake_publisher(service, folder, 'new-owner')
    new.claim()
    rejected(lambda: fake_publisher(service,folder,'other').claim())
    new.publish(snapshot, release=True)
    checks.append('active_second_writer_rejected_new_owner_can_release_identical_snapshot')
    error = requests.HTTPError('fixture',response=SimpleNamespace(status_code=429,headers={'Retry-After':'75','RateLimit':'"api";r=0;t=90'}))
    assert delay_for(error,0) == 90
    checks.append('server_retry_after_and_reset_delay_respected')
    budget = Budget(folder / 'budget')
    budget.reserve('api')
    budget.observe('api', error.response)
    assert read(budget.path)['api']['blocked_until'] >= time.time() + 89
    now = time.time()
    atomic(budget.path, {'api':dict(times=[now]*96,blocked_until=0)})
    clock=[now]
    with patch('rate_budget.time.time',side_effect=lambda:clock[0]),patch('rate_budget.time.sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
        budget.reserve('api')
    assert clock[0]>=now+300
    checks.append('shared_local_API_budget_and_server_backoff_enforced_without_network')
    return checks


def main():
    if '--fault-child' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--fault-child') + 1])
        store = Store(path, TOY, FIELDS)
        toy_commit(store, 1, sys.argv[-1])
        raise AssertionError('Fault did not terminate')
    out = ROOT / 'tread_two/lc_validation'
    out.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='acceptance_', dir=out))
    checks = offline(folder / 'offline')
    atomic(folder / 'OFFLINE.json', dict(status='passed', checks=checks, count=len(checks), network_requests=0))
    root = locate()
    weights = ROOT / 'completion_20261028/hf/report/weights.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = configuration(root, weights, device, device, engineering=True)
    atomic(folder / 'CONFIG.json', cfg)
    source = acceptance.prepare(cfg, folder / 'actual')
    restored, proof = acceptance.finish(cfg, folder / 'actual', source)
    # Rendering reads retained rows only; it performs no additional forward pass.
    target = restored.export(folder / 'final_snapshot', PILOT, 'local_acceptance_passed')
    from report import build
    build(target, cfg)
    result = dict(status='passed', offline_checks=checks, actual_model_proof=proof, source_hashes=sources(),
        contract=cfg['protocol'], fixture_directory=str(folder), final_snapshot=str(target),
        full_measured_training_started=False, new_measured_model_accuracy=None,
        actual_live_HF_tested=False, actual_dual_T4_tested=False, local_device=device,
        network_requests=0, original_workstations_changed=False,
        regression_fixed='r2 paired views, source SHA exclusion and native corrected transport')
    atomic(out / 'LOCAL_VALIDATION.json', result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_hashes','actual_model_proof')},indent=2), flush=True)


if __name__ == '__main__':
    main()

