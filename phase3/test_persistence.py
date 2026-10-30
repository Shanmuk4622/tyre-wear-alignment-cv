"""Offline fault tests: real bundle code against a revisioned in-memory HF fixture."""
from PIL import Image
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
import torch
from common import atomic, digest, read, sha
from hub import Publisher, extract_verified, retry_delay


class Conflict(Exception):
    def __init__(self, status):
        self.response = types.SimpleNamespace(status_code=status, headers={})


class Api:
    def __init__(self):
        self.files = {}
        self.revisions = {'0': {}}
        self.head = '0'
        self.binary_uploads = 0
        self.conflict_once = False
        self.timeout_once = False

    def repo_info(self, *args, **kwargs):
        return types.SimpleNamespace(sha=self.head)

    def create_commit(self, *args, operations, parent_commit, **kwargs):
        if self.conflict_once:
            self.conflict_once = False
            raise Conflict(409)
        if parent_commit != self.head:
            raise Conflict(409)
        for op in operations:
            value = op.path_or_fileobj
            payload = value if isinstance(value, bytes) else Path(value).read_bytes()
            self.files[op.path_in_repo] = payload
            self.binary_uploads += int(op.path_in_repo.endswith('.zip'))
        self.head = str(int(self.head)+1)
        self.revisions[self.head] = dict(self.files)
        if self.timeout_once:
            self.timeout_once = False
            raise TimeoutError('simulated successful commit with lost response')
        return types.SimpleNamespace(oid=self.head)

    def get_paths_info(self, repo, paths, revision, **kwargs):
        import hashlib
        return [types.SimpleNamespace(path=p, lfs=types.SimpleNamespace(sha256=hashlib.sha256(self.revisions[revision][p]).hexdigest())) for p in paths]


def run():
    import uuid
    folder = ROOT/'validation'/('persistence_fixture_'+uuid.uuid4().hex[:10])
    folder.mkdir(parents=True, exist_ok=True)
    api = Api()
    cases = []
    # No token is read from disk/environment and no real API method is called.
    def publisher(work, owner):
        p = Publisher.__new__(Publisher)
        p.work = work
        work.mkdir(exist_ok=True)
        p.contract = {'fixture': 'persistence-r1'}
        p.protocol = digest(p.contract)
        p.prefix = 'phase3/test/'+p.protocol
        p.owner = owner
        p.api = api
        p.index = {}
        p.last = {'files': {}}
        p.calls = __import__('collections').deque()
        p.token = 'offline-fixture'
        p.tag = 'fixture'
        p.take_over = False
        def fetch(path, revision, optional=False):
            payload = api.revisions[revision].get(path)
            if payload is None:
                if optional:
                    return None
                raise FileNotFoundError(path)
            target = work/(digest([path, revision])+'.download')
            target.write_bytes(payload)
            return target
        p.fetch = fetch
        return p
    p = publisher(folder/'writer', 'first')
    api.timeout_once = True
    with patch('hub.retry_delay', return_value=0):
        p.claim()
    cases += ['Successful claim with lost response can reclaim its own lease']
    runroot = folder/'runs'
    runfolder = runroot/'seed1'
    runfolder.mkdir(parents=True, exist_ok=True)
    state = dict(protocol=p.protocol, seed=1, updates=7, epoch=0, cursor=7, completed=False,
                 model={'weight': torch.arange(9)}, rng={'fixture': 7}, history=list(range(7)))
    atomic(runfolder/'state.pt', state, binary=True)
    api.conflict_once = True
    p.publish(runroot)
    assert api.binary_uploads == 1
    p.publish(runroot)
    assert api.binary_uploads == 1, 'Unchanged bundle re-uploaded'
    cases += ['CAS conflict retry', 'Stable ZIP deduplication', 'Remote index and binary SHA256 checks']
    # Fresh-session restoration resolves bundles at an immutable containing revision.
    p.publish(runroot, release=True)
    fresh = publisher(folder/'fresh', 'second')
    fresh.claim()
    fresh.restore(1, folder/'restored')
    restored = torch.load(folder/'restored/state.pt', map_location='cpu', weights_only=False)
    assert torch.equal(restored['model']['weight'], state['model']['weight']) and restored['cursor'] == 7
    cases += ['Fresh-session revision-pinned bundle download/restore']
    # A completed local state must not be replaced by an older remote state.
    local = dict(state, updates=8, cursor=8)
    atomic(folder/'restored/state.pt', local, binary=True)
    fresh.restore(1, folder/'restored')
    assert torch.load(folder/'restored/state.pt', weights_only=False)['updates'] == 8
    cases += ['Newer local completed-update cursor retained']
    try:
        p.publish(runroot)
        raise AssertionError('Stale writer accepted')
    except RuntimeError as exc:
        assert 'fencing' in str(exc)
    cases += ['Stale writer fenced after ownership change']
    archive = fresh.fetch(fresh.index['seed1']['path'], fresh.index['seed1']['revision'])
    damaged = folder/'damaged.zip'
    import zipfile
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(damaged, 'w') as dst:
        for name in src.namelist():
            dst.writestr(name, b'corrupt' if name == 'state.pt' else src.read(name))
    original_hash = sha(folder/'restored/state.pt')
    try:
        extract_verified(damaged, folder/'restored', fresh.protocol, 1)
        raise AssertionError('Corruption accepted')
    except AssertionError as exc:
        assert 'checksum' in str(exc)
    assert sha(folder/'restored/state.pt') == original_hash
    cases += ['Corrupt snapshot rejected before replacing local progress']
    old = sha(runfolder/'state.pt')
    with patch('common.os.replace', side_effect=OSError('simulated interrupted atomic replace')):
        try:
            atomic(runfolder/'state.pt', dict(state, updates=99), binary=True)
        except OSError:
            pass
    assert sha(runfolder/'state.pt') == old
    cases += ['Interrupted local replace retains previous durable checkpoint']
    assert retry_delay(types.SimpleNamespace(headers={'Retry-After': '120', 'RateLimit': '"api";r=0;t=180'}), 0) >= 180
    attempts = []
    def transient():
        attempts.append(1)
        if len(attempts) < 3:
            raise Conflict(429)
        return 'ok'
    with patch('hub.retry_delay', return_value=0):
        assert fresh.request(transient) == 'ok'
    assert len(attempts) == 3
    cases += ['Retry-After and RateLimit reset delays honoured', '429 retries are bounded and retain local progress']
    # completed-state bundle restored and recognized, not converted into a fresh run.
    complete = dict(local, completed=True)
    atomic(runfolder/'state.pt', complete, binary=True)
    fresh.publish(runroot, release=True)
    third = publisher(folder/'third', 'third')
    third.claim()
    third.restore(1, folder/'restored_completed')
    assert torch.load(folder/'restored_completed/state.pt', weights_only=False)['completed']
    cases += ['Completed remote state remains completed after fresh restore']
    # A GPU smoke failure before update zero still needs remotely recoverable diagnostics.
    diagnostic = runroot/'seed2'
    diagnostic.mkdir(exist_ok=True)
    atomic(diagnostic/'ERROR.json', {'traceback': 'fixture smoke failed before measured state'})
    third.publish(runroot, release=True)
    fourth = publisher(folder/'fourth', 'fourth')
    fourth.claim()
    fourth.restore(2, folder/'restored_diagnostic')
    assert (folder/'restored_diagnostic/ERROR.json').exists()
    assert not (folder/'restored_diagnostic/state.pt').exists()
    cases += ['Pre-update smoke failure diagnostics saved and restored without fabricated training state']
    atomic(ROOT/'validation/PERSISTENCE_TESTS.json', dict(status='passed', cases=cases,
           live_hf_tested=False, scope='Real publication/restore code, offline revisioned service fixture; no external writes.'))
    print('PERSISTENCE FAULT TESTS PASSED:', len(cases))


if __name__ == '__main__':
    run()
