"""Deterministic quota/header, HTTP-dispatch and snapshot compatibility checks."""
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/pilot'
OUT = ROOT / 'tread_two/pilot_validation/nb08_repair'
OUT.mkdir(exist_ok=True)
folder = Path(tempfile.mkdtemp(prefix='transport_', dir=OUT))
sys.path.insert(0, str(PILOT))
sys.path.insert(1, str(ROOT / 'analysis'))
import requests
import torch
from common import atomic, digest, file_meta, read, sha
from store import Store
import rate_budget
from check_tread_two_training import Service, fake_publisher

spec = importlib.util.spec_from_file_location('nb08_repair_launcher', ROOT / 'analysis/nb08_transport_launcher.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
checks = []
inventory_bytes = (PILOT / 'FILES.json').read_bytes()
inventory_sha = hashlib.sha256(inventory_bytes).hexdigest()
before = {name: file_meta(PILOT / name) for name in read(PILOT / 'FILES.json')}


class Clock:
    def __init__(self):
        self.value, self.sleeps = 10000., []
    def time(self): return self.value
    def sleep(self, seconds):
        assert 0 < seconds <= 30
        self.sleeps.append(seconds)
        self.value += seconds


clock = Clock()
work = folder / 'work'
budget = rate_budget.Budget(work)
old_ledger = dict(api=dict(times=[clock.time()-600]*96, blocked_until=0))
atomic(budget.path, old_ledger)
with patch('rate_budget.time.time', clock.time), patch('rate_budget.time.monotonic', clock.time):
    try:
        budget.reserve('api')
    except TimeoutError:
        checks.append('reproduced_original_96_per_hour_self_timeout_without_network')
    else:
        raise AssertionError('Original failure not reproduced')
context = launcher.install(PILOT, work, inventory_sha, clock.time, clock.sleep)
assert read(work / 'NB08_TRANSPORT_REPAIR/HTTP_BUDGET_BEFORE.json') == old_ledger
budget.reserve('api')
assert clock.value == 10000 and len(read(budget.path)['api']['times']) == 1
checks.append('corrected_five_minute_window_preserves_before_ledger_and_no_false_timeout')

with redirect_stdout(io.StringIO()):
    recent = rate_budget.Budget(work / 'recent')
    atomic(recent.path,dict(api=dict(times=[clock.time()]*96,blocked_until=0)))
    began = clock.time()
    recent.reserve('api')
    assert clock.time() - began == 300 and len(read(recent.path)['api']['times']) == 1
    checks.append('HTTP_requests_wait_for_300_second_window_without_resetting_or_throwing')
    commit = rate_budget.Budget(work / 'commit')
    atomic(commit.path,dict(repo_commit=dict(times=[clock.time()-600]*96,blocked_until=0)))
    began = clock.time()
    commit.reserve('repo_commit')
    assert clock.time() - began == 3000
    checks.append('separate_96_per_hour_commit_attempt_guard_waits_beyond_legacy_300_second_abort')
    limited = rate_budget.Budget(work / 'server')
    limited.observe('api', SimpleNamespace(status_code=200, headers={
        'RateLimit-Policy':'"fixed window";"api";q=40;w=300',
        'RateLimit':'"api";r=2;t=60'}))
    limited.reserve('api'); limited.reserve('api')
    began = clock.time()
    limited.reserve('api')
    assert clock.time()-began == 60 and read(limited.path)['api']['effective_limit'] == 32
    checks.append('lower_advertised_quota_and_actual_remaining_reset_take_precedence')
    reset = rate_budget.Budget(work / 'reset')
    reset.observe('api',SimpleNamespace(status_code=429,headers={'Retry-After':'750','RateLimit':'"api";r=0;t=900'}))
    began = clock.time()
    reset.reserve('api')
    assert clock.time()-began == 900
    checks.append('429_Retry_After_and_longer_server_reset_honoured_without_fake_timeout')
    dated = rate_budget.Budget(work / 'dated')
    dated.observe('resolver',SimpleNamespace(status_code=429,headers={
        'Retry-After':format_datetime(datetime.fromtimestamp(clock.time()+480,timezone.utc)),
        'RateLimit':'"resolvers";r=0;t=300'}))
    began = clock.time()
    dated.reserve('resolver')
    assert clock.time()-began == 480
    checks.append('HTTP_date_retry_after_uses_server_delay_with_short_interruptible_sleeps')

# Exercise the actual patched Requests send path; no endpoint is contacted.
http = rate_budget.Budget(work / 'http')
session = rate_budget.BudgetSession(http, work / 'TEST_HTTP.jsonl')
def response(*args, **kwargs):
    result = requests.Response()
    result.status_code = 200
    result.headers = {}
    result._content = b'{}'
    return result
with patch('requests.Session.send', response):
    session.send(requests.Request('POST','https://huggingface.co/api/datasets/a/b/commit/main').prepare())
    session.send(requests.Request('POST','https://huggingface.co/api/datasets/a/b/preupload/main').prepare())
    session.send(requests.Request('GET','https://huggingface.co/datasets/a/b/resolve/r/file.csv').prepare())
counts = read(http.path)
assert len(counts['repo_commit']['times']) == 1 and len(counts['api']['times']) == 2 and len(counts['resolver']['times']) == 1
checks.append('actual_commit_endpoint_counted_separately_from_preupload_and_resolver_HTTP')

contract, schemas = dict(sources={}, fixture='transport snapshot compatibility'), {'toy':['update','value']}
store = Store(work / 'run', contract, schemas)
for update in range(2):
    store.commit(dict(protocol=digest(contract), updates=update, phase='fixture', model={'x':torch.tensor([float(update)])}),
                 {'toy':[dict(update=update,value=update/4)]})
snapshot = store.export(folder / 'snapshot')
manifest = read(snapshot / 'MANIFEST.json')
assert manifest['contract'] == digest(contract)
assert manifest['files']['TRANSPORT_REPAIR/launcher.py']['sha256'] == sha(ROOT / 'analysis/nb08_transport_launcher.py')
assert all(file_meta(snapshot/name) == meta for name,meta in manifest['files'].items())
checks.append('repair_source_policy_and_history_exported_in_hash_verified_snapshot_without_contract_change')
restored = Store(folder / 'restored',contract,schemas)
restored.install(snapshot)
assert restored.state['updates'] == 1 and torch.equal(restored.state['model']['x'],torch.tensor([1.]))
assert list(restored.rows('toy')) == list(store.rows('toy'))
assert list((restored.root / 'session_history').glob('*-TRANSPORT-launcher.py'))
checks.append('native_state_and_direct_CSV_restore_unchanged_repair_provenance_survives_restart')
service = Service()
hub = fake_publisher(service,folder)
hub.claim()
service.lost = True
receipt = hub.publish(snapshot)
download = hub.download(receipt['revision'], folder / 'pinned',digest(contract))
fresh = Store(folder / 'fresh',contract,schemas)
fresh.install(download)
assert fresh.state['updates'] == 1 and list(fresh.rows('toy')) == list(store.rows('toy'))
hub.publish(snapshot,release=True)
checks.append('immutable_HF_fixture_upload_ambiguous_reply_and_pinned_restore_accept_repair_files')
assert all(file_meta(PILOT/name) == item for name,item in before.items()) and (PILOT/'FILES.json').read_bytes() == inventory_bytes
checks.append('all_24_frozen_source_assets_and_inventory_byte_identical')

# Error diagnostics retain safe frame names/progress and omit exception bodies.
import coordinator
secret = 'hf_fixture_DO_NOT_LOG_THIS_VALUE'
def fail(): raise RuntimeError(secret)
with patch.object(coordinator,'main', fail), redirect_stdout(io.StringIO()):
    status = launcher.main(['train','--source',str(PILOT),'--work',str(work),'--inventory-sha',inventory_sha])
assert status == 1 and secret not in context['log'].read_text(encoding='utf-8')
assert json.loads(context['log'].read_text(encoding='utf-8').splitlines()[-1])['progress']['updates'] == 1
checks.append('safe_error_diagnostics_include_stage_stack_and_saved_cursor_without_secret')
record = dict(status='passed', checks=checks, count=len(checks), source_inventory_sha256=inventory_sha,
              transport_launcher_sha256=sha(ROOT/'analysis/nb08_transport_launcher.py'), source_assets_unchanged=24,
              model_training_started=False, network_requests=0, remote_writes=0, fixture_directory=str(folder))
atomic(OUT / 'TRANSPORT_TESTS.json',record)
print(json.dumps(record,indent=2))
