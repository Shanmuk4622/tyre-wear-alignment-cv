"""Audited NB08 transport repair; frozen model/runtime files are never edited.

API/resolver HTTP requests use five-minute windows. Commit attempts keep a
separate conservative hourly budget. Server reset/Retry-After always wins.
This launcher is published alongside snapshots as explicit repair provenance.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

REVISION = 'nb08-transport-20261030-r1'
POLICY = {'api': (300, 96), 'resolver': (300, 400), 'transfer': (300, 1000), 'repo_commit': (3600, 96)}
_INSTALLED = None


def install(source, work, inventory_sha=None, clock=None, sleeper=None):
    global _INSTALLED
    source, work = Path(source).resolve(), Path(work).resolve()
    inventory_body = (source / 'FILES.json').read_bytes()
    inventory_digest = hashlib.sha256(inventory_body).hexdigest()
    if inventory_sha:
        assert inventory_digest == inventory_sha, 'Repair refuses a different frozen source inventory'
    inventory = json.loads(inventory_body)
    assert len(inventory) == 24, 'This repair is for the original NB08 source only'
    for name, item in inventory.items():
        target = (source / name).resolve()
        assert target.is_relative_to(source)
        body = target.read_bytes()
        assert len(body) == item['bytes'] and hashlib.sha256(body).hexdigest() == item['sha256'], 'Frozen source changed: ' + name
    if _INSTALLED:
        assert _INSTALLED['source'] == source and _INSTALLED['work'] == work
        return _INSTALLED
    sys.path.insert(0, str(source))
    from common import atomic, digest, event, file_meta, read, sha
    import rate_budget
    import store
    repair = work / 'NB08_TRANSPORT_REPAIR'
    repair.mkdir(parents=True, exist_ok=True)
    now = clock or time.time
    pause = sleeper or time.sleep
    log = repair / 'EVENTS.jsonl'
    own_source = Path(__file__).resolve()
    metadata = dict(revision=REVISION, source_inventory_sha256=inventory_digest,
                    launcher_sha256=sha(own_source), frozen_source_files=24,
                    policy={name:dict(window_seconds=window, local_limit=limit) for name,(window,limit) in POLICY.items()},
                    server_headers_authoritative=True, model_or_recipe_changed=False,
                    source_FILES_changed=False, contract_or_namespace_changed=False,
                    rationale='Replace erroneous 96-HTTP-requests/hour cap with bucket-specific HTTP windows; retain hourly commit-attempt guard.')
    atomic(repair / 'REPAIR.json', metadata)
    if (work / 'HF_HTTP_BUDGET.json').exists() and not (repair / 'HTTP_BUDGET_BEFORE.json').exists():
        shutil.copy2(work / 'HF_HTTP_BUDGET.json', repair / 'HTTP_BUDGET_BEFORE.json')
    event(log, 'repair_installed', revision=REVISION, inventory_sha256=inventory_digest, process=sys.argv[0])

    def reserve(self, bucket):
        assert bucket in POLICY
        default_window, default_limit = POLICY[bucket]
        next_notice = 0.
        while True:
            instant = now()
            with self.lock:
                data = read(self.path) if self.path.exists() else {}
                state = data.setdefault(bucket, dict(times=[], blocked_until=0))
                observed = state.get('server_policy', {})
                window = max(default_window, int(observed.get('window_seconds', default_window)))
                limit = min(default_limit, max(1, math.floor(.8 * int(observed.get('quota', default_limit * 2)))))
                state['times'] = [stamp for stamp in state['times'] if stamp > instant - window]
                remaining = state.get('server_remaining')
                reset_at = state.get('server_reset_at', 0.)
                if reset_at <= instant:
                    remaining = None
                    state.pop('server_remaining', None)
                local_delay = max(0., state['times'][-limit] + window - instant) if len(state['times']) >= limit else 0.
                server_delay = max(0., state.get('blocked_until', 0.) - instant,
                                   reset_at - instant if remaining is not None and remaining <= 0 else 0.)
                delay = max(local_delay, server_delay)
                state['effective_window_seconds'], state['effective_limit'] = window, limit
                if delay <= 0:
                    state['times'].append(instant)
                    if remaining is not None:
                        state['server_remaining'] = max(0, remaining - 1)
                    atomic(self.path, data)
                    return
                # Preserve the server block, history and policy. Never reset
                # the ledger, delete ownership, or convert a long wait to a
                # fake network TimeoutError.
                atomic(self.path, data)
            if instant >= next_notice:
                event(log, 'quota_wait', bucket=bucket, seconds=delay, window_seconds=window, local_limit=limit,
                      reason='server_header' if server_delay >= local_delay else 'local_window')
                print('HF', bucket, 'budget/backoff: waiting', round(delay, 1),
                      'seconds; saved progress is retained.', flush=True)
                next_notice = instant + 30
            pause(min(30., delay))

    def observe(self, bucket, response):
        headers, instant = response.headers, now()
        expected = 'resolvers' if bucket == 'resolver' else bucket
        policy = None
        for match in re.finditer(r'"([^\"]+)"\s*;\s*q=(\d+)\s*;\s*w=(\d+)', headers.get('RateLimit-Policy','')):
            if expected in match[1].split('|'):
                policy = dict(quota=int(match[2]), window_seconds=int(match[3]))
                break
        remaining, reset = None, None
        for match in re.finditer(r'"([^\"]+)"\s*;\s*r=(-?\d+)\s*;\s*t=(\d+)', headers.get('RateLimit','')):
            if expected in match[1].split('|'):
                remaining, reset = max(0, int(match[2])), int(match[3])
                break
        delay = 0.
        if response.status_code == 429:
            value = headers.get('Retry-After', '0')
            try:
                delay = max(0., float(value))
            except ValueError:
                from email.utils import parsedate_to_datetime
                delay = max(0., parsedate_to_datetime(value).timestamp() - instant)
            delay = max(delay, float(reset or 0))
        elif remaining == 0:
            delay = float(reset or 0)
        with self.lock:
            data = read(self.path) if self.path.exists() else {}
            state = data.setdefault(bucket, dict(times=[], blocked_until=0))
            if policy:
                assert policy['quota'] > 0 and policy['window_seconds'] > 0
                state['server_policy'] = policy
            if remaining is not None:
                state['server_remaining'], state['server_reset_at'] = remaining, instant + reset
            if delay:
                state['blocked_until'] = max(state.get('blocked_until',0.), instant + delay)
            atomic(self.path, data)
        if policy or delay:
            event(log, 'server_policy', bucket=bucket, policy=policy, remaining=remaining, reset_seconds=reset,
                  blocked_seconds=delay, status=response.status_code)

    original_send = rate_budget.BudgetSession.send
    def send(self, request, **kwargs):
        parsed = urlparse(request.url)
        if request.method.upper() == 'POST' and parsed.hostname == 'huggingface.co' and '/commit/' in parsed.path:
            self.budget.reserve('repo_commit')
        return original_send(self, request, **kwargs)
    rate_budget.Budget.reserve, rate_budget.Budget.observe = reserve, observe
    rate_budget.BudgetSession.send = send

    original_export, original_install = store.Store.export, store.Store.install
    def export(self, target, *args, **kwargs):
        target = original_export(self, target, *args, **kwargs)
        dest = target / 'TRANSPORT_REPAIR'
        dest.mkdir(exist_ok=True)
        shutil.copy2(own_source, dest / 'launcher.py')
        for item in repair.iterdir():
            if item.is_file() and item.suffix in ('.json','.jsonl'):
                shutil.copy2(item, dest / item.name)
        manifest = read(target / 'MANIFEST.json')
        manifest['files'] = {item.relative_to(target).as_posix():file_meta(item)
                             for item in target.rglob('*') if item.is_file() and item.name != 'MANIFEST.json'}
        atomic(target / 'MANIFEST.json', manifest)
        return target
    def install_snapshot(self, snapshot):
        result = original_install(self, snapshot)
        for item in (Path(snapshot) / 'TRANSPORT_REPAIR').glob('*'):
            if item.is_file():
                target = self.root / 'session_history' / (sha(item)[:20] + '-TRANSPORT-' + item.name)
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(item, target)
        return result
    store.Store.export, store.Store.install = export, install_snapshot
    _INSTALLED = dict(source=source, work=work, log=log, metadata=metadata, inventory=inventory_digest)
    return _INSTALLED


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','preflight','train'])
    parser.add_argument('--source', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--inventory-sha', required=True)
    args = parser.parse_args(argv)
    context = install(args.source, args.work, args.inventory_sha)
    from common import event, read
    import coordinator
    original_argv = sys.argv
    sys.argv = [str(Path(args.source) / 'coordinator.py'), args.stage, '--work', args.work]
    try:
        print('NB08 transport repair active: five-minute HTTP budgets, separate hourly commit guard.', flush=True)
        coordinator.main()
        event(context['log'], 'stage_complete', stage=args.stage)
        return 0
    except BaseException as error:
        progress = None
        pointer = Path(args.work) / 'run/POINTER.json'
        if pointer.exists():
            value = read(pointer)
            unit = read(Path(args.work) / 'run' / value['unit'])
            progress = {key:unit[key] for key in ('seq','updates','phase')}
        frames = [dict(file=Path(frame.filename).name, line=frame.lineno, function=frame.name)
                  for frame in traceback.extract_tb(error.__traceback__)]
        event(context['log'], 'stage_error', stage=args.stage, error_type=type(error).__name__, progress=progress, frames=frames)
        print('ATLAS stopped:', type(error).__name__, 'Stage:', args.stage,
              'Diagnostics:', context['log'], 'Saved state is retained.', flush=True)
        return 1
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    sys.exit(main())
