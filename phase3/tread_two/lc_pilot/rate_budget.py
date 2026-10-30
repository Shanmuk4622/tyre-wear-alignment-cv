"""Shared local HTTP budgets plus server backoff, including SDK transport calls.

Cross-host account traffic cannot be atomically counted by this session; server
headers remain authoritative. Only safe request categories/status are logged.
"""
import re
import math
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from filelock import FileLock
from common import atomic, event, read


POLICY={'api':(300,96),'resolver':(300,400),'transfer':(300,1000),'repo_commit':(3600,96)}

class Budget:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'HF_HTTP_BUDGET.json'
        self.lock = FileLock(str(self.path) + '.lock', timeout=60)

    @staticmethod
    def category(url):
        parsed = urlparse(url)
        if '/resolve/' in parsed.path:
            return 'resolver'
        return 'api' if parsed.hostname == 'huggingface.co' else 'transfer'

    def reserve(self, bucket):
        assert bucket in POLICY
        default_window, default_limit = POLICY[bucket]
        next_notice = 0.
        while True:
            instant = time.time()
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
                event(self.root / 'RATE_EVENTS.jsonl', 'quota_wait', bucket=bucket, seconds=delay, window_seconds=window, local_limit=limit,
                      reason='server_header' if server_delay >= local_delay else 'local_window')
                print('HF', bucket, 'budget/backoff: waiting', round(delay, 1),
                      'seconds; saved progress is retained.', flush=True)
                next_notice = instant + 30
            time.sleep(min(30., delay))

    def observe(self, bucket, response):
        headers, instant = response.headers, time.time()
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
            event(self.root / 'RATE_EVENTS.jsonl', 'server_policy', bucket=bucket, policy=policy, remaining=remaining, reset_seconds=reset,
                  blocked_seconds=delay, status=response.status_code)



class BudgetSession(requests.Session):
    def __init__(self, budget, log):
        super().__init__()
        self.budget, self.log = budget, Path(log)
        self.trust_env = False

    def send(self, request, **kwargs):
        parsed=urlparse(request.url)
        if request.method.upper()=='POST' and parsed.hostname=='huggingface.co' and '/commit/' in parsed.path:
            self.budget.reserve('repo_commit')
        category = self.budget.category(request.url)
        self.budget.reserve(category)
        begin = time.monotonic()
        try:
            response = super().send(request, **kwargs)
            self.budget.observe(category, response)
            event(self.log, 'http', category=category, method=request.method, status=response.status_code,
                  seconds=time.monotonic() - begin, rate_limit=response.headers.get('RateLimit'),
                  rate_policy=response.headers.get('RateLimit-Policy'), retry_after=response.headers.get('Retry-After'))
            return response
        except Exception as error:
            event(self.log, 'http_error', category=category, method=request.method, error_type=type(error).__name__)
            raise


def configure(root, log):
    # v0.36.2 is the pinned transport tested by this delivery. Do not patch
    # arbitrary newer SDK internals or install/replace Torch to satisfy it.
    import huggingface_hub
    assert huggingface_hub.__version__ == '0.36.2', 'Install only huggingface_hub==0.36.2 in the Kaggle setup cell'
    from huggingface_hub import configure_http_backend
    budget = Budget(root)
    configure_http_backend(backend_factory=lambda: BudgetSession(budget, log))
    return budget
