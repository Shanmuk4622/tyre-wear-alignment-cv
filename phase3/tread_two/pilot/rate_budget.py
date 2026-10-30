"""Shared local HTTP budgets plus server backoff, including SDK transport calls.

Cross-host account traffic cannot be atomically counted by this session; server
headers remain authoritative. Only safe request categories/status are logged.
"""
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from filelock import FileLock
from common import atomic, event, read


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
        window, limit = {'api': (3600, 96), 'resolver': (300, 400), 'transfer': (300, 1000)}[bucket]
        started = time.monotonic()
        while True:
            now = time.time()
            with self.lock:
                data = read(self.path) if self.path.exists() else {}
                state = data.setdefault(bucket, dict(times=[], blocked_until=0))
                state['times'] = [value for value in state['times'] if value > now - window]
                delay = max(0., state['blocked_until'] - now,
                            state['times'][0] + window - now if len(state['times']) >= limit else 0.)
                if delay <= 0:
                    state['times'].append(now)
                    atomic(self.path, data)
                    return
            if time.monotonic() - started + delay > 300:
                raise TimeoutError('HTTP budget/backoff deferred; local completed state remains safe')
            print('HF request budget/backoff wait:', round(delay, 1), 'seconds', flush=True)
            time.sleep(min(30., delay))

    def observe(self, bucket, response):
        headers = response.headers
        rate = headers.get('RateLimit', '')
        delay = 0.
        if response.status_code == 429 or re.search(r'\br=0(?:\D|$)', rate):
            try:
                delay = float(headers.get('Retry-After', 0))
            except ValueError:
                from email.utils import parsedate_to_datetime
                delay = max(0., parsedate_to_datetime(headers['Retry-After']).timestamp() - time.time())
            match = re.search(r'\bt=(\d+)', rate)
            delay = max(delay, float(match[1]) if match else 0.)
        if delay:
            with self.lock:
                data = read(self.path) if self.path.exists() else {}
                state = data.setdefault(bucket, dict(times=[], blocked_until=0))
                state['blocked_until'] = max(state['blocked_until'], time.time() + delay)
                atomic(self.path, data)


class BudgetSession(requests.Session):
    def __init__(self, budget, log):
        super().__init__()
        self.budget, self.log = budget, Path(log)
        self.trust_env = False

    def send(self, request, **kwargs):
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
