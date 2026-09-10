"""Fault-inject the generated NB01A read cell; no network, uploads or real waits."""
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / 'notebooks/NB01A_Baseline_Recovery.ipynb').read_text())
source = next(''.join(c['source']) for c in nb['cells']
              if 'def hf_read(' in ''.join(c['source']))

class Failure(Exception):
    def __init__(self, status, headers=None, message='retry after 128 seconds'):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status, headers=headers or {})

def body_delay(message):
    match = re.search(r'retry after (\d+) seconds', message)
    return float(match[1]) + 2 if match else None

with tempfile.TemporaryDirectory() as temp:
    ns = dict(tl=SimpleNamespace(HF_REPO_DEFAULT='owner/repo', parse_retry_after=body_delay),
              sess=SimpleNamespace(stage_dir=temp, uploader=SimpleNamespace(token='test-only')))
    with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download') as download, patch('time.sleep') as sleep:
        api.return_value.repo_info.side_effect = [Failure(429), SimpleNamespace(sha='pinned')]
        exec(source, ns)
        assert sum(c.args[0] for c in sleep.call_args_list) >= 128
        assert api.return_value.repo_info.call_count == 2
        assert api.return_value.repo_info.call_args.kwargs['token'] == 'test-only'
        api.return_value.list_repo_files.assert_not_called()
        download.side_effect = [Failure(503, {'Retry-After': '150', 'RateLimit': '"api";r=0;t=160'}, ''), temp+'/table.csv']
        sleep.reset_mock()
        ns['pull']('table.csv')
        assert sum(c.args[0] for c in sleep.call_args_list) >= 160
        assert download.call_args.kwargs['token'] == 'test-only'
        assert download.call_args.kwargs['revision'] == 'pinned'
        for error in (Failure(404), KeyboardInterrupt()):
            sleep.reset_mock()
            def fail():
                raise error
            try:
                ns['hf_read']('fail-fast', fail)
            except type(error):
                pass
            else:
                raise AssertionError('Expected immediate failure')
            sleep.assert_not_called()
        try:
            ns['hf_read']('exhausted', lambda: (_ for _ in ()).throw(Failure(429)), attempts=2)
        except RuntimeError as exc:
            assert 'after 2 attempts' in str(exc)
        else:
            raise AssertionError('Retry budget must terminate')
print('PASS: 429 recovery, authenticated/pinned downloads, server delays, no inventory, fail-fast, Stop, bounded retries')
