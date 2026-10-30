"""Notebook-only lease wait; the 27 frozen experiment sources stay unchanged."""
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from filelock import FileLock

LEASE_ERROR = ('RuntimeError: Another live gate owns this namespace; stop it '
               'cleanly or wait for its 45-minute lease expiry.')


def preflight_lease_busy(text):
    """Retry only the observed preflight claim failure, never a training error."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and lines[-1] == LEASE_ERROR
                and 'in preflight' in text and 'in claim' in text
                and 'Measured paired condition:' not in text)


def _worker_progress(work, seen):
    for path in (work / 'revision2_runs').glob('**/console.log'):
        # Logs can grow during a long run; read only their tail.
        with open(path, 'rb') as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            lines = stream.read().decode(errors='replace').splitlines()
        if not lines or seen.get(str(path)) == lines[-1]:
            continue
        seen[str(path)] = lines[-1]
        label = (path.parent.name if path.parent.name in ('baseline', 'shared_projection')
                 else path.parent.parent.name + '/' + path.parent.name)
        print(label + ': ' + lines[-1], flush=True)


def _stop_child(child):
    if child.poll() is not None:
        return
    print('Stop requested: preserving committed units and attempting HF flush. '
          'Keep the session open.', flush=True)
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for index in range(60):
        try:
            child.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            if index % 4 == 0:
                print('Still flushing or respecting server delay; do not launch '
                      'a duplicate writer.', flush=True)
    print('Coordinator is still flushing; forced termination can lose unsent work.',
          flush=True)


def launch_paired(work, retry_seconds=300, max_lease_wait_seconds=5400):
    """Recheck occupied preflight leases every five minutes, for up to 90 minutes.

    Each retry calls the unchanged coordinator, which performs its normal CAS
    claim and compatibility checks. This launcher never edits remote ownership.
    A live writer that renews its lease remains protected. Stop cancels waiting.
    """
    assert retry_seconds > 0 and max_lease_wait_seconds >= 0
    work = Path(work)
    logs = work / 'launcher_logs'
    logs.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    history = logs / (run_id + '.jsonl')

    def record(event, **values):
        with open(history, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, time_utc=time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    waited = 0.0
    attempt = 0
    child = None
    seen = {}
    # Includes the wait interval; a second launch cell cannot start another waiter.
    with FileLock(str(work / 'NB05_LAUNCHER.lock'), timeout=1):
        try:
            while True:
                attempt += 1
                log = logs / (run_id + f'_attempt_{attempt:02d}.log')
                record('coordinator_start', attempt=attempt, log=str(log))
                with open(log, 'wb', buffering=0) as output, open(log, 'rb') as reader:
                    child = subprocess.Popen(
                        [sys.executable, '-B', '-u', str(work / 'revision2/coordinator.py'),
                         '--publish'], stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    while True:
                        try:
                            code = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            code = None
                        while True:
                            chunk = reader.read(65536)
                            if not chunk:
                                break
                            print(chunk.decode(errors='replace'), end='', flush=True)
                        _worker_progress(work, seen)
                        if code is not None:
                            break
                record('coordinator_exit', attempt=attempt, exit_code=code)
                if code == 0:
                    record('completed', lease_wait_seconds=waited)
                    return
                with open(log, 'rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 32768))
                    tail = stream.read().decode(errors='replace')
                if not preflight_lease_busy(tail):
                    raise RuntimeError('NB05 stopped. Inspect the error above and ' + str(log)
                                       + '. Committed progress is preserved; this error '
                                       'is not retried automatically.')
                remaining = max_lease_wait_seconds - waited
                if remaining <= 0:
                    record('lease_wait_exhausted', lease_wait_seconds=waited)
                    raise RuntimeError('The preflight namespace is still occupied after '
                                       f'{waited / 60:g} minutes of waiting. Stop the other '
                                       'session cleanly, or let its lease expire, then rerun '
                                       'this notebook. No owner or checkpoint was overwritten.')
                delay = min(float(retry_seconds), remaining)
                record('preflight_lease_wait', attempt=attempt, seconds=delay,
                       cumulative_wait_seconds=waited)
                print('HF preflight lease is occupied. Waiting '
                      f'{delay / 60:g} minutes before a safe retry '
                      f'({waited / 60:g} of {max_lease_wait_seconds / 60:g} minutes waited). '
                      'An abandoned lease expires within 45 minutes of its last renewal; '
                      'an active session may renew it. Stop cancels this wait. '
                      'No lock is deleted and no training is started while occupied.', flush=True)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
                waited += delay
        except KeyboardInterrupt:
            record('user_stop', attempt=attempt, lease_wait_seconds=waited)
            if child is not None:
                _stop_child(child)
            raise
