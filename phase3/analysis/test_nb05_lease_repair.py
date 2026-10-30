"""Offline regression of the supplied NB05 failure and notebook-only retry."""
import ast
import base64
import contextlib
import gzip
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

assert Path(sys.prefix).name == 'cv_conda'
import nbformat
import nb05_launcher as launcher

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'revision2_validation/nb05_repair'
FAILED = OUT / 'c49292d397d0'
ORIGINAL = json.loads((FAILED / 'INSPECTION.json').read_text(encoding='utf-8'))
TRACE = next(item['text'] for item in ORIGINAL['streams'] if 'in preflight' in item['text'])


class Clock:
    def __init__(self, interrupt=False):
        self.elapsed = 0.0
        self.interrupt = interrupt

    strftime = staticmethod(time.strftime)
    gmtime = staticmethod(time.gmtime)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        if self.interrupt:
            raise KeyboardInterrupt
        self.elapsed += seconds


class Child:
    pid = 123456789

    def __init__(self, code, interrupt=False):
        self.returncode = code
        self.interrupt = interrupt

    def wait(self, timeout):
        if self.interrupt:
            raise KeyboardInterrupt
        return self.returncode

    def poll(self):
        return None if self.interrupt else self.returncode


def exercise(folder, responses, max_wait=600, clock=None, expect=None, interrupt=False):
    folder.mkdir(parents=True)
    called = []
    stopped = []
    clock = clock or Clock()

    def spawn(command, **options):
        index = len(called)
        code, text = responses[min(index, len(responses) - 1)]
        called.append(command)
        options['stdout'].write(text.encode())
        assert options['start_new_session'] and options['stderr'] == launcher.subprocess.STDOUT
        return Child(code, interrupt)

    with patch.object(launcher, 'time', clock), patch.object(launcher.subprocess, 'Popen', spawn), \
            patch.object(launcher, '_stop_child', lambda child: stopped.append(child.pid)), \
            contextlib.redirect_stdout(io.StringIO()):
        try:
            launcher.launch_paired(folder, retry_seconds=300, max_lease_wait_seconds=max_wait)
        except BaseException as exc:
            assert expect and isinstance(exc, expect), repr(exc)
        else:
            assert expect is None, 'Expected error or cancellation'
    events = [json.loads(line) for path in (folder / 'launcher_logs').glob('*.jsonl')
              for line in path.read_text(encoding='utf-8').splitlines()]
    return called, clock, events, stopped


def payload(notebook):
    result = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'b64decode' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                result.update(json.loads(gzip.decompress(base64.b64decode(node.args[0].value))))
    return {name: base64.b64decode(value) for name, value in result.items()}


def main():
    checks = []
    assert launcher.preflight_lease_busy(TRACE)
    checks.append('actual_saved_preflight_lease_failure_recognized')
    for text in ('RuntimeError: CUDA out of memory.', TRACE + '\nRuntimeError: different failure',
                 'Measured paired condition: baseline\n' + TRACE,
                 launcher.LEASE_ERROR):
        assert not launcher.preflight_lease_busy(text)
    checks.append('unrelated_training_and_incomplete_tracebacks_not_retried')
    with tempfile.TemporaryDirectory(prefix='launcher_', dir=OUT) as temp:
        temp = Path(temp)
        sys.path.insert(0, str(ROOT / 'training'))
        from test_training_evidence import Service, hub
        import evidence_hub
        service = Service()
        first = hub(service, temp, 'first')
        second = hub(service, temp, 'second')
        with patch.object(evidence_hub.time, 'time', return_value=1000):
            first.claim()
            before = service.commits
            try:
                second.claim()
            except RuntimeError as exc:
                assert str(exc) == launcher.LEASE_ERROR.removeprefix('RuntimeError: ')
            else:
                raise AssertionError('Unexpired writer was replaced')
            assert service.commits == before
        checks.append('original_frozen_claim_reproduces_live_lease_rejection_without_commit')
        with patch.object(evidence_hub.time, 'time', return_value=3701):
            second.claim()
        assert json.loads(service.files[second.destination + '/OWNER.json'])['owner'] == 'second'
        checks.append('unchanged_CAS_claim_acquires_only_after_existing_45_minute_lease_expires')
        called, clock, events, _ = exercise(temp / 'release', [(1, TRACE), (0, 'completed\n')])
        assert len(called) == 2 and clock.elapsed == 300
        assert any(e['event'] == 'preflight_lease_wait' for e in events) and events[-1]['event'] == 'completed'
        assert called[0] == called[1]
        checks.append('occupied_then_released_lease_retries_identical_coordinator_after_300_seconds')
        called, clock, events, _ = exercise(temp / 'renewed', [(1, TRACE)], expect=RuntimeError)
        assert len(called) == 3 and clock.elapsed == 600 and events[-1]['event'] == 'lease_wait_exhausted'
        checks.append('continually_renewed_lease_stops_at_wait_budget_without_takeover')
        called, clock, events, stopped = exercise(temp / 'cancel_wait', [(1, TRACE)],
            clock=Clock(interrupt=True), expect=KeyboardInterrupt)
        assert len(called) == 1 and clock.elapsed == 0 and events[-1]['event'] == 'user_stop'
        assert not stopped or stopped == [Child.pid]
        checks.append('stop_cancels_wait_before_any_retry')
        called, clock, events, stopped = exercise(temp / 'cancel_child', [(0, 'running\n')],
            expect=KeyboardInterrupt, interrupt=True)
        assert len(called) == 1 and stopped == [Child.pid]
        checks.append('stop_during_coordinator_invokes_existing_flush_signal_path')
        for index, text in enumerate(('RuntimeError: incompatible contract\n',
                                      'Measured paired condition: baseline\n' + TRACE)):
            called, clock, _, _ = exercise(temp / f'no_retry_{index}', [(1, text)], expect=RuntimeError)
            assert len(called) == 1 and clock.elapsed == 0
        checks.append('nonlease_failure_or_measured_start_exits_once_without_hidden_restart')
        called, clock, events, _ = exercise(temp / 'success', [(0, 'completed\n')])
        assert len(called) == 1 and clock.elapsed == 0 and events[-1]['event'] == 'completed'
        checks.append('successful_run_has_no_retry_or_wait')
        called, clock, events, _ = exercise(temp / 'no_budget', [(1, TRACE)], max_wait=0, expect=RuntimeError)
        assert len(called) == 1 and clock.elapsed == 0
        checks.append('zero_budget_reports_occupied_namespace_without_sleep')
        folder = temp / 'locked'
        folder.mkdir()
        with launcher.FileLock(str(folder / 'NB05_LAUNCHER.lock'), timeout=1):
            try:
                launcher.launch_paired(folder)
            except Exception as exc:
                from filelock import Timeout
                assert isinstance(exc, Timeout)
            else:
                raise AssertionError('Duplicate local launcher acquired the lock')
        checks.append('duplicate_local_launcher_fenced_during_wait_or_execution')
        # Exercise real subprocess stdout/file transport, not only process doubles.
        folder = temp / 'real_process'
        (folder / 'revision2').mkdir(parents=True)
        script = ('from pathlib import Path\nimport sys\n'
                  'p=Path(__file__).parent.parent/"attempt_count.txt"\n'
                  'n=int(p.read_text()) if p.exists() else 0\n'
                  'p.write_text(str(n+1))\n'
                  f'print({TRACE!r} if n==0 else "completed",flush=True)\n'
                  'sys.exit(1 if n==0 else 0)\n')
        (folder / 'revision2/coordinator.py').write_text(script, encoding='utf-8')
        clock = Clock()
        with patch.object(launcher, 'time', clock), contextlib.redirect_stdout(io.StringIO()):
            launcher.launch_paired(folder, retry_seconds=300, max_lease_wait_seconds=600)
        assert (folder / 'attempt_count.txt').read_text() == '2' and clock.elapsed == 300
        logs = sorted((folder / 'launcher_logs').glob('*.log'))
        assert len(logs) == 2 and TRACE in logs[0].read_text(encoding='utf-8')
        assert 'completed' in logs[1].read_text(encoding='utf-8')
        checks.append('real_subprocess_output_capture_and_retry_transport_passed_in_fresh_workspace')
    notebook_path = ROOT / 'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert all(not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == 'code')
    archived = nbformat.read(FAILED / 'executed.ipynb', as_version=4)
    assert payload(notebook) == payload(archived)
    assert len(payload(notebook)) == 27
    assert all(data == (ROOT / name).read_bytes() for name, data in payload(notebook).items())
    checks.append('all_27_frozen_experiment_sources_exactly_match_failed_notebook_and_readable_sources')
    source = (ROOT / 'analysis/nb05_launcher.py').read_text(encoding='utf-8')
    assert source in notebook.cells[1].source
    checks.append('clean_valid_notebook_embeds_exact_tested_launcher_source')
    delivery = json.loads((ROOT / 'revision2_validation/DELIVERY.json').read_text())
    previous = json.loads((FAILED / 'ORIGINAL_DELIVERY.json').read_text())
    assert delivery['sources'] == previous['sources']
    assert hashlib.sha256((FAILED / 'executed.ipynb').read_bytes()).hexdigest() == ORIGINAL['sha256']
    checks.append('original_failure_archived_and_all_source_contract_hashes_preserved')
    # Existing actual-model proofs remain bound to the identical runtime source set.
    validation = json.loads((ROOT / 'revision2_validation/VALIDATION.json').read_text())
    assert validation['source_hashes'] == delivery['sources']
    checks.append('previous_real_model_recovery_proofs_remain_bound_to_unchanged_sources')
    result = dict(status='passed', date='2026-10-29', checks=checks, check_count=len(checks),
                  notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                  failed_notebook_sha256=ORIGINAL['sha256'],
                  launcher_sha256=delivery['launcher_sha256'], experiment_sources_unchanged=27,
                  network_requests=0, live_HF_lease_retest=False, Kaggle_execution_retest=False,
                  model_updates_executed=0, source_checkpoint_compatibility_preserved=True,
                  wait_seconds=300, max_wait_seconds=5400)
    (OUT / 'REGRESSION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
