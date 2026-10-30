"""Exercise the same catchable-stop marker path used by the notebook owner."""
import os
import subprocess
import sys
import time
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/pilot'
sys.path.insert(0, str(PILOT))
from common import atomic, read
from schemas import SCHEMAS
from store import Store
import acceptance

record = read(ROOT / 'tread_two/pilot_validation/LOCAL_VALIDATION.json')
folder = Path(record['fixture_directory'])
cfg = read(folder / 'CONFIG.json')
control = folder / 'marker_control'
control.mkdir(exist_ok=True)
cfg = dict(cfg, control_root=str(control))
case = folder / 'marker_pause'
case.mkdir(exist_ok=True)
atomic(case / 'CONFIG.json', cfg)
environment = dict(os.environ)
environment.pop('HF_TOKEN', None)
with (case / 'console.log').open('ab') as stream:
    child = subprocess.Popen([sys.executable,'-B','-u',str(PILOT/'worker.py'),'--config',str(case/'CONFIG.json'),'--folder',str(case)],
                              stdout=stream, stderr=subprocess.STDOUT, env=environment)
    started = time.monotonic()
    while child.poll() is None:
        if (case / 'POINTER.json').exists():
            unit = read(case / read(case / 'POINTER.json')['unit'])
            if unit['updates'] >= 1:
                atomic(control / 'STOP_REQUEST.json', dict(reason='same stop marker as catchable notebook signal'))
                break
        assert time.monotonic() - started < 180, 'Timed out waiting for first update'
        time.sleep(.02)
    child.wait(timeout=180)
assert child.returncode == 91, 'Stop must pause an owned worker without corrupting progress'
paused = Store(case, cfg['contract'], SCHEMAS)
assert 1 <= paused.state['updates'] < 4
paused_updates = paused.state['updates']
(control / 'STOP_REQUEST.json').unlink()
restored = acceptance.launch(cfg, case)
expected = Store(folder / 'actual/expected', cfg['contract'], SCHEMAS)
proof = acceptance.compare(expected, restored)
proof.update(stop_marker_path_tested=True, paused_at_completed_update=paused_updates,
             kernel_UI_Stop_tested=False, live_HF_flush_tested=False)
atomic(ROOT / 'tread_two/pilot_validation/STOP_VALIDATION.json', proof)
print('Catchable-stop marker, completed-update pause and fresh-process continuation: passed')
