"""Offscreen native UI integration check with real inference and a video fixture."""
import os
import sys
import faulthandler
faulthandler.dump_traceback_later(180, repeat=False)
if '--native' not in sys.argv:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
print('UI check: importing desktop runtime', flush=True)
import json
import time
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import cv2
import numpy as np
from app import Station
from bootstrap import ROOT
from theme import apply_theme


def main():
    print('UI check: creating application', flush=True)
    app = QApplication([])
    apply_theme(app)
    window = Station()
    print('UI check: window created', flush=True)
    errors = []
    window.show_error = errors.append
    window.show()
    def until(predicate, seconds=180):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise RuntimeError(errors[-1])
            if predicate():
                return
            time.sleep(.015)
        raise TimeoutError('UI did not reach expected state')
    summary = json.loads((ROOT/'results'/'smoke-check.json').read_text())
    window.load_source(summary['image'])
    assert window.rgb is not None and not window.save_btn.isEnabled()
    window.inspect(compare=True)
    print('UI check: real four-model inference', flush=True)
    until(lambda: window.result is not None and not window.busy())
    assert len(window.result['models']) == 4
    assert window.model_list.count() == 4 and window.overlay_select.count() == 3
    before = window.result['frame_sha256']
    window.note.setText('UI verification · same-frame comparison')
    window.save_evidence()
    window.restore_capture(window.tray.item(0))
    assert window.result['frame_sha256'] == before
    assert window.note.text() == 'UI verification · same-frame comparison'
    window.overlay_select.setCurrentIndex(2)
    app.processEvents()
    window.grab().save(str(ROOT/'results'/'station-preview.png'))
    # A real decoded video, generated from the same study frame for UI testing.
    video = ROOT/'.cache'/'ui-video.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (320, 240))
    assert writer.isOpened()
    fixture = cv2.resize(cv2.cvtColor(window.rgb, cv2.COLOR_RGB2BGR), (320, 240))
    for _ in range(25):
        writer.write(fixture)
    writer.release()
    window.load_source(str(video))
    until(lambda: window.result is not None and not window.busy() and not window.stream.paused.is_set())
    window.toggle_freeze()
    assert window.stream.paused.is_set()
    held = window.rgb.copy()
    until(lambda: True)
    np.testing.assert_array_equal(window.rgb, held)
    window.toggle_freeze()
    assert not window.stream.paused.is_set()
    window.live_btn.setChecked(True)
    window.toggle_live()
    until(lambda: window.result is not None)
    until(lambda: not window.busy())
    window.stop_stream()
    until(lambda: not window.busy())
    window.close()
    app.processEvents()
    (ROOT/'results'/'ui-check.json').write_text(json.dumps(dict(status='passed', platform=app.platformName(),
        checks=['image load', 'real four-model GPU worker', 'comparison rows', 'overlay switching',
                'evidence save/restore', 'native screenshot', 'video decoding', 'freeze/resume', 'live inference', 'shutdown'],
        camera='Physical webcam not exercised'), indent=2), encoding='utf-8')
    print('PASS: desktop GPU worker, evidence restore, overlay, video freeze/resume and live inference. Physical webcam not tested.')
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
