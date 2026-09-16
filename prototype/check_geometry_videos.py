"""Bounded real-video geometry smoke check; pause after four results."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import time
from PySide6.QtWidgets import QApplication
from app import Station
from bootstrap import ROOT

app = QApplication([])
window = Station()
completed = []
errors = []

def received(rgb, result, masks, compare, source):
    completed.append(result)
    if len(completed) >= 4:
        window.live_enabled = False
        window.stream.paused.set()

window.worker.complete.connect(received)
window.worker.failed.connect(errors.append)
try:
    for path in sorted((ROOT.parent/'Videos').glob('*.mp4')):
        completed.clear()
        window.load_source(str(path))
        deadline = time.monotonic()+75
        while len(completed) < 4 or window.busy():
            app.processEvents()
            assert not errors, errors
            assert time.monotonic() < deadline, path.name
            time.sleep(.01)
        assert set(window.geometry_frames) == set(window.masks)
        assert window.viewer.processed is not None
        fit = window.current_edge_fit()
        assert fit is not None and fit['mode'] == 'boundary'
        from geometry_comparison import compare_fits
        comparison, summary = compare_fits(window.captured_rgb, window.masks)
        assert set(comparison) == set(window.masks)
        assert all(set(methods) == {'boundary', 'rim'} for methods in comparison.values())
        if fit['valid']:
            assert len(fit['lines']) == 3
        for g in window.geometry_frames.values():
            assert not g['stable'] or (g['valid'] and g['count'] >= 3)
        print(f"PASS {path.name}: four GPU results, same-frame geometry overlay; edges: {fit['reason']}", flush=True)
        window.stop_stream()
finally:
    window.live_enabled = False
    window.close()
    app.processEvents()
