"""Verify metadata orientation, light UI and the additional video controls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import time
import json
from PySide6.QtWidgets import QApplication
import cv2
import numpy as np
from bootstrap import ROOT
from app import Station
from theme import apply_theme
from video import configure_capture, portrait_frame

def main():
    metadata = []
    for path in sorted((ROOT.parent/'Videos').glob('*.mp4')):
        raw = cv2.VideoCapture(str(path))
        angle = configure_capture(raw)
        ok, frame = raw.read()
        assert ok
        raw.release()
        portrait, applied = portrait_frame(frame, angle)
        reference = cv2.VideoCapture(str(path))
        assert reference.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, expected = reference.read()
        reference.release()
        assert ok and portrait.shape[0] > portrait.shape[1]
        np.testing.assert_array_equal(portrait, expected)
        metadata.append(dict(video=path.name, clockwise_rotation=applied, portrait_size=list(portrait.shape[:2])))
    app = QApplication([])
    apply_theme(app)
    window = Station()
    errors=[]
    window.worker.failed.connect(errors.append)
    window.show()
    def until(predicate, timeout=60):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            app.processEvents()
            if errors:
                raise RuntimeError(errors[-1])
            if predicate():
                return
            time.sleep(.01)
        raise TimeoutError('Portrait control check timed out')
    try:
        window.load_source(str(ROOT.parent/'Videos'/'Video3.mp4'))
        until(lambda: window.result is not None and not window.busy())
        assert window.result['analysis_rotation_degrees'] == 0
        assert not window.result['orientation_scores']
        assert window.result['video_rotation_clockwise'] == 90
        assert window.captured_rgb.shape[0] > window.captured_rgb.shape[1]
        window.step_frame(1)
        until(lambda: not window._seek_pending)
        assert window.stream.paused.is_set()
        before=window.frame_time_ms
        window.step_frame(1)
        until(lambda: not window._seek_pending)
        assert abs((window.frame_time_ms-before)-1000/window.video_fps)<3
        window.step_frame(-1)
        until(lambda: not window._seek_pending)
        assert abs(window.frame_time_ms-before)<3
        window.speed.setCurrentIndex(window.speed.findData(.25))
        assert window.stream.speed == .25
        window.loop.setChecked(True)
        assert window.stream.loop
        window.jump_video(5000)
        until(lambda: not window._seek_pending)
        assert abs(window.frame_time_ms-before-5000)<100
        window.inspect()
        until(lambda: window.result is not None and not window.busy())
        window.view_mode.setCurrentText('Original')
        assert window.viewer.processed is None
        window.view_mode.setCurrentText('Compare wipe')
        assert window.viewer.processed is not None and window.viewer.wipe == .5
        window.opacity.setValue(50)
        window.show_tyre.setChecked(False)
        window.show_tread.setChecked(False)
        actual = window.viewer.qimage(window.captured_rgb)
        assert actual == window.viewer.processed
        window.show_tyre.setChecked(True)
        window.show_tread.setChecked(True)
        window.view_mode.setCurrentText('Full overlay')
        window.opacity.setValue(26)
        fit_width=window.viewer.image_rect().width()
        window.zoom.setCurrentIndex(window.zoom.findData(2.))
        assert window.viewer.image_rect().width() == 2*fit_width
        window.zoom.setCurrentIndex(0)
        assert app.palette().window().color().lightness() > 200
        window.grab().save(str(ROOT/'results'/'portrait-light-workstation.png'))
        window.save_evidence()
        # Exercise the real end-to-start loop without relying on an inference result.
        window.speed.setCurrentIndex(window.speed.findData(2.))
        window.seek_video(window.duration_ms-2*1000/window.video_fps)
        until(lambda: not window._seek_pending)
        window.stream.paused.clear()
        until(lambda: (window.frame_time_ms or 0)<1000)
        report=dict(status='passed', orientation=metadata,
            checks=['file metadata equals OpenCV display orientation', 'portrait model input', 'no model rotation search',
                    'previous/next frame', 'speed control', 'five-second jump', 'loop across video end',
                    'original/overlay/wipe', 'mask toggles', 'overlay opacity', 'display zoom', 'light palette', 'evidence export'])
        (ROOT/'results'/'portrait-controls-check.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        window.live_enabled=False
        until(lambda: not window.busy())
        window.close()
        app.processEvents()

if __name__ == '__main__':
    main()
