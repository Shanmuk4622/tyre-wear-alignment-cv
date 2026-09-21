"""Offline export checks: sampling, duration, rendering, and failure cleanup."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
from PySide6.QtWidgets import QApplication
import tempfile
from pathlib import Path
import cv2
import numpy as np
from media_export import export_video


class ProbeEngine:
    def __init__(self, fail=False):
        self.levels = []
        self.fail = fail

    def inspect(self, rgb, *args, **kwargs):
        self.levels.append(float(rgb.mean()))
        if self.fail:
            raise RuntimeError('test failure')
        mask = np.zeros(rgb.shape[:2], bool)
        mask[20:-20, 30:-30] = True
        return {'assist_used': False}, {'test': np.array([mask, mask])}


def main():
    from app import Station
    from edge_diagram import EdgeDiagram
    from edge_geometry import edge_fit
    app = QApplication([])
    with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'.cache') as tmp:
        folder = Path(tmp)
        for fps in (30, 5):
            source = folder/f'input-{fps}.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), fps, (120, 160))
            assert writer.isOpened()
            for i in range(fps):
                writer.write(np.full((160, 120, 3), 30+i*5, np.uint8))
            writer.release()
            job = dict(source=str(source), destination=str(folder/f'out-{fps}.mp4'), classifier='test', region='test',
                       overlay=None, threshold=.25, assist=False, learned='off', opacity=.5,
                       layers=[True, True], geometry=True, edges='boundary', show_learned=False)
            engine = ProbeEngine()
            assert export_video(job, engine) == 10
            np.testing.assert_allclose(engine.levels, [30+int(i*fps/10)*5 for i in range(10)], atol=3)
            cap = cv2.VideoCapture(job['destination'])
            assert cap.get(cv2.CAP_PROP_FPS) == 10
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 10
            ok, frame = cap.read()
            cap.release()
            assert ok and frame.shape[:2] == (160, 120)
            assert frame.std() > 10, 'Overlay must change the source'
            job['destination'] = str(folder/'failure.mp4')
            try:
                export_video(job, ProbeEngine(fail=True))
                raise AssertionError('Failure not propagated')
            except RuntimeError as exc:
                assert str(exc) == 'test failure'
            assert not Path(job['destination']).exists()
            assert not Path(job['destination']).with_suffix('.partial.mp4').exists()
        window = Station(device='cpu')
        try:
            window.resize(1280, 760)
            window.show()
            app.processEvents()
            assert window.height() <= 760, 'Controls must not force an oversized window'
            window.left_scroll.ensureWidgetVisible(window.export_btn)
            app.processEvents()
            position = window.export_btn.mapTo(window.left_scroll.viewport(), window.export_btn.rect().center())
            assert window.left_scroll.viewport().rect().contains(position)
            rgb = np.full((160, 120, 3), 45, np.uint8)
            marked = rgb.copy()
            marked[:, :, 1] = 200
            window.captured_rgb = rgb
            window.result_source = 'sample.png'
            window.viewer.set_frame(rgb, marked)
            window.download_path = lambda suffix, source: folder/('shown'+suffix)
            for mode in ('Full overlay', 'Original', 'Compare wipe'):
                window.viewer.mode = mode
                window.viewer.wipe = .3
                window.download_frame()
                saved = cv2.cvtColor(cv2.imread(str(folder/'shown.png')), cv2.COLOR_BGR2RGB)
                expected = rgb.copy() if mode != 'Full overlay' else marked.copy()
                if mode == 'Compare wipe':
                    expected[:, 36:] = marked[:, 36:]
                np.testing.assert_array_equal(saved, expected)
            mask = np.ones(rgb.shape[:2], bool)
            fit = edge_fit(rgb, mask)
            assert not fit['valid']
            diagram = EdgeDiagram(rgb, fit, 'Test', mask=mask)
            assert 'FIT WITHHELD' in diagram.fit_notice.text()
            diagram.close()
            # Exercise the actual independent process, then cancel. The preview
            # and its source must remain untouched; partial files must vanish.
            window.source = str(source)
            window.download_video()
            assert window.export_process is not None
            assert window.viewer.original.width() == 120
            window.cancel_export()
            import time
            deadline = time.monotonic()+15
            while window.export_process is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.02)
            assert window.export_process is None
            assert 'cancelled' in window.export_status.text()
            assert not (folder/'shown.partial.mp4').exists()
        finally:
            window.close()
            app.processEvents()
    print('PASS: 30/5 fps sampling to 10 fps, duration, overlays, failed-export cleanup, exact PNG views, withheld-fit diagram, background cancellation')


def real_check():
    """Short real-model export via the same UI process used by the operator."""
    import time
    from app import Station
    from theme import apply_theme
    from video import configure_capture, portrait_frame
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    folder = bootstrap.ROOT/'results'/'export-check'
    folder.mkdir(parents=True, exist_ok=True)
    source = next((bootstrap.ROOT.parent/'Videos').glob('*.mp4'))
    cap = cv2.VideoCapture(str(source))
    rotation = configure_capture(cap)
    writer = None
    clip = folder/'short-portrait.avi'
    try:
        for i in range(7):
            ok, frame = cap.read()
            assert ok
            if i % 3:
                continue
            frame, _ = portrait_frame(frame, rotation)
            frame = cv2.resize(frame, (360, 640))
            if writer is None:
                writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10., (360, 640))
            writer.write(frame)
    finally:
        cap.release()
        if writer:
            writer.release()
    window = Station()
    window.show()
    errors = []
    window.worker.failed.connect(errors.append)

    def wait_for(condition, seconds=180):
        deadline = time.monotonic()+seconds
        while not condition():
            app.processEvents()
            assert not errors, errors
            assert time.monotonic() < deadline, window.export_status.text()
            time.sleep(.01)

    try:
        window.load_source(str(clip))
        wait_for(lambda: window.result is not None and not window.busy())
        window.live_enabled = False
        window.stream.paused.set()
        window.view_mode.setCurrentText('Full overlay')
        # Never overwrite an earlier run's result during export.
        window.download_path = lambda suffix, source: folder/(f'real-overlay-{time.time_ns()}'+suffix)
        window.download_frame()
        window.download_video()
        destination = window.export_destination
        # Continue interacting with the station while the separate CPU job runs.
        window.opacity.setValue(25)
        window.explain_edges()
        wait_for(lambda: window.export_process is None)
        assert 'Saved to Downloads' in window.export_status.text(), window.export_status.text()
        cap = cv2.VideoCapture(str(destination))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert cap.get(cv2.CAP_PROP_FPS) == 10
        ok, frame = cap.read()
        cap.release()
        assert ok and frame.shape[0] > 640 and frame.shape[1] == 360
        cv2.imwrite(str(folder/'exported-frame.png'), frame)
        window.grab().save(str(folder/'station.png'))
        window.edge_dialog.grab().save(str(folder/'edge-diagram.png'))
        print('PASS: real GPU inspection + independent CPU export, paired learned models, 3-frame portrait MP4, PNG, controls responsive; '+str(destination))
    finally:
        if window.export_process is not None:
            window.cancel_export()
            wait_for(lambda: window.export_process is None, 15)
        wait_for(lambda: not window.busy())
        window.close()
        app.processEvents()


if __name__ == '__main__':
    import sys
    real_check() if '--real' in sys.argv else main()
