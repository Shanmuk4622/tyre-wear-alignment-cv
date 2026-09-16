"""GPU UI checks on supplied portrait videos and evidence restoration."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import json
import time
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem
from app import Station
from theme import apply_theme
from bootstrap import ROOT


def main():
    app = QApplication([]); apply_theme(app)
    window = Station(); window.show()
    errors, completed, report = [], [], []
    window.worker.failed.connect(errors.append)
    def received(rgb, result, masks, compare, source):
        if result.get('generation') == window.generation:
            completed.append(result)
        if len(completed) >= 4 and window.stream:
            window.live_enabled = False
            window.stream.paused.set()
    window.worker.complete.connect(received)
    def until(condition, timeout=90):
        deadline = time.monotonic()+timeout
        while not condition():
            app.processEvents()
            assert not errors, errors
            assert time.monotonic() < deadline, 'UI timeout'
            time.sleep(.01)
    try:
        window.learned_mode.setCurrentIndex(window.learned_mode.findData('paired'))
        for video in sorted((ROOT.parent/'Videos').glob('*.mp4')):
            completed.clear()
            window.load_source(str(video))
            until(lambda: len(completed) >= 4 and not window.busy())
            learned = window.learned_frame
            assert set(learned['models']) == {'hrnet', 'matched'}
            assert learned['frame_size'] == window.result['frame_size']
            assert window.result['analysis_rotation_degrees'] == 0
            assert window.captured_rgb.shape[0] > window.captured_rgb.shape[1]
            count = learned['stable_frames']
            window.opacity.setValue(20); window.render_capture()
            assert window.learned_frame['stable_frames'] == count  # rendering is not another frame
            with_points = window.viewer.processed.copy()
            window.show_learned.setChecked(False)
            assert window.viewer.processed != with_points
            window.show_learned.setChecked(True)
            window.view_mode.setCurrentText('Original')
            assert window.viewer.processed is None
            window.view_mode.setCurrentText('Full overlay')
            # Seeking clears previous smoothing and cannot reuse a stale result.
            window.seek_video(1500)
            assert window.learned_frame is None
            until(lambda: not window._seek_pending)
            window.inspect()
            until(lambda: not window.busy() and window.learned_frame is not None)
            assert window.learned_frame['stable_frames'] <= 1
            window.explain_learned(); app.processEvents()
            window.learned_dialog.grab().save(str(ROOT/'results'/f'{video.stem}-learned-diagram.png'))
            window.learned_dialog.close()
            before = set((ROOT/'results').glob('*/inspection.json'))
            window.save_evidence()
            new = set((ROOT/'results').glob('*/inspection.json'))-before
            assert len(new) == 1
            saved = new.pop()
            record = json.loads(saved.read_text())
            assert record['learned_geometry'] == window.learned_frame
            assert (saved.parent/'learned-boundaries.png').exists()
            assert 'learned-boundaries.png' in (saved.parent/'card.html').read_text(encoding='utf-8')
            item = QListWidgetItem(); item.setData(Qt.UserRole, str(saved.parent))
            window.restore_capture(item)
            assert window.learned_frame == record['learned_geometry']
            window.grab().save(str(ROOT/'results'/f'{video.stem}-learned-workstation.png'))
            report.append(dict(video=video.name, flags=record['learned_geometry']['flags'],
                model_ms={n:r['inference_ms'] for n,r in record['learned_geometry']['models'].items()},
                full_pipeline_peak_mb=record['peak_vram_mb'], evidence=str(saved.parent)))
            print('PASS '+video.name, flush=True)
        window.learned_mode.setCurrentIndex(0)
        window.inspect(); until(lambda: not window.busy())
        assert window.learned_frame is None and not window.engine.learned.cache
        (ROOT/'results'/'learned-ui-check.json').write_text(json.dumps(dict(status='passed', videos=report,
            checks=['GPU paired video', 'portrait inputs', 'toggle', 'original view', 'seek resets', 'frozen explanation',
                    'JSON/PNG evidence', 'restoration', 'off releases model references']), indent=2), encoding='utf-8')
    finally:
        window.live_enabled = False
        until(lambda: not window.busy())
        window.close(); app.processEvents()


if __name__ == '__main__':
    main()
