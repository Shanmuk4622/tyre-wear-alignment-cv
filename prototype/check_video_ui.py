"""Exercise actual supplied clips, seeking, replay and stale-result rejection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import json
import time
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from app import Station
from theme import apply_theme
from bootstrap import ROOT

def main():
    app = QApplication([])
    apply_theme(app)
    window = Station()
    window.show()
    errors, completed, report = [], [], []
    window.show_error = errors.append
    window.worker.failed.connect(errors.append)
    window.worker.complete.connect(lambda rgb, result, masks, compare, source: completed.append(result))
    worker = window.worker
    def until(condition, timeout=90):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            app.processEvents()
            if errors:
                raise RuntimeError(errors[-1])
            if condition():
                return
            time.sleep(.01)
        raise TimeoutError(f'Video UI timeout: busy={window.busy()}, live={window.live_enabled}, '
            f'paused={window.stream.paused.is_set() if window.stream else None}, generation={window.generation}, '
            f'frame_ms={window.frame_time_ms}, result_ms={window.result.get("source_time_ms") if window.result else None}')
    try:
        for path in sorted((ROOT.parent/'Videos').glob('*.mp4')):
            start_count=len(completed)
            window.load_source(str(path))
            print(f'{path.name}: playing', flush=True)
            until(lambda: len(completed)>=start_count+7 and not window.busy())
            assert window.worker is worker
            assert window.rgb.shape[1] <= 1280
            assert window.viewer.processed is not None
            assert window.viewer.preview is not None
            assert set(window.geometry_frames) == set(window.masks)
            for name, geometry in window.geometry_frames.items():
                if 'mask' in geometry:
                    assert geometry['mask'].shape == window.captured_rgb.shape[:2]
                if geometry['stable']:
                    assert geometry['valid'] and geometry['count'] >= 3
            times=list(window.result_times)
            hz=(len(times)-1)/(times[-1]-times[0])
            # Seek while inference is in progress; the prior result must not
            # overwrite the newly sought frame, even though the file is unchanged.
            until(window.busy)
            window.timeline.setValue(500)
            window.seek_video()
            generation=window.generation
            print(f'{path.name}: seeking', flush=True)
            until(lambda: window.result is not None and window.result['generation']==generation and not window.busy())
            assert window.result['source_time_ms'] >= window.duration_ms*.45
            assert window.result['source_frame_size'][0] >= window.result['frame_size'][0]
            if not window.stream.paused.is_set():
                window.toggle_freeze()
            assert window.stream.paused.is_set()
            assert window.viewer.processed is not None
            window.grab().save(str(ROOT/'results'/f'{path.stem}-video-workstation.png'))
            stamp=window.result['source_time_ms']
            window.save_evidence()
            window.replay()
            print(f'{path.name}: replaying from {stamp}', flush=True)
            until(lambda: window.result is not None and (window.result.get('source_time_ms') or 0)<stamp and not window.busy())
            report.append(dict(video=path.name, measured_updates_per_second=round(hz,2),
                               seek_time_ms=stamp, persistent_worker=True, resized_frame=list(window.rgb.shape[:2])))
            print(json.dumps(report[-1]), flush=True)
            window.live_enabled=False
            until(lambda: not window.busy())
            window.stop_stream()
        (ROOT/'results'/'video-ui-check.json').write_text(json.dumps(dict(status='passed', videos=report,
            checks=['automatic live start', 'warmup hold', 'persistent GPU thread', 'bounded working resolution',
                    'same-frame overlays', 'source preview', 'seeking during inference', 'freeze retains overlay',
                    'evidence capture', 'replay']), indent=2))
    finally:
        window.live_enabled=False
        until(lambda: not window.busy())
        window.close()
        app.processEvents()

if __name__ == '__main__':
    main()
