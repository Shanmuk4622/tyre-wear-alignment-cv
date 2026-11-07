"""Exercise both native tabs with existing images and a local video fixture."""
import argparse
import os
from pathlib import Path
import time
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda:0'))
parser.add_argument('--native', action='store_true')
args = parser.parse_args()
if not args.native:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
proof = Path(__file__).resolve().parents[1] / 'workstation_validation' / f'unified_{uuid.uuid4().hex[:8]}'
os.environ['TREAD_PHASE3_BENCH_ROOT'] = str(proof/'model_bench')

from app import Station
from imaging import fingerprint
from legacy_bench import BENCH, ORIGINAL, load
from paths import PHASE3, read, sha, write_json
from theme import apply
from workspace import Workspace
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication([])
    apply(app)
    proof.mkdir(parents=True)
    original = read(PHASE3 / 'workstation_validation/ORIGINAL_BASELINE.json')
    custom = Station(args.device, capture_root=proof/'captures', session_root=proof/'sessions')
    window = Workspace(custom, args.device)
    window.resize(1440, 900)
    window.show()
    checks = []
    errors = []

    def passed(message):
        checks.append(message)
        print('PASS:', message, flush=True)

    def until(predicate, message, timeout=160):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if errors or custom.errors:
                raise RuntimeError(errors or custom.errors)
            if predicate():
                return
            time.sleep(.02)
        raise TimeoutError(message)

    try:
        until(lambda: custom.model_ready, 'TREAD-ONE model did not initialize')
        assert custom.sample_rows, 'Existing study input is required'
        custom.load_example()
        custom.inspect_btn.click()
        until(lambda: custom.result is not None, 'TREAD-ONE inspection')
        frame = custom.rgb.copy()
        custom.view.setCurrentText('Signed evidence')
        custom.note.setText('Phase 3 integration check')
        custom.capture_btn.click()
        until(lambda: custom.last_export is not None and not custom.exporter.isRunning(), 'CSV capture')
        custom.restore_capture(custom.last_export)
        assert fingerprint(custom.rgb) == fingerprint(frame)
        assert custom.note.text() == 'Phase 3 integration check'
        window.grab().save(str(proof/'tread_one.png'))
        passed('TREAD-ONE real prediction, signed evidence and full CSV restore')

        from PySide6.QtWidgets import QPushButton
        controls = {widget.text(): widget for widget in custom.findChildren(QPushButton)}
        controls['Compare boundaries and rim…'].click()
        comparison = custom.comparison_dialog
        assert comparison.isVisible() and list(comparison.fits) == ['TREAD-ONE']
        assert np.array_equal(comparison.rgb, frame)
        comparison.close()
        controls['Wheel alignment…'].click()
        dialog = custom.alignment_dialog
        assert dialog.isVisible() and window.bench is None and window.tabs.currentIndex() == 0
        dialog.capture()
        assert np.array_equal(dialog.rgb, frame) and dialog.result is None
        assert 'camera calibration' in dialog.status.text()
        targets_fixture = ORIGINAL / 'results' / 'alignment-synthetic-targets.png'
        if targets_fixture.is_file():
            dialog.live.setChecked(True)
            custom.load_source(str(targets_fixture))
            dialog.tick()
            assert dialog.result is None and 'Waiting for an analysed' in dialog.preview.text()
            dialog.live.setChecked(False)
            dialog.set_profile(dict(version=1, board=load('alignment').BOARD,
                                    camera_label='Synthetic test camera', image_size=[1280, 960],
                                    camera_matrix=[[1200., 0., 640.], [0., 1200., 480.], [0., 0., 1.]],
                                    distortion=[0.] * 5, rms_px=0., views=12))
            dialog.confirm.setChecked(True)
            dialog.capture()
            assert abs(dialog.result['camber_deg'] - 2) < .7 and abs(dialog.result['toe_in_deg'] - 1) < .7
            dialog.save_result()
            evidence = next((BENCH/'results'/'alignment').glob('*/alignment.json'))
            assert read(evidence)['setup_confirmed']
            passed('TREAD-ONE measures and saves the existing synthetic alignment target fixture')
            custom.restore_capture(custom.last_export)
        dialog.close()
        app.processEvents()
        passed('Direct TREAD-ONE alignment and rim tools use the displayed native frame')

        window.transfer.click()
        assert not window.retry_bench.isVisible(), window.bench_status.text()
        until(lambda: window.bench is not None, 'Embedded bench did not open')
        bench = window.bench
        bench.worker.failed.disconnect(bench.show_error)
        bench.worker.failed.connect(errors.append)
        assert window.tabs.currentIndex() == 1 and bench.window() is window
        assert np.array_equal(bench.rgb, frame)
        assert load('app').ROOT == BENCH and load('evidence').ROOT == BENCH
        assert load('alignment_ui').ROOT == BENCH and load('geometry_comparison').ROOT == BENCH
        assert load('registry').checkpoint('mobilenetv4').is_relative_to(ORIGINAL)
        assert load('phase2_adapter').enabled()
        bench.learned_mode.setCurrentIndex(bench.learned_mode.findData('paired'))
        bench.compare_btn.click()
        until(lambda: bench.result is not None and not bench.busy(), 'Four-model comparison')
        assert len(bench.result['models']) == 4 and np.array_equal(bench.captured_rgb, frame)
        assert bench.result.get('learned_geometry', {}).get('models'), 'Learned geometry missing'
        passed('Same RGB handoff, four original models and paired learned geometry')
        bench.note.setText('Model bench integration check')
        bench.save_btn.click()
        entry = bench.tray.item(0)
        capture = Path(entry.data(256))
        assert capture.is_relative_to(BENCH) and (capture/'card.html').is_file()
        bench.restore_capture(entry)
        assert bench.note.text() == 'Model bench integration check'
        passed('Original HTML evidence capture and restore under Phase 3')
        window.grab().save(str(proof/'model_bench.png'))

        bench.open_alignment()
        assert bench.alignment_dialog.isVisible()
        bench.alignment_dialog.close()
        targets = load('alignment').export_targets(proof/'targets')
        assert targets.is_file()
        bench.compare_geometry()
        assert bench.comparison_dialog.isVisible() and len(bench.masks) >= 2
        from PySide6.QtWidgets import QDialog
        for dialog in bench.findChildren(QDialog):
            dialog.close()
        passed('Calibrated alignment dialog, printable targets and boundary comparison')

        window.transfer.click()
        assert window.tabs.currentIndex() == 0 and np.array_equal(custom.rgb, frame)
        assert custom.result is None
        passed('Reverse RGB handoff clears the prior prediction')

        video = proof / 'portrait.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 6, (192, 256))
        assert writer.isOpened()
        pixels = cv2.cvtColor(cv2.resize(frame, (192, 256)), cv2.COLOR_RGB2BGR)
        for _ in range(8):
            writer.write(pixels)
        writer.release()
        window.tabs.setCurrentIndex(1)
        bench.learned_mode.setCurrentIndex(0)
        bench.load_source(str(video))
        until(lambda: bench.rgb is not None and bench.stream is not None and not bench.busy(),
              'Embedded video playback')
        if not bench.stream.paused.is_set():
            bench.toggle_freeze()
        before = bench.frame_time_ms
        bench.step_frame(1)
        until(lambda: not bench._seek_pending and bench.frame_time_ms > before, 'Embedded frame step')
        bench.inspect_btn.click()
        until(lambda: not bench.busy() and bench.captured_rgb is not None, 'Inspection of the stepped frame')
        time_ms = bench.result['source_time_ms']
        window.transfer.click()
        assert np.array_equal(custom.rgb, bench.captured_rgb)
        assert custom.source['timestamp_seconds'] == time_ms / 1000
        window.tabs.setCurrentIndex(1)
        bench.download_path = lambda suffix, _: proof/f'exported{suffix}'
        bench.download_frame()
        assert (proof/'exported.png').is_file()
        bench.export_fps.setValue(3)
        bench.download_video()
        until(lambda: bench.export_process is None, 'Owned video export', timeout=240)
        assert (proof/'exported.mp4').stat().st_size > 0, bench.export_status.text()
        check_video = cv2.VideoCapture(str(proof/'exported.mp4'))
        try:
            assert check_video.get(cv2.CAP_PROP_FPS) == 3
            assert check_video.get(cv2.CAP_PROP_FRAME_COUNT) == 4
        finally:
            check_video.release()
        passed('Recorded-video frame stepping, PNG and real full-video MP4 export')
        for filename, expected in original.items():
            assert sha(PHASE3.parent/filename) == expected, filename
        passed(f'{len(original)} original source and model files retained byte for byte')
    finally:
        if window.bench is not None and window.bench.export_process is not None:
            window.bench.cancel_export()
        deadline = time.monotonic() + 15
        while not window.close():
            app.processEvents()
            if time.monotonic() >= deadline:
                raise TimeoutError('Unified window did not close')
            time.sleep(.04)
        app.processEvents()
    assert not custom.worker.isRunning() and not bench.worker.isRunning()
    checks.append('Owned workers stopped on unified close')
    report = dict(status='passed', platform=app.platformName(), checks=checks,
                  proof=str(proof), physical_camera_tested=False)
    write_json(proof/'VALIDATION.json', report)
    print(proof/'VALIDATION.json', flush=True)


if __name__ == '__main__':
    main()
