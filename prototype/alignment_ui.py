"""Local calibration, target measurement and independent-reference comparison."""
import json
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QPixmap, QImage, QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit)
from alignment import calibrate, validate_profile, measure, export_targets
from bootstrap import ROOT
from engine import read_image


class CalibrationWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, paths, label):
        super().__init__()
        self.paths, self.label = paths, label

    def run(self):
        try:
            self.ready.emit(calibrate((read_image(p) for p in self.paths), self.label))
        except Exception as e:
            self.failed.emit(str(e))


class AlignmentDialog(QDialog):
    def __init__(self, station):
        super().__init__(station)
        self.station = station
        self.profile = self.result = self.rgb = self.annotated = None
        self.worker = None
        self.last_frame = None
        self.setWindowTitle('Calibrated alignment bench')
        self.resize(1050, 830)
        layout = QVBoxLayout(self)
        title = QLabel('GROUND TARGET → VEHICLE FRAME ← WHEEL TARGET → CAMBER / TOE')
        title.setWordWrap(True)
        title.setStyleSheet('font-size:18px;font-weight:600;color:#17697b')
        layout.addWidget(title)
        def row():
            r = QHBoxLayout(); layout.addLayout(r); return r
        def action(r, text, callback):
            b = QPushButton(text); b.clicked.connect(callback); r.addWidget(b); return b
        r = row()
        action(r, '1 · Printable targets', self.targets)
        self.camera_label = QLineEdit()
        self.camera_label.setPlaceholderText('Camera / lens / resolution / zoom identification')
        r.addWidget(self.camera_label, 1)
        self.calibrate_btn = action(r, '2 · Calibrate from photos', self.build_profile)
        action(r, 'Load profile', self.load_profile)
        action(r, 'Save profile', self.save_profile)
        self.profile_status = QLabel('No camera calibration loaded. Use 10+ distinct ground-target photos with varied positions and tilts.')
        self.profile_status.setWordWrap(True); layout.addWidget(self.profile_status)
        guide = QLabel('GROUND: flat, verified level; printed right = vehicle-forward; printed up = vehicle-left.\n'
                       'WHEEL: separate target rigidly parallel to the wheel plane. BOTH targets visible in the same frame.\n'
                       'Use the calibrated lens, zoom, focus and orientation. Disable variable digital stabilisation/cropping.')
        guide.setWordWrap(True); layout.addWidget(guide)
        self.confirm = QCheckBox('I verified the reference axes, wheel-plane mounting and matching camera setup')
        self.confirm.toggled.connect(self.invalidate)
        layout.addWidget(self.confirm)
        r = row()
        self.side = QComboBox(); self.side.addItems(['left', 'right']); r.addWidget(self.side)
        self.side.currentIndexChanged.connect(self.invalidate)
        action(r, '3 · Measure workstation frame', self.capture)
        action(r, 'Open target photo', self.open_photo)
        self.live = QCheckBox('Follow analysed video frames'); r.addWidget(self.live)
        self.preview = QLabel('Both targets and calibration are needed for an angle result.')
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumHeight(300)
        layout.addWidget(self.preview, 1)
        self.status = QLabel('Measurement unavailable'); self.status.setWordWrap(True)
        layout.addWidget(self.status)
        r = row()
        self.reference = QCheckBox('Independent reference available'); r.addWidget(self.reference)
        self.ref_camber = QDoubleSpinBox(); self.ref_toe = QDoubleSpinBox()
        for spin, title in [(self.ref_camber, 'Reference camber '), (self.ref_toe, 'Reference toe-in ')]:
            spin.setRange(-20, 20); spin.setDecimals(3); spin.setPrefix(title); spin.setSuffix('°'); r.addWidget(spin)
            spin.valueChanged.connect(self.show_result)
        self.reference.toggled.connect(self.show_result)
        r = row()
        self.note = QLineEdit(); self.note.setPlaceholderText('Wheel / vehicle ID, reference instrument and capture notes'); r.addWidget(self.note)
        action(r, 'Save alignment evidence', self.save_result)
        action(r, 'Close', self.close)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(750)

    def invalidate(self, *args):
        self.result = None
        self.status.setText('Measurement unavailable — capture again after changing the setup')

    def targets(self):
        page = export_targets(ROOT/'data'/'alignment-targets')
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))

    def build_profile(self):
        if self.worker and self.worker.isRunning():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select 10+ calibration photos', '', 'Images (*.png *.jpg *.jpeg)')
        if not paths:
            return
        self.invalidate()
        self.profile = None
        self.calibrate_btn.setEnabled(False)
        self.profile_status.setText('Calibrating camera from target views…')
        self.worker = CalibrationWorker(paths, self.camera_label.text())
        self.worker.ready.connect(self.set_profile)
        self.worker.failed.connect(lambda message: self.profile_status.setText('Calibration unavailable: '+message))
        self.worker.finished.connect(lambda: self.calibrate_btn.setEnabled(True))
        self.worker.start()

    def set_profile(self, profile):
        validate_profile(profile)
        self.profile = profile
        self.confirm.setChecked(False)
        self.invalidate()
        self.profile_status.setText(f"{profile.get('camera_label', '')} · {profile['views']} views · calibration RMS {profile['rms_px']:.3f}px · {profile['image_size']}")

    def load_profile(self):
        if self.worker and self.worker.isRunning():
            self.profile_status.setText('Wait for the current calibration to finish before loading a profile.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load calibration', '', 'JSON (*.json)')
        if path:
            self.profile = None; self.invalidate()
            try:
                self.set_profile(json.loads(Path(path).read_text(encoding='utf-8')))
            except Exception as e:
                self.profile_status.setText(str(e))

    def save_profile(self):
        if self.profile is None:
            self.status.setText('Create or load a calibration first'); return
        path, _ = QFileDialog.getSaveFileName(self, 'Save camera calibration', str(ROOT/'data'/'camera-calibration.json'), 'JSON (*.json)')
        if path:
            Path(path).write_text(json.dumps(self.profile, indent=2), encoding='utf-8')

    def capture(self):
        rgb = self.station.captured_rgb
        if rgb is None:
            rgb = self.station.rgb
        if rgb is not None:
            self.evaluate(rgb.copy())

    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open photo showing both targets', '', 'Images (*.png *.jpg *.jpeg)')
        if path:
            self.live.setChecked(False)
            try:
                self.evaluate(read_image(path))
            except Exception as e:
                self.invalidate(); self.status.setText(str(e))

    def tick(self):
        if self.live.isChecked() and self.station.captured_rgb is not self.last_frame:
            self.last_frame = self.station.captured_rgb
            self.capture()

    def evaluate(self, rgb):
        self.rgb, self.result = rgb, None
        shown = rgb
        try:
            if self.profile is None:
                raise ValueError('Load or create camera calibration first')
            self.result, shown = measure(rgb, self.profile, self.side.currentText(), self.confirm.isChecked())
            if self.station.captured_rgb is not None and np.array_equal(rgb, self.station.captured_rgb):
                from geometry_comparison import compare_fits
                fits, summary = compare_fits(rgb, self.station.masks)
                self.result['supporting_image_geometry'] = fits
                self.result['image_geometry_summary'] = summary
                self.result['supporting_learned_geometry'] = self.station.learned_frame
            self.show_result()
        except Exception as e:
            self.status.setText('Measurement unavailable: '+str(e))
        self.annotated = shown
        a = np.ascontiguousarray(shown)
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(920, 410, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_result(self, *args):
        if self.result is None:
            return
        r = self.result
        text = f"Camber {r['camber_deg']:+.3f}° · Toe-in {r['toe_in_deg']:+.3f}° · Target residuals {r['ground_rms_px']:.2f} / {r['wheel_rms_px']:.2f}px"
        if self.reference.isChecked():
            text += f"\nDifference from reference: camber {r['camber_deg']-self.ref_camber.value():+.3f}° / toe {r['toe_in_deg']-self.ref_toe.value():+.3f}°"
        self.status.setText(text+'\nTarget-assisted research measurement; no validated accuracy or vehicle tolerance verdict. Axes drawn are target-local XYZ.')

    def save_result(self):
        if self.result is None:
            self.status.setText('No valid measurement to save'); return
        from PIL import Image
        folder = ROOT/'results'/'alignment'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        folder.mkdir(parents=True)
        record = dict(self.result, note=self.note.text(), setup_confirmed=True)
        if self.reference.isChecked():
            record['independent_reference'] = dict(camber_deg=self.ref_camber.value(), toe_in_deg=self.ref_toe.value())
        (folder/'alignment.json').write_text(json.dumps(record, indent=2, default=lambda v: v.tolist()), encoding='utf-8')
        Image.fromarray(self.rgb).save(folder/'frame.png')
        Image.fromarray(self.annotated).save(folder/'target-axes.png')
        self.show_result()
        self.setWindowTitle('Calibrated alignment — saved '+folder.name)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.status.setText('Calibration is running; close when it finishes.')
            event.ignore(); return
        self.timer.stop(); event.accept()
