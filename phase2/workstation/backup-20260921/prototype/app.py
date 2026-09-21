"""Tread Station — a native, local inspection and model comparison console."""
import bootstrap
import argparse
import json
import sys
import threading
import queue
import time
import uuid
from pathlib import Path
from collections import deque

# Load the isolated Qt DLLs before Conda OpenCV can load its own Qt build.
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QRectF, QUrl, QPointF, QProcess, QStandardPaths
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QImage, QPixmap, QShortcut, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QComboBox, QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QSplitter, QScrollArea, QSpinBox, QMessageBox, QLayout, QSlider, QDoubleSpinBox, QCheckBox, QDialog)
import cv2
import numpy as np

from bootstrap import ROOT
from registry import MODELS, checkpoint
from engine import Engine, LABELS, overlay, read_image
from evidence import save_capture
from theme import apply_theme
from video import configure_capture, portrait_frame
from shape_geometry import GeometryTracker, draw_geometry
from edge_geometry import edge_fit, draw_edge_fit
from learned_geometry import PointTracker, draw_learned


def label(text, kind=None):
    w = QLabel(text)
    if kind:
        w.setObjectName(kind)
    w.setWordWrap(True)
    return w


def button(text, callback, primary=False):
    w = QPushButton(text)
    if primary:
        w.setObjectName('primary')
    w.clicked.connect(callback)
    return w


def panel():
    w = QFrame()
    w.setObjectName('panel')
    layout = QVBoxLayout(w)
    layout.setContentsMargins(18, 18, 18, 18)
    layout.setSpacing(12)
    return w, layout


class Viewer(QWidget):
    """Native canvas with an interactive original/overlay inspection wipe."""
    def __init__(self):
        super().__init__()
        self.original = self.processed = None
        self.preview = None
        self.wipe = .50
        self.mode = 'Compare wipe'
        self.zoom = 1.
        self.pan = QPointF()
        self.drag_start = QPointF()
        self.setMinimumSize(400, 310)
        self.setMouseTracking(True)
        self.setToolTip('Drag across the image to reveal the original beneath the overlay.')

    @staticmethod
    def qimage(rgb):
        a = np.ascontiguousarray(rgb)
        return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()

    def set_frame(self, rgb, processed=None):
        self.original = self.qimage(rgb)
        self.processed = self.qimage(processed) if processed is not None else None
        self.update()

    def set_preview(self, rgb=None):
        self.preview = self.qimage(cv2.resize(rgb, (round(rgb.shape[1]*160/max(rgb.shape[:2])), round(rgb.shape[0]*160/max(rgb.shape[:2]))))) if rgb is not None else None
        self.update()

    def image_rect(self):
        area = self.rect().adjusted(22, 36, -22, -36)
        size = self.original.size().scaled(area.size(), Qt.KeepAspectRatio)
        width, height = size.width()*self.zoom, size.height()*self.zoom
        return QRectF(area.center().x()-width/2+self.pan.x(), area.center().y()-height/2+self.pan.y(), width, height)

    def mousePressEvent(self, event):
        self.drag_start = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.original:
            if self.mode != 'Compare wipe':
                if self.zoom > 1:
                    self.pan += event.position()-self.drag_start
                    self.drag_start = event.position()
                    self.update()
                return
            r = self.image_rect()
            self.wipe = max(0, min(1, (event.position().x()-r.x()) / r.width()))
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#edf3f7'))
        p.setPen(QPen(QColor('#e0e9ef'), 1))
        for x in range(0, self.width(), 30):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 30):
            p.drawLine(0, y, self.width(), y)
        if not self.original:
            p.setPen(QColor('#5d7888'))
            p.setFont(QFont('Segoe UI', 15))
            p.drawText(self.rect(), Qt.AlignCenter, 'Bring a tyre into view.\n\nOpen a photo, a video, or the camera.')
            return
        r = self.image_rect()
        p.drawImage(r, self.original)
        if self.processed and self.mode == 'Full overlay':
            p.drawImage(r, self.processed)
        elif self.processed and self.mode == 'Compare wipe':
            split = r.x()+r.width()*self.wipe
            p.save()
            p.setClipRect(QRectF(split, r.y(), r.right()-split, r.height()))
            p.drawImage(r, self.processed)
            p.restore()
            p.setPen(QPen(QColor('#087f83'), 2))
            p.drawLine(int(split), int(r.y()), int(split), int(r.bottom()))
            p.setBrush(QColor('#087f83'))
            p.drawEllipse(QRectF(split-5, r.center().y()-15, 10, 30))
        p.setFont(QFont('Consolas', 9))
        p.setPen(QColor('#426575'))
        p.drawText(22, 22, 'FRAME / ' + self.mode.upper() + ('     DRAG TO REVEAL REGIONS' if self.mode == 'Compare wipe' else '     DRAG TO PAN' if self.zoom>1 else ''))
        p.drawText(22, self.height()-13, f'{self.original.width()} × {self.original.height()} px    •    FULL FRAME INFERENCE')
        if self.preview is not None:
            size = self.preview.size().scaled(130, 160, Qt.KeepAspectRatio)
            box = QRectF(self.width()-size.width()-20, 40, size.width(), size.height())
            p.drawImage(box, self.preview)
            p.fillRect(int(box.x()), int(box.bottom()), int(box.width()), 22, QColor('#dae9ee'))
            p.drawText(int(box.x())+5, int(box.bottom())+15, 'SOURCE')


class InferenceThread(QThread):
    complete = Signal(object, object, object, bool, str)
    failed = Signal(str)
    progress = Signal(str)
    idle = Signal()

    def __init__(self, engine):
        super().__init__()
        self.setStackSize(32 * 1024 * 1024)
        self.engine = engine
        self.jobs = queue.Queue(maxsize=1)

    def submit(self, **job):
        self.jobs.put_nowait(job)

    def shutdown(self):
        self.jobs.put_nowait(None)

    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                rgb = job['rgb']
                result, masks = self.engine.inspect(rgb, job['classifier'], job['region'], job['compare'],
                    self.progress.emit, rotation=job['rotation'], threshold=job['threshold'], assist=job['assist'], learned=job.get('learned', 'off'))
                result.update(source_time_ms=job['frame_time_ms'], generation=job['generation'],
                              source_frame_size=job['source_size'],
                              video_rotation_clockwise=job['video_rotation'], encoded_frame_size=job['encoded_size'])
                masks = {name: np.array(mm, dtype=bool, copy=True) for name, mm in masks.items()}
                self.complete.emit(rgb, result, masks, job['compare'], job['source'])
            except Exception as e:
                self.failed.emit(f'{type(e).__name__}: {e}')
            finally:
                self.idle.emit()


class StreamThread(QThread):
    frame = Signal()
    notice = Signal(str)
    info = Signal(float, float)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.notified = False
        self.seek_ms = None
        self.epoch = 0
        self.rotation_degrees = 0
        self.encoded_size = None
        self.speed = 1.
        self.loop = False

    def take_frame(self):
        with self.lock:
            result, self.latest, self.notified = self.latest, None, False
            return result

    def seek(self, milliseconds):
        with self.lock:
            self.seek_ms = milliseconds
            self.epoch += 1
            self.latest = None

    def run(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW) if isinstance(self.source, int) else cv2.VideoCapture(self.source)
        try:
            if not cap.isOpened():
                self.notice.emit('Could not open the source. Check the camera number or choose another file.')
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            file_rotation = configure_capture(cap)
            self.info.emit(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps*1000 if fps > 0 else 0, fps)
            period = 1 / min(60, fps if 1 <= fps <= 120 else 30)
            while not self.stop_event.is_set():
                with self.lock:
                    seek, self.seek_ms, epoch = self.seek_ms, None, self.epoch
                if seek is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, seek)
                if self.paused.is_set() and seek is None:
                    self.msleep(40)
                    continue
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if self.loop and not isinstance(self.source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.notice.emit('End of video / source stopped. The last frame remains available.')
                    self.paused.set()
                    continue
                self.encoded_size = [frame.shape[1], frame.shape[0]]
                frame, self.rotation_degrees = portrait_frame(frame, file_rotation)
                source_size = [frame.shape[1], frame.shape[0]]
                scale = min(1., 1280 / max(frame.shape[:2]))
                if scale < 1:
                    frame = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    if epoch != self.epoch:
                        continue
                    self.latest = (rgb, cap.get(cv2.CAP_PROP_POS_MSEC), source_size, epoch)
                    notify = not self.notified
                    self.notified = True
                if notify:
                    self.frame.emit()
                self.stop_event.wait(max(0, period/self.speed-(time.perf_counter()-start)))
        finally:
            cap.release()


class Station(QMainWindow):
    def __init__(self, device='auto'):
        super().__init__()
        self.engine = Engine(device)
        self.rgb = self.captured_rgb = None
        self.result, self.masks = None, {}
        self.geometry_tracks, self.geometry_frames = {}, {}
        self.geometry_context = None
        self.edge_cache = {}
        self.point_tracker = PointTracker()
        self.learned_frame = None
        self.source = self.result_source = ''
        self.worker = self.stream = None
        self.export_process = None
        self.export_cancelled = False
        self._job_busy = False
        self.generation = 0
        self.overlay_preference = None
        self.warming = False
        self._seek_pending = False
        self._has_fresh_frame = False
        self.source_size = None
        self.duration_ms = 0.
        self.video_fps = 30.
        self.result_times = deque(maxlen=20)
        self.frame_time_ms = None
        self.live_enabled = False
        self.last_inference = 0
        self.setWindowTitle('Tread Station · Tyre inspection bench')
        self.resize(1440, 920)
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(24, 20, 24, 14)
        main.setSpacing(16)
        header = QHBoxLayout()
        titlebox = QVBoxLayout()
        brand = label('TYRE STUDY / FIELD INSTRUMENT 01', 'eyebrow')
        brand.setWordWrap(False)
        titlebox.addWidget(brand)
        titlebox.addWidget(label('Tread Station', 'title'))
        header.addLayout(titlebox)
        header.addStretch()
        self.hardware = label('LOCAL INFERENCE   /   ' + self.engine.device.upper(), 'eyebrow')
        self.hardware.setWordWrap(False)
        header.addWidget(self.hardware)
        main.addLayout(header)
        tools = QHBoxLayout()
        self.open_btn = button('+  Photo / video', self.open_file)
        tools.addWidget(self.open_btn)
        tools.addWidget(button('Study sample', self.open_sample))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 9)
        self.camera_index.setPrefix('Camera ')
        tools.addWidget(self.camera_index)
        tools.addWidget(button('Connect', self.open_camera))
        self.freeze_btn = button('Freeze   [Space]', self.toggle_freeze)
        tools.addWidget(self.freeze_btn)
        self.live_btn = button('Live predictions', self.toggle_live)
        self.live_btn.setCheckable(True)
        tools.addWidget(self.live_btn)
        tools.addStretch()
        self.state_label = label('READY TO CAPTURE', 'eyebrow')
        self.state_label.setWordWrap(False)
        self.state_label.setMinimumWidth(190)
        tools.addWidget(self.state_label)
        main.addLayout(tools)
        transport = QHBoxLayout()
        self.replay_btn = button('Replay', self.replay)
        transport.addWidget(self.replay_btn)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.setEnabled(False)
        self.timeline.sliderReleased.connect(self.seek_video)
        transport.addWidget(self.timeline, 1)
        self.video_clock = label('No video loaded', 'muted')
        transport.addWidget(self.video_clock)
        main.addLayout(transport)
        playback = QHBoxLayout()
        playback.addWidget(button('−5 s', lambda: self.jump_video(-5000)))
        playback.addWidget(button('Previous frame', lambda: self.step_frame(-1)))
        playback.addWidget(button('Next frame', lambda: self.step_frame(1)))
        playback.addWidget(button('+5 s', lambda: self.jump_video(5000)))
        playback.addWidget(label('Playback speed', 'muted'))
        self.speed = QComboBox()
        for rate in (.25, .5, 1., 1.5, 2.):
            self.speed.addItem(f'{rate:g}×', rate)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self.playback_options)
        playback.addWidget(self.speed)
        self.loop = QCheckBox('Loop video')
        self.loop.toggled.connect(self.playback_options)
        playback.addWidget(self.loop)
        playback.addStretch()
        playback.addWidget(label('PORTRAIT / DISPLAY = MODEL INPUT', 'eyebrow'))
        main.addLayout(playback)
        split = QSplitter(Qt.Horizontal)
        main.addWidget(split, 1)
        left = QWidget()
        leftlayout = QVBoxLayout(left)
        leftlayout.setContentsMargins(0, 0, 0, 0)
        self.viewer = Viewer()
        leftlayout.addWidget(self.viewer, 1)
        bottom = QHBoxLayout()
        self.frame_label = label('No frame selected', 'muted')
        bottom.addWidget(self.frame_label, 1)
        self.overlay_select = QComboBox()
        self.overlay_select.setMinimumWidth(235)
        self.overlay_select.addItem('Original', '')
        self.overlay_select.currentIndexChanged.connect(self.render_capture)
        self.overlay_select.activated.connect(self.choose_overlay)
        bottom.addWidget(self.overlay_select)
        leftlayout.addLayout(bottom)
        layers = QHBoxLayout()
        self.view_mode = QComboBox()
        for title in ('Full overlay', 'Original', 'Compare wipe'):
            self.view_mode.addItem(title)
        self.view_mode.currentIndexChanged.connect(self.render_capture)
        layers.addWidget(self.view_mode)
        self.show_tyre = QCheckBox('Tyre')
        self.show_tread = QCheckBox('Tread')
        for check in (self.show_tyre, self.show_tread):
            check.setChecked(True)
            check.toggled.connect(self.render_capture)
            layers.addWidget(check)
        layers.addWidget(label('Overlay strength', 'muted'))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 70)
        self.opacity.setValue(26)
        self.opacity.valueChanged.connect(self.render_capture)
        layers.addWidget(self.opacity, 1)
        layers.addWidget(label('Zoom', 'muted'))
        self.zoom = QComboBox()
        for title, scale in [('Fit', 1.), ('150%', 1.5), ('200%', 2.), ('300%', 3.)]:
            self.zoom.addItem(title, scale)
        self.zoom.currentIndexChanged.connect(self.set_zoom)
        layers.addWidget(self.zoom)
        leftlayout.addLayout(layers)
        self.show_geometry = QCheckBox('Track geometry on image')
        self.show_geometry.setChecked(True)
        self.show_geometry.toggled.connect(self.render_capture)
        geometry_controls = QHBoxLayout()
        geometry_controls.addWidget(self.show_geometry)
        self.edge_mode = QComboBox()
        self.edge_mode.addItem('Boundary lines / tread view', 'boundary')
        self.edge_mode.addItem('Rim candidate / side view', 'rim')
        self.edge_mode.addItem('Image edge fitting off', '')
        self.edge_mode.setToolTip('Choose the visible view. Rim mode requires a visible rim; fits remain camera-relative.')
        self.edge_mode.currentIndexChanged.connect(self.render_capture)
        geometry_controls.addWidget(self.edge_mode, 1)
        geometry_controls.addWidget(button('Explain edge fit', self.explain_edges))
        leftlayout.addLayout(geometry_controls)
        self.edge_status = label('Image edges will appear after inspection.', 'muted')
        leftlayout.addWidget(self.edge_status)
        self.geometry_btn = button('Shape Compass — explain geometry', self.explain_geometry)
        leftlayout.addWidget(self.geometry_btn)
        alignment_controls = QHBoxLayout()
        alignment_controls.addWidget(button('Compare boundaries + rim', self.compare_geometry))
        alignment_controls.addWidget(button('Calibrated alignment', self.open_alignment))
        leftlayout.addLayout(alignment_controls)
        self.note = QLineEdit()
        self.note.setPlaceholderText('Inspection note — lighting, tyre position, observed issue…')
        leftlayout.addWidget(self.note)
        self.save_btn = button('Save evidence card   [Ctrl+S]', self.save_evidence)
        self.save_btn.setEnabled(False)
        leftlayout.addWidget(self.save_btn)
        downloads = QHBoxLayout()
        downloads.addWidget(button('Download shown frame', self.download_frame))
        self.export_btn = button('Download video · 10 fps', self.download_video)
        downloads.addWidget(self.export_btn)
        self.cancel_export_btn = button('Cancel export', self.cancel_export)
        self.cancel_export_btn.setEnabled(False)
        downloads.addWidget(self.cancel_export_btn)
        leftlayout.addLayout(downloads)
        self.export_status = label('Downloads: PNG frame or full silent MP4 · background CPU export.', 'muted')
        leftlayout.addWidget(self.export_status)
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.NoFrame)
        self.left_scroll.setWidget(left)
        split.addWidget(self.left_scroll)
        right, rl = panel()
        rl.setSizeConstraint(QLayout.SetMinimumSize)
        right.setMinimumWidth(340)
        right.setMaximumWidth(440)
        rl.addWidget(label('01 / INSPECTION RECIPE', 'eyebrow'))
        self.classifier = QComboBox()
        self.region = QComboBox()
        for name, spec in MODELS.items():
            (self.classifier if spec['task']=='classifier' else self.region).addItem(spec['title'], name)
        self.region.addItem('No region overlay (fastest)', '')
        self.region.setCurrentIndex(self.region.findData('segformer_b0'))
        rl.addWidget(self.classifier)
        rl.addWidget(self.region)
        self.learned_mode = QComboBox()
        self.learned_mode.addItem('Learned boundaries off', 'off')
        self.learned_mode.addItem('HRNet · six tread-boundary points', 'hrnet')
        self.learned_mode.addItem('HRNet + Matched SegFormer · compare points', 'paired')
        from prepare_learned import path as learned_path
        if learned_path('hrnet').exists():
            self.learned_mode.setCurrentIndex(2 if learned_path('matched').exists() else 1)
        self.learned_mode.currentIndexChanged.connect(self.reset_analysis)
        rl.addWidget(self.learned_mode)
        self.show_learned = QCheckBox('Show learned point overlay')
        self.show_learned.setChecked(True)
        self.show_learned.toggled.connect(self.render_capture)
        rl.addWidget(self.show_learned)
        rl.addWidget(button('Explain learned overlay', self.explain_learned))
        self.learned_status = label('Amber dots: HRNet · cyan squares: matched SegFormer · purple: tread centreline. Image-space proposals, not measured alignment.', 'muted')
        rl.addWidget(self.learned_status)
        rl.addWidget(label('Video orientation follows the file’s portrait display tag. Models process this same vertical frame.', 'muted'))
        sensitivity = QHBoxLayout()
        sensitivity.addWidget(label('YOLO minimum score', 'muted'))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.05, .90)
        self.threshold.setSingleStep(.05)
        self.threshold.setValue(.25)
        self.threshold.setToolTip('Lower values accept weaker candidates and can add false positives. Recorded with every capture.')
        sensitivity.addWidget(self.threshold)
        rl.addLayout(sensitivity)
        self.assist = QCheckBox('Use labelled SegFormer assist when YOLO misses')
        self.assist.setChecked(True)
        rl.addWidget(self.assist)
        for combo in (self.region, self.classifier):
            combo.currentIndexChanged.connect(self.reset_analysis)
        self.threshold.valueChanged.connect(self.reset_analysis)
        self.assist.toggled.connect(self.reset_analysis)
        cadence_row = QHBoxLayout()
        cadence_row.addWidget(label('Live inspection cadence', 'muted'))
        self.cadence = QSpinBox()
        self.cadence.setRange(1, 10)
        self.cadence.setValue(5)
        self.cadence.setSuffix(' / sec')
        self.cadence.setToolTip('Maximum requested updates. Slow inference skips frames instead of building a queue.')
        cadence_row.addWidget(self.cadence)
        rl.addLayout(cadence_row)
        self.inspect_btn = button('Inspect frame   [Enter]', self.inspect, True)
        rl.addWidget(self.inspect_btn)
        self.compare_btn = button('Compare all four on this frame', lambda: self.inspect(compare=True))
        rl.addWidget(self.compare_btn)
        rl.addWidget(label('02 / MODEL OBSERVATIONS', 'eyebrow'))
        self.prediction = label('Waiting for a frame', 'big')
        rl.addWidget(self.prediction)
        self.observations = label('Full-image classification. Amber marks the tyre; mint marks the tread.', 'muted')
        rl.addWidget(self.observations)
        self.model_list = QListWidget()
        self.model_list.setFixedHeight(170)
        rl.addWidget(self.model_list)
        self.agreement = label('Freeze a frame, then compare to expose model disagreements.', 'muted')
        rl.addWidget(self.agreement)
        self.quality_label = label('Capture hints appear after inspection.', 'muted')
        rl.addWidget(self.quality_label)
        rl.addWidget(label('03 / EVIDENCE TRAY', 'eyebrow'))
        self.tray = QListWidget()
        self.tray.setFixedHeight(115)
        self.tray.itemClicked.connect(self.restore_capture)
        self.tray.itemDoubleClicked.connect(self.open_card)
        rl.addWidget(self.tray)
        rl.addWidget(button('Open saved records', self.open_results))
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setMinimumWidth(365)
        self.right_scroll.setMaximumWidth(450)
        self.right_scroll.setWidget(right)
        split.addWidget(self.right_scroll)
        split.setSizes([960, 365])
        main.addWidget(label('RESEARCH PROTOTYPE   ·   Low / mid / high mileage proxy — not measured tread depth or a roadworthiness verdict.', 'muted'))
        self.statusBar().showMessage('Open a source → inspect → freeze and compare → save evidence. Everything stays on this computer.')
        for key, func in [('Return', self.inspect), ('Space', self.toggle_freeze), ('Ctrl+S', self.save_evidence), ('Ctrl+O', self.open_file)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(func)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.live_tick)
        self.timer.start(50)
        self.refresh_tray()
        self.worker = InferenceThread(self.engine)
        self.worker.complete.connect(self.on_result, Qt.QueuedConnection)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self.show_error, Qt.QueuedConnection)
        self.worker.idle.connect(self.job_finished, Qt.QueuedConnection)
        self.worker.start()

    def busy(self):
        return self._job_busy

    def reset_analysis(self, value=None):
        self.generation += 1
        self.point_tracker = PointTracker()
        self.learned_frame = None
        self.geometry_tracks.clear()
        self.geometry_frames.clear()
        self._has_fresh_frame = self.rgb is not None
        self.result_times.clear()

    def clear_result(self):
        self.point_tracker = PointTracker()
        self.learned_frame = None
        self.edge_cache.clear()
        self.edge_status.setText('Image edges will appear after inspection.')
        self.geometry_tracks.clear()
        self.geometry_frames.clear()
        self.overlay_preference = None
        self.result, self.masks, self.captured_rgb = None, {}, None
        self.save_btn.setEnabled(False)
        self.model_list.clear()
        self.overlay_select.clear()
        self.overlay_select.addItem('Original', '')
        self.prediction.setText('Ready to inspect')
        self.agreement.setText('Compare uses the same frozen frame for all four models.')

    def stop_stream(self):
        self.generation += 1
        self.warming = False
        if self.stream and self.stream.isRunning():
            self.stream.stop_event.set()
            if not self.stream.wait(2000):
                self.statusBar().showMessage('Camera is still closing. Try again in a moment.')
                return False
        self.stream = None
        self.live_enabled = False
        self.live_btn.setChecked(False)
        self.freeze_btn.setText('Freeze   [Space]')
        self.viewer.set_preview()
        return True

    def open_file(self):
        if self.busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Choose an inspection source', str(ROOT/'data'/'incoming'),
            'Images and videos (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv);;All files (*)')
        if path:
            self.load_source(path)

    def open_sample(self):
        if self.busy():
            return
        folder = Path(r'D:\Dataset Download\Tire Dataset Prepared\FINAL\images\clean\fold_1')
        path, _ = QFileDialog.getOpenFileName(self, 'Choose an existing study image (internal data)', str(folder), 'Images (*.jpg *.png *.jpeg)')
        if path:
            self.load_source(path)

    def load_source(self, path):
        if not self.stop_stream():
            return
        self.clear_result()
        self.note.clear()
        self.rgb = None
        self.frame_time_ms = None
        self.source = str(Path(path).resolve())
        if Path(path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
            self.region.setCurrentIndex(self.region.findData('yolo26n_seg'))
            self.viewer.wipe = 0.
            self.start_stream(self.source)
        else:
            self.viewer.wipe = .5
            self.timeline.setEnabled(False)
            self.video_clock.setText('Still image')
            try:
                self.rgb = read_image(path)
                self.source_size = [self.rgb.shape[1], self.rgb.shape[0]]
                self.viewer.set_frame(self.rgb)
                self.frame_label.setText(Path(path).name)
                self.state_label.setText('STILL FRAME')
            except Exception as e:
                self.show_error(str(e))

    def open_camera(self):
        if self.busy() or not self.stop_stream():
            return
        self.clear_result()
        self.source = f'camera:{self.camera_index.value()}'
        self.region.setCurrentIndex(self.region.findData('yolo26n_seg'))
        self.viewer.wipe = 0.
        self.rgb = None
        self.frame_time_ms = None
        self.start_stream(self.camera_index.value())

    def start_stream(self, source):
        self.stream = StreamThread(source)
        self.playback_options()
        self.stream.frame.connect(self.on_frame, Qt.QueuedConnection)
        self.stream.notice.connect(self.statusBar().showMessage)
        self.stream.finished.connect(self.stream_finished)
        self.stream.info.connect(self.stream_info)
        self.live_enabled = True
        self.live_btn.setChecked(True)
        self.warming = True
        self._has_fresh_frame = False
        self._seek_pending = False
        self.result_times.clear()
        self.stream.start()
        self.state_label.setText('SOURCE RUNNING')

    @Slot()
    def on_frame(self):
        if not self.stream:
            return
        packet = self.stream.take_frame()
        if packet is None or packet[3] != self.stream.epoch:
            return
        if self.stream.paused.is_set() and not self._seek_pending:
            return
        rgb, timestamp, self.source_size, _ = packet
        self._seek_pending = False
        self.rgb = rgb
        self.frame_time_ms = timestamp
        self._has_fresh_frame = True
        if self.warming:
            self.stream.paused.set()
        self.video_clock.setText(f'{timestamp/1000:.1f}s / {self.duration_ms/1000:.1f}s')
        if self.duration_ms and not self.timeline.isSliderDown():
            self.timeline.setValue(int(timestamp/self.duration_ms*1000))
        self.viewer.set_preview(rgb if self.live_enabled else None)
        # During live inference the canvas intentionally shows the last analysed
        # frame, avoiding stale masks painted over newer pixels.
        if not self.live_enabled:
            self.viewer.set_frame(rgb)
            self.frame_label.setText(f'{Path(self.source).name}  ·  {timestamp / 1000:.2f}s  ·  source preview')
        elif self.captured_rgb is None:
            self.viewer.set_frame(rgb)

    @Slot(float, float)
    def stream_info(self, duration, fps):
        self.duration_ms = duration
        self.video_fps = fps if fps > 0 else 30.
        self.timeline.setEnabled(not self.source.startswith('camera:') and duration > 0)

    def seek_video(self, milliseconds=None):
        if not self.stream or not self.timeline.isEnabled():
            return
        self.generation += 1
        self._seek_pending = True
        self._has_fresh_frame = False
        self.warming = self.live_enabled
        self.clear_result()
        position = self.duration_ms*self.timeline.value()/1000 if milliseconds is None else milliseconds
        self.stream.seek(max(0, min(self.duration_ms-1000/self.video_fps, position)))

    def playback_options(self, value=None):
        if self.stream:
            self.stream.speed = self.speed.currentData()
            self.stream.loop = self.loop.isChecked()

    def jump_video(self, milliseconds):
        self.seek_video((self.frame_time_ms or 0)+milliseconds)

    def step_frame(self, direction):
        if not self.stream or not self.timeline.isEnabled():
            return
        self.live_enabled = False
        self.live_btn.setChecked(False)
        self.warming = False
        self.stream.paused.set()
        self.freeze_btn.setText('Resume   [Space]')
        self.jump_video(direction*1000/self.video_fps)

    def replay(self):
        if not self.stream or not self.timeline.isEnabled():
            return
        self.timeline.setValue(0)
        self.seek_video()
        self.live_enabled = True
        self.live_btn.setChecked(True)
        self.warming = True
        self.stream.paused.clear()
        self.freeze_btn.setText('Freeze   [Space]')

    @Slot()
    def stream_finished(self):
        if self.stream and not self.stream.isRunning():
            self.live_enabled = False
            self.live_btn.setChecked(False)
            self.state_label.setText('SOURCE ENDED / FRAME HELD')

    def toggle_freeze(self):
        if not self.stream or not self.stream.isRunning():
            return
        if self.warming:
            self.warming = False
            self.live_enabled = False
            self.live_btn.setChecked(False)
            self.stream.paused.set()
            self.freeze_btn.setText('Resume   [Space]')
            return
        if self.stream.paused.is_set():
            self.clear_result()
            self.stream.paused.clear()
            self.live_enabled = True
            self.live_btn.setChecked(True)
            self.freeze_btn.setText('Freeze   [Space]')
            self.state_label.setText('SOURCE RUNNING')
        else:
            was_live = self.live_enabled
            self.stream.paused.set()
            if self.live_enabled and self.captured_rgb is not None:
                self.rgb = self.captured_rgb.copy()
                self.frame_time_ms = self.result.get('source_time_ms')
            self.live_enabled = False
            self.warming = False
            self.live_btn.setChecked(False)
            self.freeze_btn.setText('Resume   [Space]')
            self.state_label.setText('FROZEN FRAME')
            if self.rgb is not None and not self.busy():
                if was_live and self.result is not None:
                    self.render_capture()
                else:
                    self.viewer.set_frame(self.rgb)
            self.viewer.set_preview()

    def toggle_live(self):
        self.live_enabled = self.live_btn.isChecked()
        if self.live_enabled and (not self.stream or not self.stream.isRunning()):
            self.live_enabled = False
            self.live_btn.setChecked(False)
            self.statusBar().showMessage('Open a video or connect a camera first.')
        elif self.live_enabled:
            self.stream.paused.clear()
            self.freeze_btn.setText('Freeze   [Space]')

    def live_tick(self):
        if (self.live_enabled and self._has_fresh_frame and self.stream
                and (not self.stream.paused.is_set() or self.warming) and not self.busy()
                and time.monotonic()-self.last_inference >= 1/self.cadence.value()):
            self.inspect(live=True)

    def inspect(self, checked=False, compare=False, live=False):
        if self.rgb is None or self.busy():
            return
        if not live and self.stream and self.stream.isRunning():
            if self.live_enabled and self.captured_rgb is not None:
                self.rgb = self.captured_rgb.copy()
                self.frame_time_ms = self.result.get('source_time_ms')
            self.stream.paused.set()
            self.warming = False
            self.live_enabled = False
            self.live_btn.setChecked(False)
            self.freeze_btn.setText('Resume   [Space]')
        self.last_inference = time.monotonic()
        self.state_label.setText('COMPARING SAME FRAME…' if compare else 'INSPECTING…')
        self.inspect_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        self._job_busy = True
        self._has_fresh_frame = False
        self.worker.submit(rgb=self.rgb.copy(), classifier=self.classifier.currentData(), region=self.region.currentData(),
            compare=compare, source=self.source, frame_time_ms=self.frame_time_ms, generation=self.generation,
            source_size=self.source_size, rotation=0, threshold=self.threshold.value(), assist=self.assist.isChecked(),
            video_rotation=self.stream.rotation_degrees if self.stream else 0,
            encoded_size=self.stream.encoded_size if self.stream else self.source_size, learned=self.learned_mode.currentData())

    @Slot()
    def job_finished(self):
        self._job_busy = False
        self.inspect_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)

    @Slot(object, object, object, bool, str)
    def on_result(self, rgb, result, masks, compare, source):
        if result.get('generation', self.generation) != self.generation:
            return
        if self.stream and self.stream.paused.is_set():
            self.rgb = rgb.copy()
            self.frame_time_ms = result.get('source_time_ms')
        self.captured_rgb, self.result, self.masks, self.result_source = rgb, result, masks, source
        self.edge_cache.clear()
        context = (self.generation, source)
        if context != self.geometry_context or not self.stream:
            self.geometry_tracks.clear()
        self.geometry_context = context
        stamp = result.get('source_time_ms')
        if stamp is None:
            stamp = time.monotonic()*1000
        learned = result.get('learned_geometry')
        if not self.stream:
            self.point_tracker = PointTracker()
        self.learned_frame = learned if learned and not self.stream and 'display_points' in learned else self.point_tracker.update(learned, stamp, context)
        if self.learned_frame:
            hr = self.learned_frame['models']['hrnet']
            widths = ' / '.join('—' if v is None else f'{v:.0f}' for v in hr['widths_px'])
            state = 'REVIEW: '+'; '.join(learned['flags']) if learned['flags'] else f"{self.learned_frame.get('stable_frames', 1)} consistent frames (not confidence)"
            times = ' · '.join(f'{n} {r["inference_ms"]:.0f}ms' for n, r in learned['models'].items())
            self.learned_status.setText(f'Raw HRNet tread widths U/M/L: {widths} px\n{state}\n{times}\nAmber dots / cyan squares / purple centreline. Red = review. Image-relative only.')
        else:
            self.learned_status.setText('Learned boundaries off. Select HRNet or the matched comparison to add points.')
        self.geometry_tracks = {name: self.geometry_tracks.get(name, GeometryTracker()) for name in masks}
        self.geometry_frames = {name: self.geometry_tracks[name].update(mm[0], stamp) for name, mm in masks.items()}
        if self.warming and self.live_enabled and self.stream:
            self.warming = False
            self.stream.paused.clear()
        self.result_times.append(time.monotonic())
        self.overlay_select.blockSignals(True)
        self.overlay_select.clear()
        self.overlay_select.addItem('Original', '')
        for name in masks:
            self.overlay_select.addItem(MODELS[name]['title'] + ' overlay', name)
        if masks:
            preferred = self.overlay_preference
            if preferred is None:
                preferred = 'segformer_b0' if result.get('assist_used') else next(iter(masks))
            index = self.overlay_select.findData(preferred)
            self.overlay_select.setCurrentIndex(index if index >= 0 else 1)
        self.overlay_select.blockSignals(False)
        self.render_capture()
        self.model_list.clear()
        self.model_list.setFixedHeight(112 * len(result['models']))
        predictions = []
        for r in result['models']:
            text = f'{r["title"]}   /   {r["inference_ms"]:.0f} ms'
            if r['task'] == 'classifier':
                predictions.append(r['label'])
                text += '\n' + r['label'] + '\nL / M / H scores: ' + ' / '.join(f'{v:.2f}' for v in r['scores'])
            else:
                text += f'\nTyre {r["coverage"]["tyre"]:.1%} · Tread {r["coverage"]["tread"]:.1%} of frame'
                if not r['target_found']:
                    text += '\nNo tyre region detected — review the target'
                if 'detector' in r:
                    text += f'\nYOLO max score {r["detector"]["max_confidence"]:.3f} / minimum {r["detector"]["threshold"]:.2f}'
                if r.get('assist_reason'):
                    text += '\nSEGFORMER ASSIST — separate model'
            item = QListWidgetItem(text)
            item.setToolTip(json.dumps(r['provenance'], indent=2))
            self.model_list.addItem(item)
        self.prediction.setText(predictions[0] if len(set(predictions)) == 1 else 'Models disagree')
        region_records = [r for r in result['models'] if r['task'] == 'regions']
        if region_records and not any(r['target_found'] for r in region_records):
            self.prediction.setText('No tyre region located')
        self.observations.setText('Ordinal model scores are uncalibrated. Compare a second view before drawing conclusions.')
        a = result['agreement']
        text = ('Classifiers agree on the proxy label.' if a.get('classifiers_agree') else 'Classifier disagreement: keep this frame for review.') if 'classifiers_agree' in a else 'Run the comparison bench for a second opinion.'
        if 'region_iou' in a:
            value = a['region_iou']['tread']
            text += f'\nTread overlap between models: {value:.1%}.' if value is not None else '\nNeither model found a tread region.'
            text += ' This measures agreement, not accuracy.'
        self.agreement.setText(text)
        self.quality_label.setText('CAPTURE / ' + (' · '.join(result['quality']['hints']) or 'No heuristic flags') + f'\nFocus {result["quality"]["focus_score"]:.0f} · Light {result["quality"]["brightness"]:.0f}/255')
        self.state_label.setText('COMPARISON READY' if compare else 'LIVE RESULT' if self.live_enabled else 'INSPECTION READY')
        self.frame_label.setText(f'Analysed frame {result["frame_sha256"][:10]} · {result["total_ms"]:.0f} ms total (includes loading)')
        if self.stream:
            hz = (len(self.result_times)-1)/(self.result_times[-1]-self.result_times[0]) if len(self.result_times)>1 else 0
            self.frame_label.setText(f'Analysed at {(result.get("source_time_ms") or 0)/1000:.2f}s · {hz:.1f} updates/s · Same portrait frame')
        if result.get('assist_used'):
            self.state_label.setText('ASSIST / REVIEW')
            self.observations.setText('YOLO missed one or both regions. The displayed SegFormer suggestion can include background or miss blurred tread. Review it before interpreting the proxy label.')
        self.hardware.setText(f'{result["device_name"]}  /  {result["peak_vram_mb"]:.0f} MB allocated peak')
        self.save_btn.setEnabled(True)
        if compare:
            QTimer.singleShot(0, lambda: self.right_scroll.verticalScrollBar().setValue(max(0, self.prediction.y()-45)))
        self.statusBar().showMessage('Drag the image divider to inspect region boundaries. Save preserves these exact analysed pixels.')

    def render_capture(self, index=0):
        if self.captured_rgb is None:
            return
        name = self.overlay_select.currentData()
        mode = self.view_mode.currentText()
        self.viewer.mode = mode
        processed = overlay(self.captured_rgb, self.masks[name], self.opacity.value()/100,
            (self.show_tyre.isChecked(), self.show_tread.isChecked())) if name in self.masks and mode != 'Original' else None
        if self.show_geometry.isChecked() and mode != 'Original':
            processed = draw_geometry(processed if processed is not None else self.captured_rgb, self.geometry_frames.get(name))
        fit = self.current_edge_fit()
        if fit is not None:
            detail = fit['reason']
            if fit['valid']:
                detail += f" · Fit residual {fit['residual_px']:.1f}px"
                if 'angle' in fit:
                    detail += f" · Midline {fit['angle']:+.1f}° (image-relative)"
            self.edge_status.setText(detail)
            if mode != 'Original':
                processed = draw_edge_fit(processed if processed is not None else self.captured_rgb, fit)
        else:
            self.edge_status.setText('Image edge fitting off.')
        if self.show_learned.isChecked() and mode != 'Original' and self.learned_frame is not None:
            processed = draw_learned(processed if processed is not None else self.captured_rgb, self.learned_frame)
        self.viewer.wipe = .5 if mode == 'Compare wipe' else 0.
        self.viewer.set_frame(self.captured_rgb, processed)

    def current_edge_fit(self):
        mode = self.edge_mode.currentData()
        if not mode or self.captured_rgb is None:
            return None
        name = self.overlay_select.currentData()
        key = (name, mode)
        if key not in self.edge_cache:
            mm = self.masks.get(name)
            self.edge_cache[key] = edge_fit(self.captured_rgb, mm[0] if mm is not None else None, mode)
        return self.edge_cache[key]

    def explain_edges(self):
        fit = self.current_edge_fit()
        if fit is None:
            self.statusBar().showMessage('Inspect a frame and select an edge-fitting view first.')
            return
        from edge_diagram import EdgeDiagram
        mm = self.masks.get(self.overlay_select.currentData())
        self.edge_dialog = EdgeDiagram(self.captured_rgb.copy(), fit, self.overlay_select.currentText(), self,
                                      mask=mm[0] if mm is not None else None)
        self.edge_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.edge_dialog.show()

    def compare_geometry(self):
        if self.captured_rgb is None:
            self.statusBar().showMessage('Inspect a frame first to compare its geometry.')
            return
        from geometry_comparison import GeometryComparison
        self.comparison_dialog = GeometryComparison(self.captured_rgb.copy(), self.masks, self)
        self.comparison_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.comparison_dialog.show()

    def open_alignment(self):
        from alignment_ui import AlignmentDialog
        self.alignment_dialog = AlignmentDialog(self)
        self.alignment_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.alignment_dialog.show()

    def explain_learned(self):
        if self.captured_rgb is None or self.learned_frame is None:
            self.statusBar().showMessage('Enable learned boundaries and inspect a frame first.')
            return
        from learned_diagram import LearnedDiagram
        self.learned_dialog = LearnedDiagram(self.captured_rgb.copy(), self.learned_frame, self)
        self.learned_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.learned_dialog.show()

    def explain_geometry(self):
        from geometry_diagram import GeometryDialog
        name = self.overlay_select.currentData()
        masks = self.masks.get(name)
        rgb = self.captured_rgb.copy() if self.captured_rgb is not None else None
        mask = masks[0].copy() if masks is not None else None
        self.geometry_dialog = GeometryDialog(rgb, mask, name or 'No selected mask', self)
        self.geometry_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.geometry_dialog.show()

    def set_zoom(self, index):
        self.viewer.zoom = self.zoom.currentData()
        self.viewer.pan = QPointF()
        self.viewer.update()

    def choose_overlay(self, index):
        self.overlay_preference = self.overlay_select.currentData()
        self.render_capture()

    def download_path(self, suffix, source):
        folder = Path(QStandardPaths.writableLocation(QStandardPaths.DownloadLocation) or Path.home()/'Downloads')
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if source and not source.startswith('camera:') else 'camera'
        return folder / f'{stem}-overlay-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}{suffix}'

    def download_frame(self):
        if self.captured_rgb is None or self.viewer.original is None:
            self.export_status.setText('Inspect a frame before downloading it.')
            return
        try:
            # Save the exact analysed frame and current wipe, without UI chrome
            # or zoom cropping. QImage copies freeze it against new live results.
            image = self.viewer.original.copy()
            if self.viewer.processed is not None and self.viewer.mode != 'Original':
                painter = QPainter(image)
                if self.viewer.mode == 'Compare wipe':
                    split = round(image.width()*self.viewer.wipe)
                    painter.setClipRect(split, 0, image.width()-split, image.height())
                painter.drawImage(0, 0, self.viewer.processed)
                painter.end()
            path = self.download_path('.png', self.result_source)
            if not image.save(str(path), 'PNG'):
                raise OSError('Could not write the PNG to Downloads')
            self.export_status.setText(f'Saved to Downloads: {path.name}')
        except Exception as exc:
            self.export_status.setText(f'Download failed: {exc}')

    def download_video(self):
        if self.export_process is not None:
            return
        if not Path(self.source).is_file() or Path(self.source).suffix.lower() not in ('.mp4', '.avi', '.mov', '.mkv'):
            self.export_status.setText('Open a video file to export it. For a live camera, download the shown frame.')
            return
        try:
            destination = self.download_path('.mp4', self.source)
            # Snapshot controls once; subsequent inspection changes stay local.
            name = self.overlay_preference
            region = name if name else self.region.currentData()
            job = dict(source=self.source, destination=str(destination), classifier=self.classifier.currentData(),
                region=region, overlay=name, threshold=self.threshold.value(), assist=self.assist.isChecked(),
                learned=self.learned_mode.currentData(), opacity=self.opacity.value()/100,
                layers=[self.show_tyre.isChecked(), self.show_tread.isChecked()], geometry=self.show_geometry.isChecked(),
                edges=self.edge_mode.currentData(), show_learned=self.show_learned.isChecked())
            self.export_job_path = ROOT/'.cache'/f'export-{uuid.uuid4().hex}.json'
            self.export_job_path.write_text(json.dumps(job), encoding='utf-8')
            self.export_destination = destination
            self.export_cancelled = False
            self.export_output = ''
            self.export_errors = ''
            process = QProcess(self)
            self.export_process = process
            process.setWorkingDirectory(str(ROOT))
            process.readyReadStandardOutput.connect(self.export_progress)
            process.readyReadStandardError.connect(self.export_error_output)
            process.finished.connect(self.export_finished)
            process.errorOccurred.connect(self.export_process_error)
            self.export_btn.setEnabled(False)
            self.cancel_export_btn.setEnabled(True)
            self.export_status.setText('Export starting on CPU · full clip, 10 fps, no audio · inspection remains available.')
            process.start(sys.executable, ['-u', str(ROOT/'media_export.py'), str(self.export_job_path)])
        except Exception as exc:
            self.export_status.setText(f'Could not start export: {exc}')
            self.export_process = None
            self.export_btn.setEnabled(True)
            self.cancel_export_btn.setEnabled(False)

    def export_progress(self):
        self.export_output += bytes(self.export_process.readAllStandardOutput()).decode('utf-8', errors='replace')
        while '\n' in self.export_output:
            line, self.export_output = self.export_output.split('\n', 1)
            try:
                item = json.loads(line)
                self.export_status.setText(f'Exporting on CPU: {item["done"]}/{item["total"]} frames · 10 fps · you can keep inspecting.')
            except (ValueError, KeyError, TypeError):
                pass

    def export_error_output(self):
        self.export_errors = (self.export_errors + bytes(self.export_process.readAllStandardError()).decode('utf-8', errors='replace'))[-4000:]

    def export_process_error(self, error):
        if error == QProcess.FailedToStart:
            self.export_errors = self.export_process.errorString()
            self.export_finished(-1)

    def export_finished(self, exit_code, exit_status=None):
        if self.export_process is None:
            return
        self.export_error_output()
        success = exit_code == 0 and self.export_destination.exists()
        if success:
            message = f'Saved to Downloads: {self.export_destination.name} · 10 fps, no audio'
        elif self.export_cancelled:
            message = 'Video export cancelled. No incomplete download was kept.'
        else:
            message = 'Video export failed: ' + (self.export_errors.strip().splitlines()[-1] if self.export_errors.strip() else 'No completed video was produced.')
        self.export_destination.with_suffix('.partial.mp4').unlink(missing_ok=True)
        self.export_job_path.unlink(missing_ok=True)
        self.export_process.deleteLater()
        self.export_process = None
        self.export_btn.setEnabled(True)
        self.cancel_export_btn.setEnabled(False)
        self.export_status.setText(message)

    def cancel_export(self):
        if self.export_process is not None:
            self.export_cancelled = True
            self.export_process.kill()
            self.export_status.setText('Cancelling background export…')

    def save_evidence(self):
        if self.result is None or self.captured_rgb is None:
            return
        try:
            record = dict(self.result, display_options=dict(opacity=self.opacity.value()/100,
                visible=[self.show_tyre.isChecked(), self.show_tread.isChecked()], mode=self.view_mode.currentText()))
            record['learned_geometry'] = self.learned_frame
            fit = self.current_edge_fit()
            if fit is not None:
                record['edge_geometry'] = json.loads(json.dumps(dict(fit,
                    model=self.overlay_select.currentData(), algorithm='image-edges-v1'),
                    default=lambda value: value.tolist()))
            path = save_capture(self.captured_rgb, record, self.masks, self.result_source, self.note.text())
            self.refresh_tray()
            self.statusBar().showMessage(f'Saved {path.name}. Click a tray entry to restore; double-click to open its evidence card.')
        except Exception as e:
            self.show_error(str(e))

    def refresh_tray(self):
        self.tray.clear()
        for path in sorted((ROOT/'results').glob('*/inspection.json'), reverse=True)[:50]:
            try:
                record = json.loads(path.read_text())
                first = record['models'][0]
                item = QListWidgetItem(path.parent.name + '\n' + first.get('label', 'Region inspection'))
                item.setData(Qt.UserRole, str(path.parent))
                self.tray.addItem(item)
            except (ValueError, KeyError):
                continue

    def restore_capture(self, item):
        if self.busy() or not self.stop_stream():
            return
        try:
            folder = Path(item.data(Qt.UserRole))
            record = json.loads((folder/'inspection.json').read_text())
            rgb = read_image(folder/'frame.png')
            masks = {}
            for r in record['models']:
                name = r['model']
                if r['task'] == 'regions':
                    masks[name] = np.array([read_image(folder/f'{name}-{k}.png')[:, :, 0]>0 for k in ('tyre', 'tread')])
            self.rgb, self.source = rgb, record['source']
            self.source_size = record.get('source_frame_size') or record.get('frame_size')
            self.note.setText(record.get('operator_note', ''))
            record['generation'] = self.generation
            self.on_result(rgb, record, masks, len(record['models']) == 4, self.source)
        except Exception as e:
            self.show_error(str(e))

    def open_card(self, item):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(item.data(Qt.UserRole))/'card.html')))

    def open_results(self):
        (ROOT/'results').mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT/'results')))

    @Slot(str)
    def show_error(self, message):
        self.live_enabled = False
        self.live_btn.setChecked(False)
        self.state_label.setText('NEEDS ATTENTION')
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, 'Inspection could not finish', message)

    def closeEvent(self, event):
        if self.export_process is not None:
            self.export_status.setText('Export is running. Let it finish or use Cancel export before closing.')
            event.ignore()
            return
        for dialog in self.findChildren(QDialog):
            calibration_worker = getattr(dialog, 'worker', None)
            if calibration_worker is not None and calibration_worker.isRunning():
                self.statusBar().showMessage('Camera calibration is running. Close when it finishes.')
                event.ignore()
                return
        if self.busy():
            self.live_enabled = False
            self.statusBar().showMessage('Finishing the current inference. Close again when it completes.')
            event.ignore()
            return
        if not self.stop_stream():
            event.ignore()
            return
        self.worker.shutdown()
        if not self.worker.wait(2000):
            event.ignore()
            return
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    window = Station(args.device)
    window.show()
    if args.image:
        window.load_source(str(args.image))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
