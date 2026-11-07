"""The single Phase 3 window and explicit frame handoff between its workspaces."""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QApplication, QFrame, QLabel, QMainWindow, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)

from imaging import fingerprint
from legacy_bench import open_bench


class Workspace(QMainWindow):
    def __init__(self, custom, device='auto'):
        super().__init__()
        self.custom = custom
        self.device = device
        self.bench = None
        self.bench_closed = False
        self.previous_tab = 0
        self.setWindowTitle('Tread Station | Phase 3')
        self.setMinimumSize(1000, 630)
        area = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1470, area.width()-24), min(920, area.height()-45))

        self.tabs = QTabWidget()
        self.tabs.setObjectName('workspaces')
        self.setCentralWidget(self.tabs)
        self.custom.setWindowFlags(Qt.Widget)
        self.tabs.addTab(self.custom, 'TREAD-ONE')
        self.attach_status(self.custom, 0)
        self.placeholder = QWidget()
        layout = QVBoxLayout(self.placeholder)
        self.bench_status = QLabel('Loading the model bench…')
        self.bench_status.setWordWrap(True)
        layout.addStretch()
        layout.addWidget(self.bench_status, alignment=Qt.AlignCenter)
        retry = QPushButton('Retry')
        retry.clicked.connect(self.ensure_bench)
        self.retry_bench = retry
        retry.hide()
        layout.addWidget(retry, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.tabs.addTab(self.placeholder, 'Model bench')
        self.transfer = QPushButton('Send frame to model bench')
        self.transfer.setToolTip('Send the displayed analysis snapshot, or the current frame before inspection.')
        self.transfer.clicked.connect(self.transfer_frame)
        self.tabs.setCornerWidget(self.transfer, Qt.TopRightCorner)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage(self.custom.statusBar().currentMessage())

    def attach_status(self, page, index):
        bar = page.statusBar()
        bar.hide()
        bar.messageChanged.connect(
            lambda message: self.statusBar().showMessage(message) if self.tabs.currentIndex() == index else None)

    def ensure_bench(self):
        if self.bench is not None:
            return self.bench
        self.retry_bench.hide()
        self.bench_status.setText('Loading the model bench…')
        try:
            bench = open_bench(self.device)
        except Exception as exc:
            self.bench_status.setText(f'The model bench could not open.\n{exc}')
            self.retry_bench.show()
            return None
        root = bench.centralWidget()
        layout = root.layout()
        layout.setContentsMargins(10, 8, 10, 7)
        layout.setSpacing(7)
        header = layout.itemAt(0).layout()
        header.setContentsMargins(0, 0, 0, 0)
        for heading in root.findChildren(QLabel):
            replacements = {
                'TYRE STUDY / FIELD INSTRUMENT 01': '',
                'Tread Station': 'Inspection',
                '01 / INSPECTION RECIPE': 'Model settings',
                '02 / MODEL OBSERVATIONS': 'Observations',
                '03 / EVIDENCE TRAY': 'Saved records',
                'PORTRAIT / DISPLAY = MODEL INPUT': 'File orientation',
            }
            if heading.text() in replacements:
                value = replacements[heading.text()]
                if value:
                    heading.setText(value)
                else:
                    heading.hide()
        bench.state_label.setMinimumWidth(145)
        bench.right_scroll.setMinimumWidth(310)
        self.organize_bench(bench)
        bench.setWindowFlags(Qt.Widget)
        self.bench = bench
        active = self.tabs.currentIndex()
        self.tabs.blockSignals(True)
        self.tabs.removeTab(1)
        self.tabs.addTab(bench, 'Model bench')
        self.tabs.setCurrentIndex(active)
        self.tabs.blockSignals(False)
        self.attach_status(bench, 1)
        self.sync_shortcuts(active)
        if active == 1:
            self.statusBar().showMessage(bench.statusBar().currentMessage())
        return bench

    @staticmethod
    def organize_bench(bench):
        """Reflow the existing controls without duplicating their handlers."""
        def transfer_item(item, target):
            child = item.layout()
            if child is not None:
                child.setParent(None)
                target.addLayout(child)
            elif item.widget() is not None:
                target.addWidget(item.widget())
            else:
                target.addItem(item)

        def page(tabs, title, scrollable=False):
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(5)
            if scrollable:
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QFrame.NoFrame)
                scroll.setWidget(content)
                tabs.addTab(scroll, title)
            else:
                tabs.addTab(content, title)
            return layout

        left = bench.left_scroll.widget().layout()
        tools = QTabWidget()
        geometry = page(tools, 'Geometry and alignment')
        evidence = page(tools, 'Evidence and export')
        # The first three items are the viewport, overlay chooser and common display controls.
        while left.count() > 3:
            item = left.takeAt(3)
            target = geometry if geometry.count() < 4 else evidence
            transfer_item(item, target)
        geometry.addStretch()
        left.addWidget(tools)

        right = bench.right_scroll.takeWidget()
        settings = right.layout()
        inspector = QTabWidget()
        pages = {title: page(inspector, title, scrollable=True)
                 for title in ('Model settings', 'Observations', 'Saved records')}
        target = pages['Model settings']
        while settings.count():
            item = settings.takeAt(0)
            heading = item.widget()
            if isinstance(heading, QLabel) and heading.text() in pages:
                target = pages[heading.text()]
                heading.hide()
            else:
                transfer_item(item, target)
        for target in pages.values():
            target.addStretch()
        bench.right_scroll.setWidget(inspector)
        bench.inspector_tabs = inspector
        bench.worker.complete.connect(lambda *_: inspector.setCurrentIndex(1) if bench.stream is None else None)
        bench.model_list.setObjectName('benchModels')

        def size_observations():
            model_list = bench.model_list
            rows = sum(model_list.sizeHintForRow(index) for index in range(model_list.count()))
            model_list.setFixedHeight(max(90, min(460, rows + 4)))
            if bench.result is not None:
                result = bench.result
                bench.hardware.setText(f"{result['device_name']} · {result['peak_vram_mb']:.0f} MB")

        bench.model_list.model().rowsInserted.connect(lambda *_: QTimer.singleShot(0, size_observations))
        right.deleteLater()

    def sync_shortcuts(self, index):
        for page, active in ((self.custom, index == 0), (self.bench, index == 1)):
            if page is not None:
                for shortcut in page.findChildren(QShortcut):
                    shortcut.setEnabled(active)

    def tab_changed(self, index):
        if index < 0:
            return
        if index != self.previous_tab:
            if index == 1 and self.custom.media.cap is not None:
                if self.custom.media.meta.get('media_kind') == 'camera':
                    self.custom.stop_media()
                else:
                    self.custom.media.pause()
                    self.custom.play_btn.setText('Play')
            if index == 0 and self.bench is not None and self.bench.stream is not None:
                if isinstance(self.bench.stream.source, int):
                    self.bench.stop_stream()
                else:
                    self.bench.stream.paused.set()
                    self.bench.live_enabled = False
                    self.bench.live_btn.setChecked(False)
                    self.bench.freeze_btn.setText('Resume   [Space]')
        self.previous_tab = index
        self.sync_shortcuts(index)
        page = self.custom if index == 0 else self.bench
        self.statusBar().showMessage(page.statusBar().currentMessage() if page else 'Opening the model bench…')
        self.transfer.setText('Send frame to model bench' if index == 0 else 'Send frame to TREAD-ONE')
        if index == 1 and self.bench is None:
            QTimer.singleShot(0, self.ensure_bench)

    def transfer_frame(self):
        if self.tabs.currentIndex() == 0:
            if self.custom.rgb is None:
                self.statusBar().showMessage('Select a TREAD-ONE frame first.', 5000)
                return
            bench = self.ensure_bench()
            if bench is None:
                self.tabs.setCurrentIndex(1)
                return
            if bench.busy() or not bench.stop_stream():
                self.statusBar().showMessage('The model bench is busy. Send the frame when it is ready.', 5000)
                return
            rgb, source = self.custom.rgb, self.custom.source
            bench.clear_result()
            bench.note.clear()
            bench.rgb = rgb.copy()
            bench.source = str(source.get('path') or 'TREAD-ONE snapshot')
            bench.source_size = [rgb.shape[1], rgb.shape[0]]
            timestamp = source.get('timestamp_seconds')
            bench.frame_time_ms = timestamp * 1000 if timestamp is not None else None
            bench.timeline.setEnabled(False)
            bench.video_clock.setText('Frozen frame from TREAD-ONE')
            bench.viewer.set_frame(bench.rgb)
            bench.frame_label.setText(Path(bench.source).name)
            bench.state_label.setText('FRAME RECEIVED')
            self.custom.session.event('frame_sent_to_model_bench', fingerprint(rgb))
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage('Frame received by the model bench. Choose Inspect or Compare.', 7000)
            return
        bench = self.bench
        if bench is None or bench.rgb is None:
            self.statusBar().showMessage('Select a model bench frame first.', 5000)
            return
        if bench.captured_rgb is not None:
            rgb, path, time_ms = bench.captured_rgb, bench.result_source, bench.result.get('source_time_ms')
        else:
            rgb, path, time_ms = bench.rgb, bench.source, bench.frame_time_ms
        kind = ('camera' if path.startswith(('camera:', 'Camera ')) else
                'video' if time_ms is not None else 'photo')
        source = dict(path=path, media_kind=kind, timestamp_seconds=time_ms / 1000 if time_ms is not None else None,
                      source_media_sha256=None, source_hash_reason='Exact RGB snapshot from the model bench',
                      timestamp_source='Original model bench snapshot clock' if time_ms is not None else None)
        self.custom.receive_bench_frame(rgb, source)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage('Frame received by TREAD-ONE. Choose Inspect to run the model.', 7000)

    def closeEvent(self, event):
        if self.custom.alignment_busy():
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage('Wait for wheel calibration to finish before closing.', 6000)
            event.ignore()
            return
        if self.custom.exporter is not None and self.custom.exporter.isRunning():
            self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage('Wait for the TREAD-ONE evidence capture to finish.', 6000)
            event.ignore()
            return
        if self.bench is not None and not self.bench_closed:
            if not self.bench.close():
                self.tabs.setCurrentIndex(1)
                event.ignore()
                return
            self.bench_closed = True
        if not self.custom.close():
            event.ignore()
            QTimer.singleShot(1100, self.close)
            return
        event.accept()
