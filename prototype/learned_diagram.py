"""Explain fixed-row learned geometry on the exact analysed frame."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton
from learned_geometry import draw_learned


class LearnedDiagram(QDialog):
    def __init__(self, rgb, record, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Learned tread geometry — six points, three guide rows')
        self.resize(1000, 780)
        layout = QVBoxLayout(self)
        title = QLabel('FULL FRAME → FIXED GUIDE ROWS → SIX BOUNDARIES → WIDTHS + CENTRELINE')
        title.setStyleSheet('font-size:16px;font-weight:600;color:#17697b'); layout.addWidget(title)
        row = QHBoxLayout(); layout.addLayout(row, 1)
        a = np.ascontiguousarray(draw_learned(rgb, record))
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
        view = QLabel(); view.setAlignment(Qt.AlignCenter)
        view.setPixmap(QPixmap.fromImage(image).scaled(470, 590, Qt.KeepAspectRatio, Qt.SmoothTransformation)); row.addWidget(view, 1)
        detail = QVBoxLayout(); row.addLayout(detail, 1)
        explanation = QLabel('Amber dots: HRNet boundary proposals\nCyan squares: matched SegFormer boundary extraction\nPurple: midpoint line between left and right\nRed: proposals needing review\n\nThree guide rows are fixed near 25%, 50%, 75% image height. The model learns horizontal positions, not arbitrary 2-D landmarks.\n\nWidths on the image follow the displayed points; the table below preserves raw predictions. Temporal smoothing changes display only.')
        explanation.setWordWrap(True); detail.addWidget(explanation)
        hr = record['models']['hrnet']; seg = record['models'].get('matched')
        table = QTableWidget(6, 4)
        table.setHorizontalHeaderLabels(['Point', 'HRNet x', 'Matched x', 'Δ px'])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, name in enumerate(hr['names']):
            p = seg['points'][i] if seg else None
            for col, text in enumerate([name, f"{hr['points'][i][0]:.1f}", '—' if p is None else f'{p[0]:.1f}',
                                        '—' if p is None else f"{abs(hr['points'][i][0]-p[0]):.1f}"]):
                table.setItem(i, col, QTableWidgetItem(text))
        table.resizeColumnsToContents(); detail.addWidget(table)
        warning = QLabel('; '.join(record['flags']) or 'No heuristic flags. This is not verified visibility, confidence or physical alignment.')
        warning.setWordWrap(True); detail.addWidget(warning)
        layout.addWidget(QLabel('Seed 1 / epoch 60 · full RGB image resized to 384 × 512 · no crop or inferred calibration'))
        close = QPushButton('Close'); close.clicked.connect(self.close); layout.addWidget(close)
