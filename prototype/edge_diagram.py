"""Image-based explanation of a frozen edge-fitting result."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
import numpy as np
from edge_geometry import draw_edge_fit


class EdgeDiagram(QDialog):
    def __init__(self, rgb, fit, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Image edges → geometric evidence')
        self.resize(1050, 650)
        layout = QVBoxLayout(self)
        title = QLabel('IMAGE → ACCEPTED EDGES → GEOMETRIC FIT')
        title.setStyleSheet('font-size: 22px; font-weight: 600; color: #174e60')
        layout.addWidget(title)
        layout.addWidget(QLabel(model+' · '+fit['mode']+' · exact inspected frame'))
        panels = QHBoxLayout()
        items = [('1 / SOURCE', rgb, 'The model restricts the search region.'),
                 ('2 / EDGE SUPPORT', draw_edge_fit(rgb, fit, True), 'Green: supports a fit. Red: rejected by fit.'),
                 ('3 / FIT HYPOTHESIS', draw_edge_fit(rgb, fit),
                  'Cyan: sides. Magenta: midline.' if fit['mode'] == 'boundary' else 'Magenta: inner ellipse candidate.')]
        for i, (name, im, caption) in enumerate(items):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            a = np.ascontiguousarray(im)
            image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
            view = QLabel()
            view.setAlignment(Qt.AlignCenter)
            view.setPixmap(QPixmap.fromImage(image).scaled(300, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            column.addWidget(view, 1)
            caption_label = QLabel(caption)
            caption_label.setWordWrap(True)
            column.addWidget(caption_label)
            panels.addLayout(column, 1)
            if i < 2:
                panels.addWidget(QLabel('→'))
        layout.addLayout(panels, 1)
        detail = fit['reason']
        if fit['valid']:
            detail += f" · Typical fit residual: {fit['residual_px']:.1f} source pixels"
        summary = QLabel(detail)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        layout.addWidget(QLabel('Supported image shape → camera-relative evidence → camber / toe still unresolved'))
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        layout.addWidget(close)
