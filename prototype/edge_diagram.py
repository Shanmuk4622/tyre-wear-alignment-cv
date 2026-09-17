"""Image-based explanation of a frozen edge-fitting result."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
import numpy as np
import cv2
from edge_geometry import draw_edge_fit


class EdgeDiagram(QDialog):
    def __init__(self, rgb, fit, model, parent=None, mask=None):
        super().__init__(parent)
        self.setWindowTitle('Image edges → geometric evidence')
        self.resize(1050, 650)
        layout = QVBoxLayout(self)
        title = QLabel('IMAGE → ACCEPTED EDGES → GEOMETRIC FIT')
        title.setStyleSheet('font-size: 22px; font-weight: 600; color: #174e60')
        layout.addWidget(title)
        layout.addWidget(QLabel(model+' · '+fit['mode']+' · exact inspected frame'))
        guided = rgb.copy()
        if mask is not None:
            contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            thickness = max(2, round(max(rgb.shape[:2])/180))
            cv2.drawContours(guided, contours, -1, (235, 160, 35), thickness)
            h, w = mask.shape
            for clipped, a, b in ((mask[:, 0].any(), (2, 0), (2, h-1)),
                                  (mask[:, -1].any(), (w-3, 0), (w-3, h-1))):
                if clipped:
                    cv2.line(guided, a, b, (225, 70, 80), thickness*2)
        valid = fit['valid']
        self.fit_notice = QLabel('FIT AVAILABLE · inspect the coloured evidence below' if valid else
                                'FIT WITHHELD · '+fit['reason']+'\nAmber = predicted search outline. Red frame edge = clipped mask. No accepted geometry is drawn.')
        self.fit_notice.setWordWrap(True)
        self.fit_notice.setStyleSheet('background: #fff2dc; color: #713d0a; padding: 12px; font-weight: 600;')
        layout.addWidget(self.fit_notice)
        panels = QHBoxLayout()
        items = [('1 / SEARCH REGION', guided, 'Amber: model search outline, not a measured boundary.'),
                 ('2 / EDGE SUPPORT', draw_edge_fit(guided, fit, True),
                  'Green: supports a fit. Red dots: rejected by fit.' if len(fit['points']) else 'No edge support accepted. Check the highlighted region and frame edges.'),
                 ('3 / FIT HYPOTHESIS' if valid else '3 / FIT WITHHELD', draw_edge_fit(rgb, fit) if valid else guided,
                  ('Cyan: sides. Magenta: midline.' if fit['mode'] == 'boundary' else 'Magenta: inner ellipse candidate.') if valid else
                  'No fitted lines. Include both tyre sides with background margin; inspect a clearer frame.')]
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
