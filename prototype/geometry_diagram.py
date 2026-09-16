"""Native visual explanation, using a snapshot of the selected model's mask."""
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QImage
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
import numpy as np
from shape_geometry import shape_geometry


class GeometryCanvas(QLabel):
    def __init__(self, rgb, mask, model):
        super().__init__()
        self.setMinimumSize(880, 620)
        self.rgb = rgb
        self.model = model
        self.geometry = shape_geometry(mask) if mask is not None else dict(valid=False, reason='Inspect a frame with a tyre region model')
        self.roll = 0

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(self.width()/1000, self.height()/700)
        p.fillRect(QRectF(0, 0, 1000, 700), QColor('#f1f6fa'))
        def text(x, y, w, h, value, size=12, color='#244353'):
            p.setPen(QColor(color))
            p.setFont(QFont('Segoe UI', size))
            p.drawText(QRectF(x, y, w, h), Qt.TextWordWrap | Qt.AlignLeft, value)
        def line(a, b, color='#098781', dashed=False):
            pen = QPen(QColor(color), 3)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(*a), QPointF(*b))
        text(25, 16, 950, 36, 'SHAPE COMPASS  /  Geometry from this captured frame', 18)
        text(25, 58, 950, 32, f'{self.model}  •  Frozen snapshot  •  Image coordinates', 10)
        g = self.geometry
        for i, title in enumerate(('1  MODEL → SILHOUETTE', '2  PIXELS → SHAPE AXIS', '3  AXIS → APPARENT TILT')):
            x = 25+i*325
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawRoundedRect(QRectF(x, 98, 300, 390), 12, 12)
            text(x+14, 112, 280, 30, title, 11)
            if i < 2 and self.rgb is not None:
                a = self.rgb.copy() if i == 0 else np.full_like(self.rgb, 237)
                if 'mask' in g:
                    m = g['mask']
                    a[m] = (a[m]*.45 + np.array([15, 164, 149])*.55).astype(np.uint8) if i == 0 else (15, 164, 149)
                a = np.ascontiguousarray(a)
                im = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
                scale = min(266/a.shape[1], 245/a.shape[0])
                r = QRectF(x+150-a.shape[1]*scale/2, 155, a.shape[1]*scale, a.shape[0]*scale)
                p.drawImage(r, im)
                if i == 1 and 'axis' in g:
                    c = np.array([r.x(), r.y()])+g['center']*scale
                    d = g['axis']*95
                    line(c-d, c+d, '#d88013')
                    line((c[0], 157), (c[0], 397), '#547488', True)
            if i == 0:
                text(x+14, 416, 276, 65, 'Keep the largest connected region.\nThis is the visible tyre silhouette.', 11)
            elif i == 1:
                text(x+14, 416, 276, 65, 'Amber: longest spread of mask pixels (PCA). Dashed: image vertical.', 11)
        cx, cy = 825, 280
        angle = g.get('angle', 0)
        a = np.radians(angle)
        r = np.radians(self.roll)
        line((cx, cy+100), (cx, cy-100), '#547488', True)
        line((cx-np.sin(r)*100, cy+np.cos(r)*100), (cx+np.sin(r)*100, cy-np.cos(r)*100), '#486bdd')
        if 'axis' in g:
            line((cx-np.sin(a)*100, cy+np.cos(a)*100), (cx+np.sin(a)*100, cy-np.cos(a)*100), '#d88013')
        value = ((angle-self.roll+90) % 180)-90
        text(692, 390, 270, 88, f'{value:+.1f}° apparent axis tilt\nBlue reference: {self.roll:+d}° (assumed)' if g['valid'] else 'ANGLE WITHHELD\n'+g['reason'], 12)
        text(25, 505, 950, 40, 'WHY THIS IS A SHAPE MEASUREMENT', 13)
        text(25, 547, 295, 90, 'Wheel pose\n       + camera pose\n       + visible / missing boundary', 12)
        text(340, 558, 300, 70, '→  Same 2-D silhouette can have\n     different 3-D explanations', 12)
        text(680, 550, 290, 90, 'Camber / toe: unresolved\nNext: stable video + rim outline\nLater: measured reference', 12, '#9b5718')
        text(25, 656, 950, 30, 'Slider changes the assumed reference only. It does not calibrate the camera or alter the image.', 10)
        p.end()


class GeometryDialog(QDialog):
    def __init__(self, rgb, mask, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Shape Compass — explain the geometry')
        self.resize(1050, 790)
        layout = QVBoxLayout(self)
        self.canvas = GeometryCanvas(rgb, mask, model)
        layout.addWidget(self.canvas)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Assumed vertical in image'))
        self.roll = QSlider(Qt.Horizontal)
        self.roll.setRange(-30, 30)
        self.roll.valueChanged.connect(self.change_roll)
        controls.addWidget(self.roll)
        reset = QPushButton('Reset reference')
        reset.clicked.connect(lambda: self.roll.setValue(0))
        controls.addWidget(reset)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        controls.addWidget(close)
        layout.addLayout(controls)

    def change_roll(self, value):
        self.canvas.roll = value
        self.canvas.update()
