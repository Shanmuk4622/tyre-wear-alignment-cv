"""Compare unlike geometric hypotheses without averaging their angles."""
import json
from datetime import datetime
import uuid
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem
from edge_geometry import edge_fit, draw_edge_fit
from bootstrap import ROOT


def compare_fits(rgb, masks):
    fits = {name: {mode: edge_fit(rgb, mm[0], mode) for mode in ('boundary', 'rim')} for name, mm in masks.items()}
    messages = []
    regions = [np.asarray(mm[0], bool) for mm in masks.values()]
    same_region = all(np.count_nonzero(a & b)/max(np.count_nonzero(a | b), 1) >= .5
                      for i, a in enumerate(regions) for b in regions[i+1:])
    if not same_region:
        messages.append('Region masks disagree or select different areas; cross-model agreement withheld.')
    boundaries = [v['boundary']['angle'] for v in fits.values() if v['boundary']['valid']]
    rims = [v['rim']['axis_ratio'] for v in fits.values() if v['rim']['valid']]
    if same_region and len(boundaries) > 1:
        delta = max(abs((a-b+90) % 180-90) for a in boundaries for b in boundaries)
        messages.append(f"Boundary axes across models: {delta:.1f}° spread"+(' — DISAGREEMENT' if delta > 5 else ' — similar image geometry'))
    if same_region and len(rims) > 1:
        delta = max(rims)-min(rims)
        messages.append(f"Rim axis ratios across models: {delta:.3f} spread"+(' — DISAGREEMENT' if delta > .1 else ' — similar image geometry'))
    if not messages:
        messages.append('Insufficient supported fits for cross-model agreement. Run Compare all four to include both region models.')
    messages.append('Boundary midline and rim axis ratio describe different shapes: no averaged angle or inferred camber/toe.')
    return fits, '\n'.join(messages)


class GeometryComparison(QDialog):
    def __init__(self, rgb, masks, station):
        super().__init__(station)
        self.rgb = rgb
        self.fits, summary = compare_fits(rgb, masks)
        self.setWindowTitle('Boundary ↔ rim comparison')
        self.resize(1080, 810)
        layout = QVBoxLayout(self)
        title = QLabel('SAME FRAME → TWO SHAPE HYPOTHESES → CHECK THEIR SUPPORT')
        title.setStyleSheet('font-size:18px;font-weight:600;color:#17697b'); layout.addWidget(title)
        self.models = QComboBox(); self.models.addItems(list(self.fits)); layout.addWidget(self.models)
        images = QHBoxLayout(); layout.addLayout(images, 1)
        self.views = []
        for name in ('BOUNDARIES / cyan sides + magenta midline', 'RIM / magenta candidate ellipse'):
            column = QVBoxLayout(); images.addLayout(column, 1)
            column.addWidget(QLabel(name))
            view = QLabel(); view.setAlignment(Qt.AlignCenter); column.addWidget(view, 1); self.views.append(view)
        table = QTableWidget(len(self.fits)*2, 4)
        table.setHorizontalHeaderLabels(['Model / method', 'Supported?', 'Image measurement', 'Fit evidence'])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        for i, (name, methods) in enumerate(self.fits.items()):
            for j, (mode, fit) in enumerate(methods.items()):
                measurement = '—'
                if fit['valid']:
                    measurement = f"Midline {fit['angle']:+.2f}°" if mode == 'boundary' else f"Axis ratio {fit['axis_ratio']:.3f}"
                evidence = f"{fit['residual_px']:.2f}px residual; " if fit['valid'] else ''
                for col, value in enumerate([name+' / '+mode, 'Yes (hypothesis)' if fit['valid'] else 'Withheld', measurement, evidence+fit['reason']]):
                    table.setItem(i*2+j, col, QTableWidgetItem(value))
        table.resizeColumnsToContents(); table.setMaximumHeight(190); layout.addWidget(table)
        explanation = QLabel(summary); explanation.setWordWrap(True); layout.addWidget(explanation)
        row = QHBoxLayout(); layout.addLayout(row)
        for text, callback in [('Save comparison', self.save), ('Open calibrated alignment', station.open_alignment), ('Close', self.close)]:
            b = QPushButton(text); b.clicked.connect(callback); row.addWidget(b)
        self.models.currentTextChanged.connect(self.render)
        self.render()

    def render(self, *args):
        methods = self.fits.get(self.models.currentText())
        if methods is None:
            return
        for view, mode in zip(self.views, ('boundary', 'rim')):
            a = np.ascontiguousarray(draw_edge_fit(self.rgb, methods[mode]))
            im = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
            view.setPixmap(QPixmap.fromImage(im).scaled(470, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save(self):
        from PIL import Image
        folder = ROOT/'results'/'geometry-comparisons'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        folder.mkdir(parents=True)
        Image.fromarray(self.rgb).save(folder/'frame.png')
        (folder/'comparison.json').write_text(json.dumps(self.fits, default=lambda v: v.tolist(), indent=2), encoding='utf-8')
        self.grab().save(str(folder/'comparison.png'))
        self.setWindowTitle('Comparison saved: '+str(folder))
