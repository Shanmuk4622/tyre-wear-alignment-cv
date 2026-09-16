import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import cv2
import numpy as np
from geometry_comparison import compare_fits
from bootstrap import ROOT


def main():
    rgb = np.full((400, 400, 3), 35, np.uint8)
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 175, 1, -1)
    cv2.ellipse(rgb, (200, 200), (100, 130), 20, 0, 360, (220, 220, 220), 3)
    masks = {name: np.array([mask, mask]) for name in ('model_a', 'model_b')}
    fits, summary = compare_fits(rgb, masks)
    assert all(set(v) == {'boundary', 'rim'} for v in fits.values())
    assert 'similar image geometry' in summary
    assert 'no averaged angle' in summary
    bad = np.zeros_like(mask); bad[:, :40] = 1
    _, conflict = compare_fits(rgb, dict(masks, model_b=np.array([bad, bad])))
    assert 'agreement withheld' in conflict
    from PySide6.QtWidgets import QApplication
    from app import Station
    from theme import apply_theme
    app = QApplication([]); apply_theme(app)
    station = Station(device='cpu')
    try:
        station.captured_rgb, station.masks = rgb, masks
        station.compare_geometry()
        dialog = station.comparison_dialog
        app.processEvents()
        dialog.models.setCurrentIndex(1)
        dialog.save()
        assert dialog.grab().save(str(ROOT/'results'/'geometry-comparison-check.png'))
        dialog.close()
    finally:
        station.close(); app.processEvents()
    print('PASS: method separation, supported rim comparison, mask disagreement rejection, native comparison and export')


if __name__ == '__main__':
    main()
