"""Known-shape checks and native diagram/button smoke test."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import cv2
import numpy as np
from shape_geometry import shape_geometry
from PySide6.QtWidgets import QApplication
from app import Station
from theme import apply_theme
from bootstrap import ROOT


def ellipse(angle=0, axes=(45, 110)):
    m = np.zeros((320, 240), np.uint8)
    cv2.ellipse(m, (120, 160), axes, angle, 0, 360, 1, -1)
    return m.astype(bool)


def main():
    for rotation in (-25, 0, 25):
        g = shape_geometry(ellipse(rotation))
        assert g['valid'], g
        assert abs(g['angle']-rotation) < .5, g['angle']
    assert not shape_geometry(ellipse(axes=(70, 70)))['valid']
    assert not shape_geometry(np.zeros((320, 240), bool))['valid']
    clipped = ellipse()
    clipped[:160, 115:125] = True
    assert not shape_geometry(clipped)['valid']
    fragments = np.zeros((320, 240), bool)
    fragments[30:140, 20:70] = True
    fragments[170:280, 150:200] = True
    assert not shape_geometry(fragments)['valid']
    app = QApplication([])
    apply_theme(app)
    station = Station(device='cpu')
    try:
        station.geometry_btn.click()
        app.processEvents()
        assert not station.geometry_dialog.canvas.geometry['valid']
        station.geometry_dialog.close()
        station.captured_rgb = np.full((320, 240, 3), 210, np.uint8)
        station.masks = {'synthetic': np.array([ellipse(20), ellipse(20)])}
        station.overlay_select.addItem('Synthetic verification', 'synthetic')
        station.overlay_select.setCurrentIndex(station.overlay_select.findData('synthetic'))
        station.geometry_btn.click()
        app.processEvents()
        dialog = station.geometry_dialog
        assert dialog.canvas.geometry['valid']
        dialog.roll.setValue(12)
        app.processEvents()
        assert dialog.canvas.roll == 12
        out = ROOT/'results'/'geometry-diagram-check.png'
        out.parent.mkdir(exist_ok=True)
        assert dialog.grab().save(str(out))
        dialog.close()
        from engine import read_image
        from geometry_diagram import GeometryDialog
        saved = next((ROOT/'results').glob('*/segformer_b0-tyre.png'), None)
        if saved is not None:
            real = GeometryDialog(read_image(saved.parent/'frame.png'),
                                  read_image(saved)[:, :, 0] > 0, 'segformer_b0 • saved capture')
            real.show()
            app.processEvents()
            assert real.grab().save(str(ROOT/'results'/'geometry-real-capture.png'))
            real.close()
    finally:
        station.close()
        app.processEvents()
    print('PASS: known signed axes; rejection gates; empty/valid button; slider; diagram screenshot')


if __name__ == '__main__':
    main()
