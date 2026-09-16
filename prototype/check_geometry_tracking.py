"""Temporal geometry regression checks, independent of model accuracy."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import numpy as np
from check_geometry import ellipse
from shape_geometry import GeometryTracker, draw_geometry
from PySide6.QtWidgets import QApplication
from app import Station
from theme import apply_theme
from bootstrap import ROOT


def main():
    t = GeometryTracker()
    for i, angle in enumerate([10, 12, 9, 11]):
        g = t.update(ellipse(angle), i*200)
        assert g['stable'] == (i >= 2)
    assert 9 < g['smooth_angle'] < 12
    assert t.update(ellipse(45), 800)['count'] == 1
    assert not t.update(np.zeros((320, 240), bool), 1000)['valid']
    assert t.update(ellipse(10), 1200)['count'] == 1
    assert t.update(ellipse(10), 0)['count'] == 1  # replay
    assert t.update(ellipse(10), 5000)['count'] == 1  # long gap
    t = GeometryTracker()
    for i, angle in enumerate([89, -89, 88]):
        g = t.update(ellipse(angle), i*200)
    assert g['stable'] and abs(g['smooth_angle']) > 85  # undirected axis wrap
    rgb = np.full((320, 240, 3), 210, np.uint8)
    unchanged = rgb.copy()
    assert not np.array_equal(draw_geometry(rgb, g), rgb)
    np.testing.assert_array_equal(rgb, unchanged)
    app = QApplication([])
    apply_theme(app)
    station = Station(device='cpu')
    try:
        station.captured_rgb = rgb
        station.masks = {'test': np.array([ellipse(10), ellipse(10)])}
        station.geometry_frames = {'test': g}
        station.overlay_select.addItem('Test mask', 'test')
        station.overlay_select.setCurrentIndex(station.overlay_select.findData('test'))
        station.view_mode.setCurrentText('Full overlay')
        station.render_capture()
        with_geometry = station.viewer.processed.copy()
        station.show_geometry.setChecked(False)
        assert station.viewer.processed != with_geometry
        station.show_geometry.setChecked(True)
        station.view_mode.setCurrentText('Original')
        assert station.viewer.processed is None
        station.clear_result()
        assert not station.geometry_frames and not station.geometry_tracks
        # Native canvas preview of the overlay on a known synthetic shape.
        from PIL import Image
        Image.fromarray(draw_geometry(rgb, g)).save(ROOT/'results'/'geometry-tracking-check.png')
    finally:
        station.close()
        app.processEvents()
    print('PASS: smoothing, reacquisition, replay/gap resets, angle wrap, immutable pixels, toggle, Original view')


if __name__ == '__main__':
    main()
