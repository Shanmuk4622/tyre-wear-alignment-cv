"""Known image edges, unsupported masks and native controls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import cv2
import numpy as np
import json
from edge_geometry import edge_fit, robust_line, draw_edge_fit
from bootstrap import ROOT


def main():
    rng = np.random.default_rng(3)
    y = np.linspace(0, 250, 100)
    p = np.column_stack((.2*y+30+rng.normal(0, .3, 100), y))
    p[::4, 0] += 30
    f = robust_line(p)
    assert abs(f['a']-.2) < .01 and .7 < f['fraction'] < .8
    rgb = np.full((400, 300, 3), 230, np.uint8)
    mask = np.zeros((400, 300), np.uint8)
    cv2.fillPoly(mask, [np.array([[110, 35], [220, 35], [175, 365], [65, 365]])], 1)
    rgb[mask > 0] = 45
    fit = edge_fit(rgb, mask, 'boundary')
    assert fit['valid'], fit['reason']
    expected = np.degrees(np.arctan(45/330))
    assert abs(fit['angle']-expected) < 1
    large = edge_fit(cv2.resize(rgb, (900, 1200)), cv2.resize(mask, (900, 1200), interpolation=cv2.INTER_NEAREST))
    assert large['valid'] and abs(large['angle']-expected) < 1
    assert not edge_fit(np.full_like(rgb, 100), mask)['valid']  # mask alone is not evidence
    assert not edge_fit(rgb, None)['valid']
    cv2.rectangle(mask, (0, 100), (120, 200), 1, -1)
    assert not edge_fit(rgb, mask)['valid']
    side = np.full((400, 400, 3), 35, np.uint8)
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 175, 1, -1)
    cv2.ellipse(side, (200, 200), (100, 130), 20, 0, 360, (220, 220, 220), 3)
    rim = edge_fit(side, mask, 'rim')
    assert rim['valid'], rim['reason']
    assert abs(rim['axis_ratio']-100/130) < .05
    assert rim['coverage'] >= .8
    partial = np.full_like(side, 35)
    cv2.ellipse(partial, (200, 200), (100, 130), 20, 0, 100, (220, 220, 220), 3)
    assert not edge_fit(partial, mask, 'rim')['valid']
    original = side.copy()
    assert draw_edge_fit(side, rim).shape == side.shape
    np.testing.assert_array_equal(original, side)
    restored = json.loads(json.dumps(rim, default=lambda value: value.tolist()))
    np.testing.assert_array_equal(draw_edge_fit(side, restored), draw_edge_fit(side, rim))
    from PySide6.QtWidgets import QApplication
    from app import Station
    from theme import apply_theme
    app = QApplication([])
    apply_theme(app)
    window = Station(device='cpu')
    try:
        window.captured_rgb = side
        window.masks = {'test': np.array([mask, mask])}
        window.overlay_select.addItem('Synthetic rim', 'test')
        window.overlay_select.setCurrentIndex(window.overlay_select.findData('test'))
        window.edge_mode.setCurrentIndex(window.edge_mode.findData('rim'))
        assert window.current_edge_fit()['valid']
        cached = window.current_edge_fit()
        window.opacity.setValue(20)
        assert cached is window.current_edge_fit()
        window.explain_edges()
        app.processEvents()
        assert window.edge_dialog.grab().save(str(ROOT/'results'/'edge-fit-explanation.png'))
        window.edge_dialog.close()
        window.view_mode.setCurrentText('Original')
        assert window.viewer.processed is None
        window.edge_mode.setCurrentIndex(window.edge_mode.findData(''))
        assert window.current_edge_fit() is None
    finally:
        window.close()
        app.processEvents()
    print('PASS: outliers, signed midline, blank/clipped masks, ellipse, partial arc rejection, cache, diagram, Original view')


if __name__ == '__main__':
    main()
