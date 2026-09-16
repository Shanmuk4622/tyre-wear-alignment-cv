"""Synthetic camera/target validation; does not establish workshop accuracy."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import cv2
import numpy as np
from alignment import BOARD, board, detect, calibrate, measure, solve_pose, angles_from_normal, camera_for, export_targets
from bootstrap import ROOT

K = np.array([[1200., 0, 640], [0, 1200., 480], [0, 0, 1]])
SIZE = (1280, 960)


def render_target(canvas, rotation, translation, wheel=False):
    target = board(wheel).generateImage((600, 840), marginSize=0)
    obj = np.array([[0, 0, 0], [.15, 0, 0], [.15, .21, 0], [0, .21, 0]], np.float32)
    dst, _ = cv2.projectPoints(obj, cv2.Rodrigues(rotation)[0], np.array(translation, float), K, np.zeros(5))
    transform = cv2.getPerspectiveTransform(np.array([[0, 0], [599, 0], [599, 839], [0, 839]], np.float32), dst.reshape(4, 2))
    warped = cv2.warpPerspective(target, transform, SIZE, borderValue=255)
    canvas[:] = np.minimum(canvas, warped[:, :, None])


def scene(camber=2., toe=1., side='left'):
    image = np.full((960, 1280, 3), 255, np.uint8)
    ground = cv2.Rodrigues(np.array([-.8, .1, .1]))[0]
    c, t = np.radians([camber, toe])
    normal = np.array([np.sin(t)*np.cos(c), (1 if side == 'left' else -1)*np.cos(t)*np.cos(c), -np.sin(c)])
    n = ground @ np.diag([1., -1., -1.]) @ normal
    if n[2] < 0:
        n = -n
    x = np.array([1., 0, 0]); x -= n*np.dot(x, n); x /= np.linalg.norm(x)
    wheel = np.column_stack((x, np.cross(n, x), n))
    render_target(image, ground, [-.24, -.1, 1.])
    render_target(image, wheel, [.08, -.06, 1.], True)
    return image


def main():
    profile = dict(version=1, board=BOARD, camera_label='SYNTHETIC TEST CAMERA', image_size=list(SIZE),
                   camera_matrix=K.tolist(), distortion=[0.]*5, rms_px=0., views=12)
    for side in ('left', 'right'):
        for camber, toe in [(-3, -2), (0, 0), (2, 1)]:
            c, t = np.radians([camber, toe])
            n = [np.sin(t)*np.cos(c), (1 if side == 'left' else -1)*np.cos(t)*np.cos(c), -np.sin(c)]
            a = angles_from_normal(n, side)
            assert abs(a['camber_deg']-camber) < 1e-8
            assert abs(a['toe_in_deg']-toe) < 1e-8
    rgb = scene()
    r, overlay = measure(rgb, profile, 'left', True)
    assert abs(r['camber_deg']-2) < .7, r
    assert abs(r['toe_in_deg']-1) < .7, r
    right, _ = measure(scene(-2, -1, 'right'), profile, 'right', True)
    assert abs(right['camber_deg']+2) < .7 and abs(right['toe_in_deg']+1) < .7, right
    for bad, confirmed in [(np.full_like(rgb, 255), True), (rgb, False)]:
        try:
            measure(bad, profile, 'left', confirmed)
            raise AssertionError('Unsupported input accepted')
        except ValueError:
            pass
    try:
        camera_for(profile, np.zeros((1280, 960, 3), np.uint8))
        raise AssertionError('Orientation mismatch accepted')
    except ValueError:
        pass
    scaled, _ = camera_for(profile, np.zeros((480, 640, 3), np.uint8))
    assert scaled[0, 0] == 600
    obj = board().getChessboardCorners().reshape(-1, 1, 3)
    img, _ = cv2.projectPoints(obj, np.array([.12, .1, 0.]), np.array([0., 0., 4.]), K, np.zeros(5))
    try:
        solve_pose(obj, img, K, np.zeros(5))
        raise AssertionError('Ambiguous distant target accepted')
    except ValueError as e:
        assert 'ambiguous' in str(e)
    # Test intrinsic estimation from rendered printed-target images.
    images = []
    rng = np.random.default_rng(7)
    for i in range(16):
        im = np.full_like(rgb, 255)
        rotation = cv2.Rodrigues(np.array([rng.uniform(-.65, .65), rng.uniform(-.55, .55), rng.uniform(-.25, .25)]))[0]
        render_target(im, rotation, [rng.uniform(-.26, .12), rng.uniform(-.2, .08), rng.uniform(.8, 1.15)])
        images.append(im)
    estimated = calibrate(images, 'SYNTHETIC rendered target calibration')
    assert abs(estimated['camera_matrix'][0][0]-1200)/1200 < .04, estimated
    assert abs(estimated['camera_matrix'][1][1]-1200)/1200 < .04
    try:
        calibrate([images[0]]*12, 'Repeated image')
        raise AssertionError('Duplicate calibration photos accepted')
    except ValueError:
        pass
    export_targets(ROOT/'results'/'alignment-test-targets')
    from PIL import Image
    Image.fromarray(rgb).save(ROOT/'results'/'alignment-synthetic-targets.png')
    Image.fromarray(overlay).save(ROOT/'results'/'alignment-synthetic-axes.png')
    from PySide6.QtWidgets import QApplication
    from app import Station
    from theme import apply_theme
    app = QApplication([]); apply_theme(app)
    window = Station(device='cpu')
    try:
        window.open_alignment()
        dialog = window.alignment_dialog
        from unittest.mock import patch
        import time
        input_dir = ROOT/'results'/'calibration-test-photos'
        input_dir.mkdir(exist_ok=True)
        paths = []
        for i, image in enumerate(images):
            path = input_dir/f'{i:02d}.png'
            Image.fromarray(image).save(path); paths.append(str(path))
        with patch('alignment_ui.QFileDialog.getOpenFileNames', return_value=(paths, '')):
            dialog.build_profile()
        deadline = time.monotonic()+40
        while dialog.worker.isRunning() or dialog.profile is None:
            app.processEvents()
            assert time.monotonic() < deadline, dialog.profile_status.text()
            time.sleep(.01)
        assert dialog.profile['views'] == 16
        dialog.set_profile(profile); dialog.confirm.setChecked(True)
        dialog.evaluate(rgb)
        assert dialog.result is not None
        dialog.reference.setChecked(True); dialog.ref_camber.setValue(2); dialog.ref_toe.setValue(1)
        assert 'Difference from reference' in dialog.status.text()
        dialog.save_result()
        app.processEvents()
        assert dialog.grab().save(str(ROOT/'results'/'alignment-bench-check.png'))
        dialog.confirm.setChecked(False)
        assert dialog.result is None
        dialog.close()
    finally:
        window.close(); app.processEvents()
    print('PASS: signed camber/toe both sides, rendered dual targets, camera calibration, missing targets, setup gates, scaling/orientation, native UI and evidence')


if __name__ == '__main__':
    main()
