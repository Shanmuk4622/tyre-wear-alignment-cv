"""Target-assisted wheel-plane geometry in an explicitly defined vehicle frame."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

BOARD = dict(squares=[5, 7], square_m=.03, marker_m=.022, dictionary='DICT_4X4_100')


def board(wheel=False):
    ids = np.arange(50 if wheel else 0, 67 if wheel else 17, dtype=np.int32)
    return cv2.aruco.CharucoBoard((5, 7), .03, .022,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100), ids)


def detect(rgb, wheel=False):
    b = board(wheel)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or len(ids) < 8:
        raise ValueError(('Wheel' if wheel else 'Ground')+' target: at least 8 corners must be visible')
    obj, img = b.matchImagePoints(corners, ids)
    if np.linalg.matrix_rank(obj.reshape(-1, 3)[:, :2]-obj.reshape(-1, 3)[:, :2].mean(0)) < 2:
        raise ValueError('Target corners are collinear')
    return obj, img


def validate_profile(profile):
    if profile.get('version') != 1 or profile.get('board') != BOARD:
        raise ValueError('Use a Tread Station calibration profile with the matching target')
    k = np.asarray(profile['camera_matrix'], float)
    d = np.asarray(profile['distortion'], float).reshape(-1)
    size = np.asarray(profile['image_size'], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Invalid camera matrix')
    if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all() or size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError('Invalid distortion or image size')
    if not np.isfinite(profile.get('rms_px', np.inf)) or profile['rms_px'] > 1.5 or profile.get('views', 0) < 10:
        raise ValueError('Calibration needs at least 10 views and RMS no greater than 1.5px')
    return k, d


def camera_for(profile, rgb):
    k, d = validate_profile(profile)
    w, h = profile['image_size']
    sx, sy = rgb.shape[1]/w, rgb.shape[0]/h
    if abs(sx-sy)/max(sx, sy) > .002:
        raise ValueError('Frame aspect/orientation differs from calibration; use the calibrated capture mode')
    k = k.copy()
    k[0] *= sx
    k[1] *= sy
    return k, d


def calibrate(images, label):
    objects, points, hashes = [], [], []
    size = None
    for rgb in images:
        current = (rgb.shape[1], rgb.shape[0])
        if size is not None and current != size:
            raise ValueError('All calibration images must use the same resolution and orientation')
        size = current
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in hashes:
            continue
        obj, img = detect(rgb)
        objects.append(obj)
        points.append(img)
        hashes.append(digest)
    if len(objects) < 10:
        raise ValueError('Capture at least 10 distinct target views at varied positions and tilts')
    rms, k, d, rotations, translations = cv2.calibrateCamera(objects, points, size, None, None)
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rotations])
    spread = np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max()
    centers = np.array([p.reshape(-1, 2).mean(0) for p in points])/size
    if spread < 15 or np.max(np.ptp(centers, axis=0)) < .2:
        raise ValueError('Views need more tilt and movement across the frame; repeat near-identical views are insufficient')
    per_view = []
    for obj, img, r, t in zip(objects, points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, k, d)
        per_view.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-img.reshape(-1, 2))**2, axis=1)))))
    profile = dict(version=1, board=BOARD, camera_label=label, image_size=list(size),
                   camera_matrix=k.tolist(), distortion=d.reshape(-1).tolist(), rms_px=float(rms),
                   views=len(objects), per_view_rms_px=per_view, image_hashes=hashes)
    validate_profile(profile)
    return profile


def solve_pose(obj, img, k, d):
    _, rotations, translations, _ = cv2.solvePnPGeneric(obj, img, k, d, flags=cv2.SOLVEPNP_IPPE)
    poses = []
    for r, t in zip(rotations, translations):
        rotation = cv2.Rodrigues(r)[0]
        if np.min((obj.reshape(-1, 3) @ rotation.T+t.reshape(1, 3))[:, 2]) <= 0:
            continue
        predicted, _ = cv2.projectPoints(obj, r, t, k, d)
        error = float(np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2)-img.reshape(-1, 2))**2, axis=1))))
        poses.append(dict(rotation=rotation, rvec=r, tvec=t, rms_px=error))
    poses.sort(key=lambda p: p['rms_px'])
    if not poses or poses[0]['rms_px'] > 1.5:
        raise ValueError('Target pose has too much reprojection error')
    if len(poses) > 1:
        delta = np.degrees(np.arccos(np.clip(poses[0]['rotation'][:, 2] @ poses[1]['rotation'][:, 2], -1, 1)))
        if poses[1]['rms_px']-poses[0]['rms_px'] < .3 and delta > 1:
            raise ValueError('Planar target pose is ambiguous; use a clearer oblique view')
    return poses[0]


def angles_from_normal(normal, side):
    n = np.asarray(normal, float)
    if side not in ('left', 'right') or n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9:
        raise ValueError('Invalid wheel side or normal')
    n = n/np.linalg.norm(n)
    sign = 1 if side == 'left' else -1
    if n[1]*sign < 0:
        n = -n
    if abs(n[1]) < .7:
        raise ValueError('Wheel target is not approximately in the wheel plane / vehicle frame')
    # X forward, Y vehicle-left, Z up; outward-facing wheel normal.
    return dict(camber_deg=float(np.degrees(np.arctan2(-n[2], np.hypot(n[0], n[1])))),
                toe_in_deg=float(np.degrees(np.arctan2(n[0], abs(n[1])))))


def measure(rgb, profile, side, confirmed=False):
    if not confirmed:
        raise ValueError('Confirm the ground axes, rigid wheel-plane mounting and matching camera setup')
    k, d = camera_for(profile, rgb)
    ground_points = detect(rgb)
    ground = solve_pose(*ground_points, k, d)
    wheel = solve_pose(*detect(rgb, True), k, d)
    # Printed board +X = forward; printed -Y = left; printed -Z = up.
    world_to_camera = ground['rotation'] @ np.diag([1., -1., -1.])
    normal = world_to_camera.T @ wheel['rotation'][:, 2]
    angles = angles_from_normal(normal, side)
    result = dict(angles, side=side, ground_rms_px=ground['rms_px'], wheel_rms_px=wheel['rms_px'],
                  ground_corners=len(ground_points[0]), frame_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                  calibration=profile, method='two-charuco-targets-v1',
                  limitation='Target-assisted research measurement. Mounting, reference and camera accuracy require independent validation.')
    out = rgb.copy()
    for pose in (ground, wheel):
        cv2.drawFrameAxes(out, k, d, pose['rvec'], pose['tvec'], .09, 3)
    text = f"Camber {result['camber_deg']:+.2f} deg | Toe-in {result['toe_in_deg']:+.2f} deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(.8, (out.shape[1]-20)/cv2.getTextSize(text, font, 1., 1)[0][0])
    for y, line in [(28, text), (52, 'Calibrated targets - research measurement')]:
        cv2.putText(out, line, (10, y), font, scale, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, scale, (25, 70, 90), 1, cv2.LINE_AA)
    return result, out


def export_targets(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, wheel in [('ground', False), ('wheel', True)]:
        png = board(wheel).generateImage((1500, 2100), marginSize=0, borderBits=1)
        (folder/f'{name}.png').write_bytes(cv2.imencode('.png', png)[1].tobytes())
    (folder/'print-targets.html').write_text('''<!doctype html><meta charset="utf-8">
<style>@page{size:A4;margin:12mm}section{page-break-after:always}img{width:150mm;height:210mm}body{font:12px sans-serif}</style>
<section><h2>GROUND TARGET · 5 × 7 squares · each square 30 mm</h2>
<p>Print at 100%, no fit-to-page. Measure squares with a ruler. Mount flat on a verified level plane.<br>
Printed RIGHT (+X) points vehicle-forward. Printed UP (-Y) points vehicle-left.<br>Visible face points UP (-Z). Keep both targets visible together.</p><img src="ground.png"></section>
<section><h2>WHEEL TARGET · distinct marker IDs</h2><p>Print at 100%. Rigidly mount parallel to the verified wheel plane.<br>
A loose target or target resting on tyre rubber does not establish the wheel plane.</p><img src="wheel.png"></section>''', encoding='utf-8')
    return folder/'print-targets.html'
