"""Image-edge hypotheses inside a model-guided region, not alignment angles."""
import cv2
import numpy as np


def robust_line(points, tolerance=3.):
    """Deterministic two-point consensus followed by least squares: x = a*y+b."""
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 15:
        return None
    rng = np.random.default_rng(42)
    best = np.zeros(len(p), bool)
    for _ in range(80):
        q, r = p[rng.choice(len(p), 2, replace=False)]
        if abs(q[1]-r[1]) < 15:
            continue
        a = (q[0]-r[0])/(q[1]-r[1])
        b = q[0]-a*q[1]
        inside = abs(p[:, 0]-a*p[:, 1]-b)/np.sqrt(1+a*a) < tolerance
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 15:
        return None
    a, b = np.polyfit(p[best, 1], p[best, 0], 1)
    residual = abs(p[:, 0]-a*p[:, 1]-b)/np.sqrt(1+a*a)
    inside = residual < tolerance
    return dict(a=float(a), b=float(b), inliers=inside,
                residual=float(np.median(residual[inside])), fraction=float(inside.mean()))


def edge_fit(rgb, mask, mode='boundary'):
    result = dict(valid=False, mode=mode, reason='No tyre mask', points=[], accepted=[], lines=[])
    if mask is None or np.count_nonzero(mask) < 100:
        return result
    # Bound work independently of source resolution; map every result back.
    scale = min(1., 640/max(rgb.shape[:2]))
    size = (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale))
    im = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return result
    k = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (labels == k).astype(np.uint8)
    if m.sum()/max(np.count_nonzero(mask)*scale*scale, 1) < .85:
        result['reason'] = 'Fragmented mask: select a clearer frame'
        return result
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 100)
    y, x = np.nonzero(m)
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    if mode == 'boundary':
        if x0 == 0 or x1 == m.shape[1]-1:
            result['reason'] = 'Side boundary clipped by frame'
            return result
        rows = np.unique(np.linspace(y0+.15*(y1-y0), y0+.85*(y1-y0), 100).astype(int))
        sides = [[], []]
        band = max(4, round((x1-x0)*.06))
        for row in rows:
            xs = np.flatnonzero(m[row])
            if not len(xs):
                continue
            for side, target in enumerate((xs[0], xs[-1])):
                candidates = np.flatnonzero(edges[row, max(0, target-band):target+band+1])+max(0, target-band)
                if len(candidates):
                    sides[side].append((candidates[np.argmin(abs(candidates-target))], row))
        fits = [robust_line(p) for p in sides]
        result['points'] = [np.asarray(p).reshape(-1, 2)/scale for p in sides]
        result['accepted'] = [f['inliers'] if f else np.zeros(len(p), bool) for f, p in zip(fits, sides)]
        if any(f is None for f in fits):
            result['reason'] = 'Not enough image edges on both sides'
            return result
        for p, f in zip(sides, fits):
            accepted = np.asarray(p)[f['inliers']]
            if f['fraction'] < .65 or len(accepted)/len(rows) < .5 or np.ptp(accepted[:, 1]) < .5*(y1-y0):
                result['reason'] = 'Boundary support too short or inconsistent'
                return result
        slopes = [f['a'] for f in fits]
        angles = -np.degrees(np.arctan(slopes))
        if max(abs(angles)) > 45 or abs(angles[0]-angles[1]) > 12:
            result['reason'] = 'Side fits disagree: no midline'
            return result
        ends = np.array([rows[0], rows[-1]])
        segments = [np.column_stack((f['a']*ends+f['b'], ends))/scale for f in fits]
        if np.any(segments[1][:, 0]-segments[0][:, 0] < 10/scale):
            result['reason'] = 'Side fits cross or collapse'
            return result
        result.update(valid=True, reason='Two supported boundaries; inspect the midline',
                      lines=segments+[np.mean(segments, axis=0)],
                      angle=float(-np.degrees(np.arctan(np.mean(slopes)))),
                      residual_px=float(np.mean([f['residual'] for f in fits])/scale))
        return result
    # Side-view candidates come from image edges well INSIDE the tyre mask.
    # Never interpret the tyre's outer silhouette as a detected rim.
    radius = max(3, round(min(x1-x0, y1-y0)*.05))
    interior = cv2.erode(m, np.ones((2*radius+1, 2*radius+1), np.uint8))
    edges[interior == 0] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in sorted(contours, key=len, reverse=True)[:30]:
        p = np.ascontiguousarray(contour[:, 0, :][::2], dtype=np.float32)
        if len(p) < 40:
            continue
        try:
            ellipse = cv2.fitEllipse(p)
        except cv2.error:
            continue
        (cx, cy), (wa, ha), rotation = ellipse
        if not np.isfinite([cx, cy, wa, ha, rotation]).all():
            continue
        if min(wa, ha) < .2*min(x1-x0, y1-y0) or max(wa, ha) > max(x1-x0, y1-y0) or min(wa, ha)/max(wa, ha) < .2:
            continue
        a = np.radians(rotation)
        local = (p-[cx, cy]) @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        norm = local/np.array([wa/2, ha/2])
        residual = abs(np.linalg.norm(norm, axis=1)-1)*min(wa, ha)/2
        accepted = residual < 3
        if accepted.sum() < 35:
            continue
        sectors = np.floor((np.arctan2(norm[accepted, 1], norm[accepted, 0])+np.pi)/(2*np.pi)*24).astype(int) % 24
        coverage = len(np.unique(sectors))/24
        if coverage < .8 or accepted.mean() < .8:
            continue
        theta = np.linspace(0, 2*np.pi, 100)
        ring = np.column_stack((wa/2*np.cos(theta), ha/2*np.sin(theta))) @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])+[cx, cy]
        ij = np.round(ring).astype(int)
        if (ij < 0).any() or (ij[:, 0] >= m.shape[1]).any() or (ij[:, 1] >= m.shape[0]).any() or np.mean(interior[ij[:, 1], ij[:, 0]]) < .95:
            continue
        candidates.append((coverage*wa*ha, p, accepted, ring, residual, coverage, wa, ha))
    if not candidates:
        result['reason'] = 'No supported inner ellipse: show the complete rim'
        return result
    _, p, accepted, ring, residual, coverage, wa, ha = max(candidates, key=lambda c: c[0])
    result.update(valid=True, reason='Inner ellipse candidate — visually confirm it is the rim',
                  points=[p/scale], accepted=[accepted], ellipse=ring/scale,
                  residual_px=float(np.median(residual[accepted])/scale), coverage=coverage,
                  axis_ratio=float(min(wa, ha)/max(wa, ha)))
    return result


def draw_edge_fit(rgb, result, points_only=False):
    out = rgb.copy()
    radius = max(1, round(max(rgb.shape[:2])/700))
    for points, accepted in zip(result['points'], result['accepted']):
        for p, inside in zip(points, accepted):
            cv2.circle(out, tuple(np.round(p).astype(int)), radius,
                       (15, 190, 125) if inside else (225, 70, 80), -1)
    if result['valid'] and not points_only:
        for i, segment in enumerate(result['lines']):
            cv2.line(out, tuple(np.round(segment[0]).astype(int)), tuple(np.round(segment[1]).astype(int)),
                     (200, 70, 215) if i == 2 else (30, 185, 215), radius+1, cv2.LINE_AA)
        if 'ellipse' in result:
            cv2.polylines(out, [np.round(result['ellipse']).astype(np.int32)], True, (200, 70, 215), radius+1, cv2.LINE_AA)
    return out
