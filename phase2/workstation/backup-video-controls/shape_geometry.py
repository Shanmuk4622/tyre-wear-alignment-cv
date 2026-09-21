"""Inspectable 2-D silhouette geometry; never a physical alignment estimator."""
import cv2
import numpy as np


class GeometryTracker:
    """Conservative same-view continuity; stability is not alignment accuracy."""
    def __init__(self):
        self.previous = None
        self.vector = None
        self.count = 0
        self.stamp = None

    def update(self, mask, stamp):
        g = shape_geometry(mask)
        previous = self.previous
        continuous = previous is not None and self.stamp is not None and 0 < stamp-self.stamp <= 2000
        if continuous and g['valid'] and previous['mask'].shape == g['mask'].shape:
            intersection = np.count_nonzero(g['mask'] & previous['mask'])
            union = np.count_nonzero(g['mask'] | previous['mask'])
            delta = abs((g['angle']-previous['angle']+90) % 180-90)
            continuous = intersection/max(union, 1) >= .55 and delta <= 15 and abs(g['ratio']-previous['ratio']) <= .15
        else:
            continuous = False
        self.stamp = stamp
        if not g['valid']:
            self.previous, self.vector, self.count = None, None, 0
            return dict(g, stable=False, count=0)
        radians = np.radians(g['angle']*2)
        current = np.array([np.cos(radians), np.sin(radians)])
        self.vector = .65*self.vector + .35*current if continuous else current
        self.count = self.count+1 if continuous else 1
        self.previous = g
        angle = float(np.degrees(np.arctan2(self.vector[1], self.vector[0]))/2)
        return dict(g, stable=self.count >= 3, count=self.count, smooth_angle=angle)


def draw_geometry(rgb, geometry):
    """Draw on the matching analysed image only, never a newer preview frame."""
    out = rgb.copy()
    h, w = out.shape[:2]
    scale = max(.45, min(w/800, h/700))
    thick = max(1, round(scale*2))
    g = geometry
    if g and 'mask' in g:
        contours, _ = cv2.findContours(g['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (50, 155, 180), thick)
    if g and g['valid']:
        c = g['center']
        length = min(h, w)*.24
        center = tuple(np.round(c).astype(int))
        cv2.line(out, (center[0], round(c[1]-length)), (center[0], round(c[1]+length)), (90, 120, 155), thick)
        angles = [(g['angle'], (195, 195, 195), thick)]
        if g['stable']:
            angles.append((g['smooth_angle'], (235, 145, 20), thick+1))
        for angle, color, width in angles:
            a = np.radians(angle)
            d = np.array([np.sin(a), -np.cos(a)])*length
            cv2.line(out, tuple(np.round(c-d).astype(int)), tuple(np.round(c+d).astype(int)), color, width, cv2.LINE_AA)
        cv2.circle(out, center, thick+3, (15, 125, 135), -1)
        message = (f"Stable shape {g['smooth_angle']:+.1f} deg | {g['count']} frames" if g['stable'] else f"Acquiring shape | {g['count']}/3 frames")
        legend = 'Amber: smoothed | Gray: raw | Blue: image vertical'
    else:
        message = 'Shape withheld: '+(g['reason'] if g else 'No tyre mask')
        legend = 'Waiting for a complete, consistent silhouette'
    lines = [message, legend, 'Camera-relative shape; not camber / toe']
    # Fit labels to the image width, including narrow portrait frames.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(.55*scale, (w-20)/max(cv2.getTextSize(s, font, 1, 1)[0][0] for s in lines))
    row = max(15, round(25*scale))
    cv2.rectangle(out, (0, 0), (w, 3*row+12), (245, 249, 252), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8, (i+1)*row), font, font_scale, (35, 65, 80), 1, cv2.LINE_AA)
    return out


def shape_geometry(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count < 2:
        return dict(valid=False, reason='No tyre mask — inspect a frame first')
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == k
    y, x = np.nonzero(clean)
    if len(x) < 100:
        return dict(valid=False, reason='Too few mask pixels for shape geometry')
    points = np.column_stack((x, y)).astype(float)
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    axis = vectors[:, -1]
    if axis[1] > 0:
        axis = -axis
    angle = float(np.degrees(np.arctan2(axis[0], -axis[1])))
    ratio = float(np.sqrt(values[0] / max(values[1], 1e-9)))
    clipped = bool(clean[0].any() or clean[-1].any() or clean[:, 0].any() or clean[:, -1].any())
    retained = float(clean.sum() / binary.astype(bool).sum())
    reason = ('Mask touches frame edge' if clipped else
              'Several large mask fragments' if retained < .9 else
              'Nearly round: dominant axis is ambiguous' if ratio > .85 else '')
    return dict(valid=not reason, reason=reason, mask=clean, center=center,
                axis=axis, angle=angle, ratio=ratio, retained=retained)
