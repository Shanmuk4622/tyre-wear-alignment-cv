"""Same-frame presentation labels; image tilt is not wheel alignment."""
import math
import cv2
import numpy as np


def tread_angle(record):
    if not record:
        return None
    hr = record.get('models', {}).get('hrnet', {})
    if not hr.get('ordered'):
        return None
    points = record.get('display_points', hr.get('points'))
    try:
        p = np.asarray(points, dtype=float)
        if p.shape != (6, 2) or not np.isfinite(p).all():
            return None
        if np.any(p[1::2, 0] <= p[::2, 0]):
            return None
        centers = (p[::2] + p[1::2]) / 2
        y = centers[:, 1] - centers[:, 1].mean()
        if float(y @ y) < 1:
            return None
        slope = float(y @ (centers[:, 0] - centers[:, 0].mean()) / (y @ y))
        # Positive: upper centreline leans right of lower centreline.
        return math.degrees(math.atan(-slope))
    except (ValueError, TypeError):
        return None


def footer_lines(result, learned, seconds, fps):
    lines = [f'TREAD STATION  |  {seconds:.2f} s  |  Export {fps:g} fps']
    records = result.get('models', [])
    cls = next((r for r in records if r.get('task') == 'classifier'), None)
    region = [r.get('title', r.get('model', 'Regions')) for r in records if r.get('task') == 'regions']
    if region:
        lines.append('Regions: ' + ' + '.join(region).replace('\u00b7', '-'))
    if cls:
        scores = cls.get('scores', [])
        pred = cls.get('prediction', -1)
        score = f' | model score {scores[pred]*100:.0f}%' if 0 <= pred < len(scores) else ''
        lines.append(cls.get('label', 'Mileage proxy') + score)
    angle = tread_angle(learned)
    lines.append('Tread tilt: unavailable' if angle is None else f'Tread tilt: {angle:+.1f} deg vs image vertical')
    if learned:
        widths = learned['models']['hrnet'].get('widths_px', [])
        lines.append('Raw widths U/M/L: ' + ' / '.join('--' if v is None else f'{v:.0f}' for v in widths) + ' px')
        flags = learned.get('flags', [])
        lines.append('Learned points: review required' if flags else 'Learned points: no heuristic flags')
    lines.append('Image geometry only - not camber/toe or a safety verdict')
    return lines


def draw_footer(rgb, lines):
    """Append a readable panel so labels cannot hide the tyre or fitted lines."""
    h, w = rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.3, min(.7, w / 900))
    margin = max(8, round(w*.025))
    # Fixed canvas across changing predictions/assist paths, required by MP4.
    lines = list(lines[:7]) + [''] * max(0, 7-len(lines))
    row = max(18, round(scale*36))
    panel = np.full((row*7+margin*2, w, 3), (18, 34, 44), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w-1, 3), (30, 190, 170), -1)
    for i, line in enumerate(lines):
        fitted = min(scale, (w-2*margin)/max(1, cv2.getTextSize(line, font, 1., 1)[0][0]))
        cv2.putText(panel, line, (margin, margin+(i+1)*row-5), font, fitted,
                    (90, 220, 200) if i == 0 else (230, 239, 243), 1, cv2.LINE_AA)
    return np.concatenate([rgb, panel], axis=0)
