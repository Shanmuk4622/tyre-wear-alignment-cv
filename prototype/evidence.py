"""Portable evidence cards, with original pixels and independently saved masks."""
import html
import json
import os
import uuid
from datetime import datetime

from PIL import Image
from bootstrap import ROOT
from engine import overlay


def save_capture(rgb, result, masks, source, note=''):
    capture_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    folder = ROOT / 'results' / capture_id
    folder.mkdir(parents=True)
    Image.fromarray(rgb).save(folder / 'frame.png')
    display = result.get('display_options', {})
    for name, mm in masks.items():
        Image.fromarray(overlay(rgb, mm, display.get('opacity', .26), display.get('visible', (True, True)))).save(folder / f'{name}-overlay.png')
        for label, mask in zip(['tyre', 'tread'], mm):
            Image.fromarray(mask.astype('uint8') * 255).save(folder / f'{name}-{label}.png')
    report = dict(result, capture_id=capture_id, source=source, operator_note=note)
    tmp = folder / 'inspection.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, folder / 'inspection.json')
    esc = html.escape
    rows = []
    for r in result['models']:
        detail = r.get('label', 'Tyre / tread localization')
        if r.get('assist_reason'):
            detail += ' · SEGFORMER ASSIST: ' + r['assist_reason']
        if 'scores' in r:
            detail += ' · scores L/M/H: ' + ' / '.join(f'{v:.3f}' for v in r['scores'])
        rows.append(f'<tr><td>{esc(r["title"])}</td><td>{esc(detail)}</td><td>{r["inference_ms"]:.0f} ms</td></tr>')
    pictures = '<figure><img src="frame.png"><figcaption>Recorded analysis frame (video may be resized)</figcaption></figure>'
    for name in masks:
        pictures += f'<figure><img src="{name}-overlay.png"><figcaption>{esc(name)} · amber tyre / mint tread</figcaption></figure>'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Tread Station · {capture_id}</title>
<style>body{{background:#f4f7fa;color:#213447;font:16px system-ui;max-width:1100px;margin:50px auto;padding:24px}}h1{{font-size:42px}}small{{color:#16767c}}table{{width:100%;border-collapse:collapse;background:white}}td{{padding:16px;border-bottom:1px solid #d7e1e7}}.images{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:16px 0;flex:1;min-width:240px}}img{{width:100%}}figcaption,p{{color:#536d7d}}code{{overflow-wrap:anywhere}}</style>
<small>TREAD STATION / INSPECTION RECORD</small><h1>Evidence, frame by frame.</h1>
<p>{esc(capture_id)} · {esc(result['device_name'])} · {esc(result['created_at'])}</p>
<p>{esc(result['limitation'])}</p><table>{''.join(rows)}</table><div class="images">{pictures}</div>
<p>Video position: {esc(str(result.get('source_time_ms', 'not applicable')))} ms ·
Model-view rotation: {result.get('analysis_rotation_degrees', 0)}° · YOLO minimum score: {result.get('yolo_threshold', .25)}.
File display rotation applied before inference: {result.get('video_rotation_clockwise', 0)}° clockwise.
Recorded frame size: {esc(str(result.get('frame_size')))}; source frame size: {esc(str(result.get('source_frame_size') or result.get('frame_size')))}.</p>
<p>Model agreement: {esc(json.dumps(result['agreement']))}</p><p>Operator note: {esc(note or '—')}</p>
<p>Capture hints: {esc('; '.join(result['quality']['hints']) or 'No heuristic flags')}</p>
<p>Source: {esc(source)}</p><p>Frame SHA256: <code>{result['frame_sha256']}</code></p>
<p>Full checkpoint provenance, runtime, timings and scores: <a href="inspection.json">inspection.json</a>.
Timings include preprocessing and postprocessing, exclude loading, and are not camera FPS. This card is a single capture, not an accuracy evaluation.</p></html>'''
    (folder / 'card.html').write_text(page, encoding='utf-8')
    return folder
