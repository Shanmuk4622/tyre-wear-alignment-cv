"""Read-only source videos -> resumable, native-resolution Phase 2 PNG candidates.

Run with cv_conda. No training, model inference, labeling, or remote writes.
Uses presentation timestamps, not nominal FPS arithmetic, for VFR sources.
"""
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RATE = 2
VERSION = 'phase2-native-2fps-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def probe(path, frames=False):
    entries = ('frame=best_effort_timestamp_time' if frames else
               'stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:'
               'stream_side_data=rotation:format=duration,size')
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', entries, '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(cmd, text=True, encoding='utf-8'))


def rotate(frame, degrees):
    return {0: lambda x: x,
            90: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            180: lambda x: cv2.rotate(x, cv2.ROTATE_180),
            270: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)}[degrees](frame)


def extract(video, video_id):
    source_hash = sha(video)
    metadata = probe(video)
    stream = metadata['streams'][0]
    pts = [float(r['best_effort_timestamp_time']) for r in probe(video, True)['frames']]
    if not pts or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError('Missing or non-monotonic frame timestamps; inspect source before extraction')
    relative = [x - pts[0] for x in pts]
    duration = float(stream['duration'])
    ff_rotation = int(next((s['rotation'] for s in stream.get('side_data_list', []) if 'rotation' in s), 0))
    clockwise = (-ff_rotation) % 360
    if clockwise not in (0, 90, 180, 270):
        raise ValueError('Non-quarter-turn source orientation needs explicit handling')
    selected = {}
    skipped = []
    for k in range(math.ceil(duration * RATE)):
        target = k / RATE
        idx = bisect.bisect_left(relative, target - 1e-8)
        if idx >= len(pts):
            skipped.append(target)
            continue
        if idx in selected:
            raise ValueError('Sampling would duplicate a decoded source frame')
        selected[idx] = target
    folder = ROOT / 'data' / 'new_frames' / video_id
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    records, decoded = [], 0
    try:
        while True:
            ok, encoded = cap.read()
            if not ok:
                break
            idx = decoded
            decoded += 1
            if idx not in selected:
                continue
            if encoded.shape[:2] != (stream['height'], stream['width']):
                raise ValueError('Decoder applied unexpected scaling/orientation')
            bgr = rotate(encoded, clockwise)
            h, w = bgr.shape[:2]
            target = selected[idx]
            name = f'phase2_{video_id}_t{int(round(target*1000)):07d}_f{idx:06d}'
            output = folder / (name + '.png')
            pixel_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
            if output.exists():
                old = cv2.imread(str(output), cv2.IMREAD_COLOR)
                if old is None or old.shape != bgr.shape or not np.array_equal(old, bgr):
                    raise ValueError(f'Existing candidate differs; refusing overwrite: {output}')
            else:
                tmp = folder / (name + '.part.png')
                if not cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise OSError(f'Failed writing {tmp}')
                tmp.replace(output)
            small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (288, 512))
            tiny = cv2.resize(small, (9, 8))
            bits = (tiny[:, 1:] > tiny[:, :-1]).ravel()
            dhash = f'{sum(int(bit) << i for i, bit in enumerate(bits)):016x}'
            records.append(dict(frame_id=name, video_id=video_id, source_video=video.name,
                source_video_sha256=source_hash, source_frame_index=idx,
                target_seconds=target, actual_seconds=relative[idx], absolute_pts_seconds=pts[idx],
                timestamp_error_seconds=relative[idx]-target, clockwise_rotation=clockwise,
                width=w, height=h, relative_path=output.relative_to(ROOT).as_posix(),
                png_sha256=sha(output), decoded_bgr_sha256=pixel_hash, bytes=output.stat().st_size,
                dhash64=dhash, brightness=float(small.mean()),
                focus_laplacian_variance=float(cv2.Laplacian(small, cv2.CV_64F).var()),
                physical_tyre_id=None, original_group_id=None, mileage_proxy=None,
                label_status='unlabelled', split='UNASSIGNED', review_status='pending'))
            if len(records) % 10 == 0:
                print(f'{video.name}: {len(records)}/{len(selected)} native PNGs verified', flush=True)
    finally:
        cap.release()
    if decoded != len(pts) or len(records) != len(selected):
        raise ValueError(f'Incomplete decode {video.name}: {decoded}/{len(pts)} frames, {len(records)}/{len(selected)} samples')
    if sha(video) != source_hash:
        raise ValueError('Source changed while decoding')
    write_json(ROOT / 'manifests' / f'phase2_{video_id}_probe.json', metadata)
    return records, dict(video_id=video_id, source_video=video.name, source_video_sha256=source_hash,
        video_duration_seconds=duration, decoded_frames=decoded, extracted_frames=len(records),
        clockwise_rotation=clockwise, output_width=records[0]['width'], output_height=records[0]['height'],
        skipped_targets_past_last_pts=skipped, sampling='first source frame at or after k/2 seconds from first PTS')


def sheets(records):
    out = ROOT / 'review'
    out.mkdir(exist_ok=True)
    names = []
    for video in sorted({r['video_id'] for r in records}):
        rows = [r for r in records if r['video_id'] == video]
        for page, begin in enumerate(range(0, len(rows), 24), 1):
            subset = rows[begin:begin+24]
            canvas = Image.new('RGB', (1200, 70 + math.ceil(len(subset)/6)*335), '#f0f3f6')
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 12), f'PHASE 2 | {video} | candidate review | page {page} | native originals retained', fill='#172838')
            draw.text((16, 34), 'No labels or train/test split assigned. Two frames/sec are correlated observations.', fill='#172838')
            for j, r in enumerate(subset):
                x, y = (j % 6)*200, 70 + (j//6)*335
                with Image.open(ROOT / r['relative_path']) as im:
                    thumb = ImageOps.contain(im.convert('RGB'), (190, 280))
                canvas.paste(thumb, (x+(200-thumb.width)//2, y))
                draw.text((x+6, y+283), f'{r["target_seconds"]:.1f}s / frame {r["source_frame_index"]}', fill='#172838')
                draw.text((x+6, y+301), f'focus {r["focus_laplacian_variance"]:.0f}', fill='#172838')
            name = f'phase2_{video}_contact_{page:02d}.jpg'
            canvas.save(out/name, quality=92)
            names.append(name)
    html = '<!doctype html><meta charset="utf-8"><title>Phase 2 candidate frames</title><style>body{font:18px system-ui;margin:24px;background:#eef2f5;color:#172838}img{max-width:100%;display:block;margin:20px 0}a{color:#1459aa}</style><h1>Phase 2 — new video frames only</h1><p>Review gallery, not a labeling tool. Native PNGs are in ../data/new_frames/. Labels and splits remain unassigned.</p>'
    html += ''.join(f'<h2>{n}</h2><a href="{n}"><img src="{n}" loading="lazy"></a>' for n in names)
    (out/'phase2_review.html').write_text(html, encoding='utf-8')
    return names


def main():
    if shutil.disk_usage(ROOT).free < 4*1024**3:
        raise OSError('Need at least 4 GiB free before native PNG extraction')
    paths = sorted((REPO/'Videos').glob('*.mp4'), key=lambda p:p.name.lower())
    if [p.name.lower() for p in paths] != ['video1.mp4', 'video2.mp4', 'video3.mp4']:
        raise ValueError('Expected exactly the three inspected source clips')
    started = time.time()
    all_rows, summaries = [], []
    for i, p in enumerate(paths, 1):
        rows, summary = extract(p, f'video{i}')
        all_rows.extend(rows)
        summaries.append(summary)
    write_json(ROOT/'manifests'/'phase2_frames.json', dict(version=VERSION, target_fps=RATE, frames=all_rows))
    with (ROOT/'manifests'/'phase2_frames.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    # Review queue is user-owned after creation: rerunning extraction never resets it.
    queue = ROOT/'manifests'/'phase2_review_queue.csv'
    if not queue.exists():
        with queue.open('w', newline='', encoding='utf-8') as f:
            fields = ['frame_id','video_id','target_seconds','relative_path','keep','reason','physical_tyre_id','mileage_proxy','notes']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for r in all_rows:
                writer.writerow({k:r.get(k, '') for k in fields})
    counts = {}
    for r in all_rows:
        counts.setdefault(r['decoded_bgr_sha256'], []).append(r['frame_id'])
    duplicates = [v for v in counts.values() if len(v)>1]
    contact_sheets = sheets(all_rows)
    result = dict(version=VERSION, status='extracted_candidates_not_labelled_or_split',
        videos=summaries, total_frames=len(all_rows), total_png_bytes=sum(r['bytes'] for r in all_rows),
        max_timestamp_error_seconds=max(r['timestamp_error_seconds'] for r in all_rows),
        exact_duplicate_groups=duplicates, contact_sheets=contact_sheets,
        elapsed_seconds=round(time.time()-started, 2), opencv_version=cv2.__version__,
        sources_rehashed_unchanged=True, labels_invented=False,
        frames_manifest_sha256=sha(ROOT/'manifests'/'phase2_frames.json'))
    write_json(ROOT/'manifests'/'phase2_extraction_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
