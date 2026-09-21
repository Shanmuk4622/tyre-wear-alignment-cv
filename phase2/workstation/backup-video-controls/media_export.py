"""Independent, bounded-CPU video export; never uses the live engine or stream."""
import bootstrap
from PySide6.QtGui import QImage  # Load the isolated Qt DLLs before OpenCV.
import json
import math
import sys
from pathlib import Path
import cv2
import torch
from engine import Engine, overlay
from video import configure_capture, portrait_frame
from shape_geometry import GeometryTracker, draw_geometry
from edge_geometry import edge_fit, draw_edge_fit
from learned_geometry import PointTracker, draw_learned


def export_video(job, engine=None, progress=lambda done, total: None):
    """Sample the full clip at 100 ms intervals; keep its duration at 10 fps.

    Lower-rate inputs repeat their latest frame. Decode sequentially to avoid
    keyframe seeking errors. The output is silent and capped at 1280 px, exactly
    like the workstation's analysis frames. Publish only a completed MP4.
    """
    destination = Path(job['destination'])
    partial = destination.with_suffix('.partial.mp4')
    cap = cv2.VideoCapture(job['source'])
    writer = None
    count = 0
    tracks, point_tracker = {}, PointTracker()
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open the selected video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError('Video has no usable duration or frame rate')
        total = math.ceil(frames / fps * 10 - 1e-7)
        rotation = configure_capture(cap)
        engine = engine or Engine('cpu')
        progress(0, total)
        index = -1
        frame = None
        for sample in range(total):
            target = min(int(sample * fps / 10 + 1e-7), int(frames)-1)
            while index < target:
                ok, decoded = cap.read()
                if not ok:
                    raise ValueError('Video ended before its declared duration; export not saved')
                frame = decoded
                index += 1
            upright, _ = portrait_frame(frame, rotation)
            scale = min(1., 1280 / max(upright.shape[:2]))
            if scale < 1:
                upright = cv2.resize(upright, (round(upright.shape[1]*scale), round(upright.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
            result, masks = engine.inspect(rgb, job['classifier'], job['region'],
                threshold=job['threshold'], assist=job['assist'], learned=job['learned'])
            name = job['overlay']
            if name is None:
                name = 'segformer_b0' if result.get('assist_used') else job['region']
            mm = masks.get(name)
            out = overlay(rgb, mm, job['opacity'], job['layers']) if mm is not None else rgb.copy()
            stamp = sample * 100.
            if job['geometry']:
                geometry = tracks.setdefault(name, GeometryTracker()).update(mm[0], stamp) if mm is not None else None
                out = draw_geometry(out, geometry)
            if job['edges']:
                fit = edge_fit(rgb, mm[0] if mm is not None else None, job['edges'])
                out = draw_edge_fit(out, fit)
            learned = point_tracker.update(result.get('learned_geometry'), stamp, job['source'])
            if job['show_learned'] and learned is not None:
                out = draw_learned(out, learned)
            # MPEG-4 requires even dimensions. Pad, never stretch the overlay.
            h, w = out.shape[:2]
            out = cv2.copyMakeBorder(out, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)
            if writer is None:
                writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*'mp4v'), 10., (out.shape[1], out.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('MP4 encoder could not start')
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            count += 1
            progress(count, total)
        writer.release()
        writer = None
        # Verify the complete encoded output before presenting it as a download.
        verify = cv2.VideoCapture(str(partial))
        try:
            decoded_count = 0
            while verify.grab():
                decoded_count += 1
            if decoded_count != count or count == 0:
                raise RuntimeError('The encoded video failed its frame-count check')
        finally:
            verify.release()
        if destination.exists():
            raise FileExistsError('Download destination already exists')
        partial.rename(destination)
        return count
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        export_video(job, progress=lambda n, total: print(json.dumps({'done': n, 'total': total}), flush=True))
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
