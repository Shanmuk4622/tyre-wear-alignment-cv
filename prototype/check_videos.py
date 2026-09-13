"""Actual supplied-video regression and diagnostic samples; not an accuracy benchmark."""
import bootstrap
import json
import threading
import traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from bootstrap import ROOT
from engine import Engine, overlay
from video import configure_capture, portrait_frame

def main():
    engine = Engine()
    out = ROOT/'results'/'video-check'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    tiles = []
    for path in sorted((ROOT.parent/'Videos').glob('*.mp4')):
        cap = cv2.VideoCapture(str(path))
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        orientation = configure_capture(cap)
        for position in (.1, .5, .9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*position))
            ok, bgr = cap.read()
            assert ok, path
            bgr, display_rotation = portrait_frame(bgr, orientation)
            scale = min(1., 1280/max(bgr.shape[:2]))
            bgr = cv2.resize(bgr, (round(bgr.shape[1]*scale), round(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            baseline, _ = engine.predict(rgb, 'yolo26n_seg')
            result, masks = engine.inspect(rgb, region='yolo26n_seg', rotation=0, assist=True)
            assert rgb.shape[0] >= rgb.shape[1]
            assert all(mm.shape == (2, *rgb.shape[:2]) for mm in masks.values())
            name = 'segformer_b0' if result['assist_used'] else 'yolo26n_seg'
            row = dict(video=path.name, position=position, baseline_found=baseline['target_found'],
                       rotation=result['analysis_rotation_degrees'], assist_used=result['assist_used'],
                       file_display_rotation_clockwise=display_rotation,
                       yolo_found=next(r['target_found'] for r in result['models'] if r['model']=='yolo26n_seg'),
                       displayed_region_model=name, overlay_found=bool(masks[name].any()),
                       total_ms=result['total_ms'], models=result['models'])
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='models'}), flush=True)
            pic = Image.fromarray(overlay(rgb, masks[name]))
            pic.thumbnail((480, 270))
            tile = Image.new('RGB', (480, 305), '#172629')
            tile.paste(pic, (0, 35))
            ImageDraw.Draw(tile).text((8, 8), f'{path.name} {position:.0%} | {name} | {result["analysis_rotation_degrees"]} deg', fill='white')
            tiles.append(tile)
        cap.release()
    assert records, 'No supplied videos found'
    sheet = Image.new('RGB', (1440, 305*((len(tiles)+2)//3)))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i%3)*480, (i//3)*305))
    sheet.save(out/'samples.jpg')
    (out/'report.json').write_text(json.dumps(dict(samples=records,
        note='Three samples per supplied video. Nonempty masks demonstrate pipeline operation, not validated segmentation accuracy.'), indent=2))

if __name__ == '__main__':
    threading.stack_size(32*1024*1024)
    errors=[]
    def run():
        try:
            main()
        except Exception as e:
            errors.append(e)
            traceback.print_exc()
    t=threading.Thread(target=run)
    t.start()
    t.join()
    if errors:
        raise SystemExit(1)
