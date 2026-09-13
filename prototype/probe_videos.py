"""Sample real videos and report region detections without changing weights."""
import bootstrap
import json
import threading
import traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from engine import Engine, overlay
from bootstrap import ROOT

def main():
    engine = Engine()
    output = ROOT / 'results' / 'video-probe'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in (ROOT.parent/'Videos').glob('*.mp4'):
        cap = cv2.VideoCapture(str(path))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        for fraction in (.1, .5, .9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*fraction))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            record, masks = engine.predict(rgb, 'yolo26n_seg')
            row = dict(video=path.name, position=fraction, fps=fps, frame_size=list(rgb.shape), **record)
            rows.append(row)
            picture = Image.fromarray(overlay(rgb, masks))
            picture.thumbnail((900, 600))
            picture.save(output/f'{path.stem}-{fraction}.jpg')
            print(json.dumps({k:row[k] for k in ('video','position','frame_size','coverage','target_found','inference_ms')}), flush=True)
        cap.release()
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*.5))
        ok, bgr = cap.read()
        cap.release()
        if ok:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (960, round(rgb.shape[0]*960/rgb.shape[1])))
            for turn in range(4):
                rotated = np.ascontiguousarray(np.rot90(rgb, turn))
                r, mm = engine.predict(rotated, 'yolo26n_seg')
                rows.append(dict(video=path.name, rotation=turn*90, **r))
                print(path.name, turn*90, r['detector'], r['coverage'], flush=True)
                Image.fromarray(np.rot90(overlay(rotated, mm), -turn)).save(output/f'{path.stem}-rotation-{turn*90}.jpg')
    (output/'report.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

if __name__ == '__main__':
    threading.stack_size(32*1024*1024)
    errors = []
    def run():
        try:
            main()
        except Exception as e:
            errors.append(e)
            traceback.print_exc()
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    if errors:
        raise SystemExit(1)
