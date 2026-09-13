"""Real-checkpoint GPU smoke check, repeatability check and local timing report."""
import bootstrap
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from engine import Engine, read_image
from evidence import save_capture
from registry import MODELS
from bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda:0'])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('Use at least two repeats to check repeatability')
    path = args.image or next(Path(r'D:\Dataset Download\Tire Dataset Prepared\FINAL\images\clean\fold_1').rglob('*.jpg'))
    rgb = read_image(path)
    engine = Engine(args.device)
    timings = {}
    for name in MODELS:
        record, masks = engine.predict(rgb, name)
        runs = []
        for _ in range(args.repeats):
            again, mm = engine.predict(rgb, name)
            runs.append(again['inference_ms'])
            if masks is not None:
                assert np.array_equal(masks, mm), f'{name}: masks changed on repeat'
            else:
                assert record['prediction'] == again['prediction']
                np.testing.assert_allclose(record['scores'], again['scores'], atol=1e-6)
        timings[name] = dict(median_ms=statistics.median(runs), min_ms=min(runs), max_ms=max(runs), repeats=args.repeats)
        print(name, timings[name], flush=True)
    result, masks = engine.inspect(rgb, compare=True)
    assert len(result['models']) == 4 and len(masks) == 2
    for mm in masks.values():
        assert mm.shape == (2, *rgb.shape[:2])
    folder = save_capture(rgb, result, masks, str(path), 'Automated smoke check on an existing internal study image; not an external validation.')
    np.testing.assert_array_equal(read_image(folder/'frame.png'), rgb)
    for name, mm in masks.items():
        for i, region in enumerate(['tyre', 'tread']):
            np.testing.assert_array_equal(read_image(folder/f'{name}-{region}.png')[:, :, 0]>0, mm[i])
    summary = dict(device=result['device_name'], image=str(path), frame_size=result['frame_size'],
        timings=timings, evidence=str(folder), peak_vram_mb=result['peak_vram_mb'],
        note='Single internal photo; warm inference includes preprocessing/postprocessing, excludes model loading. Not camera FPS or an accuracy benchmark.')
    output = ROOT/'results'/'smoke-check.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('PASS: strict loads, repeatability, all four models, evidence round-trip.', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    # Windows Python has a small main-thread stack; recent CUDA libraries can
    # exhaust it. The desktop uses the same explicit stack size on its worker.
    import threading
    import traceback
    threading.stack_size(32 * 1024 * 1024)
    errors = []
    def run():
        try:
            main()
        except BaseException as e:
            errors.append(e)
            traceback.print_exc()
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    if errors:
        raise SystemExit(1)
