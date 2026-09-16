"""Real pinned checkpoints, independent adapter checks and geometry contracts."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
# Qt must load before cv2 in the Windows application process.
from PySide6.QtWidgets import QApplication
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from learned_geometry import LearnedEngine, tensor, PointTracker, draw_learned
from bootstrap import ROOT


def main():
    torch.set_num_threads(4)
    engine = LearnedEngine('cuda:0')
    model, cfg, _ = engine.load('hrnet')
    row = next(r for r in cfg['rows'] if r['role'] == 'test')
    data_root = 'D:/Dataset Download/Tire Dataset Prepared/FINAL'
    image_path = __import__('pathlib').Path(data_root)/row['original_relative_path']
    if not image_path.exists():
        from hrnet_protocol import root_data
        data_root = root_data()
        image_path = data_root/row['original_relative_path']
    rgb = np.asarray(Image.open(image_path).convert('RGB')).copy()
    from hrnet_runtime import load_batch, positions
    original, _ = load_batch([row], data_root, cfg, device='cuda:0')
    torch.testing.assert_close(tensor(rgb, 'cuda:0'), original, rtol=0, atol=0)
    result = engine.inspect(rgb, 'paired')
    with torch.inference_mode():
        expected = positions(model(original))[0].cpu().numpy()*(rgb.shape[1]-1)
    actual = np.array(result['models']['hrnet']['points'])
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-4)
    np.testing.assert_array_equal(actual[::2, 1], row['guide_y'])
    sm, _, _ = engine.load('matched')
    with torch.inference_mode():
        native = F.interpolate(sm(original).logits[:, 1:2].float(), rgb.shape[:2], mode='bilinear', align_corners=False)
    from segformer_matched import boundaries
    expected_seg = boundaries((native[0, 0] >= 0).cpu().numpy(), row['guide_y'])
    actual_seg = result['models']['matched']['points']
    for a, b in zip(actual_seg, expected_seg):
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a[0]-b*(rgb.shape[1]-1)) < 1e-4
    repeated = engine.inspect(rgb, 'paired')
    assert repeated['models']['hrnet']['points'] == result['models']['hrnet']['points']
    assert repeated['models']['matched']['points'] == result['models']['matched']['points']
    blank = engine.inspect(np.zeros_like(rgb), 'paired')
    assert any('Low image contrast' in s for s in blank['flags'])
    assert len(blank['models']['hrnet']['points']) == 6
    tracker = PointTracker()
    clean = dict(result, flags=[])
    for i in range(3):
        tracked = tracker.update(clean, i*200, ('test', 1))
    assert tracked['stable_frames'] == 3
    assert tracker.update(clean, 0, ('test', 1))['stable_frames'] == 1
    assert tracker.update(blank, 200, ('test', 1))['stable_frames'] == 0
    assert tracker.update(clean, 400, ('test', 1))['stable_frames'] == 1
    np.testing.assert_array_equal(rgb, np.asarray(Image.open(image_path).convert('RGB')))
    Image.fromarray(draw_learned(rgb, result)).save(ROOT/'results'/'learned-image-check.png')
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        r = engine.inspect(rgb, 'paired')
        timings.append({name: value['inference_ms'] for name, value in r['models'].items()})
    report = dict(status='passed', image=str(image_path), precision='FP32',
        checks=['strict checkpoint loading', 'original preprocessing exact', 'original coordinate decoding',
                'matched native-logit boundary extraction', 'repeatability', 'blank proposals flagged',
                'temporal resets', 'immutable source pixels'],
        timings_ms=timings, gpu=torch.cuda.get_device_name(), peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        result=result)
    (ROOT/'results'/'learned-integration-check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'result'}), flush=True)


if __name__ == '__main__':
    main()
