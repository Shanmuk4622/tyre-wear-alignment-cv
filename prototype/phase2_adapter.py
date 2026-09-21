"""Verified Phase 2 inference, preserving the exact training input contract."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / 'phase2'
REGISTRY = ROOT / 'workstation' / 'registry.json'
ALIASES = {'mobilenetv4': 'mobilenetv4', 'resnet50': 'resnet50',
           'yolo26n_seg': 'yolo26m', 'segformer_b0': 'segformer',
           'hrnet': 'hrnet', 'matched': 'segformer'}

def enabled():
    return (ROOT / 'workstation' / 'ACTIVE').exists()

def spec(name):
    return json.loads(REGISTRY.read_text(encoding='utf-8'))[ALIASES[name] + '-combined']

def load(name, device):
    meta = spec(name)
    path = ROOT / 'workstation' / 'models' / ALIASES[name] / 'weights.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Phase 2 checkpoint checksum mismatch: ' + name)
    for filename in ('phase2_models.py', 'phase2_training_data.py'):
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != meta['contract']['runtime_sources'][filename]:
            raise ValueError('Phase 2 inference source differs from the trained contract')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import phase2_models as models
    model_name = ALIASES[name]
    if model_name == 'segformer':
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        config = SegformerConfig.from_dict(json.loads((path.parent / 'MODEL_CONFIG.json').read_text()))
        model = SegformerForSemanticSegmentation(config).to(device)
    else:
        model = models.build(model_name, path.parent, pretrained=False, device=device)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return model, None, meta

def tensor(rgb, name, device):
    hw = spec(name)['contract']['input_hw']
    a = np.asarray(Image.fromarray(rgb).resize(tuple(hw[::-1]), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device) / 255.
    if ALIASES[name] != 'yolo26m':
        x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
    return x

@torch.inference_mode()
def predict(model, rgb, name, device, threshold=.25):
    import phase2_models as models
    x = tensor(rgb, name, device)
    model_name = ALIASES[name]
    record = {}; masks = None
    if model_name in ('mobilenetv4', 'resnet50'):
        probabilities = models.predict(model, {'img': x}, model_name)[0]
        pred = int(probabilities.argmax())
        record.update(prediction=pred, label=['Low mileage proxy', 'Mid mileage proxy', 'High mileage proxy'][pred],
                      scores=probabilities.cpu().tolist(), head='softmax', input_resolution=384)
    elif model_name == 'segformer':
        small = models.predict(model, {'img': x}, model_name).float()
        masks = F.interpolate(small, rgb.shape[:2], mode='nearest')[0].bool().cpu().numpy()
    elif model_name == 'yolo26m':
        from ultralytics.utils.ops import process_mask
        out = model(x); pair = out[0] if isinstance(out[0], tuple) else out
        detections, proto = pair
        det = detections[0]; accepted = det[det[:, 4] >= threshold]
        small = torch.zeros((2, *x.shape[-2:]), dtype=torch.bool, device=device)
        if len(accepted):
            instances = process_mask(proto[0], accepted[:, 6:], accepted[:, :4], x.shape[-2:], upsample=True).bool()
            for c in (0, 1):
                hit = accepted[:, 5].long() == c
                if hit.any(): small[c] = instances[hit].any(0)
        masks = F.interpolate(small[None].float(), rgb.shape[:2], mode='nearest')[0].bool().cpu().numpy()
        record['detector'] = dict(candidates=len(det), accepted=len(accepted), threshold=threshold,
            max_confidence=float(det[:, 4].max()) if len(det) else 0.,
            class_max={label: float(det[det[:, 5].long() == c, 4].max()) if (det[:, 5].long() == c).any() else 0.
                       for c, label in enumerate(['tyre', 'tread'])})
    return record, masks

@torch.inference_mode()
def learned_inspect(engine, rgb, mode):
    from learned_geometry import describe
    import cv2
    h, w = rgb.shape[:2]
    guides = [round(y / 1535 * (h - 1)) for y in (384, 768, 1151)]
    result = dict(mode=mode, models={}, frame_size=[w, h], input_hw=[512, 384], precision='FP32',
                  limitation='Phase 2 training-cohort adaptation. Unconditional image-space proposals, not visibility confidence or physical alignment.')
    for name in (['hrnet', 'matched'] if mode == 'paired' else ['hrnet']):
        start = time.perf_counter()
        if name not in engine.cache: engine.cache[name] = load(name, engine.device)
        model, _, meta = engine.cache[name]
        if str(engine.device).startswith('cuda'): torch.cuda.synchronize()
        loaded = time.perf_counter()
        if name == 'hrnet':
            import phase2_models as models
            xs = models.predict(model, {'img': tensor(rgb, name, engine.device)}, 'hrnet')[0].cpu().tolist()
            points = [[float(v * (w-1)), guides[i//2]] for i, v in enumerate(xs)]
        else:
            _, masks = predict(model, rgb, name, engine.device)
            points = []
            for y in guides:
                xs = np.flatnonzero(masks[1, y])
                for side in (0, -1):
                    value = int(xs[side]) if len(xs) else None
                    points.append(None if value is None or value in (0, w-1) else [value, y])
        if str(engine.device).startswith('cuda'): torch.cuda.synchronize()
        result['models'][name] = dict(points=points, names=spec('hrnet')['contract']['classes'], provenance=meta,
            inference_ms=round((time.perf_counter()-loaded)*1000, 2), load_ms=round((loaded-start)*1000, 2), **describe(points, w))
    hr = result['models']['hrnet']; flags = []
    if not hr['ordered']: flags.append('Crossed or collapsed HRNet boundaries')
    if any(p[0] < .01*(w-1) or p[0] > .99*(w-1) for p in hr['points']): flags.append('HRNet proposal near frame edge')
    if float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).std()) < 8: flags.append('Low image contrast; HRNet still predicts coordinates')
    if 'matched' in result['models']:
        seg = result['models']['matched']; differences = [None if b is None else abs(a[0]-b[0]) for a, b in zip(hr['points'], seg['points'])]
        result.update(differences_px=differences, paired_coverage=sum(v is not None for v in differences)/6)
        if not seg['ordered']: flags.append('Phase 2 segmentation has missing/crossed boundaries')
        if any(v is not None and v > .05*(w-1) for v in differences): flags.append('HRNet / Phase 2 segmentation disagree (>5% image width)')
    result['flags'] = flags
    return result
