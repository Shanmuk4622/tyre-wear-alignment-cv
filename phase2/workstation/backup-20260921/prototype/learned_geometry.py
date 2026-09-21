"""Frozen full-image adapters and explicitly uncalibrated point overlays."""
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from prepare_learned import SPECS, path
from prepare_models import digest


def tensor(rgb, device):
    a = np.asarray(Image.fromarray(rgb).resize((384, 512), Image.Resampling.BILINEAR), dtype=np.float32).copy()/255
    x = torch.from_numpy(a).permute(2, 0, 1)[None].to(device)
    return (x-x.new_tensor([.485, .456, .406])[None, :, None, None])/x.new_tensor([.229, .224, .225])[None, :, None, None]


def describe(points, width):
    spans = [None if points[2*i] is None or points[2*i+1] is None else points[2*i+1][0]-points[2*i][0] for i in range(3)]
    widths = [v if v is not None and v > .02*(width-1) else None for v in spans]
    valid = all(v is not None for v in widths)
    return dict(widths_px=widths, ordered=valid)


class LearnedEngine:
    def __init__(self, device):
        self.device, self.cache = device, {}

    def load(self, name):
        if name in self.cache:
            return self.cache[name]
        file = path(name)
        if not file.exists():
            raise FileNotFoundError('Learned geometry weights missing. Run prepare_learned.py in cv_conda.')
        meta = json.loads(file.with_suffix('.json').read_text())
        if meta['revision'] != SPECS[name]['revision'] or digest(file) != meta['inference_sha256']:
            raise ValueError('Learned geometry checkpoint provenance mismatch')
        saved = torch.load(file, map_location='cpu', weights_only=False)
        cfg = saved['contract']
        from hrnet_protocol import canonical, sha
        if sha(canonical(cfg)) != SPECS[name]['protocol']:
            raise ValueError('Learned geometry contract mismatch')
        if name == 'hrnet':
            from hrnet_runtime import Geometry
            model = Geometry(cfg)
        else:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
            config = SegformerConfig.from_dict(saved['config']); config.num_labels = 2
            model = SegformerForSemanticSegmentation(config)
            saved['model'] = {k.removeprefix('net.'): v for k, v in saved['model'].items()}
        model.load_state_dict(saved['model'], strict=True)
        model.to(self.device).eval()
        self.cache[name] = (model, cfg, meta)
        return self.cache[name]

    @torch.inference_mode()
    def inspect(self, rgb, mode):
        if mode == 'off':
            self.cache.clear()
            return None
        if mode not in ('hrnet', 'paired'):
            raise ValueError('Unknown learned geometry mode')
        h, w = rgb.shape[:2]
        output = dict(mode=mode, models={}, frame_size=[w, h],
                      input_hw=[512, 384], precision='FP32',
                      limitation='Unconditional image-space proposals. Not visibility confidence or physical alignment.')
        for name in (['hrnet', 'matched'] if mode == 'paired' else ['hrnet']):
            start = time.perf_counter(); model, cfg, meta = self.load(name)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize()
            loaded = time.perf_counter()
            x = tensor(rgb, self.device)
            logits = model(x)
            ref = cfg['rows'][0]
            guides = [round(y/(ref['height']-1)*(h-1)) for y in ref['guide_y']]
            if name == 'hrnet':
                from hrnet_runtime import positions
                xx = positions(logits)[0].cpu().tolist()
                points = [[float(value*(w-1)), int(guides[i//2])] for i, value in enumerate(xx)]
            else:
                z = F.interpolate(logits.logits[:, 1:2].float(), (h, w), mode='bilinear', align_corners=False)
                mask = (z[0, 0] >= 0).cpu().numpy()
                points = []
                for y in guides:
                    xs = np.flatnonzero(mask[y])
                    for side in (0, -1):
                        value = int(xs[side]) if len(xs) else None
                        points.append(None if value is None or value in (0, w-1) else [value, y])
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize()
            output['models'][name] = dict(points=points, names=cfg['point_names'], provenance=meta,
                inference_ms=round((time.perf_counter()-loaded)*1000, 2), load_ms=round((loaded-start)*1000, 2),
                **describe(points, w))
        hr = output['models']['hrnet']
        flags = []
        if not hr['ordered']:
            flags.append('Crossed or collapsed HRNet boundaries')
        if any(p[0] < .01*(w-1) or p[0] > .99*(w-1) for p in hr['points']):
            flags.append('HRNet proposal near frame edge')
        if float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).std()) < 8:
            flags.append('Low image contrast; HRNet still predicts coordinates')
        if 'matched' in output['models']:
            seg = output['models']['matched']
            differences = [None if b is None else abs(a[0]-b[0]) for a, b in zip(hr['points'], seg['points'])]
            output['differences_px'] = differences
            output['paired_coverage'] = sum(v is not None for v in differences)/6
            if not seg['ordered']:
                flags.append('Matched segmentation has missing/crossed boundaries')
            if any(v is not None and v > .05*(w-1) for v in differences):
                flags.append('HRNet / matched segmentation disagree (>5% image width)')
        output['flags'] = flags
        return output


class PointTracker:
    def __init__(self):
        self.previous = self.smoothed = self.stamp = self.context = None
        self.count = 0

    def update(self, record, stamp, context):
        if record is None:
            self.__init__(); return None
        result = dict(record)
        points = np.asarray(record['models']['hrnet']['points'], float)
        continuous = (not record['flags'] and self.previous is not None and context == self.context
                      and 0 < stamp-self.stamp <= 2000 and np.max(abs(points-self.previous)) < .04*record['frame_size'][0])
        self.count = self.count+1 if continuous else 1
        self.smoothed = self.smoothed*.65+points*.35 if continuous else points.copy()
        self.previous, self.stamp, self.context = points, stamp, context
        if record['flags']:
            self.previous = None
        result['stable_frames'] = self.count if not record['flags'] else 0
        result['display_points'] = self.smoothed.tolist() if self.count >= 3 and not record['flags'] else points.tolist()
        return result


def draw_learned(rgb, record):
    if record is None:
        return rgb.copy()
    out = rgb.copy(); h, w = out.shape[:2]
    thickness = max(1, round(max(h, w)/650))
    models = record['models']
    for name, item in models.items():
        points = record.get('display_points', item['points']) if name == 'hrnet' else item['points']
        color = (230, 85, 70) if record['flags'] and name == 'hrnet' else (235, 165, 25) if name == 'hrnet' else (20, 185, 205)
        coords = [None if p is None else tuple(np.round(p).astype(int)) for p in points]
        for i, p in enumerate(coords):
            if p is None:
                continue
            if name == 'hrnet':
                cv2.circle(out, p, thickness+3, color, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(out, (p[0]-thickness-3, p[1]-thickness-3), (p[0]+thickness+3, p[1]+thickness+3), color, thickness)
        if item['ordered']:
            for indices in ((0, 2, 4), (1, 3, 5)):
                cv2.polylines(out, [np.array([coords[i] for i in indices], np.int32)], False, color, thickness, cv2.LINE_AA)
            centers = []
            for i in range(3):
                a, b = coords[2*i:2*i+2]
                cv2.line(out, a, b, color, 1, cv2.LINE_AA)
                centers.append(((a[0]+b[0])//2, a[1]))
                if name == 'hrnet':
                    text = f'{("U", "M", "L")[i]} {b[0]-a[0]} px'
                    position = (max(3, min(w-100, centers[-1][0]-35)), max(18, a[1]-10))
                    font_scale = max(.4, min(1., max(h, w)/1600))
                    cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness+2, cv2.LINE_AA)
                    cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (70, 40, 20), thickness, cv2.LINE_AA)
                if name == 'hrnet':
                    text = f'{("U", "M", "L")[i]} {b[0]-a[0]} px'
                    position = (max(3, min(w-100, centers[-1][0]-35)), max(18, a[1]-10))
                    font_scale = max(.4, min(1., max(h, w)/1600))
                    cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness+2, cv2.LINE_AA)
                    cv2.putText(out, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (70, 40, 20), thickness, cv2.LINE_AA)
            if name == 'hrnet':
                cv2.polylines(out, [np.array(centers, np.int32)], False, (160, 70, 205), thickness, cv2.LINE_AA)
    return out
