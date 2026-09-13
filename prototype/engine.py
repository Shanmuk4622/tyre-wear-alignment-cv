"""Inference shared by the desktop console and reproducible command-line checks."""
import bootstrap
import gc
import hashlib
import importlib.metadata
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from registry import MODELS, checkpoint
from prepare_models import digest

LABELS = ['Low mileage proxy', 'Mid mileage proxy', 'High mileage proxy']


def read_image(path):
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))


def quality(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (512, 512))
    focus = float(cv2.Laplacian(small, cv2.CV_64F).var())
    brightness = float(gray.mean())
    hints = []
    if focus < 55:
        hints.append('Soft detail: hold still or move closer')
    if brightness < 45:
        hints.append('Dark frame: add diffuse light')
    if float((gray > 245).mean()) > .12:
        hints.append('Bright highlights: change the light angle')
    return dict(focus_score=round(focus, 1), brightness=round(brightness, 1),
                hints=hints, note='Uncalibrated capture hints; not a quality certification')


def overlay(rgb, masks, opacity=.26, visible=(True, True)):
    out = rgb.copy()
    for mask, color, show in zip(masks, [(225, 146, 32), (16, 161, 139)], visible):
        if not show:
            continue
        out[mask] = (out[mask] * (1-opacity) + np.array(color) * opacity).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, max(1, round(rgb.shape[1] / 650)))
    return out


class Engine:
    def __init__(self, device='auto'):
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Select CPU or repair the cv_conda CUDA installation.')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.cache = OrderedDict()
        self.verified = set()

    def load(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name], 0.0
        start = time.perf_counter()
        path = checkpoint(name)
        if not path.exists():
            raise FileNotFoundError(f'{MODELS[name]["title"]} is not downloaded. Run python prepare_models.py first.')
        p = json.loads(path.with_suffix('.json').read_text())
        if p['revision'] != MODELS[name]['revision'] or p['path'] != MODELS[name]['path']:
            raise RuntimeError('Checkpoint provenance differs from the model registry')
        if name not in self.verified:
            if digest(path) != p['inference_sha256']:
                raise RuntimeError('Checkpoint checksum failed. Run prepare_models.py to repair.')
            self.verified.add(name)
        # Keep the light classifier, YOLO and explicit SegFormer assist resident.
        while len(self.cache) >= 3:
            _, old = self.cache.popitem(last=False)
            del old
            gc.collect()
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
        if name == 'yolo26n_seg':
            from ultralytics import YOLO
            model = YOLO(str(path), task='segment')
            model.to(self.device)
            item = (model, None, p)
        else:
            s = torch.load(path, map_location='cpu', weights_only=False)
            if MODELS[name]['task'] == 'classifier':
                from classification import build_classifier, build_transforms
                cfg = s['config']
                model = build_classifier(name, cfg)
                model.load_state_dict(s['model'], strict=True)
                transform = build_transforms(cfg['input_resolution'], cfg.get('preprocessing', 'raw'))
                item = (model.to(self.device).eval(), (transform, cfg), p)
            else:
                from transformers import SegformerConfig, SegformerForSemanticSegmentation
                cfg = SegformerConfig.from_dict(s['config'])
                cfg.num_labels = 2
                model = SegformerForSemanticSegmentation(cfg)
                model.load_state_dict(s['model'], strict=True)
                item = (model.to(self.device).eval(), None, p)
        self.cache[name] = item
        return item, (time.perf_counter() - start) * 1000

    def sync(self):
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def predict(self, rgb, name, threshold=.25):
        (model, extra, provenance), load_ms = self.load(name)
        self.sync()
        start = time.perf_counter()
        im = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        record = dict(model=name, title=MODELS[name]['title'], task=MODELS[name]['task'], provenance=provenance)
        masks = None
        if record['task'] == 'classifier':
            from classification import decision
            transform, cfg = extra
            scores = model(transform(im).unsqueeze(0).to(self.device)).float()
            probs, pred = decision(scores, cfg['head_type'])
            record.update(prediction=pred, label=LABELS[pred], scores=probs.cpu().tolist(),
                          head=cfg['head_type'], input_resolution=cfg['input_resolution'])
        elif name == 'segformer_b0':
            x = torch.from_numpy(np.array(im.resize((512, 512), Image.Resampling.BILINEAR))).permute(2, 0, 1)
            x = x.to(self.device).float().unsqueeze(0) / 255
            x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
            logits = model(x).logits
            # Preserve both interpolation stages used in the S5 evaluator.
            logits = F.interpolate(logits, (512, 512), mode='bilinear', align_corners=False)
            p = F.interpolate(logits.float(), (h, w), mode='bilinear', align_corners=False).sigmoid()[0]
            masks = (p >= .5).cpu().numpy().copy()
        else:
            result = model.predict(im, imgsz=512, conf=.001, device=self.device,
                                   verbose=False, retina_masks=True, max_det=100)[0]
            masks = np.zeros((2, h, w), dtype=bool)
            record['detector'] = dict(candidates=len(result.boxes),
                max_confidence=float(result.boxes.conf.max()) if len(result.boxes) else 0.,
                threshold=threshold, accepted=int((result.boxes.conf >= threshold).sum()),
                class_max={label: float(result.boxes.conf[result.boxes.cls == k].max())
                    if (result.boxes.cls == k).any() else 0. for k, label in enumerate(['tyre', 'tread'])})
            if result.masks is not None:
                for mask, score, cls in zip(result.masks.data, result.boxes.conf, result.boxes.cls):
                    cls = int(cls.item())
                    if float(score) >= threshold and cls in (0, 1):
                        masks[cls] |= (F.interpolate(mask.float()[None, None], (h, w), mode='nearest')[0, 0] >= .5).cpu().numpy()
        if masks is not None:
            record['coverage'] = dict(tyre=float(masks[0].mean()), tread=float(masks[1].mean()))
            record['tyre_found'] = bool(masks[0].any())
            record['tread_found'] = bool(masks[1].any())
            record['target_found'] = bool(masks.any())
        self.sync()
        record.update(inference_ms=round((time.perf_counter() - start) * 1000, 2), load_ms=round(load_ms, 2))
        return record, masks

    def inspect(self, rgb, classifier='mobilenetv4', region='segformer_b0', compare=False, progress=None,
                rotation=0, threshold=.25, assist=False):
        start = time.perf_counter()
        orientation_scores = {}
        if rotation == 'auto':
            if progress:
                progress('Checking tyre orientation once; playback resumes after model warmup.')
            for angle in (0, 90, 180, 270):
                oriented = np.ascontiguousarray(np.rot90(rgb, angle//90))
                r, _ = self.predict(oriented, 'yolo26n_seg', threshold)
                orientation_scores[angle] = r['detector']['max_confidence']
            rotation = max(orientation_scores, key=orientation_scores.get)
        rotation = int(rotation)
        if rotation not in (0, 90, 180, 270):
            raise ValueError('Rotation must be a quarter turn')
        working = np.ascontiguousarray(np.rot90(rgb, rotation//90))
        names = list(MODELS) if compare else [classifier] + ([region] if region else [])
        records, masks = [], {}
        if self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for name in names:
            if progress:
                progress(f'Running {MODELS[name]["title"]} — first use includes loading the model.')
            record, mm = self.predict(working, name, threshold)
            records.append(record)
            if mm is not None:
                masks[name] = np.ascontiguousarray(np.rot90(mm, -rotation//90, axes=(1, 2)))
        fallback_used = False
        if assist and not compare and region == 'yolo26n_seg':
            yolo_record = next(r for r in records if r['model'] == region)
            if not (yolo_record['tyre_found'] and yolo_record['tread_found']):
                if progress:
                    progress('YOLO missed a region; checking the separately labelled SegFormer assist.')
                record, mm = self.predict(working, 'segformer_b0')
                record['assist_reason'] = 'YOLO did not return both tyre and tread above the selected threshold'
                records.append(record)
                masks['segformer_b0'] = np.ascontiguousarray(np.rot90(mm, -rotation//90, axes=(1, 2)))
                fallback_used = True
        agreement = {}
        classifiers = [r for r in records if r['task'] == 'classifier']
        if len(classifiers) == 2:
            agreement['classifiers_agree'] = classifiers[0]['prediction'] == classifiers[1]['prediction']
        if len(masks) == 2:
            a, b = list(masks.values())
            agreement['region_iou'] = {}
            for k, label in enumerate(['tyre', 'tread']):
                union = (a[k] | b[k]).sum()
                agreement['region_iou'][label] = float((a[k] & b[k]).sum() / union) if union else None
        return dict(created_at=datetime.now(timezone.utc).isoformat(),
                    frame_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    frame_size=[rgb.shape[1], rgb.shape[0]], models=records, agreement=agreement,
                    analysis_rotation_degrees=rotation, orientation_scores=orientation_scores,
                    yolo_threshold=threshold, assist_used=fallback_used,
                    quality=quality(rgb), device=self.device,
                    device_name=torch.cuda.get_device_name() if self.device.startswith('cuda') else 'CPU',
                    total_ms=round((time.perf_counter() - start) * 1000, 2),
                    peak_vram_mb=round(torch.cuda.max_memory_allocated()/2**20, 1) if self.device.startswith('cuda') else 0,
                    runtime={p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'timm', 'ultralytics', 'transformers', 'PySide6-Essentials', 'opencv-python-headless', 'numpy']},
                    limitation='Mileage-proxy prediction; not tread depth, roadworthiness or measured wear. Scores are not calibrated. Region IoU compares models, not ground truth.'), masks
