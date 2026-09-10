"""Isolated S5 job process. Parent notebook alone owns HF uploads and their rate budget."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image
from filelock import FileLock

import s5_data as d


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def runtime_versions():
    import importlib.metadata as m
    return {k: m.version(k) for k in ['torch', 'torchvision', 'numpy', *d.PACKAGES]}


def weight_signature(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_checkpoint(state, plan, job):
    assert state['plan_hash'] == d.signature(plan), 'Different data/protocol; do not resume'
    assert state['job'] == job, 'Checkpoint belongs to another model/fold/seed'
    assert 0 <= state['epoch'] <= plan['epochs']
    assert [x['epoch'] for x in state['history']] == list(range(1, state['epoch']+1))
    assert state['runtime'] == runtime_versions(), 'Runtime changed: resume in the recorded package environment'


def publish_local(out, state, native=None):
    """One atomic generation; parent copies it under the same lock before uploading."""
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/'snapshot.lock')):
        tmp = out/'state.tmp.pt'
        if native is not None:
            # Includes native optimizer, EMA and train arguments plus explicit RNG/scaler.
            state = dict(state, native=torch.load(native, map_location='cpu', weights_only=False))
        torch.save(state, tmp)
        os.replace(tmp, out/'state.pt')
        pd.DataFrame(state['history']).to_csv(out/'epochs.csv', index=False)
        atomic_json(out/'STATUS.json', dict(status='trained' if state['epoch']==60 else 'resumable',
            epoch=state['epoch'], plan_hash=state['plan_hash'], job=state['job'],
            checkpoint_sha256=d.digest(out/'state.pt'), evaluated=False))


def state_header(plan, job, epoch, history):
    import tyrelib as tl
    return dict(plan_hash=d.signature(plan), job=job, epoch=epoch, history=history,
                runtime=runtime_versions(), rng=tl.capture_rng())


class DenseDataset:
    def __init__(self, frame, root, annotations, size, train):
        self.rows = list(frame.itertuples())
        self.root, self.annotations, self.size, self.train = Path(root), Path(annotations), size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        row = self.rows[i]
        with Image.open(self.root/row.relative_path) as im:
            image = im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(self.annotations/'clean/masks'/f'{row.image_id}.png') as mm:
            mask = np.array(mm.resize((self.size, self.size), Image.Resampling.NEAREST))
        x, y = np.array(image), d.regions(mask).astype('float32')
        if self.train and random.random() < .5:
            x, y = x[:, ::-1].copy(), y[:, :, ::-1].copy()
        x = torch.from_numpy(x.copy()).permute(2, 0, 1).float()/255
        x = (x-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        return x, torch.from_numpy(y.copy())


def make_model(plan, job, pretrained=True):
    name = job['model']
    if job['backend'] == 'semantic':
        if name.startswith('segformer'):
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
            model_id = plan['models'][name][1]
            cfg = SegformerConfig.from_pretrained(model_id, revision=plan['model_revisions'][model_id])
            cfg.num_labels = 2
            if pretrained:
                model = SegformerForSemanticSegmentation.from_pretrained(model_id, config=cfg,
                    revision=plan['model_revisions'][model_id], ignore_mismatched_sizes=True)
            else:
                model = SegformerForSemanticSegmentation(cfg)
        else:
            import segmentation_models_pytorch as smp
            cls = smp.Unet if name == 'unet_r34' else smp.DeepLabV3Plus
            model = cls(encoder_name='resnet34', encoder_weights='imagenet' if pretrained else None,
                        in_channels=3, classes=2, activation=None)
    elif job['backend'] == 'rtdetr':
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
        model_id = plan['models'][name][1]
        cfg = RTDetrV2Config.from_pretrained(model_id, revision=plan['model_revisions'][model_id])
        cfg.num_labels = 2
        cfg.id2label, cfg.label2id = {0: 'tyre', 1: 'tread'}, {'tyre': 0, 'tread': 1}
        cfg.disable_custom_kernels = True
        model = (RTDetrV2ForObjectDetection.from_pretrained(model_id, config=cfg,
            revision=plan['model_revisions'][model_id], ignore_mismatched_sizes=True)
            if pretrained else RTDetrV2ForObjectDetection(cfg))
    else:
        raise ValueError(job)
    return model


def logits(model, images):
    import torch.nn.functional as F
    output = model(images)
    output = output.logits if hasattr(output, 'logits') else output
    return F.interpolate(output, images.shape[-2:], mode='bilinear', align_corners=False)


def dense_loss(pred, target):
    import torch.nn.functional as F
    probability = pred.float().sigmoid()
    dice = (2*(probability*target).sum((0, 2, 3))+1)/(probability.sum((0, 2, 3))+target.sum((0, 2, 3))+1)
    return F.binary_cross_entropy_with_logits(pred.float(), target) + 1-dice.mean()


def coco_target(i, masks):
    annotations = []
    for cls, mask in enumerate(masks):
        b = d.box(mask)
        if b is not None:
            x0, y0, x1, y1 = b
            annotations.append(dict(id=2*i+cls+1, image_id=i, category_id=cls,
                bbox=[x0, y0, x1-x0, y1-y0], area=int(mask.sum()), iscrowd=0))
    return dict(image_id=i, annotations=annotations)


def detector_batch(rows, root, annotations, processor, device):
    images, targets = [], []
    for i, row in enumerate(rows):
        with Image.open(Path(root)/row.relative_path) as im:
            images.append(im.convert('RGB'))
        with Image.open(Path(annotations)/'clean/masks'/f'{row.image_id}.png') as mm:
            targets.append(coco_target(i, d.regions(np.array(mm))))
    data = processor(images=images, annotations=targets, return_tensors='pt')
    return {k: [{a: b.to(device) if hasattr(b, 'to') else b for a, b in item.items()} for item in v]
            if k == 'labels' else v.to(device) for k, v in data.items()}


def processor_for(plan):
    from transformers import RTDetrImageProcessor
    key = 'PekingU/rtdetr_v2_r18vd'
    return RTDetrImageProcessor.from_pretrained(key, revision=plan['model_revisions'][key],
                                               size={'height': 512, 'width': 512})


def train_torch(plan, job, root, annotations, out, smoke=False):
    import torch
    import tyrelib as tl
    from torch.utils.data import DataLoader
    tl.seed_everything(job['seed'])
    torch.set_num_threads(2)
    tr, _ = d.split_frames(plan, root, job['fold'])
    saved = torch.load(out/'state.pt', map_location='cpu', weights_only=False) if (out/'state.pt').exists() else None
    if saved:
        check_checkpoint(saved, plan, job)
        if saved['epoch'] == plan['epochs'] and not smoke:
            return
    model = make_model(plan, job, pretrained=saved is None).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=plan['epochs'])
    scaler = tl._grad_scaler(torch.device('cuda'))
    history, start = [], 0
    if saved:
        model.load_state_dict(saved['model'], strict=True)
        opt.load_state_dict(saved['optimizer']); sched.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler']); tl.restore_rng(saved['rng'])
        history, start = saved['history'], saved['epoch']
    resumed = saved is not None
    previous_identity = saved['identity'] if saved else None
    del saved
    dataset = DenseDataset(tr, root, annotations, 512, True)
    proc = processor_for(plan) if job['backend'] == 'rtdetr' else None
    rows = list(tr.itertuples())
    batch = plan[job['backend']]['batch']
    initial = previous_identity or dict(model=job['model'], implementation=type(model).__module__+'.'+type(model).__name__,
                   parameters=sum(p.numel() for p in model.parameters()), runtime=runtime_versions(),
                   initial_weights_sha256=weight_signature(model))
    atomic_json(out/'identity.json', initial)
    for epoch in range(start, start+1 if smoke else plan['epochs']):
        # Epoch-seeded order/augmentation makes a completed-epoch resume independent of loader state.
        tl.seed_everything(job['seed']*10000+epoch)
        model.train()
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)
        order = np.random.permutation(len(rows))
        count, total, steps = 0, 0., []
        started = time.monotonic()
        for step in range(math.ceil(len(rows)/batch)):
            t = time.monotonic()
            if step == 0:
                iterator = iter(loader)
            opt.zero_grad(set_to_none=True)
            with tl._autocast(torch.device('cuda')):
                if proc is None:
                    x, y = next(iterator)
                    loss = dense_loss(logits(model, x.cuda()), y.cuda())
                    n = len(x)
                else:
                    selected = [rows[int(i)] for i in order[step*batch:(step+1)*batch]]
                    inputs = detector_batch(selected, root, annotations, proc, 'cuda')
                    loss = model(**inputs).loss
                    n = len(selected)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; previous completed epoch preserved')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            total += float(loss.detach())*n; count += n
            steps.append(time.monotonic()-t)
            if step==0 or (step+1)%10==0 or step+1==len(loader):
                print(f"{job['run_id']} epoch {epoch+1}/60 batch {step+1}/{len(loader)} loss {float(loss):.4f}", flush=True)
            if len(steps) == 5:
                estimate = float(np.median(steps[2:]))*len(loader)
                if estimate > 900:
                    raise RuntimeError(f'Estimated epoch {estimate:.0f}s >15min. Stop, inspect runtime; no silent smaller model/batch.')
            if smoke and step == 5:
                break
        if smoke:
            assert len(steps) >= 5
            # Pilot-only synthetic boundary: save every state component, then test reload
            # in another process. This is never copied into a scientific run namespace.
            state = state_header(plan, job, start, history)
            state.update(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                         scaler=scaler.state_dict(), identity=initial)
            publish_local(out, state)
            atomic_json(out/'SMOKE.json', dict(initial, steps_seconds=steps,
                estimated_epoch_seconds=float(np.median(steps[2:]))*len(loader),
                peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30,
                status='passed', training_run=False, resumed=resumed))
            return
        sched.step()
        history.append(dict(epoch=epoch+1, train_loss=total/count, seconds=time.monotonic()-started,
                            lr=float(opt.param_groups[0]['lr']), examples=count))
        state = state_header(plan, job, epoch+1, history)
        state.update(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     scaler=scaler.state_dict(), identity=initial)
        publish_local(out, state)


def polygon(mask):
    """Bridge external components using Ultralytics' own converter; audit bitmap loss."""
    import cv2
    from ultralytics.data.converter import merge_multi_segment
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    if not segments:
        raise ValueError('Empty/degenerate polygon cannot represent this manual region')
    vertices = np.concatenate(merge_multi_segment(segments)) if len(segments) > 1 else segments[0]
    raster = np.zeros(mask.shape, 'uint8')
    cv2.fillPoly(raster, [vertices.astype('int32')], 1)
    iou = float((raster.astype(bool)&mask).sum()/(raster.astype(bool)|mask).sum())
    return vertices, iou


def export_yolo(plan, job, root, annotations, out):
    import yaml
    export = out/'dataset'
    seg = job['model'].endswith('_seg')
    audit = []
    for role, frame in zip(('train', 'val'), d.split_frames(plan, root, job['fold'])):
        images, labels = export/'images'/role, export/'labels'/role
        images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
        for row in frame.itertuples():
            source = (Path(root)/row.relative_path).resolve()
            dest = images/(row.image_id+source.suffix)
            if not dest.exists():
                dest.symlink_to(source)  # Linux Kaggle input is read-only; no image duplication.
            with Image.open(Path(annotations)/'clean/masks'/f'{row.image_id}.png') as mm:
                masks = d.regions(np.array(mm)); width, height = mm.size
            lines = []
            for cls, mask in enumerate(masks):
                if seg:
                    vertices, iou = polygon(mask)
                    audit.append(dict(image_id=row.image_id, role=role, region=cls, polygon_iou=iou))
                    if iou < plan['yolo']['polygon_min_iou']:
                        raise RuntimeError(f'{row.image_id} region {cls}: polygon IoU {iou:.4f}<.98; cannot silently discard manual mask detail')
                    values = (vertices/np.array([width, height])).reshape(-1)
                else:
                    x0, y0, x1, y1 = d.box(mask)
                    values = [(x0+x1)/2/width, (y0+y1)/2/height, (x1-x0)/width, (y1-y0)/height]
                lines.append(str(cls)+' '+' '.join(f'{v:.9f}' for v in values))
            (labels/(row.image_id+'.txt')).write_text('\n'.join(lines)+'\n')
    pd.DataFrame(audit, columns=['image_id', 'role', 'region', 'polygon_iou']).to_csv(out/'polygon_audit.csv', index=False)
    path = export/'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(export.resolve()), train='images/train', val='images/val',
                                      names={0:'tyre', 1:'tread'})))
    return path


def train_yolo(plan, job, root, annotations, out, smoke=False):
    import torch
    import tyrelib as tl
    from ultralytics import YOLO, settings
    settings.update({'wandb': False, 'mlflow': False, 'clearml': False, 'comet': False,
                     'neptune': False, 'hub': False, 'sync': False})
    data = export_yolo(plan, job, root, annotations, out)
    saved = torch.load(out/'state.pt', map_location='cpu', weights_only=False) if (out/'state.pt').exists() else None
    if saved:
        check_checkpoint(saved, plan, job)
        if saved['epoch'] == 60 and not smoke:
            return
        assert saved['native']['optimizer'] is not None, 'Native optimizer missing; cannot restart as a fresh job'
        native = out/'native_resume.pt'; torch.save(saved['native'], native)
        model = YOLO(str(native))
    else:
        model = YOLO(plan['models'][job['model']][1])
    history = saved['history'][:] if saved else []
    epoch_start = [0.]
    initial = dict(model=job['model'], parameters=sum(p.numel() for p in model.model.parameters()),
                   implementation=type(model.model).__module__+'.'+type(model.model).__name__, runtime=runtime_versions())
    atomic_json(out/'identity.json', initial)

    def on_start(trainer):
        initial.update(parameters=sum(p.numel() for p in trainer.model.parameters()),
                       task=trainer.args.task, classes=trainer.model.names)
        initial['initial_weights_sha256'] = saved['identity']['initial_weights_sha256'] if saved else weight_signature(trainer.model)
        atomic_json(out/'identity.json', initial)
        if saved:
            # Native Ultralytics checkpoints resume from half-precision EMA. Restore
            # the actual training weights/full optimizer instead; retain EMA separately.
            trainer.model.load_state_dict(saved['model'], strict=True)
            trainer.optimizer.load_state_dict(saved['optimizer'])
            trainer.scheduler.load_state_dict(saved['scheduler'])
            trainer.ema.ema.load_state_dict(saved['ema_model'], strict=True)
            trainer.ema.updates = saved['ema_updates']
            trainer.scaler.load_state_dict(saved['scaler'])
            if saved.get('loader_generator') is not None:
                trainer.train_loader.generator.set_state(saved['loader_generator'])
            tl.restore_rng(saved['rng'])

    def epoch_begin(trainer):
        tl.seed_everything(job['seed']*10000+trainer.epoch)
        epoch_start[0] = time.monotonic()

    def checkpoint(trainer):
        seconds = time.monotonic()-epoch_start[0]
        history.append(dict(epoch=trainer.epoch+1, seconds=seconds,
                            train_loss=float(trainer.tloss.detach().sum()),
                            lr=float(trainer.optimizer.param_groups[0]['lr'])))
        state = state_header(plan, job, trainer.epoch+1, history)
        state.update(scaler=trainer.scaler.state_dict(), identity=initial,
                     model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                     scheduler=trainer.scheduler.state_dict(), ema_model=trainer.ema.ema.state_dict(),
                     ema_updates=trainer.ema.updates,
                     loader_generator=trainer.train_loader.generator.get_state() if trainer.train_loader.generator else None)
        publish_local(out, state, trainer.last)
        if seconds > 900:
            raise RuntimeError(f'Epoch took {seconds:.0f}s >15min; checkpoint preserved, inspect runtime')
        if smoke:
            atomic_json(out/'SMOKE.json', dict(initial, status='passed', seconds=seconds,
                                              peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30,
                                              training_run=False, resumed=saved is not None))
            # Deliberate pilot only: never mark this scratch job as a full run.
            raise PilotComplete()

    model.add_callback('on_train_start', on_start)
    model.add_callback('on_train_epoch_start', epoch_begin)
    model.add_callback('on_model_save', checkpoint)
    kwargs = dict(data=str(data), epochs=60, imgsz=512, batch=4, device=0, workers=0,
        optimizer='AdamW', lr0=.0001, lrf=.01, weight_decay=.01, nbs=4,
        cos_lr=True, warmup_epochs=0., patience=0, seed=job['seed'], deterministic=True,
        cache=False, amp=True, save=True, save_period=-1, plots=False,
        project=str(out/'native'), name='train', exist_ok=True,
        mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=0., scale=0., shear=0.,
        perspective=0., flipud=0., fliplr=.5, hsv_h=0., hsv_s=0., hsv_v=0.,
        overlap_mask=False, close_mosaic=0)
    if saved:
        kwargs = dict(resume=True, device=0, workers=0, data=str(data))
    try:
        model.train(**kwargs)
    except PilotComplete:
        if not smoke:
            raise


class PilotComplete(Exception):
    pass


def rle(mask):
    from pycocotools import mask as mask_api
    result = mask_api.encode(np.asfortranarray(mask.astype('uint8')))
    result['counts'] = result['counts'].decode('ascii')
    return result


def predict_regions(model, proc, job, image):
    import torch
    import torch.nn.functional as F
    width, height = image.size
    masks = np.zeros((2, height, width), dtype=bool)
    detections = []
    with torch.inference_mode():
        if job['backend'] == 'semantic':
            x = np.array(image.resize((512, 512), Image.Resampling.BILINEAR)).copy()
            x = torch.from_numpy(x).permute(2, 0, 1).float().cuda()[None]/255
            x = (x-x.new_tensor([.485, .456, .406])[None, :, None, None])/x.new_tensor([.229, .224, .225])[None, :, None, None]
            p = F.interpolate(logits(model, x).float(), (height, width), mode='bilinear', align_corners=False).sigmoid()[0]
            masks = (p >= .5).cpu().numpy()
            for cls in (0, 1):
                b = d.box(masks[cls])
                if b:
                    detections.append(dict(label=cls, box=b, score=float(p[cls][p[cls]>=.5].mean()), mask=rle(masks[cls])))
        elif job['backend'] == 'rtdetr':
            inputs = proc(images=image, return_tensors='pt').to('cuda')
            result = proc.post_process_object_detection(model(**inputs),
                        target_sizes=torch.tensor([[height, width]], device='cuda'), threshold=.001)[0]
            for b, score, cls in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['labels'].cpu().tolist()):
                detections.append(dict(label=int(cls), box=b, score=float(score)))
        else:
            result = model.predict(image, imgsz=512, conf=.001, device=0, verbose=False,
                                   retina_masks=True, max_det=100)[0]
            for i, (b, score, cls) in enumerate(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.int().cpu().tolist())):
                record = dict(label=int(cls), box=b, score=float(score))
                if result.masks is not None:
                    m = result.masks.data[i].float()[None, None]
                    m = F.interpolate(m, (height, width), mode='nearest')[0, 0].cpu().numpy() >= .5
                    record['mask'] = rle(m)
                    if score >= .25:
                        masks[int(cls)] |= m
                detections.append(record)
    boxes = []
    for cls in (0, 1):
        candidates = [x for x in detections if x['label']==cls and x['score']>=.25]
        boxes.append(max(candidates, key=lambda x: x['score'])['box'] if candidates else None)
    if job['backend']=='semantic' or job['model'].endswith('_seg'):
        boxes = [d.box(m) for m in masks]
    return masks, boxes, detections


def coco_metrics(targets, predictions, images, segmentation):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = dict(info={}, images=images, annotations=targets,
                      categories=[dict(id=0, name='tyre'), dict(id=1, name='tread')])
    gt.createIndex()
    result = {}
    for kind in (['bbox', 'segm'] if segmentation else ['bbox']):
        records = [{k: v for k, v in p.items() if k != 'bbox'} if kind=='segm' else
                   {k: v for k, v in p.items() if k != 'segmentation'} for p in predictions]
        if not records:
            result[kind+'_map_50_95'] = 0.; result[kind+'_map_50'] = 0.
            continue
        dt = gt.loadRes(records)
        ev = COCOeval(gt, dt, kind); ev.evaluate(); ev.accumulate(); ev.summarize()
        result[kind+'_map_50_95'], result[kind+'_map_50'] = float(ev.stats[0]), float(ev.stats[1])
    return result


def evaluate(plan, job, root, annotations, out):
    import torch
    import tyrelib as tl
    from huggingface_hub import hf_hub_download
    from sklearn.metrics import f1_score, accuracy_score
    state = torch.load(out/'state.pt', map_location='cpu', weights_only=False)
    check_checkpoint(state, plan, job)
    assert state['epoch'] == 60, 'No final evaluation of a partial training run'
    if job['backend']=='yolo':
        from ultralytics import YOLO
        native = out/'eval_native.pt'; torch.save(state['native'], native)
        model, proc = YOLO(str(native)), None
        model.model.load_state_dict(state['ema_model'], strict=True)
    else:
        model = make_model(plan, job, pretrained=False)
        model.load_state_dict(state['model'], strict=True); model = model.cuda().eval()
        proc = processor_for(plan) if job['backend']=='rtdetr' else None
    del state
    _, va = d.split_frames(plan, root, job['fold'])
    paths = out/'predictions'; paths.mkdir(exist_ok=True)
    records, mask_rows, targets, predictions, images = [], [], [], [], []
    has_masks = job['backend']=='semantic' or job['model'].endswith('_seg')
    # Per-image outputs survive an interrupted evaluation locally; the parent snapshots them every 30min.
    for i, row in enumerate(va.itertuples()):
        path = paths/f'{row.image_id}.json'
        if path.exists():
            rec = json.loads(path.read_text())
            assert rec['plan_hash']==d.signature(plan) and rec['job']==job
        else:
            with Image.open(Path(root)/row.relative_path) as im:
                im = im.convert('RGB'); width, height = im.size
                masks, boxes, dets = predict_regions(model, proc, job, im)
            with Image.open(Path(annotations)/'clean/masks'/f'{row.image_id}.png') as mm:
                truth = d.regions(np.array(mm))
            rec = dict(image_id=row.image_id, plan_hash=d.signature(plan), job=job,
                width=width, height=height, predicted_boxes=boxes, detections=dets,
                mask_metrics=[d.mask_scores(a, b) for a, b in zip(truth, masks)] if has_masks else None)
            atomic_json(path, rec)
        records.append(rec)
        images.append(dict(id=i, width=rec['width'], height=rec['height']))
        with Image.open(Path(annotations)/'clean/masks'/f'{row.image_id}.png') as mm:
            truth = d.regions(np.array(mm))
        for a, mask in zip(coco_target(i, truth)['annotations'], truth):
            a['segmentation'] = rle(mask); targets.append(a)
        for det in rec['detections']:
            x0, y0, x1, y1 = det['box']
            p = dict(image_id=i, category_id=det['label'], score=det['score'], bbox=[x0, y0, x1-x0, y1-y0])
            if has_masks:
                p['segmentation'] = det['mask']
            predictions.append(p)
        if has_masks:
            mask_rows.extend(dict(image_id=row.image_id, region=['tyre','tread'][cls], **scores)
                             for cls, scores in enumerate(rec['mask_metrics']))
    scores = coco_metrics(targets, predictions, images, has_masks)
    del model, proc
    tl.release_host_memory(); torch.cuda.empty_cache()
    # Fixed matched-fold/seed classifier; never select or tune it on S5 results.
    classifier_id = plan['downstream']['classifier'].format(**job)
    from s5_notebook import retry
    file = retry(lambda: hf_hub_download('Shanmuk4622/tyre-wear-study',
        f'runs/{classifier_id}/checkpoints/ckpt_last.pt', repo_type='dataset',
        revision=plan['source_revision'], token=os.environ.get('HF_TOKEN'), local_dir=str(out/'classifier')))
    ck = torch.load(file, map_location='cpu', weights_only=False)
    cfg = ck['config']
    assert cfg['arch']=='resnet50' and cfg['fold']==job['fold'] and cfg['seed']==job['seed']
    classifier = tl.build_model('resnet50', 3, pretrained=False, head=cfg['head_type'], img_size=cfg['input_resolution'])
    classifier.load_state_dict(ck['model'], strict=True); classifier = classifier.cuda().eval()
    classifier_sha = d.digest(file)
    del ck
    tf = tl.build_transforms(cfg['input_resolution'], False, cfg.get('preprocessing', 'raw'))
    rows = []
    for row, rec in zip(va.itertuples(), records):
        with Image.open(Path(root)/row.relative_path) as im, Image.open(Path(annotations)/'clean/masks'/f'{row.image_id}.png') as mm:
            im = im.convert('RGB'); truth = d.regions(np.array(mm))
            choices = [None, *rec['predicted_boxes'], *[d.box(m) for m in truth]]
            for mode, b in zip(plan['downstream']['modes'], choices):
                cropped = im.crop(d.padded_box(b, *im.size))
                with torch.inference_mode():
                    output = classifier(tf(cropped)[None].cuda()).float()
                    prob = tl.CoralHead.probs(output) if cfg['head_type']=='coral' else output.softmax(1)
                    pred = int(tl.CoralHead.predict(output)[0]) if cfg['head_type']=='coral' else int(output.argmax(1)[0])
                rows.append(dict(image_id=row.image_id, session=row.session_group, mode=mode,
                    truth=tl.C2I[row.proxy_label], prediction=pred, fallback=mode.startswith('pred_') and b is None,
                    prob_low=float(prob[0,0]), prob_mid=float(prob[0,1]), prob_high=float(prob[0,2])))
    frame = pd.DataFrame(rows)
    roi = []
    for mode, group in frame.groupby('mode'):
        roi.append(dict(mode=mode, macro_f1=float(f1_score(group.truth, group.prediction, labels=[0,1,2], average='macro', zero_division=0)),
                        accuracy=float(accuracy_score(group.truth, group.prediction)), fallback_count=int(group.fallback.sum()), n=len(group)))
    base = next(x['macro_f1'] for x in roi if x['mode']=='full')
    for x in roi:
        x['delta_macro_f1_vs_full'] = x['macro_f1']-base
    with FileLock(str(out/'snapshot.lock')):
        frame.to_csv(out/'roi_predictions.csv', index=False)
        pd.DataFrame(roi).to_csv(out/'roi_metrics.csv', index=False)
        pd.DataFrame(mask_rows).to_csv(out/'mask_metrics.csv', index=False)
        atomic_json(out/'metrics.json', dict(job=job, plan_hash=d.signature(plan), n_validation=len(va),
            localisation=scores, classifier_run=classifier_id, classifier_checkpoint_sha256=classifier_sha,
            classifier_source_revision=plan['source_revision'], endpoint='epoch_60'))
        status = json.loads((out/'STATUS.json').read_text())
        status.update(status='completed', evaluated=True, n_validation=len(va),
                      artifact_sha256={f:d.digest(out/f) for f in ['epochs.csv','roi_predictions.csv','roi_metrics.csv','mask_metrics.csv','metrics.json']})
        atomic_json(out/'STATUS.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    req = json.loads(Path(args.request).read_text())
    plan, job, out = req['plan'], req['job'], Path(req['out'])
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'Select Kaggle T4 x2'
    assert torch.cuda.device_count()>=2 and all('T4' in torch.cuda.get_device_name(i) for i in (0,1)), 'Select T4 x2; no automatic replacement'
    assert shutil.disk_usage(out).free > 5*2**30, 'Need at least 5GiB scratch headroom'
    if req['action']=='evaluate':
        evaluate(plan, job, req['root'], req['annotations'], out)
    else:
        fn = train_yolo if job['backend']=='yolo' else train_torch
        assert req['action'] in ('train','smoke','resume_test')
        if req['action']=='resume_test':
            assert (out/'state.pt').exists(), 'Resume pilot needs its checkpoint'
        fn(plan, job, req['root'], req['annotations'], out, smoke=req['action'] in ('smoke','resume_test'))
        if req['action']=='resume_test':
            assert json.loads((out/'SMOKE.json').read_text())['resumed'], 'Pilot did not actually resume'


if __name__ == '__main__':
    main()
