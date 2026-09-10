"""S5 manual-label protocol and geometry. No GPU, training, or network side effects."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REVISION = 's5-manual-2026-09-10-r1'
SOURCE_REVISION = 'dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8'
MODELS = {
    'unet_r34': ('semantic', 'smp.Unet:resnet34:imagenet'),
    'deeplabv3plus_r34': ('semantic', 'smp.DeepLabV3Plus:resnet34:imagenet'),
    'segformer_b0': ('semantic', 'nvidia/mit-b0'),
    'segformer_b2': ('semantic', 'nvidia/mit-b2'),
    'yolo26n_det': ('yolo', 'yolo26n.pt'),
    'yolo26s_det': ('yolo', 'yolo26s.pt'),
    'yolo26n_seg': ('yolo', 'yolo26n-seg.pt'),
    'yolo26s_seg': ('yolo', 'yolo26s-seg.pt'),
    'rtdetrv2_r18': ('rtdetr', 'PekingU/rtdetr_v2_r18vd'),
}
PACKAGES = {'ultralytics': '8.4.20', 'transformers': '4.51.3',
            'segmentation-models-pytorch': '0.5.0', 'timm': '1.0.15',
            'pycocotools': '2.0.11'}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def regions(mask):
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2, 3, 4]).all():
        raise ValueError('Expected an indexed manual mask with values 0..4')
    return np.stack((mask > 0, (mask == 2) | (mask == 3)))


def box(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def padded_box(b, width, height, fraction=.05):
    if b is None:
        return [0, 0, width, height]
    x0, y0, x1, y1 = map(float, b)
    if not np.isfinite(b).all() or x1 <= x0 or y1 <= y0:
        return [0, 0, width, height]
    px, py = fraction * (x1-x0), fraction * (y1-y0)
    result = [max(0, int(np.floor(x0-px))), max(0, int(np.floor(y0-py))),
              min(width, int(np.ceil(x1+px))), min(height, int(np.ceil(y1+py)))]
    return result if result[2] > result[0] and result[3] > result[1] else [0, 0, width, height]


def mask_scores(reference, prediction):
    import cv2
    a, b = np.asarray(reference, bool), np.asarray(prediction, bool)
    if a.shape != b.shape:
        raise ValueError('Native mask geometry differs')
    inter, union, total = int((a & b).sum()), int((a | b).sum()), int(a.sum()+b.sum())
    kernel = np.ones((3, 3), np.uint8)
    ea = a & ~cv2.erode(a.astype('uint8'), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    eb = b & ~cv2.erode(b.astype('uint8'), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    near_a = cv2.dilate(ea.astype('uint8'), np.ones((5, 5), np.uint8)).astype(bool)
    near_b = cv2.dilate(eb.astype('uint8'), np.ones((5, 5), np.uint8)).astype(bool)
    precision = float((eb & near_a).sum()/max(1, eb.sum()))
    recall = float((ea & near_b).sum()/max(1, ea.sum()))
    bf = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(iou=inter/union if union else None, dice=2*inter/total if total else None,
                boundary_f1_2px=bf if total else None,
                reference_pixels=int(a.sum()), prediction_pixels=int(b.sum()))


def inspect_data(root, annotations):
    root, annotations = Path(root), Path(annotations)
    clean = pd.read_csv(root/'manifests/clean_manifest.csv')
    assert len(clean) == 418 and clean.image_id.is_unique, 'Expected 418 unique clean images'
    assert set(clean.image_kind) == {'clean_original'}
    records = []
    for row in clean.sort_values('image_id').itertuples():
        image_path, mask_path = root/row.relative_path, annotations/'clean/masks'/f'{row.image_id}.png'
        with Image.open(image_path) as im, Image.open(mask_path) as mm:
            assert im.size == mm.size, f'Image/mask size mismatch: {row.image_id}'
            width, height = im.size
            rr = regions(np.array(mm))
        assert rr[0].any() and rr[1].any(), f'Missing tyre/tread: {row.image_id}'
        image_sha = digest(image_path)
        assert image_sha == row.file_sha256, f'Image contents differ from manifest: {row.image_id}'
        records.append(dict(image_id=row.image_id, image_sha256=image_sha, mask_sha256=digest(mask_path),
                            width=width, height=height, tyre_box=box(rr[0]), tread_box=box(rr[1]),
                            proxy_label=row.proxy_label, session_group=row.session_group,
                            relative_path=row.relative_path, fold_id=int(row.fold_id)))
    splits = {}
    known = set(clean.image_id)
    for fold in (0, 1, 2):
        tr = pd.read_csv(root/f'splits/cv{fold}_train.csv')
        va = pd.read_csv(root/f'splits/cv{fold}_validation.csv')
        tr = tr.loc[tr.image_kind.eq('clean_original')]
        assert set(va.image_kind) == {'clean_original'}
        assert tr.image_id.is_unique and va.image_id.is_unique
        assert set(tr.image_id).isdisjoint(va.image_id)
        assert set(tr.session_group).isdisjoint(va.session_group)
        assert set(tr.image_id) | set(va.image_id) == known
        assert set(tr.file_sha256).isdisjoint(va.file_sha256), 'Exact image leakage'
        splits[str(fold)] = dict(train=sorted(tr.image_id), validation=sorted(va.image_id))
    return dict(records=records, splits=splits,
                manifest_sha256=digest(root/'manifests/clean_manifest.csv'),
                split_sha256={f'cv{fold}_{role}':digest(root/f'splits/cv{fold}_{role}.csv')
                              for fold in (0,1,2) for role in ('train','validation')})


def protocol(data):
    return dict(revision=REVISION, source_revision=SOURCE_REVISION,
        implementation_sha256={name:digest(Path(__file__).parent/name)
                               for name in ('s5_data.py','s5_runtime.py')},
        models={k: list(v) for k, v in MODELS.items()}, packages=PACKAGES,
        epochs=60, folds=[0, 1, 2], seeds=[1, 2, 3], label_source='existing_manual',
        data=data, training_images='clean_only_no_derived_images',
        regions=['tyre: labels 1+2+3+4', 'tread: labels 2+3'],
        semantic=dict(size=512, batch=4, loss='BCEWithLogits + soft Dice (two overlapping sigmoid channels)',
                      optimizer='AdamW', lr=.0001, weight_decay=.01, scheduler='cosine', augmentation='horizontal_flip_0.5'),
        rtdetr=dict(size=512, batch=2, optimizer='AdamW', lr=.0001, weight_decay=.01,
                    scheduler='cosine', augmentation='none', loss='native RT-DETRv2 detection loss'),
        yolo=dict(size=512, batch=4, optimizer='AdamW', lr=.0001, weight_decay=.01,
                  augmentation='horizontal_flip_0.5_only', polygon_min_iou=.98,
                  overlap_mask=False, endpoint='final_epoch_EMA'),
        endpoint='fixed_epoch_60_not_validation_selected', gpu='cuda:0_no_DataParallel',
        downstream=dict(classifier='a-resnet50-base-f{fold}-s{seed}', checkpoint='ckpt_last.pt',
                        classifier_source_revision=SOURCE_REVISION, crop_padding=.05,
                        modes=['full', 'pred_tyre', 'pred_tread', 'oracle_tyre', 'oracle_tread'],
                        missing_prediction='full_image_fallback', classifier_training='none_frozen'),
        limitations=['folds 0/2 suspected cross-tyre leakage; fold 1 tiny tyre sample',
                      'descriptive existing-fold evaluation, not independent new-tyre testing',
                      'SAM2 comparison and blind repeat annotation deferred',
                      'backend-native losses/EMA/pretraining differ; not a controlled equal-pretraining ablation'])


def jobs(plan, family=None):
    result = []
    for name, (backend, _) in plan['models'].items():
        if family and backend != family:
            continue
        for fold in plan['folds']:
            for seed in plan['seeds']:
                result.append(dict(model=name, backend=backend, fold=fold, seed=seed,
                                   run_id=f'{name}-f{fold}-s{seed}'))
    return result


def assigned(plan, family, worker, workers):
    if not 0 <= worker < workers:
        raise ValueError('Worker outside active account list')
    # Ownership never depends on which jobs have already finished.
    return [j for i, j in enumerate(jobs(plan, family)) if i % workers == worker]


def split_frames(plan, root, fold):
    clean = pd.read_csv(Path(root)/'manifests/clean_manifest.csv').set_index('image_id', drop=False)
    s = plan['data']['splits'][str(fold)]
    return clean.loc[s['train']].reset_index(drop=True), clean.loc[s['validation']].reset_index(drop=True)
