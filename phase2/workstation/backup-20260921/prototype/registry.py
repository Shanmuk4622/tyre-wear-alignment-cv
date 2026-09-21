"""Fixed, final-epoch fold-1 seed-1 artifacts. No validation-best selection."""
from bootstrap import ROOT

REPO = 'Shanmuk4622/tyre-wear-study'
REVISION = 'a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5'
CLASS_REVISION = 'dd43b231cfbdd92dd6d8c01b47166ddec4ab05f8'
S5 = 's5/s5-manual-2026-09-10-r1/1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df'
MODELS = {
    'mobilenetv4': dict(title='MobileNet V4', task='classifier', revision=CLASS_REVISION,
        path='runs/a-mobilenetv4-base-f1-s1/checkpoints/ckpt_last.pt'),
    'resnet50': dict(title='ResNet 50', task='classifier', revision=CLASS_REVISION,
        path='runs/a-resnet50-base-f1-s1/checkpoints/ckpt_last.pt'),
    'yolo26n_seg': dict(title='YOLO26 Nano', task='regions', revision=REVISION,
        path=f'{S5}/runs/yolo26n_seg-f1-s1/state.pt'),
    'segformer_b0': dict(title='SegFormer B0', task='regions', revision=REVISION,
        path=f'{S5}/runs/segformer_b0-f1-s1/state.pt'),
}

def checkpoint(name):
    return ROOT / 'checkpoints' / name / 'inference.pt'
