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

# Preserve legacy identifiers for evidence/recipes; the title identifies Medium.
from phase2_adapter import enabled as phase2_enabled, spec as phase2_spec
if phase2_enabled():
    for name, item in MODELS.items():
        selected = phase2_spec(name)
        item.update(revision=selected['revision'], path=selected['weights'], phase2=True)
        item['title'] = {'mobilenetv4': 'MobileNet V4 · Phase 2', 'resnet50': 'ResNet 50 · Phase 2',
                         'yolo26n_seg': 'YOLO26 Medium · Phase 2', 'segformer_b0': 'SegFormer B0 · Phase 2'}[name]
