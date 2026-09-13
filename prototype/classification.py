"""Small inference-only equivalents of tyrelib v12's classifier contract.

Keep these synchronized with tyrelib.build_transforms and CoralHead. No training
infrastructure or pandas/Arrow import is needed to classify one image.
"""
import numpy as np
import torch

ARCHITECTURES = {
    'mobilenetv4': 'mobilenetv4_conv_medium.e500_r256_in1k',
    'resnet50': 'resnet50',
}


def build_classifier(name, cfg):
    import timm
    return timm.create_model(ARCHITECTURES[name], pretrained=False,
                             num_classes=2 if cfg['head_type'] == 'coral' else 3)


def build_transforms(size, preprocessing):
    from torchvision import transforms as T
    ops = []
    if preprocessing == 'clahe':
        def clahe(im):
            import cv2
            from PIL import Image
            lab = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        ops.append(T.Lambda(clahe))
    elif preprocessing not in ('raw', 'grayscale'):
        raise ValueError(f'Unsupported checkpoint preprocessing: {preprocessing}')
    ops.append(T.Resize((size, size)))
    if preprocessing == 'grayscale':
        ops.append(T.Grayscale(num_output_channels=3))
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def decision(logits, head):
    if head != 'coral':
        return logits.softmax(-1)[0], int(logits.argmax(-1).item())
    cum = logits.sigmoid()
    probs = torch.stack([1-cum[:, 0], cum[:, 0]-cum[:, 1], cum[:, 1]], dim=1)
    probs = probs.clamp_min(1e-8) / probs.clamp_min(1e-8).sum(1, keepdim=True)
    # Deliberately threshold-count, not argmax of the displayed scores.
    return probs[0], int((cum > .5).sum(1).item())
