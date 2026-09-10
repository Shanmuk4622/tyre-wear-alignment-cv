"""Isolated CPU backend checks. Wheels go to a temporary directory, not the user's environment."""
import io
import json
import os
import sys
import tempfile
from pathlib import Path
import zipfile

import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tyrelib'))
import s5_data as d


def install_wheels(target):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    supported=set(sys_tags())
    for name, version in dict(d.PACKAGES, tokenizers='0.21.4', **{'ultralytics-thop':'2.0.18'}).items():
        response=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=40)
        response.raise_for_status()
        compatible=[f for f in response.json()['urls'] if f['filename'].endswith('.whl')
                    and parse_wheel_filename(f['filename'])[3] & supported]
        assert compatible, f'No compatible binary wheel: {name}'
        wheel=compatible[0]
        print('Read-only validation dependency:',wheel['filename'],flush=True)
        response=requests.get(wheel['url'],timeout=90);response.raise_for_status()
        import hashlib
        assert hashlib.sha256(response.content).hexdigest()==wheel['digests']['sha256']
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(target)


def checks():
    import numpy as np
    import torch
    import segmentation_models_pytorch as smp
    from transformers import RTDetrV2Config,RTDetrV2ForObjectDetection,RTDetrImageProcessor
    from transformers import SegformerConfig,SegformerForSemanticSegmentation
    import s5_runtime as r
    torch.set_num_threads(2)
    # Real pinned public classes, tiny CPU inputs, random weights: no experiment trained.
    for constructor in (smp.Unet,smp.DeepLabV3Plus):
        model=constructor(encoder_name='resnet34',encoder_weights=None,classes=2).eval()
        with torch.inference_mode():
            result=r.logits(model,torch.zeros(2,3,64,64))
        assert result.shape==(2,2,64,64)
        print('PASS',constructor.__name__,flush=True)
        del model
    model=SegformerForSemanticSegmentation(SegformerConfig(num_labels=2)).eval()
    with torch.inference_mode(): result=r.logits(model,torch.zeros(2,3,64,64))
    assert result.shape==(2,2,64,64)
    print('PASS SegFormer two-channel output',flush=True)
    from PIL import Image
    proc=RTDetrImageProcessor(size={'height':64,'width':64})
    target=r.coco_target(0,np.ones((2,64,64),bool))
    inputs=proc(images=[Image.new('RGB',(64,64))],annotations=[target],return_tensors='pt')
    assert inputs['labels'][0]['class_labels'].tolist()==[0,1]
    config=RTDetrV2Config(num_labels=2,num_queries=10,disable_custom_kernels=True,use_pretrained_backbone=False)
    model=RTDetrV2ForObjectDetection(config).eval()
    with torch.inference_mode(): result=model(**inputs)
    assert torch.isfinite(result.loss)
    print('PASS RT-DETRv2 processor + two-class detection loss',flush=True)
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    cfg=get_cfg(overrides={'overlap_mask':False,'optimizer':'AdamW','nbs':4,'patience':0,'warmup_epochs':0.,'imgsz':512})
    assert cfg.overlap_mask is False
    for spec in ('yolo26n.yaml','yolo26n-seg.yaml'):
        model=YOLO(spec)
        print('PASS model constructor',spec,flush=True)
    mask=np.zeros((20,20),bool);mask[2:8,2:8]=True;mask[12:18,12:18]=True
    polygon,iou=r.polygon(mask)
    assert polygon.shape[1]==2 and 0<iou<=1
    score=r.coco_metrics([dict(id=1,image_id=0,category_id=0,bbox=[2,2,6,6],area=36,iscrowd=0)],
        [dict(image_id=0,category_id=0,bbox=[2,2,6,6],score=.9)], [dict(id=0,width=20,height=20)],False)
    assert score['bbox_map_50']>.99
    print('PASS polygon conversion and COCO AP fixture',flush=True)
    masks=Path('D:/Dataset Download/Tire Dataset Prepared/annotations/clean/masks')
    if masks.exists():
        audits=[]
        for path in sorted(masks.glob('*.png')):
            with Image.open(path) as im:
                regions=d.regions(np.array(im))
            for cls,mask in enumerate(regions):
                _,iou=r.polygon(mask)
                audits.append((path.stem,cls,iou))
        failed=[x for x in audits if x[2]<.98]
        print('MANUAL POLYGON AUDIT',len(audits),'regions; min IoU',min(x[2] for x in audits),
              'below .98:',len(failed),failed[:5],flush=True)
        assert not failed, 'The current masks do not pass the planned polygon gate'


if __name__=='__main__':
    if len(sys.argv)>1:
        target=Path(sys.argv[1])
    else:
        target=Path(tempfile.mkdtemp(prefix='tyre-s5-backends-'))
    print('Isolated validation libraries:',target,flush=True)
    os.environ['YOLO_CONFIG_DIR']=str(target/'yolo_settings')
    (target/'yolo_settings').mkdir(exist_ok=True)
    if not (target/'ultralytics').exists():
        install_wheels(target)
    sys.path.insert(0,str(target))
    checks()
