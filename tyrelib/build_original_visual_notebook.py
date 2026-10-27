"""Build the standalone original-study notebook; never overwrite executed NB07."""
import ast
import base64
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def extract(path,names):
    text=path.read_text(encoding='utf-8');tree=ast.parse(text);parts=[]
    for node in tree.body:
        name=getattr(node,'name',None)
        if isinstance(node,ast.AnnAssign):name=getattr(node.target,'id',None)
        if name in names:parts.append(ast.get_source_segment(text,node))
    return '\n\n'.join(parts)


def payloads():
    support='''import gc, hashlib, json, shutil, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
REPO='Shanmuk4622/tyre-wear-study'
def _print(*args): print(*args)
'''
    support+=extract(ROOT/'phase2/phase2_visual_compare.py',{'sha','write','Publisher','boundaries','draw_lines'}).replace("'phase2/visual-comparison-v1/'","'original-study/visual-comparison-v1/'").replace('len(self.calls)>=60','len(self.calls)>=100')
    support+='\n'+extract(ROOT/'tyrelib/tyrelib.py',{'ZOO','_timm_model_candidates','build_model','build_transforms'})
    support+='\n'+extract(ROOT/'tyrelib/hrnet_runtime.py',{'Geometry','positions'})
    ast.parse(support)
    return {'original_visual_support.py':support.encode(),
        'original_visual_compare.py':(ROOT/'tyrelib/original_visual_compare.py').read_bytes(),
        'assets.json':(ROOT/'notebooks/assets/original_visual_assets.json').read_bytes(),
        'registry.json':(ROOT/'notebooks/assets/original_visual_registry.json').read_bytes()}


def main():
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('''# Original paper study — low / medium / high model comparison

**This is the corrected notebook: original final_v1 dataset + original trained models only. No Phase 2 dataset, split, weights or inference contract is used.** The executed Phase 2 notebook is preserved separately as a record of that run.

### Scope
- All **17 retained Stage-A classifier architectures**, fold 1 / seed 1. The quarantined ConvNeXt-S substitution is excluded.
- All **nine original S5 localisers**: U-Net, DeepLabV3+, SegFormer B0/B2, YOLO26 Nano/Small detection and segmentation, RT-DETRv2 R18.
- Original **HRNet W18 + matched SegFormer** geometry checkpoints, seed 1 / final epoch 60.
- These are 28 model pipelines, not every fold/seed replica. No retraining.

### Kaggle setup
1. Import this notebook. Select GPU **T4 ×2**, enable Internet and the **HF_TOKEN** secret with write access to `Shanmuk4622/tyre-wear-study`.
2. Attach the original [shanmuk4622/tire-dataset-prepared](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared), containing `FINAL/manifests/clean_manifest.csv`. **Do not attach the Phase 2 dataset for this notebook.**
3. Run all cells. The original manifest and selected source images are hash-checked.
4. Inspect classifier score sheets, nine-model overlay boards, and geometry diagrams. Download and extract the gallery ZIP; open `gallery.html`.

Default examples: three per class, chosen deterministically from the original **fold-1 validation set**, before predictions. These are internal illustrations, not an independent test or a guarantee that every low-use image will be recognised. Geometry uses a different split; its cohort label is shown separately.
''')
    code('''DATA_ROOT = ''  # Auto-detect original FINAL; otherwise enter its full Kaggle input path.
PER_CLASS = 3
CLASSIFIER_ENDPOINT = 'best'  # 'best', 'last', or 'both'
# best = original maximum-validation-QWK checkpoint, NOT best on these displayed images.
# last = fixed epoch 60. both = show both endpoints, without choosing whichever looks better.
RUN_TAG = 'original-paper-comparison'
PUBLISH_TO_HF = True
# Defaults include ALL retained classifier architectures and all original region/geometry models.
CLASSIFIERS = None  # Optional explicit subset, e.g. ['mobilenetv4','resnet50'] for a quick check.
INCLUDE_LOCALISATION = True
INCLUDE_GEOMETRY = True
''')
    md('''## 1 · Self-contained runtime and frozen provenance
Only one model occupies GPU 0 at a time; a single T4 also works. The full comparison requires several GB of checkpoint downloads, but only the active downloaded checkpoint is retained. Completed prediction files survive interruptions. Dataset images remain in read-only Kaggle input; model weights and full dataset files are not copied into the output gallery. At least the next checkpoint size plus 2 GiB reserve must be free. No 1 TB assumption.
''')
    setup='''import os, sys, subprocess, base64, json, hashlib, re
from pathlib import Path
WORK=Path('/kaggle/working/original_tyre_comparison');WORK.mkdir(parents=True,exist_ok=True)
RUNTIME=WORK/'runtime';RUNTIME.mkdir(exist_ok=True)
DEPS=WORK/'deps'
os.environ['YOLO_CONFIG_DIR']=str(WORK/'yolo_settings')
os.environ['HF_HUB_DISABLE_XET']='1'  # Use resumable HTTP downloads; no separate Xet runtime.
(WORK/'yolo_settings'/'Ultralytics').mkdir(parents=True,exist_ok=True)
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-input','--no-deps','--upgrade','--target',str(DEPS),
 'timm==1.0.15','ultralytics==8.4.20','ultralytics-thop==2.0.18','transformers==4.51.3',
 'tokenizers==0.21.4','huggingface_hub==0.36.0','safetensors==0.5.3','filelock==3.19.1',
 'segmentation-models-pytorch==0.5.0'],timeout=600)
sys.path[:0]=[str(DEPS),str(RUNTIME)]
'''
    for n,b in payloads().items():setup+=f"(RUNTIME/{n!r}).write_bytes(base64.b64decode({base64.b64encode(b).decode()!r}))\n"
    setup+='''from PIL import Image
import torch, timm, transformers, ultralytics
assert timm.__version__=='1.0.15' and transformers.__version__=='4.51.3' and ultralytics.__version__=='8.4.20'
assert torch.cuda.is_available(), 'Enable a Kaggle GPU.'
from kaggle_secrets import UserSecretsClient
TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN: raise RuntimeError('Enable HF_TOKEN in Secrets; never paste it into code.')
import original_visual_compare as studio
from original_visual_support import Publisher,write,sha
ASSETS=json.loads((RUNTIME/'assets.json').read_text());REGISTRY=json.loads((RUNTIME/'registry.json').read_text())
print('Original study only |',torch.cuda.get_device_name(0),'| PyTorch',torch.__version__)
'''
    code(setup)
    md('''## 2 · Verify old data and declare checkpoint policy
The best and final classifier checkpoints have materially different behaviour in the original study. This notebook exposes that choice rather than silently substituting a model. Original CORAL classifiers use a **threshold-count decision**; their three displayed scores are not used to replace that decision with argmax. Localisers and geometry use their original fixed-final checkpoints.
''')
    code('''from IPython.display import display, Image as DisplayImage, FileLink
import pandas as pd
assert CLASSIFIER_ENDPOINT in ('best','last','both')
assert re.fullmatch(r'[A-Za-z0-9_-]{1,60}',RUN_TAG)
archs=studio.ARCHS if CLASSIFIERS is None else CLASSIFIERS
assert len(set(archs))==len(archs) and all(a in studio.ARCHS for a in archs)
endpoints=['best','last'] if CLASSIFIER_ENDPOINT=='both' else [CLASSIFIER_ENDPOINT]
KEYS=[a+'-'+e for a in archs for e in endpoints]
if INCLUDE_LOCALISATION: KEYS += [k for k,v in REGISTRY.items() if v['task']=='localisation']
if INCLUDE_GEOMETRY: KEYS += ['geometry-hrnet','geometry-matched']
assert KEYS, 'Select at least one model.'
ROWS=studio.select(DATA_ROOT,ASSETS,PER_CLASS)
identity=dict(schema=1,tag=RUN_TAG,models={k:REGISTRY[k] for k in KEYS},
    images=[{k:r[k] for k in ['image_id','sha256','class_index','role','geometry_role']} for r in ROWS],
    runtime={p.name:sha(p) for p in RUNTIME.iterdir() if p.is_file()})
RUN_ID=RUN_TAG+'-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
PUBLISHER=Publisher(WORK,RUN_ID,TOKEN,PUBLISH_TO_HF)
PUBLISHER.restore()
write(PUBLISHER.out/'run.json',dict(identity=identity,inputs=ROWS,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
display(pd.DataFrame([dict(image=r['image_id'],label=studio.CLASSES[r['class_index']],classification_cohort=r['role'],geometry_cohort=r['geometry_role']) for r in ROWS]))
print('Pipelines:',len(KEYS),'| source: original final_v1 | classifiers:',CLASSIFIER_ENDPOINT)
print('HF output namespace:',PUBLISHER.prefix)
''')
    md('''## 3 · Run original models
All 45 available checkpoint references (17 best + 17 final + 9 localisation + 2 geometry) are pinned to immutable HF revisions and SHA256 hashes. Only the selected endpoints are downloaded. Training-state files contain the original model configuration; loading checks the recorded architecture, fold, seed and epoch, then loads tensors strictly. No fallback architecture or Phase 2 switch exists.

Progress is saved after each image/model pair. HF snapshots are batched after each group of four models and at the end of inference, at 30-minute inference checkpoints, and on catchable Stop/error. Wait for the verified upload message after Stop. Forced kernel termination/network loss can lose work since the last upload. Restart with the same settings and inputs to recover. Never run two sessions with the same RUN_TAG and settings concurrently.
''')
    code('studio.run(ROWS,REGISTRY,ASSETS,WORK,PUBLISHER,KEYS)\n')
    md('''## 4 · Compare without hiding failures
The classification sheet includes every selected architecture in a fixed order. ✓/× compares the **actual ordinal decision** with the supplied mileage label. Scores are uncalibrated.

The region board shows masks for segmenters and boxes for detectors; missing detections remain visible. Geometry compares HRNet proposals with matched SegFormer mask intersections. Missing/clipped boundaries remain missing. Boundary gap is model disagreement, not measured alignment accuracy. Do not choose examples or endpoints solely because they look correct and present them as a new evaluation.
''')
    code('''try:
    FIGURES=studio.render(ROWS,KEYS,REGISTRY,PUBLISHER.out)
    PUBLISHER.publish('original gallery complete')
except BaseException:
    try:PUBLISHER.publish('render interrupted')
    except Exception as e:print('Upload unavailable:',type(e).__name__,'Local files retained.')
    raise
for p in FIGURES: display(DisplayImage(filename=str(p)))
''')
    md('''## 5 · Download
Extract the ZIP before opening `gallery.html`. It includes figures, classification CSV, raw numerical predictions/masks/boxes, and complete checkpoint/input provenance. No paper files are modified.
''')
    code('''import zipfile
DOWNLOAD=WORK/(RUN_ID+'-gallery.zip')
with zipfile.ZipFile(DOWNLOAD,'w',zipfile.ZIP_DEFLATED) as z:
    for p in PUBLISHER.out.rglob('*'):
        if p.is_file() and p.suffix!='.tmp':z.write(p,p.relative_to(PUBLISHER.out).as_posix())
display(FileLink(str(DOWNLOAD)))
print('Kaggle Output:',DOWNLOAD.name)
print('HF: https://huggingface.co/datasets/'+studio.REPO+'/tree/main/'+PUBLISHER.prefix)
''')
    md('''### Troubleshooting
- Dataset rejected: select the original `tire-dataset-prepared` input and its FINAL directory, not `tire-dataset-prepared-phase2`.
- Import or GPU-memory error: use a fresh Kaggle kernel; Run All restores saved outputs. Preserve the existing Kaggle Torch/torchvision installation.
- HTTP 429: allow the bounded retries and Retry-After delay. Uploads are batched, with a conservative per-session application-call budget; actual server request limits still apply.
- Upload failure: keep Kaggle outputs and retry `PUBLISHER.publish('manual recovery')`. No API token is saved or printed.
- Low-use mistakes: inspect the recorded ordinal prediction and endpoint, not just the largest displayed score. An old model is not assumed correct simply because Phase 2 failed.
''')
    for i,c in enumerate(cells):c['id']=f'original-visual-{i:02d}'
    n=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.11')))
    path=ROOT/'notebooks/NB33_Original_Study_Visual_Comparison.ipynb';path.write_text(json.dumps(n,indent=1)+'\n',encoding='utf-8');print(path)


if __name__=='__main__':main()
