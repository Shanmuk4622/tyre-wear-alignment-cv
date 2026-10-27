"""Build a self-contained Kaggle inference notebook from the checked runtime."""
import ast
import base64
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    cells=[]
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        ast.parse(s)
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('''# Tyre model comparison studio
## Low · Medium · High — five models, one consistent view

This notebook **runs inference only** using all five current Phase 2 model families: MobileNet V4 Medium, ResNet50, YOLO26 Medium segmentation, SegFormer B0 and HRNet W18. It uses the same validation-selected exports as Tread Station, not every historical architecture or training seed.

Each example gets a six-panel inspection card: original photo, two probability plots, two transparent region overlays, and a learned-boundary comparison. A probability fingerprint and a light-theme HTML gallery bring the examples together. No model is selected using the displayed predictions.

### Before Run All
1. Import this notebook into Kaggle. Choose **T4 ×2** (one GPU is sufficient; inference keeps only one model on GPU 0 at a time to save memory).
2. Enable Internet and the **HF_TOKEN** secret with write permission to `Shanmuk4622/tyre-wear-study`.
3. Attach [Tire Dataset Prepared phase2](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared-phase2) directly as a Kaggle input. No dataset download from HF and no repository upload are needed.
4. Keep the defaults for three examples per category, or attach your own `low/`, `medium/`, `high/` image folders and set `CUSTOM_ROOT` below.
5. Run top to bottom. Open the inline cards, then download the gallery ZIP from the final cell.

**Interpretation:** supplied classes are mileage proxies. Medium examples from the prepared collection are training images; low/high defaults use test images. These few selected examples are qualitative illustrations, not a new accuracy benchmark. Boundary agreement is not physical alignment accuracy. Softmax scores are uncalibrated.
''')
    code('''# Change these before running. Keep RUN_TAG unchanged when resuming.
PER_CLASS = 3
DATA_ROOT = ''       # Auto-detect the attached prepared Phase 2 dataset.
CUSTOM_ROOT = ''     # Example: '/kaggle/input/my-tyres/photos' with low/medium/high.
RUN_TAG = 'first-comparison'  # New tag for a separate gallery; never run duplicate writers.
PUBLISH_TO_HF = True # Saves overlays + predictions to your existing HF dataset repository.
''')
    md('''## 1 · Prepare the small runtime
The notebook embeds the exact model code and export registry. It downloads only five checkpoint files (about 278 MiB total). Prepared images stay in read-only Kaggle input. It never trains, copies the full dataset, or assumes 1 TB of storage.
''')
    setup='''import os, sys, subprocess, base64, json, hashlib, re
from pathlib import Path
WORK = Path('/kaggle/working/tyre_visual_comparison')
WORK.mkdir(parents=True, exist_ok=True)
RUNTIME = WORK/'runtime'; RUNTIME.mkdir(exist_ok=True)
DEPS = WORK/'deps'
os.environ['YOLO_CONFIG_DIR'] = str(WORK/'yolo_settings')
(WORK/'yolo_settings'/'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
    '--disable-pip-version-check', '--no-deps', '--upgrade', '--target', str(DEPS),
    'timm==1.0.15', 'ultralytics==8.4.20', 'ultralytics-thop==2.0.18',
    'transformers==4.51.3', 'tokenizers==0.21.4', 'huggingface_hub==0.36.0',
    'safetensors==0.5.3', 'filelock==3.19.1'], timeout=600)
# Start with a fresh kernel if these packages were imported in another notebook cell.
sys.path[:0] = [str(DEPS), str(RUNTIME)]
'''
    payload={n:(ROOT/n).read_bytes() for n in ['phase2_models.py','phase2_training_data.py','phase2_visual_compare.py']}
    payload['registry.json']=(ROOT/'workstation/registry.json').read_bytes()
    payload['MODEL_CONFIG.json']=(ROOT/'workstation/models/segformer/MODEL_CONFIG.json').read_bytes()
    for n,b in payload.items(): setup+=f"(RUNTIME/{n!r}).write_bytes(base64.b64decode({base64.b64encode(b).decode()!r}))\n"
    setup+='''from PIL import Image  # Load before torch, also supports local Windows checking.
import torch, timm, transformers, ultralytics
assert timm.__version__ == '1.0.15'
assert transformers.__version__ == '4.51.3'
assert ultralytics.__version__ == '8.4.20'
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN: raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets. Never paste it here.')
import phase2_visual_compare as studio
REGISTRY = json.loads((RUNTIME/'registry.json').read_text())
print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())
if not torch.cuda.is_available(): print('CPU fallback is supported but slower. A T4 is recommended.')
'''
    code(setup)
    md('''## 2 · Choose examples before seeing predictions
Default selection takes original photos in deterministic ID order, cycling through available tyre identities. It prefers the test partition for low/high. All prepared medium tyres belong to training, and every card says so. With custom inputs, folder names supply the expected category; independence from training is unknown.
''')
    code('''import pandas as pd
from IPython.display import display, Image as DisplayImage, FileLink, HTML
ROWS = studio.select(DATA_ROOT, CUSTOM_ROOT, PER_CLASS)
assert re.fullmatch(r'[A-Za-z0-9_-]{1,60}', RUN_TAG), 'Use letters, numbers, - or _ for RUN_TAG.'
identity = dict(schema=1, tag=RUN_TAG,
    images=[dict(key=r['key'],sha=r['image_sha256'],label=r['class_index'],role=r['role']) for r in ROWS],
    weights={k:v['sha256'] for k,v in REGISTRY.items()},
    runtime={n:studio.sha(RUNTIME/n) for n in ['phase2_visual_compare.py','phase2_models.py','phase2_training_data.py','MODEL_CONFIG.json']})
RUN_ID = RUN_TAG+'-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
PUBLISHER = studio.Publisher(WORK, RUN_ID, TOKEN, PUBLISH_TO_HF)
PUBLISHER.restore()
studio.write(PUBLISHER.out/'run.json', dict(identity=identity, inputs=ROWS, registry=REGISTRY,
    environment=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')))
display(pd.DataFrame([dict(image=r['image_id'],category=studio.CLASSES[r['class_index']],cohort=r['role'],tyre=r['physical_tyre_id']) for r in ROWS]))
print('Run:', RUN_ID, '\\nOutput:', PUBLISHER.out, '\\nHF path:', PUBLISHER.prefix)
''')
    md('''## 3 · Run all five models
Weights are loaded strictly and checked against their recorded SHA256. Input sizes, normalization, mask thresholds and fixed guide rows match the trained contract. Results are atomically saved after **each image/model pair**. Rerunning skips completed pairs.

One batched HF snapshot is sent after each model, at 30-minute checkpoints during long work, and on a catchable Stop/error. Retries handle throttling and Retry-After. No upload per image. If you press Stop, allow the final upload to finish. A forced kernel kill or loss of connectivity cannot guarantee the last upload; a fresh session restores the last verified snapshot. Keep the same inputs, code and RUN_TAG to resume. Do not run two sessions with the same run ID.
''')
    code("studio.run(ROWS, REGISTRY, RUNTIME, WORK, PUBLISHER)\n")
    md('''## 4 · Inspect the comparison cards
**Cyan mask:** tyre. **Amber mask:** tread. **Magenta points/lines:** HRNet proposals. **Cyan points/lines:** boundaries extracted from the SegFormer tread mask at the same three rows.

Missing or frame-clipped mask intersections remain missing; they are not replaced by plausible-looking points. A crossed HRNet proposal is flagged. The two classifiers can agree and still be wrong. The boundary gap measures model disagreement only; the notebook does not manufacture ground truth or physical angles.
''')
    code('''try:
    PANELS = studio.render(ROWS, PUBLISHER.out, REGISTRY)
    PUBLISHER.publish('gallery complete')
except BaseException:
    try: PUBLISHER.publish('gallery interrupted')
    except Exception as error: print('Upload unavailable:', type(error).__name__, 'Local outputs retained.')
    raise
display(DisplayImage(filename=str(PUBLISHER.out/'prediction_fingerprint.png')))
for panel in PANELS: display(DisplayImage(filename=str(panel)))
''')
    md('''## 5 · Download and keep the evidence
The ZIP includes the light-theme HTML gallery, PNG cards, numerical predictions, raw masks/points, input hashes, checkpoint provenance and environment. Open `gallery.html` **after extracting the ZIP**, so its image links work. Original full-resolution source files and model weights are not duplicated in the gallery. Inference times in NPZ files are diagnostic cold/warm timings, not a benchmark.
''')
    code('''import zipfile
DOWNLOAD = WORK/(RUN_ID+'-gallery.zip')
with zipfile.ZipFile(DOWNLOAD,'w',zipfile.ZIP_DEFLATED) as z:
    for p in PUBLISHER.out.rglob('*'):
        if p.is_file() and p.suffix!='.tmp': z.write(p,p.relative_to(PUBLISHER.out).as_posix())
display(FileLink(str(DOWNLOAD)))
print('Also available from Kaggle Output:', DOWNLOAD.name)
print('HF destination:', 'https://huggingface.co/datasets/'+studio.REPO+'/tree/main/'+PUBLISHER.prefix)
''')
    md('''### Troubleshooting
- **Wrong/missing dataset:** attach the expanded prepared Phase 2 Kaggle dataset, or set DATA_ROOT to its directory containing VERSION.json. A different release is rejected. For your own photos use CUSTOM_ROOT instead.
- **GPU memory:** the notebook loads one model at a time. Restart a fresh kernel to release allocations left by other work, then Run All; saved predictions resume.
- **Dependency/import error:** use a fresh Kaggle kernel and run setup before importing Torch model libraries elsewhere. Kaggle Torch/torchvision are preserved.
- **HF 401/403:** enable the HF_TOKEN secret and its repository write access. No token is printed or saved in outputs. On 429/network failures, allow retries; local results remain available.
- **Stop:** wait for the HF verification message. For a failed upload, rerun `PUBLISHER.publish('manual recovery')` while the files still exist. A forced kill cannot run cleanup.
- **Unexpected predictions:** retain them. Inspect cohort labels, original orientation and full probabilities; do not tune thresholds or choose prettier examples and call that an evaluation.

This notebook uses the five current production exports. Earlier Phase 1 model families and all three training seeds are a different comparison scope; no paper text is changed here.
''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')))
    for i,c in enumerate(cells):c['id']=f'visual-{i:02d}'
    path=ROOT/'notebooks/phase2_NB07_Visual_Model_Comparison.ipynb'
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
