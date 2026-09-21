"""Generate new notebooks only; preserve NB00 and every executed notebook."""
import ast,base64,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=['phase2_training_data.py','phase2_models.py','phase2_worker.py','phase2_hub.py','phase2_run.py','phase2_report.py','phase2_dataset_verify.py']

def main():
 setup='''import os, sys, subprocess, base64, signal
from pathlib import Path
WORK = Path('/kaggle/working/phase2_training')
WORK.mkdir(parents=True, exist_ok=True)
RUNTIME = WORK / 'runtime'
RUNTIME.mkdir(exist_ok=True)
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN:
    raise RuntimeError('Enable the HF_TOKEN Kaggle secret; never paste it into a cell.')
'''
 for name in RUNTIME:setup+=f"(RUNTIME / {name!r}).write_bytes(base64.b64decode({base64.b64encode((ROOT/name).read_bytes()).decode()!r}))\n"
 deps='''# Preserve Kaggle torch, torchvision, NumPy and Pillow. Install only small adapters.
DEPS = WORK / 'deps'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
    '--disable-pip-version-check', '--no-deps', '--upgrade', '--target', str(DEPS),
    'timm==1.0.15', 'ultralytics==8.4.20', 'ultralytics-thop==2.0.18',
    'transformers==4.51.3', 'tokenizers==0.21.4', 'huggingface_hub==0.36.0',
    'safetensors==0.5.3', 'filelock==3.19.1'], timeout=600)

def launch(arguments):
    env = os.environ.copy()
    env['HF_TOKEN'] = TOKEN
    env['PYTHONPATH'] = str(DEPS) + os.pathsep + str(RUNTIME)
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['OMP_NUM_THREADS'] = '2'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['YOLO_CONFIG_DIR'] = str(WORK/'yolo_settings')
    (WORK/'yolo_settings'/'Ultralytics').mkdir(parents=True,exist_ok=True)
    child = subprocess.Popen([sys.executable, '-B', '-u', *arguments], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    del env
    try:
        for line in child.stdout:
            print(line, end='', flush=True)
        code = child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            print('Stop requested. Wait for safe checkpoints and the HF upload; do not force-kill the kernel.')
            for line in child.stdout:
                print(line, end='', flush=True)
            child.wait()
        raise
    if code:
        raise RuntimeError('Run stopped with an error. Keep the outputs; local progress and any verified HF snapshots are retained.')
'''
 config='''# Defaults: ONE Kaggle session using both T4s. Do not launch duplicate copies.
NUM_WORKERS = 1
WORKER_ID = 0
# For N parallel Kaggle sessions: same NUM_WORKERS=N, unique IDs 0..N-1.
# Keep this layout fixed while sessions are active. Model jobs are statically sharded.
RUN_CONDITIONS = 'combined'   # Optional controlled comparison: 'combined,old_only'
DATA_ROOT = ''               # Auto-detect the attached Phase 2 dataset.
TAKE_OVER = False            # True ONLY after the previous owner notebook has stopped.
'''
 stages=[('phase2_NB01_GPU_Resume_Smoke.ipynb','GPU smoke and independent-process resume tests','smoke','mobilenetv4,resnet50,segformer,yolo26m,hrnet'),
 ('phase2_NB02_Classification.ipynb','MobileNetV4 Medium + ResNet50 training','train','mobilenetv4,resnet50'),
 ('phase2_NB03_SegFormer.ipynb','SegFormer-B0 training','train','segformer'),
 ('phase2_NB04_YOLO26M_Seg.ipynb','YOLO26 Medium segmentation training','train','yolo26m'),
 ('phase2_NB05_HRNet.ipynb','HRNet-W18 geometry training','train','hrnet'),
 ('phase2_NB06_Evaluate_Export.ipynb','Result audit and export registry','report','')]
 for filename,title,mode,models in stages:
  cells=[]
  def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
  def code(s):ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
  md(f'''# {title}

**{'CPU' if mode=='report' else 'GPU T4 x2'} · Internet ON · HF_TOKEN · attach [Tire Dataset Prepared phase2](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared-phase2).**

NB00 has already passed and its HF report was independently verified. Do not rerun it. This notebook is self-contained; no GitHub repository upload is required. Run cells top to bottom.

Source dataset: 570 images. Frozen training overlay: **386 train / 81 validation / 103 test**, keeping all mid tyres and video frames together in train. This avoids unknown-identity overlap; held-out scores describe old low/high tyres only, not unseen mid/video performance.

Each model/condition has three seeds, 60 epochs. The default combined study is 15 runs across NB02–NB05. Optional `combined,old_only` gives the original 30-run A/B study at a larger compute cost.
''')
  if mode=='smoke':md('Run NB01 first on dual T4. It checks actual pretrained models at full configured resolution and independent-process mid-epoch resume. It does not do 60-epoch training. Training notebooks also repeat their own model test automatically, so GPU/environment changes cannot silently bypass it.')
  elif mode=='train':md('This notebook verifies the source input, constructs the frozen overlay, downloads only pinned general-pretrained weights, checks real-model resume, and then trains. One independent process owns each GPU; data loading is synchronous and stateless for reproducible mid-epoch resume. Existing tyre-trained weights are not used for initialization.')
  else:md('Run after NB02–NB05 finish. It checks completed-run manifests, recomputes stored scores, reports per-tyre and per-seed results, and selects each model’s exported seed by validation score only. Partial studies are explicitly reported as partial. It does not change the workstation.')
  code(setup);code(deps)
  if mode!='report':
   code(config)
   code(f'''arguments = [str(RUNTIME/'phase2_run.py'), {mode!r}, '--work', str(WORK),
    '--root', DATA_ROOT, '--models', {models!r}, '--conditions', RUN_CONDITIONS,
    '--worker', str(WORKER_ID), '--workers', str(NUM_WORKERS)]
if TAKE_OVER: arguments.append('--take-over')
launch(arguments)
''')
  else:
   code("DATA_ROOT = ''\nRUN_PREFIX = ''  # Auto-detect one protocol, or paste RUN_PREFIX printed by training.\nRUN_CONDITIONS = 'combined'\nlaunch([str(RUNTIME/'phase2_report.py'), '--work', str(WORK/'report'), '--root', DATA_ROOT, '--prefix', RUN_PREFIX, '--conditions', RUN_CONDITIONS])\n")
  md('''## Save / stop / resume

Every successful optimizer update gets an atomic local checkpoint containing weights, optimizer, scheduler, AMP scaler, RNG, sampler cursor, history, validation records and best weights. Rerun the same notebook/configuration after a session restart to restore the last hash-verified HF checkpoint. Completed runs are skipped after their result manifests verify.

A single coordinator uploads immutable snapshots roughly every 30 minutes, on run completion and catchable Stop/error. It uses bounded retries, rate budgeting and server backoff. Wait for the printed HF commit after stopping. A forced kernel kill cannot guarantee the final push; progress since the last verified upload can be lost. The watchdog pauses around 8.5 hours.

Checkpoints use measured scratch space; images stay in read-only Kaggle input. The notebook does not assume 1 TB is available. It pauses on insufficient free space instead of deleting unsaved progress. Only one notebook with each worker ID may run at a time for the same jobs. Remote leases fence stale writers. If a crashed session left a lease, confirm the old session is stopped before setting `TAKE_OVER=True` for recovery.

## Label decisions and limitations

SegFormer ignores contradictory pixels. YOLO consumes overlapping instance masks directly, with a documented training-only tyre union with tread; its exclusive-class auxiliary semantic branch is disabled. Source labels are unchanged. HRNet reuses original human points and uses new polygon-derived points as **weak targets weighted 0.25**, never as human-reviewed evaluation truth.

The code was locally tested; the full pretrained T4 smoke results will be produced when you run it. No accuracy improvement, complete three-class test coverage, unseen-video performance or workstation promotion is claimed before those results exist. Keep and share the executed notebook outputs if a check fails.
''')
  for i,c in enumerate(cells):c['id']=f'phase2-{i:02d}'
  nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=mode!='report',datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])),cells=cells)
  path=ROOT/'notebooks'/filename
  if path.exists() and any(c.get('outputs') or c.get('execution_count') is not None for c in json.loads(path.read_text())['cells']):
   raw=path.read_bytes();archive=path.parent/'execution_archives';archive.mkdir(exist_ok=True);saved=archive/(path.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
   if not saved.exists():saved.write_bytes(raw)
  path.write_text(json.dumps(nb,indent=1),encoding='utf-8');print(filename)
if __name__=='__main__':main()
