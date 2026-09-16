"""Build only NB30–32; never overwrite executed notebooks."""
import ast
import base64
import json
import sys
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def md(text):return dict(cell_type='markdown',metadata={},source=text.splitlines(True))
def code(text):
    ast.parse(text)
    return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=text.splitlines(True))

setup="""import os, sys, subprocess, base64, signal
from pathlib import Path
WORK=Path('/kaggle/working/segformer_matched');WORK.mkdir(exist_ok=True)
from kaggle_secrets import UserSecretsClient
TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
assert TOKEN, 'Enable HF_TOKEN in Kaggle Secrets before Run All'
"""
for name in ['segformer_matched.py','hrnet_runtime.py','hrnet_protocol.py','hrnet_amp_repair.py','segformer_resume_repair.py']:
    setup+=f"(WORK/{name!r}).write_bytes(base64.b64decode({base64.b64encode((ROOT/'tyrelib'/name).read_bytes()).decode()!r}))\n"
setup+="""# Source is also visible as ordinary .py files in the Kaggle working folder.
print('Runtime ready. This writes only the new SegFormer namespace, never HRNet runs.')
"""
deps="""# Keep Kaggle's torch, torchvision and NumPy. Isolate only these small runtime packages.
DEPS=WORK/'deps'
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps','--target',str(DEPS),
    'transformers==4.51.3','tokenizers==0.21.4','huggingface_hub==0.36.0','safetensors==0.5.3'])
"""
launch="""def launch(mode):
    env=os.environ.copy();env['HF_TOKEN']=TOKEN
    env['PYTHONPATH']=str(DEPS)+os.pathsep+str(WORK)
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['OMP_NUM_THREADS']='2'
    runtime='segformer_resume_repair.py' if mode=='train' else 'segformer_matched.py'
    child=subprocess.Popen([sys.executable,'-u',str(WORK/runtime),mode,
        '--work',str(WORK),'--root',DATA_ROOT],env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    del env
    try:
        for line in child.stdout:print(line,end='',flush=True)
        result=child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            print('Waiting for the completed-step checkpoint and HF push. Do not force-kill the kernel.')
            for line in child.stdout:print(line,end='',flush=True)
            child.wait()
        raise
    if result:raise RuntimeError('Stage stopped with an error. Keep outputs/local files; do not delete checkpoints.')
"""
common="""## What is matched—and what is not

The frozen HRNet allocation is reused exactly: 72 training / 24 validation / 24 test images,
from 8 / 2 / 2 user-confirmed different tyres. Seeds 1–3, 60 epochs, batch 2, input
512×384, AdamW 0.0001, cosine schedule, no augmentation, and final-epoch evaluation.
SegFormer-B0 starts from pinned ImageNet MiT-B0 weights, NOT an old S5 checkpoint.
It learns from the **existing dense masks on the same 72 training images**; HRNet
learned from six clicked points. Same evaluation annotations, different training supervision.
This is an exploratory system comparison, not an equal-supervision architecture claim.

No new annotations and no HRNet rerun. Only two test tyres; no significance or full-S9 claim.
"""
specs=[
 ('NB30_SegFormer_Matched_Preflight.ipynb','CPU preflight','preflight',
  'Use CPU, Internet ON, enable HF_TOKEN. Attach the same prepared dataset used by NB14, including FINAL images and annotations/clean/masks. Run All. This hashes the 120 images and 72 training masks; no weights are downloaded. After PREFLIGHT PASSED, run NB31.'),
 ('NB31_SegFormer_Matched_Training.ipynb','GPU smoke + resumable training','train',
  'REPAIRED resume check: use this new notebook in a fresh T4 session. Your completed NB30 remains valid; do not rerun it. Internet ON, HF_TOKEN enabled, same prepared dataset as NB30. Run All in ONE session only—do not launch four workers. The deterministic resize adapter keeps the strict four-step resume test, then trains all three seeds. Dual T4 is supported; one GPU is deliberately used. If it pauses as resumable, run NB31 again. Run NB32 only after all requested seeds complete.'),
 ('NB32_SegFormer_HRNet_Comparison.ipynb','CPU comparison report','compare',
  'Use CPU, Internet ON, HF_TOKEN enabled. No dataset attachment needed. Run All after NB31 completes all three seeds. Reads small frozen HRNet and fresh SegFormer metric files, checks identical targets, and publishes the paired report and chart to HF.')]
for name,title,mode,instructions in specs:
    if '--only' in sys.argv and name!=sys.argv[sys.argv.index('--only')+1]:continue
    path=ROOT/'notebooks'/name
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells']):
            archive=ROOT/'notebooks/execution_archives';archive.mkdir(exist_ok=True)
            saved=archive/(path.stem+'_'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]+'.ipynb')
            if not saved.exists():saved.write_bytes(path.read_bytes())
            print('Preserved executed notebook:',saved.name)
    cells=[md('# '+name.split('_')[0]+' — '+title+'\n\n'+instructions),md(common),
        md('## 1. Credentials and isolated runtime\nThe embedded source is versioned with the experiment contract.'),code(setup),
        md('## 2. Dependencies and dataset\nLeave DATA_ROOT empty for automatic detection of one prepared dataset.'),
        code(deps+"\nDATA_ROOT = ''\n"),md('## 3. Run this stage'),code(launch+f"\nlaunch({mode!r})\n")]
    if mode=='train':cells.append(md('## Saving and resuming\nEach completed optimizer step is saved atomically locally with optimizer, scheduler, AMP scaler, sampler cursor and RNG state. HF receives a consistent snapshot every 30 minutes, after a seed completes, and on a catchable Stop/error. A graceful 9-hour guard pauses the run. Forced kernel death or lost Internet can lose progress since the last successful HF push; nothing can guarantee upload after a hard kill. Keep the notebook running while Stop finishes uploading. Only this experiment’s small model states are retained; no dataset copy or other-model checkpoints.'))
    if mode=='compare':cells.append(code("from IPython.display import display, Image\nfor chart in (WORK/'comparison').glob('*/comparison.png'): display(Image(filename=str(chart)))\nprint('Send this executed notebook for review before deciding which geometry model to integrate.')\n"))
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11'),kaggle=dict(isInternetEnabled=True)),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):c['id']=f'cell-{i:02d}'
    # Generated output of the notebook builder, not handwritten source edits.
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(name)
