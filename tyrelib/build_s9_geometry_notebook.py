"""Generate NB23 only; leave executed notebooks intact."""
import ast
import base64
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'notebooks/NB23_S9_Geometry_Baseline.ipynb'
if dest.exists() and any(c.get('outputs') for c in json.loads(dest.read_text())['cells']):
    raise RuntimeError('Preserve executed NB23 before rebuilding')
intro='''# NB23 — S9 saved-mask geometry baseline

**CPU / Accelerator None · ONE copy · Internet ON · enable HF_TOKEN · Run All.**
**Attach nothing.** No dataset, weights, GPU or new training is needed.

Uses the verified NB22 annotation revision `4110bdcc…` and existing SegFormer-B0
seed-1 predictions, pinned to HF commit `05bf37c0…`. It does not silently select
new labels. P06 is excluded from main scores while review is unresolved; its
mask/points remain visible as a diagnostic. You can run this now with 11/12 labels.

Checks native image identity and per-run validation membership. Existing fold
0/2 leakage caveats still apply: this is not independent new-tyre accuracy.
Extracts left/right mask extrema at the same three guide rows. Exact frame-edge
and empty-row predictions are rejected and reduce coverage. No threshold tuning.

Results: SUMMARY.json, all 72 POINTS.json records with exclusion reasons,
12 mask/point PNGs (cyan human, yellow predicted), source hashes and contract.
These are predicted-mask diagrams, not original-image overlays or manual masks.
Read error together with coverage. No automated conclusion that HRNet is useful
or unnecessary; PatchCore and full S9 remain uncompleted.

Small JSON sources are cached; rerunning deterministically rebuilds the same
content-addressed output, without repeating training or downloads in-session.
A fresh session refetches only small saved predictions. Uploads one batch at
completion, every 30 minutes if needed, and on catchable Stop/error. Forced kernel
kills cannot flush. Retry Run All after a failed upload. Do not run four workers.
Source and output caps: 20 MiB each, comfortably below Kaggle working storage.
Dependencies are isolated; existing Transformers/HF packages are not downgraded.
'''
setup='''import sys, subprocess, base64, importlib
from pathlib import Path
WORK=Path('/kaggle/working/s9_geometry'); WORK.mkdir(exist_ok=True)
# Run in a child with isolated dependency precedence; preserve Kaggle's base env.
DEPS=WORK/'deps'
if not (DEPS/'pycocotools').exists() or not (DEPS/'huggingface_hub').exists():
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--target',str(DEPS),
        'numpy<3','Pillow>=10,<13','requests>=2.32,<3','pycocotools==2.0.11','huggingface_hub>=1.3,<2'])
'''
for name in ['s9_pilot.py','s9_geometry.py']:
    raw=(ROOT/'tyrelib'/name).read_bytes()
    setup+=f"(WORK/{name!r}).write_bytes(base64.b64decode({base64.b64encode(raw).decode()!r}))\n"
run='''import os, signal, time
from kaggle_secrets import UserSecretsClient
token=UserSecretsClient().get_secret('HF_TOKEN')
if not token: raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets.')
env=os.environ.copy()
env['HF_TOKEN']=token
env['PYTHONPATH']=str(DEPS)+os.pathsep+str(WORK)
code="import os,s9_geometry as g; g.run('/kaggle/working/s9_geometry',token=os.environ['HF_TOKEN'])"
child=subprocess.Popen([sys.executable,'-u','-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
del token, env
try:
    for line in child.stdout:
        print(line,end='',flush=True)
    result=child.wait()
except KeyboardInterrupt:
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        print('Stop requested: allowing child to flush completed results to HF.')
        for line in child.stdout:
            print(line,end='',flush=True)
        child.wait()
    raise
if result: raise RuntimeError('Comparison/upload did not finish. Read the preceding error; local results are retained. Retry Run All.')
from IPython.display import FileLink, display
for folder in sorted((WORK/'results').iterdir()):
    display(FileLink(str(folder/'SUMMARY.json')))
    display(FileLink(str(folder/'POINTS.json')))
print('Done. Send this executed notebook for HF verification. No HRNet training started.')
'''
cells=[]
for i,(kind,src) in enumerate([('markdown',intro),('code',setup),('code',run)]):
    c=dict(cell_type=kind,id=f's9-geometry-{i}',metadata={},source=src.splitlines(True))
    if kind=='code': ast.parse(src);c.update(execution_count=None,outputs=[])
    cells.append(c)
dest.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
    metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python',version='3.11'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True))),indent=1),encoding='utf-8')
print(dest)
