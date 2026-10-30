"""Package the CPU-only completed-ATLAS review/publication; leave NB08 untouched."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path
import nbformat

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
pilot=ROOT/'tread_two/pilot'
inventory=json.loads((pilot/'FILES.json').read_text())
files={'frozen_source/'+name:(pilot/name).read_bytes() for name in list(inventory)+['FILES.json']}
for name,meta in inventory.items(): assert hashlib.sha256(files['frozen_source/'+name]).hexdigest()==meta['sha256']
files.update({'review/'+p.name:p.read_bytes() for p in (ROOT/'tread_two/review').glob('*.py')})
files['transport/nb08_transport_launcher.py']=(ROOT/'analysis/nb08_transport_launcher.py').read_bytes()
files['FIT_MANIFEST.json']=(ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit_MANIFEST.json').read_bytes()
payload={name:base64.b64encode(value).decode() for name,value in files.items()}
encoded=base64.b64encode(json.dumps(payload,separators=(',',':')).encode()).decode()
code='''import base64,hashlib,importlib.metadata,json,os,signal,subprocess,sys,shutil,time
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB09 on Kaggle CPU.'
WORK=Path('/kaggle/working/tread_two_nb09_review')
WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>=2*1024**3,'Keep at least 2 GiB free for evidence, output and immutable publication staging.'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
try: hub=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError: hub=None
if hub!='0.36.2': subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'SESSION.lock'),timeout=1):
    for name,value in payload.items():
        target=(WORK/name).resolve()
        assert target.is_relative_to(WORK.resolve()) and '..' not in Path(name).parts
        body=base64.b64decode(value)
        if target.exists(): assert target.read_bytes()==body,'Preserve this review; its existing source differs.'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(body)
OUT=WORK/'analysis_report'
CACHE=WORK/'cache'
common=['--out',str(OUT),'--source',str(WORK/'frozen_source'),'--launcher',str(WORK/'transport/nb08_transport_launcher.py'),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--work',str(WORK/'publication')]
def launch(script,args,token=None):
    environment=dict(os.environ)
    environment.pop('HF_TOKEN',None)
    if token is not None: environment['HF_TOKEN']=token
    with FileLock(str(WORK/'SESSION.lock'),timeout=1):
        child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'review'/script),*args],env=environment,start_new_session=True)
        environment.pop('HF_TOKEN',None)
        try:
            while True:
                try: status=child.wait(timeout=15);break
                except subprocess.TimeoutExpired: pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(120):
                    try: child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%6==0: print('Waiting for safe read cleanup or completed-review HF flush; preserve local files.',flush=True)
            raise
        assert status==0,'Review stage stopped; keep working files and rerun. Completed verified reads are reused.'
print('Review source ready. CPU only; no dataset attachment or training launch. No secret read yet.')
'''.replace('__PAYLOAD__',encoded)
cells=[]
def md(value): cells.append(nbformat.v4.new_markdown_cell(value))
def py(value): ast.parse(value);cells.append(nbformat.v4.new_code_cell(value))
md('''# Phase 3 · NB09 · TREAD-TWO / ATLAS completed-run review

**CPU only · no training or new inference · direct CSV + interactive evidence · additive HF publication.**

NB08 completed all 60 epochs / 2,940 updates. Both selected/final balanced
accuracies are 50%. The actual confusion is **62 low → mid, 41 high → high**.
This notebook audits the saved run and publishes new post-hoc tables, figures,
source/provenance and a complete interactive failure/probe explorer.

## Kaggle settings

1. Accelerator **None / CPU**, Internet **ON**. No T4 allocation is needed.
2. **No dataset attachment needed**: all inputs and source images come from
   the pinned completed NB08 evidence; no new model forward pass occurs.
3. Keep at least **2 GiB free**. Selected scientific input is about 150 MB;
   local evidence, derived output and publication staging stay well below 20 GB.
4. Enable **HF_TOKEN** in Kaggle Secrets with write access to
   **Shanmuk4622/tyre-wear-study**. It is read only by the final publication cell.
5. **Run All**. The tested Hub transport dependency is pinned; Torch/CUDA and
   other model packages are not changed.

Inputs are pinned to revision `82e7e1c5774f54c786571cdef7b3b60c1e96630f`.
Every consumed file is verified by size and SHA-256. Interrupted/partial
downloads are fetched again; verified completed objects are reused. Derived
tables are atomic deterministic recomputations, not another training trajectory.
No original fit/checkpoint/CSV namespace or workstation is overwritten.
''')
py(code)
md('''## 1 · Analyse retained evidence and prepare publication

Checks cover all 29,410 predictions, 23,520 training exposures, 2,940 updates,
60 selections, 824 probe pairs, all 1,030 endpoint/probe logit decompositions,
18 literal references, all 32 patch contributions and retained dense summaries.
The full 35.5-million-row source is not all downloaded/recounted; the exact
consumed subset and exclusions are recorded. All derived metrics remain
existing-cohort development evidence with no held-out mid targets.
''')
py("launch('review_atlas.py',['--cache',str(CACHE),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--out',str(OUT)])\nlaunch('publish_atlas_review.py',common+['--prepare-only'])")
md('''## 2 · Inspect direct tables, figures and interactive records

Original and intervention scores are paired by image and checkpoint. Green/red
patch evidence and literal reference contributions are saved-score decompositions,
not causal wear measurements. Grayscale modifies query and bank; brightness
modifies query only. The atlas thumbnail remains the base atlas, explicitly labeled.
''')
py("from IPython.display import display,FileLink,Image\nimport pandas as pd\ndisplay(pd.read_csv(OUT/'cohort_metrics.csv').query(\"stage in ['selected_eval','final_eval','diagnostics']\"))\ndisplay(Image(filename=str(OUT/'figures/probes_and_affinities.png')))\ndisplay(FileLink(str(OUT/'ATLAS_Failure_Explorer.html')))\nprint((WORK/'publication/REVIEW_DELIVERY.json').read_text())")
md('''## 3 · Publish the complete review directly to HF

The destination is a new `phase3/tread-two/atlas-v1/review/<contract>` namespace.
Original NB08 training remains untouched. CSVs, JSON, Python source, PNG and HTML
are uploaded directly, never as ZIP/NPZ/database bundles. The immutable publisher
checks remote Git/LFS metadata before selecting its final manifest and releases
ownership. An identical completed manifest is reused on rerun.

This finite analysis has one major completion flush, rather than periodic
training updates. Catchable Stop/error during publication attempts to finish or
retry the same complete local review, respecting server backoff. Forced kernel
death cannot publish unsent work; rerun the final cell with retained files.
An existing live writer is never stolen; allow its release/expiry.
''')
py("from kaggle_secrets import UserSecretsClient\ntoken=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    launch('publish_atlas_review.py',common,token=token)\nfinally:\n    token=None\nprint((WORK/'publication/REVIEW_PUBLICATION.json').read_text())\nprint('Preserve this executed notebook for the independent HF completion audit. No model was trained or installed.')")
for i,cell in enumerate(cells): cell.id=f'atlas-nb09-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
nbformat.validate(nb)
target=ROOT/'notebooks/phase3_NB09_TREAD_TWO_ATLAS_Completed_Review.ipynb'
if target.exists():
    prior=nbformat.read(target,as_version=4)
    assert not any(c.get('execution_count') or c.get('outputs') for c in prior.cells if c.cell_type=='code'),'Archive executed NB09 before regeneration'
nbformat.write(nb,target)
record=dict(status='packaged_not_executed_on_Kaggle',notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),notebook=str(target),bytes=target.stat().st_size,embedded_files=len(files),source_hashes={name:hashlib.sha256(value).hexdigest() for name,value in files.items()},code_cells=4,training_updates=0,HF_publication_performed_locally=False)
out=ROOT/'tread_two/review_validation';out.mkdir(exist_ok=True)
(out/'NOTEBOOK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'},indent=2))
