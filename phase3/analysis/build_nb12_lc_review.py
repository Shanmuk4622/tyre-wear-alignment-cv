"""Self-contained CPU review/publication of completed NB11; no new fit."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'tread_two/lc_pilot';review=ROOT/'tread_two/lc_review'
done=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
assert json.loads((done/'VERIFICATION.json').read_text())['status']=='passed'
assert json.loads((ROOT/'tread_two/lc_analysis_report/ANALYSIS.json').read_text())['status']=='passed'
catalog=json.loads((source/'FILES.json').read_text())
files={'frozen_source/'+n:(source/n).read_bytes() for n in list(catalog)+['FILES.json']}
for n,m in catalog.items():assert hashlib.sha256(files['frozen_source/'+n]).hexdigest()==m['sha256']
files.update({'review/'+p.relative_to(review).as_posix():p.read_bytes() for p in review.rglob('*') if p.is_file() and (p.suffix in ('.py','.csv','.json'))})
files['FIT_MANIFEST.json']=(done/'REMOTE_MANIFEST.json').read_bytes();files['LOCAL_FULL_AUDIT.json']=(done/'VERIFICATION.json').read_bytes()
encoded=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def py(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# Phase3 · NB12 · ATLAS-LC completed review and decision

**CPU · zero fitting / zero model forwards · all saved observations and paired losses.**

NB11 completed60epochs/2,940updates. The unchanged validation rule selected epoch7
(BA56.25%). Selected/final historical endpoint BA is50%:62low→mid,41high→high.
The LC representation improves probability/appearance stability but has not
solved low-tyre transfer. This review preserves failures and checks the science
before another model is considered. It does not install a model in a workstation.

## Settings

1. Choose **None / CPU**, Internet ON. **No tyre dataset attachment or GPU.**
2. Enable Kaggle Secret **HF_TOKEN**, write access to
   `Shanmuk4622/tyre-wear-study`, for the publication cell.
3. Keep at least **2GiB free**. This notebook downloads the used saved-evidence
   subset, not the entire5.28GB fit. Native model weights are not loaded.
4. Run All. Input objects are pinned at commit `3d5ae396…`, hash verified,
   cached and reused after restart. Torch/CUDA are not changed.

The full local publication audit verified every object and every raw CSV count;
its receipt is embedded as provenance. This CPU review independently checks its
consumed subset and creates additional direct CSV/figures/interactive evidence.
These two scopes are distinct. More rows are not new independent tyres.
''')
py('''import base64,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB12 on Kaggle CPU.'
WORK=Path('/kaggle/working/tread_two_nb12_lc_review');WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>=2*1024**3,'Keep2GiB free for cache,CSV/report and immutable publication staging.'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
try: version=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError: version=None
if version!='0.36.2':subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for n,v in payload.items():
        target=(WORK/n).resolve();assert target.is_relative_to(WORK.resolve()) and '..' not in Path(n).parts
        body=base64.b64decode(v)
        if target.exists():assert target.read_bytes()==body,'Preserve existing source/progress for compatibility review.'
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
CACHE=WORK/'cache';OUT=WORK/'report';SOURCE=WORK/'frozen_source'
def run(script,arguments,token=None):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    if token is not None:environment['HF_TOKEN']=token
    with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
        try:child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'review'/script),*arguments],env=environment,start_new_session=True)
        finally:environment.pop('HF_TOKEN',None)
        try:
            while True:
                try:result=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(180):
                    try:child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%6==0:print('Waiting for atomic cache/table boundary or active HF publication; keep files.',flush=True)
            raise
        assert result==0,'Preserve cache/report and errors. Rerun unchanged; no training is repeated.'
print('Pinned completed-fit evidence review ready. CPU only, no dataset attachment or new model fit.')
'''.replace('__PAYLOAD__',encoded))
md('''## 1 · Audit the saved science and build the full review

All53,342 predictions/roles,23,520paired objectives,752,640patch distances,
960,156source-SHA permissions,47,040appearance seeds/factors,2,940updates and
all gradient summaries are checked. Both-view online metrics keep separate and
combined denominators. All1,442 endpoint/probe records reconstruct signed logits
and support/opposition from affinities,valid-band pooling,coefficients andbias.

Selection remains the original validation-selected epoch7; test/probes do not
re-select weights. Fixed v1 metrics are pinned comparison evidence. Changed
representation/bank/query/sampler/objective prevent single-factor attribution.
Raw confidence risk curves are descriptive; no deployment threshold is fitted.

Completed source downloads are cached by hash; incomplete files are atomic.
Derived CSVs are atomic per table. A stopped finite review regenerates derived
tables from verified inputs on rerun; it has no optimizer/within-epoch state.
Its completed publication occurs at the next major cell. This is distinct from
NB11's completed-update training recovery and30minute snapshot protocol.
''')
py("run('review_lc.py',['--cache',str(CACHE),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--out',str(OUT)])")
md('''## 2 · Publish validated direct records in a new namespace

Publication is additive: `phase3/tread-two/atlas-lc-r2/review/<contract>`.
Original fit objects are read-only. All derived tables are direct CSV; metadata,
code,figures and HTML are direct files. Never ZIP,NPZ or a database.

The tested r2 publisher includes native five-minute HTTP budgets,separate hourly
commit guard,server reset/Retry-After,80-object batches,immutable hash verification,
CAS ownership and deduplication. Catchable Stop during publication allows the
current complete review flush to finish; server backoff may delay it. Forced
kernel death cannot send unsent work. Rerun unchanged after restart; completed
matching publications are verified and skipped. No alternate credential lookup.
''')
py("from kaggle_secrets import UserSecretsClient\ntoken=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    run('publish_lc_review.py',['--out',str(OUT),'--source',str(SOURCE),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--work',str(WORK)],token=token)\nfinally:\n    token=None")
md('''## 3 · Inspect and keep the executed notebook

The complete explorer includes six diagnostics,all images,source references and
signed patches; atlas thumbnails show the saved original atlas. New transformed
image previews are not fabricated. The model remains research-only:low endpoint
recall is zero,no held-out mid exists,and scores are mileage proxies,not physical
wear/depth/safety truth. No additional photo collection or workstation changes.

Keep this executed NB12 and the publication receipt for independent HF review.
Do not rerun NB11 or select another test-scoring checkpoint to improve the table.
''')
py("from IPython.display import display,FileLink,Image\nprint((WORK/'REVIEW_PUBLICATION.json').read_text())\nprint((OUT/'README.md').read_text())\ndisplay(Image(filename=str(OUT/'figures/paired_learning.png')))\ndisplay(FileLink(str(OUT/'ATLAS_Failure_Explorer.html')))\ndisplay(FileLink(str(OUT/'CSV_INDEX.md')))")
for i,c in enumerate(cells):c.id=f'atlas-lc-nb12-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
nbformat.validate(nb)
path=ROOT/'notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb'
if path.exists():assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve executed NB12'
nbformat.write(nb,path)
out=ROOT/'tread_two/lc_review_validation';out.mkdir(exist_ok=True)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,embedded_files=len(files),code_cells=4,sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},model_forwards=0,training_updates=0)
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
