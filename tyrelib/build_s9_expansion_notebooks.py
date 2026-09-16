"""Build NB24/NB25 without touching executed notebooks."""
import ast,base64,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
setup='''import sys, subprocess, importlib.util, base64, json, shutil
from pathlib import Path
if importlib.util.find_spec('huggingface_hub') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','huggingface_hub>=1.3,<2'])
WORK=Path('/kaggle/working/s9_expansion'); WORK.mkdir(exist_ok=True)
'''
for name in ['s9_pilot.py','s9_expansion.py','pilot_template.html']:
    setup+=f"(WORK/{name!r}).write_bytes(base64.b64decode({base64.b64encode((ROOT/'tyrelib'/name).read_bytes()).decode()!r}))\n"
setup+='''sys.path.insert(0,str(WORK))
import importlib, s9_expansion as e, s9_pilot as p
e=importlib.reload(e)
from kaggle_secrets import UserSecretsClient
TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN: raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets.')
from huggingface_hub import hf_hub_download,HfApi
from IPython.display import display,FileLink
OUT=WORK/'outputs';OUT.mkdir(exist_ok=True)
'''
prep='''DATA_ROOT = ''  # Leave blank with ONE prepared dataset attached. All 120 images are included.
root=Path(DATA_ROOT) if DATA_ROOT else p.discover()
pilot_file=hf_hub_download(p.REPO,'s9/s9-input-pilot-r1/87933cacc9cc06234dc150fbba749e469d81e57850666d51a2f455a746d3fd96/MANIFEST.json',
    repo_type='dataset',revision='05bf37c0f2067b0119ef057a2442d7759cf9ac51',token=TOKEN)
pilot=json.loads(Path(pilot_file).read_text())
PACKAGE=None
try:
    PACKAGE=e.prepare(root,pilot,WORK/'pilot_template.html',OUT)
    shutil.copy2(WORK/'s9_expansion.py',PACKAGE/'s9_expansion.py')
    COMMIT=e.publish(PACKAGE,TOKEN,f's9/{e.VERSION}/packages/{PACKAGE.name}')
except KeyboardInterrupt:
    if PACKAGE is not None:e.publish(PACKAGE,TOKEN,f's9/{e.VERSION}/packages/{PACKAGE.name}')
    raise
display(FileLink(str(PACKAGE/'ALL_120_IMAGES.zip')))
print('Download ALL_120_IMAGES.zip; extract EVERYTHING, keep images beside ANNOTATE.html, open ANNOTATE.html.')
print('Annotate all 120 at your pace. Save JSON often; Load saved JSON to resume. No batch switching.')
print('Also tell the assistant which capture sessions share the same physical tyre, if known.')
'''
review='''ANNOTATION_JSON = ''  # Normally attach ONE annotations_*.json and leave blank.
files=[Path(ANNOTATION_JSON)] if ANNOTATION_JSON else sorted(Path('/kaggle/input').glob('**/annotations_*.json'))
if len(files)!=1:raise ValueError('Attach one annotation JSON, or set its exact path.')
if files[0].stat().st_size>1024**2:raise ValueError('Attach only the small annotation JSON.')
value=json.loads(files[0].read_text());package=value.get('package_id','')
if len(package)!=64 or any(c not in '0123456789abcdef' for c in package):raise ValueError('Invalid package ID')
rev=HfApi(token=TOKEN).repo_info(p.REPO,repo_type='dataset').sha
mf=hf_hub_download(p.REPO,f's9/{e.VERSION}/packages/{package}/MANIFEST.json',repo_type='dataset',revision=rev,token=TOKEN)
manifest=json.loads(Path(mf).read_text());RESULT=None
try:
    RESULT=e.intake(files[0],manifest,OUT)
    COMMIT=e.publish(RESULT,TOKEN,f's9/{e.VERSION}/packages/{package}/reviews/{RESULT.name}')
except KeyboardInterrupt:
    if RESULT is not None:e.publish(RESULT,TOKEN,f's9/{e.VERSION}/packages/{package}/reviews/{RESULT.name}')
    raise
display(FileLink(str(RESULT/'REVIEW.json')))
print('Send executed NB25 for review. A mechanical pass is not approval to train.')
'''
for name,title,body in [
('NB24_S9_Geometry_Annotation_Batches.ipynb', '''# NB24 — ALL 120 geometry images in ONE package

**CPU, ONE copy, Internet ON, HF_TOKEN enabled, existing Tire Dataset Prepared attached. Run All.**
No BATCH setting. Download ALL_120_IMAGES.zip, extract everything, open ANNOTATE.html.
Mark six tread-transition points or explicit uncertain/occluded/outside-frame states.
Visible issue needs a short note. Save annotation JSON, then run NB25 or return JSON.
**All 120 new images are available together. No intermediate review pause.**
The old 12 are excluded. Save one JSON across sittings; load it to resume.
Keep the images folder beside ANNOTATE.html. ZIP cap is 160 MiB, not 20 MiB:
all original-resolution JPEGs are preserved; no hidden compression/downsampling.

No model predictions/old labels are shown. Native JPEG detail is preserved; no new
photos, masks or boxes requested. Images are sampled across available sessions,
but sessions are not verified unique tyres. SESSION_IDENTITY.json is a blank
optional ledger: tell us in chat which sessions share a tyre, or leave unknown.
Train/test allocation is NOT locked until identity review; no training starts.

One HF commit for the completed package, backoff and catchable Stop flush of
completed package. No heartbeat/claim writes. These short preparation jobs have
no training loop. Forced kills cannot flush; rerun safely for the same package.
No checkpoints or dataset downloads: data is read directly from Kaggle input.
''',prep),
('NB25_S9_Geometry_Annotation_Intake.ipynb','''# NB25 — save and check the ONE 120-image annotation export

**CPU, ONE copy, Internet ON, HF_TOKEN enabled. Attach ONE exported annotation JSON. Run All.**
Run NB24 first for that package. No images/dataset/weights required here.
Validates identity, point states, coordinates, ordering and required descriptions.
Preserves partial submissions as partial; displays X/120 complete. Fix only missing
items, not the entire assignment. Uploads annotation JSON under an immutable hash.
Return when all 120 are reviewed, or save partial progress whenever you choose.

No HRNet/PatchCore training or health certification. Input metadata and review
must be approved and identity-disjoint splits frozen before training notebooks.
One batched commit at completion; catchable Stop flush and server backoff.
''',review)]:
    dest=ROOT/'notebooks'/name
    if dest.exists() and any(c.get('outputs') for c in json.loads(dest.read_text())['cells']):
        import hashlib
        raw=dest.read_bytes();archive=dest.parent/'execution_archives';archive.mkdir(exist_ok=True)
        saved=archive/(dest.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
        if not saved.exists():saved.write_bytes(raw)
    cells=[]
    for i,(kind,text) in enumerate([('markdown',title),('code',setup),('code',body)]):
        c=dict(cell_type=kind,id=f's9-expansion-{i}',metadata={},source=text.splitlines(True))
        if kind=='code':ast.parse(text);c.update(execution_count=None,outputs=[])
        cells.append(c)
    dest.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True))),indent=1),encoding='utf-8')
    print(dest.name)
