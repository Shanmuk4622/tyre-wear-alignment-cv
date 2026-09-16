"""Build only NB21/NB22; never replace executed notebook evidence without archiving."""
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def cell(kind, text):
    value=dict(cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code':
        ast.parse(text); value.update(execution_count=None,outputs=[])
    return value

def build():
    source=(ROOT/'tyrelib/s9_pilot.py').read_bytes()
    template=(ROOT/'tyrelib/pilot_template.html').read_bytes()
    bootstrap=f'''import sys, subprocess, base64, json, shutil
from pathlib import Path
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub==0.36.2', 'Pillow>=10,<13'])
WORK=Path('/kaggle/working/s9_pilot_runtime'); WORK.mkdir(exist_ok=True)
(WORK/'s9_pilot.py').write_bytes(base64.b64decode({base64.b64encode(source).decode()!r}))
(WORK/'pilot_template.html').write_bytes(base64.b64decode({base64.b64encode(template).decode()!r}))
sys.path.insert(0,str(WORK))
import importlib, s9_pilot as pilot
pilot=importlib.reload(pilot)
from kaggle_secrets import UserSecretsClient
TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN: raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets; never paste it in a cell.')
from IPython.display import FileLink, display
OUT=Path('/kaggle/working/s9_pilot_outputs'); OUT.mkdir(exist_ok=True)
'''
    prep='''# Normally leave blank: find the one attached prepared dataset automatically.
DATA_ROOT = ''
ROOT_DATA=Path(DATA_ROOT) if DATA_ROOT else pilot.discover()
PACKAGE=None
try:
    PACKAGE=pilot.prepare(ROOT_DATA, OUT, WORK/'pilot_template.html')
    for name in ['s9_pilot.py','pilot_template.html']:
        shutil.copy2(WORK/name,PACKAGE/name)
    PREFIX=f's9/{pilot.VERSION}/{PACKAGE.name}'
    COMMIT=pilot.publish(PACKAGE,TOKEN,PREFIX)  # Major-action completion: one batched commit.
except KeyboardInterrupt:
    if PACKAGE is not None and (PACKAGE/'STATUS.json').exists():
        print('Catchable Stop: attempting to publish the completed small package.')
        pilot.publish(PACKAGE,TOKEN,f's9/{pilot.VERSION}/{PACKAGE.name}')
    raise
display(FileLink(str(PACKAGE/'PILOT_12_IMAGES.zip')))
print('If the link does not download: Output > s9_pilot_outputs > package folder > PILOT_12_IMAGES.zip')
print('Extract ZIP, open ANNOTATE.html in your browser, label only 12 images, save annotation JSON.')
print('Return only that JSON for review. No more labels or training until feedback.')
print('Public HF revision:', COMMIT)
'''
    review='''# Attach only your exported annotation JSON using Kaggle Add Input / upload files.
# Leave blank when exactly one annotations_*.json is attached; otherwise set its exact path.
ANNOTATION_JSON = ''
files=sorted(Path('/kaggle/input').glob('**/annotations_*.json')) if not ANNOTATION_JSON else [Path(ANNOTATION_JSON)]
if len(files)!=1: raise ValueError('Attach ONE exported annotations_*.json, or set ANNOTATION_JSON explicitly.')
if files[0].stat().st_size>1024*1024: raise ValueError('Expected the small annotation-only JSON, not images.')
value=json.loads(files[0].read_text())
package_id=value.get('package_id','')
if len(package_id)!=64 or any(c not in '0123456789abcdef' for c in package_id): raise ValueError('Invalid package ID')
from huggingface_hub import HfApi,hf_hub_download
REV=HfApi(token=TOKEN).repo_info(pilot.REPO,repo_type='dataset').sha
PREFIX=f's9/{pilot.VERSION}/{package_id}'
path=hf_hub_download(pilot.REPO, f'{PREFIX}/MANIFEST.json', repo_type='dataset', revision=REV, token=TOKEN)
manifest=json.loads(Path(path).read_text())
if manifest.get('package_id')!=package_id: raise ValueError('Remote package ID mismatch')
identity={k:v for k,v in manifest.items() if k!='package_id'}
if pilot.digest(pilot.canonical(identity))!=package_id: raise ValueError('Remote manifest content hash mismatch')
package=WORK/package_id; package.mkdir(exist_ok=True)
shutil.copy2(path,package/'MANIFEST.json')
RESULT=None
try:
    RESULT=pilot.review(files[0],package,OUT)
    pilot.write_json(RESULT/'STATUS.json',dict(status='mechanical_review_only',source_revision=REV,
        package_id=package_id,human_review_required=True,training_approved=False,full_s9_complete=False))
    COMMIT=pilot.publish(RESULT,TOKEN,f'{PREFIX}/reviews/{RESULT.name}')
except KeyboardInterrupt:
    if RESULT is not None:
        pilot.publish(RESULT,TOKEN,f'{PREFIX}/reviews/{RESULT.name}')
    raise
display(FileLink(str(RESULT/'REVIEW.json')))
print('Send your annotation JSON / this HF revision to the assistant for visual review:',COMMIT)
print('Mechanical validation does NOT approve labels, healthy references, training or full S9 completion.')
'''
    titles=[('NB21_S9_Annotation_Pilot.ipynb', '''# NB21 — S9 small annotation pilot (PREPARE)

**CPU · ONE copy · Internet ON · HF_TOKEN enabled · attach Tire Dataset Prepared · Run All.**
No GPU, weights or extra dataset download. Uses only 12 existing original JPEGs, one per capture session.
Creates a lossless offline annotation page, three schematic worked examples and an identity manifest.
The downloadable ZIP is capped at20MiB: it stops rather than silently reducing necessary image detail.

After Run All: download `PILOT_12_IMAGES.zip`, extract, open `ANNOTATE.html`, and follow its guide.
Save the annotation-only JSON every few images. Return that JSON to the assistant, or run NB22 on CPU.
**Stop after the 12-image pilot. Do not label more or train HRNet/PatchCore until human review.**

Six proposed 2-D tread-boundary points test label visibility/repeatability; they are not wheel angles.
Visual surface observations are NOT certified health labels. Independent records may be unknown.
The native prototype and its calibrated alignment workflow are untouched. Existing models remain unchanged.

HF persistence: one batched commit at completion, retry/backoff for transient errors, catchable Stop flush
when a complete package exists. No claim/heartbeat commits or training epochs. This short CPU action has
no scheduled long training loop; rerun safely if interrupted before completion. Forced kills cannot flush.
No labels exist until you work in the offline page; the notebook cannot save unsent browser edits.
''',prep),('NB22_S9_Pilot_Review.ipynb','''# NB22 — S9 annotation-only intake (REVIEW)

**CPU · ONE copy · Internet ON · HF_TOKEN enabled · attach ONLY your exported JSON · Run All.**
Run NB21 first so its manifest is public. No original images, dataset attachment or model checkpoint needed.
The file should be named `annotations_<package>.json` (browser suffixes are fine).

Checks package/image identity, six point names, visibility, native coordinates, left/right ordering and
completion. Partial work is preserved as partial. Uploads only labels and small review/status JSON files.
Then send the JSON or HF revision to the assistant for actual visual review. **No training and no next batch
is authorised by a mechanical pass.** “No visible issue” is never converted to “verified healthy”.

One batched major-completion upload; catchable Stop attempts publication of already validated local output.
Rate-limit backoff still applies. Rerun the last cell after network failure; immutable annotation hashes
keep separate export generations. No prior scientific results are overwritten.
''',review)]
    for name,title,run in titles:
        dest=ROOT/'notebooks'/name
        if dest.exists():
            raw=dest.read_bytes();old=json.loads(raw)
            if any(c.get('outputs') for c in old['cells']):
                archive=dest.parent/'execution_archives';archive.mkdir(exist_ok=True)
                saved=archive/(dest.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
                if not saved.exists(): saved.write_bytes(raw)
        cells=[cell('markdown',title),cell('code',bootstrap),cell('code',run)]
        for i,c in enumerate(cells): c['id']=f's9-pilot-{i}'
        nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                    language_info=dict(name='python',version='3.11'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True)),
                nbformat=4,nbformat_minor=5)
        dest.write_text(json.dumps(nb,indent=1),encoding='utf-8')
        print(name)

if __name__=='__main__': build()
