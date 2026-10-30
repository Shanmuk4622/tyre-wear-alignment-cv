"""Package exact isolated source and literal bank as a self-contained Kaggle ipynb."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

import nbformat

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/pilot'
catalog = json.loads((PILOT / 'FILES.json').read_text(encoding='utf-8'))
files = {name: (PILOT / name).read_bytes() for name in catalog}
files['FILES.json'] = (PILOT / 'FILES.json').read_bytes()
for name, item in catalog.items():
    assert hashlib.sha256(files[name]).hexdigest() == item['sha256'] and len(files[name]) == item['bytes']
payload = {name: base64.b64encode(value).decode() for name, value in files.items()}
encoded = base64.b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode()
transport = (ROOT / 'analysis/nb08_transport_launcher.py').read_bytes()
transport_encoded = base64.b64encode(transport).decode()
transport_sha = hashlib.sha256(transport).hexdigest()
inventory_sha = hashlib.sha256(files['FILES.json']).hexdigest()

cells = []
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text):
    ast.parse(text)
    cells.append(nbformat.v4.new_code_cell(text))

md('''# Phase 3 · NB08 · TREAD-TWO / ATLAS

**One new classifier fit · existing data · full direct CSV evidence · completed-update recovery.**

The declared TREAD-ONE seed-3/epoch-20 dense anchor stays frozen. A custom local
encoder compares 32 atlas patches with 18 literal training-source reference cases.
There are **81,274 new trainable parameters** and 1,453,433 total parameters.
No workstation is changed. Mileage classes remain proxies, not depth/safety labels.

Before Run All:

1. Select **GPU T4 ×2** and enable **Internet**.
2. Attach the **expanded** `shanmuk4622/tire-dataset-prepared-phase2` input directly
   from Kaggle. Do not download/copy another dataset or attach a ZIP-only version.
3. Enable Kaggle Secret **HF_TOKEN**, with write access to
   `Shanmuk4622/tyre-wear-study`. No alternative credential source is used.
4. Start with at least **12 GiB free**. The output guard is 16 GB within the
   user's 20 GB output budget; actual mounts/free space are measured. Larger
   scratch is not assumed. Scientific tables are not dropped to fit storage.
5. Choose **Run All**. The live recovery/space gate must pass before the measured
   cell starts. Keep this executed notebook for the completion audit.

**GPU ownership:** GPU 0 trains the one candidate; GPU 1 evaluates an immutable
model copy at barriers. This is not DDP or two independent measured fits.
FP32 is deliberate; no Torch upgrade or AMP compatibility repair is needed.

**30 October repair:** the model/runtime/bank/recipe and run namespace are
unchanged. An audited external launcher corrects the accidental hourly HTTP
request cap: API/resolver requests use five-minute windows, commit attempts
keep a separate hourly guard, and server delay headers remain authoritative.
Already completed recovery gates are reused; native/CSV progress is preserved.
''')

bootstrap = '''import base64, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path
assert Path('/kaggle/working').exists(), 'Run this notebook on Kaggle.'
WORK = Path('/kaggle/working/tread_two_atlas')
SOURCE = WORK / 'source'
WORK.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(exist_ok=True)
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    hub_version = importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError:
    hub_version = None
if hub_version != '0.36.2':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', 'huggingface_hub==0.36.2'])
payload = json.loads(base64.b64decode('__PAYLOAD__'))
for name, value in payload.items():
    target = (SOURCE / name).resolve()
    assert target.is_relative_to(SOURCE.resolve()) and '..' not in Path(name).parts
    body = base64.b64decode(value)
    if target.exists():
        assert target.read_bytes() == body, 'Existing source differs. Preserve the run and request a compatibility review.'
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
catalog = json.loads((SOURCE / 'FILES.json').read_text())
assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == item['sha256'] for name,item in catalog.items())
print('Exact isolated source and reference bank ready:', len(catalog), 'files. No model has been fitted by this cell.')
REPAIR_PAYLOAD = '__TRANSPORT__'
repair_body = base64.b64decode(REPAIR_PAYLOAD)
assert hashlib.sha256(repair_body).hexdigest() == '__TRANSPORT_SHA__'
TRANSPORT_LAUNCHER = WORK / 'nb08_transport_launcher.py'
if TRANSPORT_LAUNCHER.exists():
    assert TRANSPORT_LAUNCHER.read_bytes() == repair_body, 'A different transport overlay exists; preserve it for compatibility review.'
else:
    TRANSPORT_LAUNCHER.write_bytes(repair_body)
SOURCE_INVENTORY_SHA = '__INVENTORY_SHA__'
print('Transport repair ready. All 24 frozen source/asset files remain unchanged.')
'''.replace('__PAYLOAD__', encoded)
bootstrap = bootstrap.replace('__TRANSPORT__', transport_encoded).replace('__TRANSPORT_SHA__', transport_sha).replace('__INVENTORY_SHA__', inventory_sha)
code(bootstrap)
md('''## 2 · Safe launcher

The publisher receives the Kaggle secret in memory. Training workers have no
token. Stop requests tell the owned process to finish its current atomic unit
and attempt a flush; server backoff may delay that flush. Do not close the
kernel while it reports that publication is pending.''')
code('''import signal, time
from kaggle_secrets import UserSecretsClient

def launch(stage):
    token = UserSecretsClient().get_secret('HF_TOKEN')
    environment = dict(os.environ)
    environment['HF_TOKEN'] = token
    child = subprocess.Popen([sys.executable, '-B', '-u', str(TRANSPORT_LAUNCHER), stage, '--source', str(SOURCE),
                              '--work', str(WORK), '--inventory-sha', SOURCE_INVENTORY_SHA],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)
    environment.pop('HF_TOKEN', None)
    token = None
    try:
        for line in child.stdout:
            print(line, end='', flush=True)
        status = child.wait()
        if status:
            raise RuntimeError('ATLAS stopped safely. Inspect NB08_TRANSPORT_REPAIR/EVENTS.jsonl and retained worker logs; keep this executed notebook. Saved progress is preserved.')
    except KeyboardInterrupt:
        print('Stop requested. Waiting for the current unit and immediate HF flush attempt...', flush=True)
        child.send_signal(signal.SIGINT)
        started = time.monotonic()
        while child.poll() is None and time.monotonic()-started < 900:
            time.sleep(1)
            if int(time.monotonic()-started) % 30 == 0:
                print('Still waiting for safe shutdown/publication; local state is retained.', flush=True)
        if child.poll() is None:
            print('Publication is still pending. Keep the session/files; forced kernel death cannot save unsent work.', flush=True)
        raise
    finally:
        if child.poll() is not None:
            child.stdout.close()

print('Launcher ready. Next: verify data and fetch the exact native anchor.')
''')
md('''## 3 · Verify existing data and prepare the frozen contract

This hashes the attached release, verifies historical roles and bank provenance,
and downloads only the exact 5.3 MiB native anchor from its pinned HF revision.
It records the real environment and storage; no measured optimizer update occurs.''')
code("launch('prepare')")
md('''## 4 · Automatic live recovery and storage gate

Short **engineering fixtures**, using training queries and two validation
examples only, test the actual batch of eight. The gate compares uninterrupted,
catchable-pause/pinned-HF-restore and abrupt-termination recovery, including
model/optimizer/scheduler/scaler/RNG, CSV rows and native evidence maps.

The publisher stages direct files, verifies Git/LFS hashes, restores the pinned
update-1 source in a fresh process and records a measured storage extrapolation.
These four-update fixtures are not classifier results. A verified completed gate
is reused on restart. An incomplete gate cannot authorize the long run.''')
code("launch('preflight')")
md('''## 5 · One measured seed-1 classifier fit

60 epochs ×49 updates ×8 balanced query draws; 378 query images plus eight bank
photos as support. All three proxy classes are trained; historical validation
and test have only low/high targets. The inspected old endpoint is development
evidence. Selection is validation observed-class BA, then NLL, then earliest epoch.

Every completed update commits native state and CSV byte cursors. Normal HF
publication is about every 30 minutes and at run completion/Stop/error.
Rerunning this cell restores/continues the same contract; completed work is
verified and skipped. Selected/final test endpoints and four declared probes
are resumable and retained. No test-based checkpoint choice occurs.''')
code("launch('train')")
md('''## 6 · Inspect the retained report

The model exposes coverage, raw scores, literal reference cases and signed
evidence. Unsupported atlases remain in the metric denominators. Point errors
separate human and weak targets; incompatible guide rows are marked unavailable.
The explorer/curves are derived from saved evidence and require no retraining.''')
code('''import pandas as pd
from IPython.display import display, FileLink, Image
receipt = json.loads((WORK/'LAST_REMOTE.json').read_text())
latest = json.loads((WORK/'LATEST_SNAPSHOT.json').read_text())
SNAPSHOT = Path(latest['path'])
print(json.dumps(receipt, indent=2))
tables = [pd.read_csv(p) for p in sorted((SNAPSHOT/'csv').glob('epoch_*/metrics.csv'))]
metrics = pd.concat(tables, ignore_index=True)
display(metrics[(metrics['level']=='cohort') & metrics['stage'].isin(['selected_eval','final_eval'])])
display(Image(filename=str(SNAPSHOT/'report/training_curves.png')))
display(FileLink(str(SNAPSHOT/'report/ATLAS_Evidence_Explorer.html')))
display(FileLink(str(SNAPSHOT/'SCHEMA.json')))
print('Preserve this executed notebook. A completion audit must independently verify the pinned HF revision and scientific records.')
''')
md('''## Restart, storage and troubleshooting

| Situation | What to do |
|---|---|
| Normal restart / lost session | Enable the same input, T4×2 and secret, then Run All unchanged. Remote state is checked before resume/skip. |
| Catchable Stop | Wait for the flush result. The next run starts at the next completed update; an incomplete update is replayed. |
| Forced kernel/process death | Only the last verified remote snapshot is guaranteed recoverable; unsent local work may be lost. |
| Another active writer | The launcher waits up to 90 minutes for release/expiry. Do not delete ownership or open the same fit in two sessions. |
| Local HTTP budget / 429 | The repaired launcher waits for five-minute HTTP windows and actual server resets without converting a long quota wait to a fake network timeout. Commit attempts have a separate 96/hour guard. |
| Disk guard | Keep current files; flush/inspect space. Do not delete CSV/history or change the contract to force continuation. |
| Source/environment mismatch | Preserve files and this notebook; request a compatibility review. Do not delete the HF run or silently start it over. |
| Worker failure | Inspect `/kaggle/working/tread_two_atlas/run/console.log`, `NUMERICAL_EVENTS.jsonl`, `LIFECYCLE.jsonl` and `LAST_REMOTE.json`. |

HF layout: `phase3/tread-two/atlas-v1/engineering/<contract>` and
`phase3/tread-two/atlas-v1/fit/<contract>`. CSVs remain direct `.csv`; native
recovery/maps use `.pt`/`.npy`; manifests/source are direct JSON/Python. No ZIP,
NPZ or database is uploaded. Historical study/model/workstation files are untouched.

The 18 cases come from only 1/3/4 original training tyres. Paint, glare,
identity and view shortcuts remain possible. Exact evidence accounting is
not causal reasoning, calibration or proof of physical wear measurement.
''')
notebook = nbformat.v4.new_notebook(cells=cells, metadata=dict(
    kernelspec=dict(name='python3', display_name='Python 3', language='python'),
    kaggle=dict(isInternetEnabled=True, isGpuEnabled=True, datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
nbformat.validate(notebook)
target = ROOT / 'notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb'
archived_execution = None
if target.exists() and any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(target,as_version=4).cells if c.cell_type=='code'):
    original = target.read_bytes()
    identity = hashlib.sha256(original).hexdigest()
    archive = ROOT / 'tread_two/pilot_validation/nb08_repair' / identity[:12] / 'executed.ipynb'
    assert archive.exists() and archive.read_bytes() == original, 'Preserve an executed notebook before regeneration'
    archived_execution = identity
nbformat.write(notebook, target)
output = ROOT / 'tread_two/pilot_validation'
output.mkdir(exist_ok=True)
record = dict(status='notebook_schema_source_and_syntax_valid', notebook=str(target), notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size, code_cells=sum(c.cell_type=='code' for c in cells), embedded_source_files=len(files),
    sources={name:hashlib.sha256(value).hexdigest() for name,value in files.items()},
    transport_launcher_sha256=transport_sha, repair_revision='nb08-transport-20261030-r1',
    frozen_inventory_sha256=inventory_sha, archived_execution_sha256=archived_execution,
    measured_fit_started=False, Kaggle_execution_tested=False, live_HF_recovery_tested_locally=False)
(output / 'NOTEBOOK_DELIVERY.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in record.items() if key!='sources'},indent=2))
