"""Generate two compact, self-contained Kaggle notebooks from readable runtime."""
import ast
import base64
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
assert Path(sys.prefix).name == 'cv_conda', 'Activate cv_conda before building'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md(source):
    return dict(cell_type='markdown', metadata={}, source=source)


def code(source):
    ast.parse(source)
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source)


def build():
    output = ROOT/'notebooks'
    output.mkdir(exist_ok=True)
    files = {p.name: p.read_bytes() for p in sorted((ROOT/'runtime').glob('*.py'))}
    # Exact reused source copies must match the protected, already-tested Phase 2 contracts.
    for name in ('phase2_training_data.py', 'phase2_dataset_verify.py'):
        assert files[name] == (ROOT.parent/'phase2'/name).read_bytes(), 'Vendored source drift: '+name
    bundle = dict(version='tread-one-delivery-r1', generator_sha256=sha(__file__),
                  sources_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                  note='Readable runtime source is authoritative. Notebook outer-file hashes are in releases/DELIVERY.json.')
    payload_files = dict(files, **{'BUNDLE.json': json.dumps(bundle, sort_keys=True).encode()})
    payload = base64.b64encode(gzip.compress(json.dumps({name: base64.b64encode(value).decode() for name, value in payload_files.items()}, sort_keys=True).encode(), mtime=0)).decode()
    bootstrap = '''# Install only the tested Hub/locking tools; retain Kaggle's CUDA PyTorch.
import base64, gzip, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path
for package, version in [('huggingface_hub', '0.36.2'), ('filelock', '3.29.5')]:
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', package+'=='+version])
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
RUNTIME = Path('/kaggle/working/treadone_runtime')
RUNTIME.mkdir(parents=True, exist_ok=True)
payload = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for name, encoded in payload.items():
    assert Path(name).name == name
    value = base64.b64decode(encoded)
    destination = RUNTIME/name
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_bytes(value)
    temporary.replace(destination)
bundle = json.loads((RUNTIME/'BUNDLE.json').read_text())
for name, expected in bundle['sources_sha256'].items():
    assert hashlib.sha256((RUNTIME/name).read_bytes()).hexdigest() == expected, name
print('Verified', len(bundle['sources_sha256']), 'runtime files. HF_TOKEN loaded without displaying it.')
'''.replace('PAYLOAD', repr(payload))
    launch = '''import signal, time
def launch(arguments):
    process = subprocess.Popen([sys.executable, '-B', '-u', str(RUNTIME/'run.py'), *arguments],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except KeyboardInterrupt:
        print('Stop requested. Wait for the checkpoint/HF flush; a second Stop or kernel kill can interrupt cleanup.', flush=True)
        process.send_signal(signal.SIGINT)
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        raise
    if status:
        raise RuntimeError('Run stopped with an error. Read the retained console/checkpoint and last verified HF revision; rerun unchanged to resume.')
'''
    for mode in ('preflight', 'train'):
        title = 'NB00 — data integrity and safe split' if mode == 'preflight' else 'NB01 — TREAD-ONE training, evaluation and export'
        intro = f'''# {title}

**One custom architecture • three seeds • 60 epochs • no pretrained weights.**

Use **{'CPU / Accelerator None' if mode == 'preflight' else 'GPU T4 ×2'}**, Internet ON, Kaggle secret **HF_TOKEN** with write permission to **Shanmuk4622/tyre-wear-study**.
Attach [Tire Dataset Prepared phase2](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared-phase2).
No repository/dataset ZIP upload is needed in addition to that input. Run All in order, one copy of NB01 only.

The dataset is 570 images of 12 tyres; the frozen 386/81/103 split holds out only old low/high tyres. All known mid/video material is training-only. Classification is a mileage proxy. Boundaries and masks do not measure physical alignment or tread depth.

{'This cell sequence verifies all source files and publishes a compact report. Its CPU runtime anchor is separate from training.' if mode == 'preflight' else 'Two independent seed processes use the two T4s; seed 3 queues as a slot becomes free. Each seed runs a real-model, fresh-process resume gate before its first measured update. A failed gate prevents long training.'}

**Recovery:** local checkpoint after each completed optimizer update; consistent HF bundles every 30 minutes, at seed/cell completion and catchable Stop/error. Restore uses the same RUN_TAG/source/environment. Hard kernel death can lose work since the last verified upload. Watch for `HF snapshot verified` and retain the printed revision.

The exact additive maps explain class logits at the evidence layer. They do not prove physical wear reasoning or pixel-level causality; final output includes background/occlusion diagnostics and failures.
'''
        config = f'''# Keep these unchanged when resuming the same experiment.
DATA_ROOT = ''  # Auto-detect VERSION.json in the attached Kaggle input.
RUN_TAG = 'treadone-r1'
TAKE_OVER = False  # True ONCE only after the older copy is stopped and has flushed.
ALLOW_SINGLE_GPU = False  # Default training expects T4 x2; no silent fallback.
WORK = Path('/kaggle/working/treadone_{mode}')
WORK.mkdir(parents=True, exist_ok=True)
arguments = [{mode!r}, '--root', DATA_ROOT, '--work', str(WORK), '--tag', RUN_TAG]
if TAKE_OVER:
    arguments.append('--take-over')
if ALLOW_SINGLE_GPU:
    arguments.append('--allow-single-gpu')
'''
        cells = [md(intro), md('## 1. Bootstrap and secrets\nDo not paste token values into cells. The embedded payload is generated from readable Phase 3 source.'),
                 code(bootstrap), md('## 2. Configure and define the safe launcher'), code(config), code(launch),
                 md('## 3. Execute\nStop once and wait for the verified HF flush. A server backoff or network outage may delay it.'), code('launch(arguments)')]
        if mode == 'train':
            final = '''from IPython.display import display, Image
report = json.loads((WORK/'report/REPORT.json').read_text())
print('Status:', report['status'])
print('Validation-selected export:', report['export_selection'])
print(json.dumps(report['summary'], indent=2))
for path in sorted((WORK/'report/gallery').glob('*.png')):
    display(Image(filename=str(path)))
import zipfile
target = Path('/kaggle/working/TREAD_ONE_Results.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in (WORK/'report').rglob('*'):
        if path.is_file():
            archive.write(path, 'report/'+str(path.relative_to(WORK/'report')))
    for name in ('CONTRACT.json', 'OVERLAY.json', 'ENVIRONMENT.json', 'LAST_REMOTE.json', 'DATA_PREFLIGHT.json'):
        archive.write(WORK/name, name)
print('Download:', target)
print('Full training state/traces and raw test outputs are also in the verified HF seed bundles.')
'''
            cells += [md('## 4. Inspect and download\nThese are task-specific metrics, not a single shared accuracy. The gallery uses the declared validation winner and retains errors.'), code(final)]
        else:
            cells += [code("print(json.dumps(json.loads((WORK/'DATA_PREFLIGHT.json').read_text()), indent=2))\nprint('Preflight complete. Continue with NB01 in a dual-T4 session; no training was performed here.')")]
        for i, cell in enumerate(cells):
            cell['id'] = f'treadone-{mode}-{i:02d}'
        notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
            kernelspec=dict(display_name='Python 3', language='python', name='python3'),
            language_info=dict(name='python'), kaggle=dict(isInternetEnabled=True, isGpuEnabled=mode == 'train',
            datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
        name = 'phase3_NB00_Data_Preflight.ipynb' if mode == 'preflight' else 'phase3_NB01_TREAD_ONE.ipynb'
        (output/name).write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
        import nbformat
        nbformat.validate(nbformat.read(output/name, as_version=4))
    release = ROOT/'releases'
    release.mkdir(exist_ok=True)
    selected = list(output.glob('*.ipynb'))+[ROOT/name for name in ('TREAD_ONE_RUN_GUIDE.md', 'TREAD_ONE_DESIGN.md', 'TREAD_ONE_EXPERIMENT.md', 'TREAD_ONE_VALIDATION.md') if (ROOT/name).exists()]
    selected += [ROOT/'validation'/name for name in ('LOCAL_TESTS.json', 'PERSISTENCE_TESTS.json', 'LIFECYCLE_TESTS.json', 'model_contracts.json') if (ROOT/'validation'/name).exists()]
    delivery = dict(version='tread-one-r1', files={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in selected},
                    runtime_sources=bundle['sources_sha256'], generator_sha256=sha(__file__))
    (release/'DELIVERY.json').write_text(json.dumps(delivery, indent=2)+'\n')
    with zipfile.ZipFile(release/'TREAD_ONE_Kaggle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in selected:
            archive.write(p, str(p.relative_to(ROOT)))
        archive.write(release/'DELIVERY.json', 'DELIVERY.json')
    print('Built and schema-validated both self-contained notebooks:', release/'TREAD_ONE_Kaggle.zip')


if __name__ == '__main__':
    build()
