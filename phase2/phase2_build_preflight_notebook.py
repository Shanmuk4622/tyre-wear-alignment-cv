"""Generate CPU Kaggle integrity preflight; never starts training or assigns splits."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    dataset=ROOT/'data/phase2_dataset_v1'
    sums_hash=hashlib.sha256((dataset/'SHA256SUMS.txt').read_bytes()).hexdigest()
    verifier=(ROOT/'phase2_dataset_verify.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
    cells=[]
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        compile(s,'<phase2-cell>','exec')
        cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('''# Phase 2 NB00 — verify the uploaded dataset

**CPU, one notebook session. No training, no new annotation, no random splitting.**

1. Upload the supplied ZIP as **Tire Dataset Prepared phase2** and attach that dataset here.
2. Enable Internet and the Kaggle secret **HF_TOKEN** for Shanmuk4622.
3. Run All. The notebook checks the exact release and all image/mask files, then publishes one small verification report to the existing HF dataset repository under a new phase2 path.
4. An integrity PASS is not permission to train: unresolved identity/label/geometry gates are printed separately. Save the executed notebook outputs.

The ZIP should be expanded by Kaggle into the attached input tree. If only a ZIP appears, inspect its contents in the dataset uploader; do not copy the entire dataset into the limited output area. This notebook never modifies the attached data.
''')
    code('''from pathlib import Path
import hashlib, json, os, sys, time, uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
if not HF_TOKEN:
    raise RuntimeError('Enable the HF_TOKEN Kaggle secret before running.')
HF_REPO = 'Shanmuk4622/tyre-wear-study'
OUTPUT = Path('/kaggle/working/phase2_preflight')
OUTPUT.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
api = HfApi(token=HF_TOKEN)

def retry(operation):
    for attempt in range(6):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            if status not in (None, 429, 500, 502, 503, 504) or attempt == 5:
                raise
            hint = getattr(response, 'headers', {}).get('Retry-After', '0')
            try:
                server_wait = float(hint)
            except ValueError:
                try: server_wait = max(0., (parsedate_to_datetime(hint)-datetime.now(timezone.utc)).total_seconds())
                except Exception: server_wait = 0.
            delay = max(server_wait, min(300., 10.*2**attempt))
            print(f'HF temporary error (HTTP {status}); local report retained. Waiting {delay:.0f}s.')
            deadline = time.monotonic()+delay
            while time.monotonic()<deadline: time.sleep(min(2., max(0.,deadline-time.monotonic())))

def publish_report(report):
    local = OUTPUT/'phase2_preflight_report.json'
    local.write_text(json.dumps(report, indent=2), encoding='utf-8')
    remote = f'phase2/source-v1/{EXPECTED_CHECKSUM_SHA}/preflight/{RUN_ID}/report.json'
    commit = retry(lambda: api.create_commit(repo_id=HF_REPO, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))],
        commit_message='Phase 2 dataset integrity preflight '+RUN_ID))
    # Cache is a few KB; only a small report is downloaded, not images or weights.
    downloaded = retry(lambda: hf_hub_download(repo_id=HF_REPO, repo_type='dataset', filename=remote,
        revision=commit.oid, token=HF_TOKEN, cache_dir='/kaggle/working/phase2_preflight/hf_cache'))
    assert hashlib.sha256(Path(downloaded).read_bytes()).digest() == hashlib.sha256(local.read_bytes()).digest()
    print('Verified HF report commit:', commit.oid)
    (OUTPUT/'phase2_hf_commit.json').write_text(json.dumps({'revision':commit.oid,'path':remote},indent=2))
''')
    code(f"EXPECTED_CHECKSUM_SHA = {sums_hash!r}\n"+verifier)
    code('''report = dict(run_id=RUN_ID, expected_checksum_manifest_sha256=EXPECTED_CHECKSUM_SHA,
              python=sys.version, status='started', training_started=False)
failure = None
try:
    candidates = []
    for version_file in Path('/kaggle/input').rglob('VERSION.json'):
        try: metadata = json.loads(version_file.read_text())
        except (ValueError, OSError): continue
        if metadata.get('dataset_title') == 'Tire Dataset Prepared phase2':
            candidates.append(version_file.parent)
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one expanded Phase 2 dataset, found {len(candidates)}. Attach the correct version.')
    DATASET = candidates[0]
    observed = hashlib.sha256((DATASET/'SHA256SUMS.txt').read_bytes()).hexdigest()
    if observed != EXPECTED_CHECKSUM_SHA:
        raise RuntimeError('Attached dataset differs from the delivered release; do not bypass this check.')
    report['verification'] = verify(DATASET)
    report['status'] = 'integrity_pass_training_gates_pending'
except BaseException as exc:
    failure = exc
    report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'verification_failed'
    report['error_type'] = type(exc).__name__
    report['error'] = str(exc)[:1000]
finally:
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    try:
        publish_report(report)
    except BaseException as publication_error:
        print('HF publication not verified:', type(publication_error).__name__)
        print('Local report is retained in', OUTPUT, '- retry publication without rechecking the data.')
        if failure is None: failure = publication_error
if failure is not None:
    raise failure
print('SOURCE DATASET VERIFIED. Training is still gated; inspect the reported next steps.')
''')
    md('''## Interruption and remaining work

Catchable interruption attempts to publish the partial verification report. A forced kernel kill or network failure cannot guarantee that push; no training progress exists to lose in this CPU notebook. Normal verification should finish in minutes and uses one commit; never commits once per image. Server backoff takes precedence over immediate upload.

If only publication failed, rerun `publish_report(report)` while the local report remains available. Keep the report/commit ID. Next: resolve the video-to-original tyre mapping, freeze the split, resolve mask conflicts for the selected trainers, review new geometry points, then build/run the model-specific GPU/resume smoke notebooks. Do not run old training notebooks against this new schema.
''')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),cells=cells)
    for i,c in enumerate(cells):c['id']=f'phase2-preflight-{i:02d}'
    dest=ROOT/'notebooks'/'phase2_NB00_Dataset_Preflight.ipynb'
    dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(nb,indent=1),encoding='utf-8')
    print('Generated',dest,'with pinned release checksum',sums_hash)


if __name__=='__main__':main()
