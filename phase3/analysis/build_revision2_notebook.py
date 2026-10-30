"""Build only the new NB05 paired pilot; preserve every user-executed notebook."""
import ast,base64,gzip,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def main():
 target=ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb'
 if target.exists():
  previous=nbformat.read(target,as_version=4)
  if any(c.get('outputs') or c.get('execution_count') is not None for c in previous.cells if c.cell_type=='code'):
   identity=hashlib.sha256(target.read_bytes()).hexdigest()
   archive=ROOT/'revision2_validation/nb05_repair'/identity[:12]/'executed.ipynb'
   assert archive.exists() and hashlib.sha256(archive.read_bytes()).hexdigest()==identity,'Archive the user-executed NB05 before rebuilding'
 base=json.loads((ROOT/'training_gates/DELIVERY.json').read_text())['embedded_sources']
 assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in base.items())
 (ROOT/'revision2/BASE_SOURCES.json').write_text(json.dumps(base,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
 files={n:(ROOT/n).read_bytes() for n in base}
 files.update({p.relative_to(ROOT).as_posix():p.read_bytes() for p in (ROOT/'revision2').glob('*') if p.suffix in ('.py','.json')})
 encoded=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode(),mtime=0)).decode()
 cells=[nbformat.v4.new_markdown_cell('''# TREAD-ONE NB05 — one paired shared-gradient comparison

**This notebook launches the next measured study after automatic preflight:**
one unchanged TREAD-ONE architecture, **seed 1**, two matched conditions,
**60 epochs / 2,940 optimizer updates each**. Baseline uses the original summed
gradient; `shared_projection` applies symmetric conflict projection only to
shared modules. Initial weights, samples, augmentation, losses, optimizer,
validation selection and endpoints are matched. It is one unreplicated paired
development pilot, not established generalization or an improved model claim.

## Kaggle setup

1. **T4 ×2**, Internet **ON**; attach exactly one expanded
   `shanmuk4622/tire-dataset-prepared-phase2` dataset.
2. Enable write-authorized Kaggle Secret **HF_TOKEN** for
   `Shanmuk4622/tyre-wear-study`.
3. Use a clean working area with **12 GiB free**. The prepared dataset stays in
   input; no pretrained weights or old training ZIP is downloaded.
4. **Run All**. No dependency downgrade or new package installation.

**29 October lease-wait repair:** if another session holds the automatic
preflight's HF writer lease, this notebook waits and safely retries every five
minutes, for up to 90 minutes of waiting. Stop cancels waiting. An abandoned
lease expires within 45 minutes of its last renewal; an active session can
renew it. Stop an unused older session cleanly. Do not delete OWNER.json or
change the experiment namespace. Training errors are not automatically retried.
Attempt logs remain in `/kaggle/working/treadone_r2_paired/launcher_logs/`.
The 27 experiment sources and checkpoint/source contract are unchanged.

The first short automatic preflight exercises both conditions with validation
fixtures, native/CSV recovery and actual pinned HF download. It must pass before
the two measured runs start. The old NB04 and r1 data/checkpoints are preserved.

## Retention and resume

Every actual augmented training occurrence, loss sufficient statistic, pixel
and point error, latent/evidence statistic and task-gradient observation is
retained. Full validation runs over all 81 images each epoch. Selected and final
endpoints retain predictions and native maps for all 103 previously inspected
development test images. Source/test limitations remain explicit.

Local atomic journal units keep native optimizer/scheduler/AMP/RNG state and
observation cursors together. Every published metric table is a **direct CSV**;
native PT/NPY and JSON supplement it. No HF ZIP/NPZ or SQLite upload.

HF publication is batched every **30 minutes**, after major stages, or on a
catchable Stop/error. Immutable direct files are staged in bounded batches; the
snapshot manifest selects only verified objects. Forced kernel death cannot
flush unsent progress. Native local states rotate to two; all scalar records,
declared bounded maps and selected-weight improvements remain retained.

Rerun this notebook unchanged to restore compatible published state. A changed
software/source contract is rejected rather than silently starting over.
Keep the session open while Stop flushes and server backoff is active. Do not
launch another writer for the same condition.

The model outputs remain classification, overlapping tyre/tread masks,
mask-derived boxes, ordered points and exact additive signed evidence. No app
model is replaced automatically. More independent tyre data remains needed.
''')]
 setup='''import base64,gzip,json,os,signal,subprocess,sys,time
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from filelock import FileLock
WORK=Path('/kaggle/working/treadone_r2_paired')
WORK.mkdir(parents=True,exist_ok=True)
os.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
sources=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
with FileLock(str(WORK/'R2_SESSION.lock'),timeout=1):
    for name,value in sources.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        target=WORK/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(value))
print('TREAD-ONE NB05 ready; secret loaded without displaying it.')
'''.replace('PAYLOAD',repr(encoded))
 launcher=ROOT/'analysis/nb05_launcher.py'
 setup+='\n'+launcher.read_text(encoding='utf-8')+'\n\ndef launch():\n    return launch_paired(WORK)\n'
 cells += [nbformat.v4.new_code_cell(setup),nbformat.v4.new_markdown_cell('## Automatic live preflight, then the paired 60-epoch runs\nBoth conditions use the same frozen architecture. This cell starts the declared measured study only after preflight passes.'),nbformat.v4.new_code_cell('launch()'),nbformat.v4.new_markdown_cell('## Results and interactive progress reports\nInterpret this as a one-seed development comparison. Installation requires separate review. No test-based checkpoint or seed reselection.'),nbformat.v4.new_code_cell('''from IPython.display import display,FileLink
latest=json.loads((WORK/'revision2_runs/LATEST_PAIR.json').read_text())
folder=Path(latest['path'])
result=json.loads((folder/'PAIR_DELIVERY.json').read_text())
assert result['status']=='completed'
print(json.dumps(result,indent=2))
for condition in ('baseline','shared_projection'):
    snapshot=json.loads((folder/condition/'LATEST_SNAPSHOT.json').read_text())['path']
    display(FileLink(str(Path(snapshot)/'TREAD_ONE_R2_Progress.html')))
    display(FileLink(str(Path(snapshot)/'artifacts/PAIR_COMPARISON.csv')))
print('Preserve this executed notebook for independent publication/result audit.')
''')]
 for i,c in enumerate(cells):
  c['id']=f'treadone-r2-paired-{i:02d}'
  if c.cell_type=='code':ast.parse(c.source)
 nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)))
 nbformat.validate(nb);target=ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb';nbformat.write(nb,target)
 out=ROOT/'revision2_validation';out.mkdir(exist_ok=True)
 (out/'DELIVERY.json').write_text(json.dumps(dict(status='schema_syntax_and_payload_valid',notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),notebook_bytes=target.stat().st_size,
  sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},source_files=len(files),launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
  repair='notebook-only-preflight-lease-wait-20261029',measured_study_started=False,Kaggle_execution_tested=False),indent=2)+'\n')
 print('Built NB05 only:',target,'bytes',target.stat().st_size,'sources',len(files))

if __name__=='__main__':main()
