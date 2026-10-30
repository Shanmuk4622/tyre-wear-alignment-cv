"""NB04 is an engineering acceptance gate, never a new measured training study."""
import ast,base64,gzip,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def main():
 frozen=json.loads((ROOT/'completion_20261028/hf/CONTRACT.json').read_text())['sources']
 assert all(hashlib.sha256((ROOT/'runtime'/name).read_bytes()).hexdigest()==value for name,value in frozen.items())
 (ROOT/'training/FROZEN.json').write_text(json.dumps(frozen,sort_keys=True,separators=(',',':'))+'\n')
 files=[p for parent in ('runtime','training') for p in sorted((ROOT/parent).glob('*')) if p.is_file() and p.suffix in ('.py','.json')]+[ROOT/'diagnostic/metrics.py']
 payload={str(p.relative_to(ROOT)).replace('\\','/'):base64.b64encode(p.read_bytes()).decode() for p in files}
 encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
 cells=[nbformat.v4.new_markdown_cell('''# TREAD-ONE NB04 — richer evidence and optimizer-recovery gate

**Engineering test only. It does not launch another 60-epoch study, modify r1
checkpoints, choose a new model or replace the workstation models.**

This tests real TREAD-ONE optimizer updates with richer direct CSV, actual
augmented-image predictions/targets, numerical/gradient/activation/resource
records, native maps and full optimizer/scheduler/AMP/RNG recovery state.

## Settings and run order

1. Kaggle **T4 ×2**, Internet **ON**. Both available T4 devices run independent
   compute-only recovery tests; one parent coordinates HF operations.
2. Attach `shanmuk4622/tire-dataset-prepared-phase2` directly from Kaggle.
3. Enable Kaggle Secret **HF_TOKEN**, write access to `Shanmuk4622/tyre-wear-study`.
4. Keep at least **12 GiB free** in working storage. The input dataset is not copied.
5. Run All. Existing packages are retained; no global dependency downgrade.

Each engineering case stops at **three** updates, crossing a declared two-batch
surrogate epoch boundary. The gate compares frozen r1 update math, uninterrupted
instrumented updates, fresh-process pause/resume, an abrupt native-process exit,
and reconstruction from directly stored snapshots. These repeated fixtures are
recovery evidence, not independent model experiments or training-data growth.

HF proof uploads a one-update state, downloads that pinned commit to a new
recovery base, continues to update three, and checks exact numerical state
against the uninterrupted run. Direct CSV and native `.pt`/`.npy` files share a
manifest and committed cursor; no ZIP/NPZ is published.

## Stop and resume

Use this notebook unchanged to resume. Compatible saved cases skip completed
updates. Local native recovery states rotate to the latest two; **all CSV and
declared gate maps remain retained**. HF commits preserve immutable history.
The parent attempts a consistent snapshot on catchable stop/error, major stage
completion and every 30 minutes, respecting server backoff. A forced kernel or
container death cannot flush unsent work. Only one parent may own a namespace;
clean completion releases its 45-minute writer lease.

The pass result validates this engineered logging/recovery path. It is not a
claim of long-run model quality, continuous GPU energy measurement, universal
network reliability, or a completed r2 study. A measured recipe and bounded map
schedule must be frozen before any later long training run.
''')]
 source='''import base64,gzip,json,os,signal,subprocess,sys,time
from pathlib import Path
from kaggle_secrets import UserSecretsClient
WORK=Path('/kaggle/working/treadone_evidence_gate')
WORK.mkdir(parents=True,exist_ok=True)
os.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
sources=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for name,value in sources.items():
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    target=WORK/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(value))
print('TREAD-ONE evidence gate ready. HF_TOKEN loaded without displaying it.')

def launch():
    child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'training/gate.py'),'--publish'],start_new_session=True)
    try:
        code=child.wait()
    except KeyboardInterrupt:
        print('Stop requested: waiting for completed-update state/CSV snapshot and HF flush.')
        os.killpg(child.pid,signal.SIGINT)
        for i in range(48):
            try:
                child.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                if i%4==0:print('Still waiting for a safe snapshot or server backoff; keep the session open.')
        if child.poll() is None:print('The coordinator is still flushing; do not launch a duplicate writer.')
        raise
    if code:raise RuntimeError('Gate failed or paused. Read the traceback and retained coordinator.log; rerun unchanged to recover compatible cases.')
'''.replace('PAYLOAD',repr(encoded))
 cells.append(nbformat.v4.new_code_cell(source))
 cells.append(nbformat.v4.new_markdown_cell('## Run the real-model gate\nNo old training bundle or pretrained model download is required. The prepared Kaggle input supplies all images/labels.'))
 cells.append(nbformat.v4.new_code_cell('launch()'))
 cells.append(nbformat.v4.new_markdown_cell('## Inspect the gate and publication receipts\nPass/fail concerns engineering correctness. These few fixture losses are not study results.'))
 cells.append(nbformat.v4.new_code_cell('''from IPython.display import display,FileLink
result=json.loads((WORK/'training_gates/GATE_COMPLETION.json').read_text())
assert result['status']=='passed'
print(json.dumps(result,indent=2))
for device in result['devices']:
    folder=Path(device['path'])
    snapshot=json.loads((folder/'LATEST_ENVELOPE.json').read_text())['path']
    display(FileLink(str(Path(snapshot)/'cases/live_resume/EVIDENCE_GATE.html')))
    print('Verified HF receipt:',device['publication'])
print('Next: review the frozen logging schema and choose one controlled model/objective change; no new study was started by NB04.')
'''))
 for i,c in enumerate(cells):
  c['id']=f'treadone-evidence-gate-{i:02d}'
  if c.cell_type=='code':ast.parse(c.source)
 nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
     kaggle=dict(isInternetEnabled=True,isGpuEnabled=True,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
 nbformat.validate(nb);target=ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb';nbformat.write(nb,target)
 out=ROOT/'training_gates';out.mkdir(exist_ok=True)
 report=dict(status='schema_syntax_and_payload_valid',notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),notebook_bytes=target.stat().st_size,
   embedded_sources={n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in payload.items()},kaggle_execution_tested=False,live_hf_gate_tested=False)
 (out/'DELIVERY.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Built NB04 only:',target,'bytes:',target.stat().st_size)

if __name__=='__main__':main()
