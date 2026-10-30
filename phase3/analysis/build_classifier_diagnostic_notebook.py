"""Build NB03 only; preserve all executed NB00/NB01/NB02 artifacts."""
import ast
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def main():
    sources={str(p.relative_to(ROOT)).replace('\\','/'):base64.b64encode(p.read_bytes()).decode()
             for parent in ('runtime','diagnostic') for p in sorted((ROOT/parent).glob('*')) if p.suffix in ('.py','.html')}
    fetch=ROOT/'analysis/fetch_completed_results.py'
    sources['analysis/fetch_completed_results.py']=base64.b64encode(fetch.read_bytes()).decode()
    payload=base64.b64encode(gzip.compress(json.dumps(sources).encode(),mtime=0)).decode()
    cells=[nbformat.v4.new_markdown_cell('''# TREAD-ONE NB03 — classifier diagnosis and image explorer

This is the next **TREAD-ONE** step. It uses the six existing selected/final
checkpoints and performs **no training, optimizer update, new test evaluation or
checkpoint selection**. It measures nominal train/validation predictions,
defined gate/background/occlusion interventions and bounded fixed-batch gradients.

## Kaggle settings and order

1. Attach `shanmuk4622/tire-dataset-prepared-phase2` directly from Kaggle.
2. Internet **ON**, accelerator **T4 ×2**; **GPU 0 only** is used sequentially for
   consistent diagnostic inference. GPU 1 is unused; this is not distributed training.
3. Enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`.
4. Run All. Existing Kaggle packages are retained; no shared dependency downgrade.
5. Keep at least **6 GiB free** in `/kaggle/working`. Approximately 390 MiB of
   old source evidence is read; all new outputs are direct CSV/JSON/native NPY/HTML.

The report includes probabilities, per-tyre/domain failures, mask/gate/evidence
overlays and interactive opacity/class controls. More logged records are not
additional independent tyres. User-assisted new-data collection is a later step.

## Persistence and stop behavior

Local complete inference units contain at most 32 images, with all five variants.
HF snapshots publish after each major checkpoint job, every 30 minutes and on
catchable Stop/error. Direct CSV plus JSON cursors reconstruct progress in a fresh
compatible session; only the last incomplete local unit is replayed.

Use this notebook unchanged to resume. Recipe, source and environment identities
are frozen into a diagnostic namespace; different environments do not silently
merge results. Only one notebook may own that namespace. A hard-killed writer's
lease expires after 45 minutes; catchable clean stop releases it immediately.
Hard kernel/container death cannot publish unsent work. Server backoff still applies.
''')]
    setup='''import base64, gzip, json, os, signal, subprocess, sys, time
from pathlib import Path
from kaggle_secrets import UserSecretsClient
WORK=Path('/kaggle/working/treadone_diagnostic')
WORK.mkdir(parents=True,exist_ok=True)
os.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
payload=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for name,body in payload.items():
    target=WORK/name
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(body))
print('TREAD-ONE diagnostic sources ready; HF_TOKEN loaded without displaying it.')

def launch(name,*arguments):
    process=subprocess.Popen([sys.executable,'-B','-u',str(WORK/name),*arguments],start_new_session=True)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        print('Stop requested. Asking the diagnostic to persist and publish completed units; keep this session open.')
        os.killpg(process.pid,signal.SIGINT)
        for attempt in range(36):
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if attempt%4==0: print('Waiting for the final snapshot/server backoff...')
        if process.poll() is None:
            print('The child is still flushing. Do not start a duplicate notebook; forced kernel death can lose unpublished work.')
        raise
    if code:
        raise RuntimeError(f'{name} exited with status {code}; completed local/remote units remain resumable. Read the diagnostic traceback above.')
'''.replace('PAYLOAD',repr(payload))
    cells.append(nbformat.v4.new_code_cell(setup))
    cells.append(nbformat.v4.new_markdown_cell('## 1. Verify/retrieve the pinned r1 source evidence\nLegacy ZIPs are downloaded only as read-only historical input. No new ZIP is uploaded. Cached complete downloads are reused.'))
    cells.append(nbformat.v4.new_code_cell("launch('analysis/fetch_completed_results.py','--bundles')"))
    cells.append(nbformat.v4.new_markdown_cell('## 2. Run or resume the fixed-checkpoint diagnosis\nFull train/validation cohorts, three seeds, both saved endpoints, five variants. It restores compatible remote progress before deciding what to skip.'))
    cells.append(nbformat.v4.new_code_cell("launch('diagnostic/run.py','--publish')"))
    cells.append(nbformat.v4.new_markdown_cell('## 3. Inspect and retain the visual report\nDownload/open the self-contained HTML for the image explorer. It needs no external service. If Kaggle blocks interactive inline output, the downloaded HTML remains interactive in your browser.'))
    cells.append(nbformat.v4.new_code_cell('''import html
import pandas as pd
from IPython.display import HTML, FileLink, display
completed=sorted((WORK/'diagnostics/classifier_r1').glob('*/COMPLETE.json'),key=lambda p:p.stat().st_mtime)
assert completed, 'Diagnosis is not complete yet; rerun the diagnostic cell to resume.'
completion=json.loads(completed[-1].read_text())
snapshot=Path(completion['snapshot'])
report=snapshot/'TREAD_ONE_Classifier_Diagnostic.html'
display(pd.read_csv(snapshot/'summary/cohort_group_metrics.csv').query("level == 'cohort' and variant == 'normal'"))
display(FileLink(str(report)))
SHOW_INLINE_PREVIEW=False  # Full explorer is about 31 MiB; keep ordinary notebook outputs compact.
if SHOW_INLINE_PREVIEW:
    display(HTML('<iframe title="TREAD-ONE diagnostic explorer" sandbox="allow-scripts" style="width:100%;height:1100px;border:0" srcdoc="'+html.escape(report.read_text(),quote=True)+'"></iframe>'))
print('Download/open the HTML for interactive images. Set SHOW_INLINE_PREVIEW=True only if the notebook renderer supports the large inline report.')
print((completed[-1].parent/'PUBLICATION.json').read_text())
print('Preserved output folder:',snapshot)
'''))
    for i,cell in enumerate(cells):
        cell['id']=f'treadone-diagnostic-{i:02d}'
        if cell.cell_type=='code': ast.parse(cell.source)
    notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
         kaggle=dict(isInternetEnabled=True,isGpuEnabled=True,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
    nbformat.validate(notebook)
    target=ROOT/'notebooks/phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb'
    nbformat.write(notebook,target)
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    report=dict(status='schema_syntax_and_embedded_source_valid',notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
          sources={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in sources.items()},
          kaggle_execution_tested=False,live_hf_diagnostic_publication_tested=False)
    (out/'NB03_DELIVERY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Built NB03:',target,'bytes:',target.stat().st_size)


if __name__=='__main__': main()
