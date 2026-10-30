"""Build the CPU-only additive CSV recovery notebook; never rebuild r1 notebooks."""
import ast
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import nbformat
ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ('fetch_completed_results.py', 'audit_and_export_csv.py', 'publish_recovered_csv.py')
    payload = {name: base64.b64encode((ROOT/'analysis'/name).read_bytes()).decode() for name in names}
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    cells = [nbformat.v4.new_markdown_cell('''# NB02 — completed-run audit and direct CSV recovery

**CPU only. No training, model changes, test-based reselection or ZIP upload.**

**Repair r2: fresh-session output directories are created before retrieval.**
Re-import this file and Run All. Existing complete downloads in the same session
are reused; no training notebook or completed checkpoint needs rerunning.

The r1 run is already complete. This notebook reads its immutable HF evidence,
checks every checkpoint/archive member, independently recalculates saved scores,
recovers detailed CSV and publishes those CSV files directly in one additive commit.

1. Accelerator **None / CPU**; Internet ON.
2. Attach **shanmuk4622/tire-dataset-prepared-phase2** from Kaggle.
3. Enable secret **HF_TOKEN**, write access to **Shanmuk4622/tyre-wear-study**.
4. Run All in order. The final publication cell adds a `posthoc-csv/2026-10-28/`
   folder under the original protocol; it does not overwrite model/checkpoint files.

It downloads about **390 MiB** of legacy evidence once. Local ZIPs are read only
to recover the historical format; **no ZIP/NPZ/checkpoint is uploaded by this notebook**.
Native input masks stay read-only. About 3 GiB free working space is required.
Existing Kaggle PyTorch/Hub dependencies are retained; no global package downgrade.

Source revision: `a0296f18e02b12d28f7e51377783d68538032209`.
Scores remain low/high-only, two-test-tyre internal results. Added metrics are
explicitly post-hoc derivations, not previously unlogged observations.
''')]
    setup = '''import base64, gzip, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient
import torch, numpy, PIL, requests, huggingface_hub
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
WORK = Path('/kaggle/working/treadone_csv_recovery')
WORK.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(WORK).free > 3*1024**3, 'Need 3 GiB working free for recovery/verification.'
scripts = WORK/'analysis'
scripts.mkdir(parents=True, exist_ok=True)
(WORK/'completion_20261028').mkdir(parents=True, exist_ok=True)
payload = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
for name, value in payload.items():
    assert Path(name).name == name
    (scripts/name).write_bytes(base64.b64decode(value))
print('CPU recovery sources extracted. HF_TOKEN loaded without displaying it.')
def launch(script, *arguments):
    subprocess.run([sys.executable, '-B', '-u', str(scripts/script), *arguments], check=True)
'''.replace('PAYLOAD', repr(encoded))
    cells.append(nbformat.v4.new_code_cell(setup))
    cells.append(nbformat.v4.new_markdown_cell('## 1. Retrieve and verify the historical evidence\nA restart reuses completed local downloads; immutable source hashes remain fixed.'))
    cells.append(nbformat.v4.new_code_cell("launch('fetch_completed_results.py', '--bundles')"))
    cells.append(nbformat.v4.new_markdown_cell('## 2. Audit and write CSV\nThis checks 8,820 updates, 180 validation epochs, raw test predictions and source-mask hashes; it performs no optimizer step.'))
    cells.append(nbformat.v4.new_code_cell("launch('audit_and_export_csv.py')\nlaunch('publish_recovered_csv.py', '--prepare-only')\nprint((WORK/'completion_20261028/CSV_README.md').read_text())"))
    cells.append(nbformat.v4.new_markdown_cell('## 3. Publish directly readable CSV\nOne batched commit, verified file hashes; no ZIP upload and no modification of the original training namespace files. Retry the same cell after a transient publication-only failure.'))
    cells.append(nbformat.v4.new_code_cell("launch('publish_recovered_csv.py')\nprint((WORK/'completion_20261028/CSV_PUBLICATION.json').read_text())"))
    cells.append(nbformat.v4.new_markdown_cell('## 4. Inspect local results\nThe same individual CSV files remain in Kaggle Output. The dataset viewer can browse the new HF folder directly.'))
    cells.append(nbformat.v4.new_code_cell("import pandas as pd\ndisplay(pd.read_csv(WORK/'completion_20261028/csv/run_summary.csv'))\nprint('CSV folder:', WORK/'completion_20261028/csv')"))
    for i, cell in enumerate(cells):
        cell['id'] = f'treadone-csv-{i:02d}'
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    nb = nbformat.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', name='python3', language='python'),
        kaggle=dict(isInternetEnabled=True, isGpuEnabled=False, datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
    nbformat.validate(nb)
    target = ROOT/'notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb'
    nbformat.write(nb, target)
    report = dict(status='schema_and_syntax_valid', revision='r2-fresh-session-directories', notebook=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  embedded_sources={name: hashlib.sha256((ROOT/'analysis'/name).read_bytes()).hexdigest() for name in names},
                  local_audit_executed=True, kaggle_execution_tested=False, live_publication_tested=False)
    (ROOT/'completion_20261028/CSV_NOTEBOOK_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Built new CPU notebook only:', target)


if __name__ == '__main__':
    main()
