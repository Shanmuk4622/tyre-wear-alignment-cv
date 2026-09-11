"""Generate only NB13--NB17. Executed earlier notebooks are never overwritten."""
import base64
import json
import hashlib
from pathlib import Path
import build_notebooks as b
import build_closure_notebooks as c

HERE = Path(__file__).parent


def save_cpu(name, cells):
    c.save(name, cells)
    path = b.OUT/name
    nb = json.loads(path.read_text(encoding='utf-8'))
    nb['metadata'].pop('accelerator', None)
    nb['metadata']['kaggle'].pop('accelerator', None)
    nb['metadata']['kaggle']['isGpuEnabled'] = False
    path.write_text(json.dumps(nb, indent=1), encoding='utf-8')


def embedded():
    return '\n'.join(f"(WORK/{name!r}).write_bytes(base64.b64decode({base64.b64encode((HERE/name).read_bytes()).decode()!r}))"
        for name in ['s5_data.py', 's5_runtime.py', 's5_notebook.py']) + "\nimport importlib\nimport s5_data as sd\nimport s5_notebook as sn\nimportlib.reload(sd); importlib.reload(sn)\n"


INSTALL = r'''# Keep Kaggle's CUDA torch/torchvision and NumPy; never upgrade them implicitly.
import importlib.metadata as metadata
import subprocess, sys, os
scratch = Path('/kaggle/temp/tyre_s5')
scratch.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(scratch/'hf_cache')
os.environ['TORCH_HOME'] = str(scratch/'torch_cache')
os.environ['YOLO_CONFIG_DIR'] = str(scratch/'yolo_config')
(scratch/'yolo_config').mkdir(exist_ok=True)
constraints = scratch/'constraints.txt'
constraints.write_text('\n'.join(f'{p}=={metadata.version(p)}' for p in ('torch','torchvision','numpy'))+'\n')
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',
    '--disable-pip-version-check','--timeout','60','-c',str(constraints),
    *[f'{p}=={v}' for p,v in sd.PACKAGES.items()]],check=True,timeout=600)
print('Pinned S5 packages installed; Kaggle CUDA packages retained.')
'''


def start(title):
    cells = c.start(title, 's5', data=True)
    cells[1] = b.code(b.bootstrap_cell().replace("'pyarrow', 'timm'", "'pyarrow'"))
    cells.insert(2, b.code(embedded()))
    # Data preparation uses the existing attached Kaggle package, not a full HF dataset clone.
    cells.append(b.code("assert ANN_ROOT is not None, 'Attach Tire Dataset Prepared including annotations/clean/masks'\n"))
    return cells


def build():
    # Preserve uploaded execution evidence before replacing generated notebook cells.
    for path in b.OUT.glob('NB1[3-7]_S5_*.ipynb'):
        old = path.read_bytes()
        nb = json.loads(old)
        if any(cell.get('outputs') or cell.get('execution_count') is not None for cell in nb['cells']):
            archive = b.OUT/'execution_archives'
            archive.mkdir(exist_ok=True)
            target = archive/(path.stem+'_'+hashlib.sha256(old).hexdigest()[:12]+'.ipynb')
            if not target.exists():
                target.write_bytes(old)
            print('Preserved executed notebook:', target.name)
    cells = start('''# NB13 — S5 data and protocol preparation (CPU)

Run this **once**, before NB14/15/16. Internet ON, HF_TOKEN enabled, Tire Dataset
Prepared attached. No annotation and no GPU training. Uses the 418 existing clean
manual masks, not SAM2 or augmented-image pseudo-ground truth.

The new dense-task recipe is **9 model/task configurations × 3 folds × 3 seeds ×
60 epochs = 81 runs**. This is not another classification rerun. Fold 0/2 leakage
flags remain; results are descriptive. Model IDs and exact input bytes are frozen.
Training uses clean originals, with documented online flips where supported;
it does not silently reuse the much larger classification derivative recipe.
''')
    cells += [b.code("PLAN, PREFIX = sn.prepare(sess, DATA_ROOT, ANN_ROOT)\nprint('Copy this PREFIX into NB14, NB15, NB16 and NB17:', PREFIX)\nassert sess.finish()")]
    save_cpu('NB13_S5_Prepare.ipynb', cells)
    for name, family, title in [('NB14_S5_Semantic.ipynb','semantic','Semantic segmentation — U-Net, DeepLabV3+, SegFormer B0/B2'),
                                ('NB15_S5_YOLO.ipynb','yolo','YOLO26-n/s detection and instance segmentation'),
                                ('NB16_S5_RTDETRv2.ipynb','rtdetr','RT-DETRv2-R18 detection')]:
        cells = start(f'''# {name.split('_')[0]} — S5 {title}

1. Finish NB13 first. Leave PREFIX blank to discover its unique matching protocol
   from HF automatically. Multiple matches require an explicit PREFIX. Attach the same dataset/masks.
2. Select **T4 ×2**, Internet ON, HF_TOKEN enabled. GPU0 is used; GPU1 intentionally
   idle to avoid the previous DataParallel slowdown. No automatic smaller model.
3. First run **MODE='PILOT'** in one copy. This tests every model in this family,
   checkpoints and reloads a pilot in isolated scratch, and uploads logs. Pilots
   are separate from scientific runs and never counted as completed training.
   With four accounts configured, only the first (worker0) runs PILOT; others skip it.
4. After pilots pass, change MODE to 'TRAIN'. One worker is the default. For four
   copies set the SAME four ACTIVE_KAGGLE_ACCOUNTS in every Session cell, and a
   DIFFERENT ACCOUNT in each. Run only this family across those four copies.
   Do not run the other S5 families concurrently or duplicate an account.
5. Rerun TRAIN after a session ends. Completed jobs skip; unfinished jobs resume
   from the latest HF-published completed epoch. Then continue with the next family.

Normal pushes every30min, plus pilot/training-cell completion and catchable Stop.
There is no per-claim commit. A hard kernel/OS kill cannot flush; up to30min of
unpublished progress may need repeating. Stop during an epoch replays that epoch,
not individual batches. A runtime/package/protocol mismatch stops instead of resetting.
The first run is a Kaggle validation step: local checks do not certify GPU execution.
''')
        if family=='yolo':
            cells[0] = b.md('''# NB15 — corrected YOLO26 detection/segmentation (AUTO)

**Run All. No manual PILOT → TRAIN switch is required. Do not rerun NB13.**

- Select T4×2, Internet ON, HF_TOKEN enabled, and attach Tire Dataset Prepared.
- Leave PREFIX blank and MODE='AUTO'. The notebook checks the corrected
  flip-only policy in short pilots, then automatically starts/resumes full training.
- One copy works with the default account settings. For four copies, use the same
  four ACTIVE_KAGGLE_ACCOUNTS in each, and different ACCOUNT values (acct1–acct4).
  Start acct1 first: it handles pilot checks; the others wait for those checks,
  then each starts its own training shard. Do not run duplicate accounts.
- Normal HF pushes every30min; major completion and catchable Stop flush.
  Restart AUTO after a session ends to resume the latest published completed epoch.
  A hard kernel kill cannot flush unfinished work.

This repair disables implicit Blur/MedianBlur/grayscale/CLAHE. A loader-level
check must print `flip-only-r2 VERIFIED` before updates. Old pilot evidence is
retained separately, not mistaken for corrected-policy validation. Same four
models,36 runs,60 epochs,512px,batch4; GPU0 used and GPU1 intentionally idle.
The correction was tested locally; its T4 pilot is validated by this notebook.
''')
        if family=='rtdetr':
            cells[0] = b.md('''# NB16 — RT-DETRv2 resume repair

**Use a fresh Kaggle session and Run All. MODE='TRAIN' is already selected.**
The existing HF pilot is verified; NB13 and the completed training runs need not be repeated.

Select T4×2, Internet ON, HF_TOKEN, Tire Dataset Prepared. Leave PREFIX blank.
One worker is configured by default: keep it unless you deliberately use separate
accounts with matching active-account lists. Do not run duplicate copies of the same worker.

The saved checkpoint used a different NumPy version from the fresh Kaggle session.
This notebook restores the EXACT saved NumPy in an isolated child-process directory,
without replacing notebook packages or the CUDA stack and without weakening the
checkpoint check. Fresh jobs use the pilot's runtime; completed jobs skip.

Continue TRAIN after session limits. Last published completed epoch resumes;
30min normal HF pushes, major completion and catchable Stop flush. Forced kills
cannot flush. No model, batch, resolution, epoch-budget or protocol change.
''')
        default_mode = 'AUTO' if family=='yolo' else ('TRAIN' if family=='rtdetr' else 'PILOT')
        cells += [b.code(INSTALL), b.code(f"PREFIX = ''  # automatic discovery; set explicitly only if HF has multiple matching protocols\nMODE = {default_mode!r}  # AUTO validates corrected YOLO pilots then trains; other families use PILOT first\nFAMILY = {family!r}\nPLAN = sn.load_plan(sess, PREFIX, DATA_ROOT, ANN_ROOT)\nPREFIX = sn.plan_prefix(PLAN)\nprint('Using frozen protocol:', PREFIX)\n"),
            b.code("sn.run_family(sess, PLAN, PREFIX, DATA_ROOT, ANN_ROOT, FAMILY, MODE)\nassert sess.finish(), 'Retry the final flush before closing'\n")]
        c.save(name, cells)
    cells = start('''# NB17 — S5 public-HF completion audit and report (CPU)

Run after NB14, NB15 and NB16 finish. This notebook does not retrain models.
It checks all81 jobs against a pinned HF snapshot, checkpoint hashes, complete
60-epoch histories, held-out prediction coverage, and recomputed paired ROI F1.
Partial results are reported as partial, never green. The per-image localisation
and frozen-classifier evaluation is performed by the training notebooks in a
separate process after each trained model, so a failed evaluation needs no retraining.

SAM2/manual comparison remains deferred. S9 is not completed by S5.
''')
    cells += [b.code("PREFIX = ''  # automatically find the unique matching NB13 protocol on HF\nPLAN = sn.load_plan(sess, PREFIX, DATA_ROOT, ANN_ROOT)\nPREFIX = sn.plan_prefix(PLAN)\n"),
              b.code("REPORT = sn.report(sess, PLAN, PREFIX)\nassert sess.finish()\nprint(REPORT.to_string(index=False))")]
    save_cpu('NB17_S5_Report.ipynb', cells)


if __name__ == '__main__':
    build()
