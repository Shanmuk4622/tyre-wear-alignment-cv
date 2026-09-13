"""Generate only NB18, preserving all executed earlier notebooks."""
import base64
import json
from pathlib import Path
import build_notebooks as b
import build_closure_notebooks as c

def build():
    old=b.OUT/'NB18_S9_Fusion_Analysis.ipynb'
    if old.exists():
        raw=old.read_bytes();prior=json.loads(raw)
        if any(c.get('outputs') for c in prior['cells']):
            import hashlib
            archive=b.OUT/'execution_archives';archive.mkdir(exist_ok=True)
            target=archive/(old.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
            if not target.exists():target.write_bytes(raw)
    cells=c.start('''# NB18 — S9 exploratory fusion and reporting (CPU)

**Kaggle CPU, Internet ON, HF_TOKEN enabled; Run All in ONE copy.**
No dataset attachment, GPU, training or new annotation needed. NB13–NB17 need
not be rerun. Reads the verified S5 report/predictions at a fixed HF revision.

Tests five fixed arms for all81 S5 runs: full image, tyre crop, tread crop,
equal tyre+tread averaging, equal full+tyre+tread averaging. Saves per-image
probabilities, per-run metrics and per-model/fold three-seed summaries to HF.
No tuned weights, winner selection, significance claim or new-tyre claim.
Repair: reads each classifier's frozen configuration and uses its recorded
CORAL ordinal or softmax decision rule. Old four-run analysis is retained on
HF; corrected CPU analyses use a new code-version namespace. No model retraining.

This is an exploratory component study, NOT the full planned S9 pipeline.
HRNet landmark labels and a verified healthy pool for PatchCore are unavailable.
Existing tyre masks and low-wear proxy labels must not be substituted for them.

Rerun after interruption: verified completed run analyses skip recomputation.
Normal pushes every30min; completion/catchable Stop flush. Forced kernel kills
cannot flush unpublished work. Only small prediction files are downloaded into
scratch; no checkpoint weights or full datasets. Keep account defaults.
''','s9analysis')
    cells[1]=b.code(b.bootstrap_cell().replace("'pyarrow', 'timm'","'pyarrow'"))
    source=Path(__file__).with_name('s9_evidence.py').read_bytes()
    cells += [b.code(f"(WORK/'s9_evidence.py').write_bytes(base64.b64decode({base64.b64encode(source).decode()!r}))\nimport s9_evidence as s9\n"),
              b.code("PREFIX = s9.run(sess)\nassert sess.finish(), 'Retry final flush before closing'\nprint('Published:', PREFIX)\n")]
    name='NB18_S9_Fusion_Analysis.ipynb';c.save(name,cells)
    path=b.OUT/name;nb=json.loads(path.read_text());nb['metadata'].pop('accelerator',None)
    nb['metadata']['kaggle'].pop('accelerator',None);nb['metadata']['kaggle']['isGpuEnabled']=False
    path.write_text(json.dumps(nb,indent=1),encoding='utf-8')

if __name__=='__main__':build()
