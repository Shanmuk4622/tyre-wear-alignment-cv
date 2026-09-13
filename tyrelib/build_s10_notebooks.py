"""Generate NB19/NB20 only. Preserve executed notebook evidence."""
import base64
import hashlib
import json
from pathlib import Path
import build_notebooks as b
import build_closure_notebooks as c


def build():
    for number,action,title in [(19,'evidence','Verified evidence package'),(20,'report','Final figures and results draft')]:
        name=f'NB{number}_S10_'+('Evidence' if number==19 else 'Report')+'.ipynb'
        p=b.OUT/name
        if p.exists():
            raw=p.read_bytes();old=json.loads(raw)
            if any(x.get('outputs') for x in old['cells']):
                archive=b.OUT/'execution_archives';archive.mkdir(exist_ok=True)
                dest=archive/(p.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
                if not dest.exists():dest.write_bytes(raw)
        cells=c.start(f'''# NB{number} — S10 {title}

**Kaggle CPU, one copy, Internet ON, HF_TOKEN enabled — Run All.**
No dataset attachment, GPU training, new annotation or previous notebook rerun.
Run NB19 FIRST, then NB20 in a CPU session. No paths/configuration to copy.

NB19 collects pinned verified tables and10 existing NB10R figures with hashes.
NB20 requires NB19's matching evidence manifest and produces four new figures,
an HTML report and Markdown results/limitations draft. Source records are never
overwritten. The full project is NOT certified complete by either notebook:
HRNet/PatchCore deferred, H2 inconclusive, H3 untested, video/factorial gaps open.

No expensive recomputation: only small tables/figures, no model checkpoints.
HF cache resumes downloads; outputs use a code-versioned namespace. Normal
pushes30min, major completion and catchable Stop flush. Forced kills cannot
flush. Rerun the same notebook after interruption; keep one worker. Human
review of scientific claims/references/venue formatting is still required.
''','s10report')
        cells[1]=b.code(b.bootstrap_cell().replace("'pyarrow', 'timm'","'pyarrow'"))
        source=Path(__file__).with_name('s10_reporting.py').read_bytes()
        cells += [b.code(f"(WORK/'s10_reporting.py').write_bytes(base64.b64decode({base64.b64encode(source).decode()!r}))\nimport s10_reporting as s10\n"),
                  b.code(f"PREFIX = s10.{action}(sess)\nassert sess.finish(), 'Retry final flush before closing'\nprint(PREFIX)\n")]
        c.save(name,cells);nb=json.loads(p.read_text());nb['metadata'].pop('accelerator',None)
        nb['metadata']['kaggle'].pop('accelerator',None);nb['metadata']['kaggle']['isGpuEnabled']=False
        p.write_text(json.dumps(nb,indent=1),encoding='utf-8')

if __name__=='__main__':build()
