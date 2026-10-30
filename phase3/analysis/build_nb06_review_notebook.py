"""Compact Kaggle CPU review/publication; preserve the executed NB05 exactly."""
import ast,base64,gzip,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def main():
 sources=json.loads((ROOT/'revision2_validation/DELIVERY.json').read_text())['sources']
 names=list(sources)+['analysis/'+n for n in ('nb05_public_reads.py','verify_nb05_publication.py','verify_nb05_preflight.py','audit_nb05_results.py','render_nb05_review.py','publish_nb05_review.py','nb05_launcher.py')]
 names+=['notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb','revision2_validation/DELIVERY.json']
 files={n:(ROOT/n).read_bytes() for n in names}
 encoded=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode(),mtime=0)).decode()
 setup='''import base64,gzip,hashlib,json,os,signal,shutil,subprocess,sys
from pathlib import Path
from filelock import FileLock
WORK=Path('/kaggle/working/treadone_nb06_review')
WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>6*1024**3,'Need at least 6 GiB free for verified object/cache links and the review.'
payload=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
with FileLock(str(WORK/'NB06_SESSION.lock'),timeout=1):
    for name,value in payload.items():
        p=(WORK/name).resolve()
        assert p.is_relative_to(WORK.resolve()) and not Path(name).is_absolute()
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
print('Read-only review sources ready. No HF secret is accessed until publication.')
def launch(name,*args):
    with FileLock(str(WORK/'NB06_SESSION.lock'),timeout=1):
        child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'analysis'/name),*args],start_new_session=True)
        try:
            while True:
                try:code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for index in range(60):
                    try:child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if index%4==0:print('Allow the current public-read cleanup or publication flush to finish.')
            raise
        assert code==0,'Review stage stopped; preserve logs and rerun. Verified completed downloads are reused.'
'''.replace('PAYLOAD',repr(encoded))
 text='''# TREAD-ONE NB06 — completed paired review and failure atlas

**CPU only. No training, new inference, checkpoint selection or app installation.**
NB05 completed both 60-epoch conditions. The projected candidate's selected
development-test BA was **17.073%**, versus **50%** for the baseline; its human
point error was also higher. This notebook verifies/reviews that completed
result and publishes a directly browsable audit and interactive image atlas.

## Kaggle setup

1. Accelerator **None / CPU**; Internet **ON**. No T4 allocation is needed.
2. Attach exactly one expanded **shanmuk4622/tire-dataset-prepared-phase2** input.
3. Allow at least **6 GiB free working space**. About **2.05 GB** of final source
   evidence is fetched and checked; logical object/cache links are larger than
   physical storage. Optional preflight fixtures add a bounded smaller download.
4. Enable Kaggle Secret **HF_TOKEN** with write access to
   **Shanmuk4622/tyre-wear-study** for the final publication cell only.
5. **Run All**. Use current Kaggle packages; no installation/downgrade.

Read stages explicitly use the public pinned repository with no token-file or
.env lookup. The measured training files stay in their original namespaces.
The review is additive under `phase3/tread-one/r2/review/paired-seed1/<contract>`.
Metrics are direct CSV, with JSON, PNG and a self-contained HTML viewer; no ZIP,
NPZ, database or model-weight upload. The notebook's compressed source payload
is only packaging of executable notebook sources, not a hidden HF data archive.

## What is retained and what resume means here

The original 23 CSV families, native recovery/checkpoint identities and maps are
verified against pinned manifests. Post-hoc tables include class/tyre errors,
validation stability, gradient/clipping history, calibration, confidence-risk
diagnostics and paired image deltas. The atlas includes all 103 test images and
all four selected/final condition endpoints, masks, six points and signed class
evidence. It uses saved outputs, not a new model forward pass.

Interrupted reads reuse completed verified objects on rerun; an incomplete file
is fetched again. Derived tables/figures are deterministic recomputations.
Local analysis files are saved before publication; the final stage attempts a
consistent flush on catchable Stop/error, respecting HF backoff. Forced process
death cannot flush. Rerun the same stage after an interruption. Do not run
duplicate writers; an old unexpired publication lease remains protected.

This is one paired seed on an already inspected low/high test cohort from two
tyres. Richer rows/plots do not add independent subjects. Do not choose a new
checkpoint using these test results. Independent tyre collection is the next
data gate; physical wear/depth/alignment/safety are not established.
'''
 cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(setup)]
 for title,code in [
  ('1. Check the pinned manifests and fetch verified evidence',"launch('verify_nb05_publication.py','--metadata')\nlaunch('verify_nb05_publication.py')"),
  ('2. Independently verify preflight recovery and recompute the science',"launch('verify_nb05_preflight.py')\nlaunch('audit_nb05_results.py')"),
  ('3. Build the figures, complete image atlas and direct-file review',"launch('render_nb05_review.py')\nlaunch('publish_nb05_review.py','--prepare-only')"),
  ('4. Publish the completed review directly to HF',"from kaggle_secrets import UserSecretsClient\nos.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    launch('publish_nb05_review.py')\nfinally:\n    os.environ.pop('HF_TOKEN',None)"),
  ('5. Inspect results and preserve this executed notebook',"from IPython.display import display,FileLink\nimport pandas as pd\nOUT=WORK/'revision2_validation/nb05_publication'\ndisplay(pd.read_csv(OUT/'analysis/endpoint_metrics.csv'))\ndisplay(FileLink(str(OUT/'analysis/TREAD_ONE_NB05_Comparison_Explorer.html')))\nprint((OUT/'REVIEW_PUBLICATION.json').read_text())\nprint('Keep this executed notebook for independent audit. No model was trained or installed.')")]:
  cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(code)])
 for i,cell in enumerate(cells):
  cell['id']=f'treadone-nb06-review-{i:02d}'
  if cell.cell_type=='code':ast.parse(cell.source)
 notebook=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
 nbformat.validate(notebook);target=ROOT/'notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb'
 if target.exists():
  previous=nbformat.read(target,as_version=4)
  assert not any(c.get('outputs') or c.get('execution_count') for c in previous.cells if c.cell_type=='code'),'Preserve executed NB06 before rebuilding'
 nbformat.write(notebook,target)
 result=dict(status='schema_syntax_payload_valid',notebook_bytes=target.stat().st_size,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
  embedded_files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},source_NB05_sha256=hashlib.sha256(files['notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb']).hexdigest(),
  optimizer_updates=0,new_inference=False,Kaggle_executed=False,live_review_publication_tested=False)
 out=ROOT/'revision2_validation/nb06_delivery';out.mkdir(exist_ok=True);(out/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('Built CPU NB06:',target.stat().st_size,'bytes;',len(files),'embedded files')

if __name__=='__main__':main()
