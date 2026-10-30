"""Create only NB07: CPU collection/audit + direct HF CSV, no training."""
import ast,base64,gzip,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def main():
 names=list(json.loads((ROOT/'revision2_validation/DELIVERY.json').read_text())['sources'])
 names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'data_gate').glob('*') if p.suffix in ('.py','.json')]
 names+=['analysis/nb05_public_reads.py']
 names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'data_intake').glob('*') if p.suffix in ('.csv','.json')]
 files={n:(ROOT/n).read_bytes() for n in names};encoded=base64.b64encode(gzip.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode(),mtime=0)).decode()
 setup='''import base64,gzip,json,os,signal,subprocess,sys,shutil
from pathlib import Path
from filelock import FileLock
WORK=Path('/kaggle/working/treadone_nb07_data_gate')
WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>1024**3,'Need at least 1 GiB free for the gate/cache; source images remain in Kaggle Input.'
sources=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
with FileLock(str(WORK/'NB07_SESSION.lock'),timeout=1):
    for name,value in sources.items():
        p=(WORK/name).resolve();assert p.is_relative_to(WORK.resolve())
        # Preserve an operator's already edited intake on rerun.
        if name.startswith('data_intake/') and p.exists():continue
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
(WORK/'data_intake/incoming').mkdir(exist_ok=True)
candidates=[]
for p in Path('/kaggle/input').rglob('INTAKE_VERSION.json'):
    if json.loads(p.read_text()).get('schema')=='tyre-prospective-intake-v1':candidates.append(p.parent)
assert len(candidates)<=1,'Attach at most one prospective intake dataset or select it explicitly in INTAKE.'
INTAKE=candidates[0] if candidates else WORK/'data_intake'
print('CPU data gate ready. Intake:',INTAKE,'; no model training or inference is launched.')
def launch(name,*args):
    with FileLock(str(WORK/'NB07_SESSION.lock'),timeout=1):
        child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'data_gate'/name),*map(str,args)],start_new_session=True)
        try:
            while True:
                try:code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(60):
                    try:child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%4==0:print('Allow completed-file cleanup or the publication flush to finish.')
            raise
        assert code==0,'Stage stopped; preserve logs and rerun to reuse completed image/file caches.'
'''.replace('PAYLOAD',repr(encoded))
 intro='''# TREAD-ONE NB07 — data and protocol gate

**CPU only. Collects richer CSV evidence; no neural inference or training.**
This implements the next data/identity/protocol milestone after NB06. It audits
the existing 570 images / 12 original tyres, registers supplied new intake,
and records whether a prospective dataset is ready. Empty intake correctly
produces a completed audit with `prospective_training_ready=false`.

## Kaggle setup

1. Accelerator **None / CPU**; Internet **ON**.
2. Attach `shanmuk4622/tire-dataset-prepared-phase2` directly from Kaggle.
3. Optionally attach one new intake dataset containing `INTAKE_VERSION.json`,
   the tyre/session/image CSVs, `incoming/` native files and optional actual
   `protocol_policy.json`. With no new dataset, the bundled empty templates
   are used. Confirmed clip-parent mappings can be added to the resolution CSV.
4. Keep **1 GiB free**. Source images stay read-only. About 120 pinned training
   CSV shards are fetched for saved-exposure aggregation; no model-weight ZIP
   or full previous 2 GB experiment download is needed. No 1 TB scratch assumed.
5. Enable Kaggle Secret **HF_TOKEN** for final publication to
   `Shanmuk4622/tyre-wear-study`, then **Run All**. No package installation.

The CSVs contain actual source registries, fixed-grid quality descriptors,
native mask/point geometry and provenance, similarity review candidates,
conservative identity components, class/subject support, saved training exposure,
validated intake and explicit readiness gaps. These are not new independent
tyres or additional physical wear ground truth. Quality features are audit
outputs and are not silently fed to the models.

## Resume and output policy

Each completed source-image descriptor is cached atomically against source file,
annotation, code and environment hashes. Verified completed exposure downloads
are reused. An incomplete image/file is recomputed/downloaded on rerun. Input
CSVs/native files are never edited; derived records live in a versioned output.
An audit result can be useful and publishable while prospective training is not
ready. Required data, labels/identity, annotation and frozen-policy conditions
remain visible; unknown inputs are not filled with fake measurements.

Tables are directly browsable CSV. JSON schema/contracts, PNG figures and a
self-contained HTML explorer supplement them. Publication is additive under
`phase3/tread-one/data-gate/v1/<contract>`, uses bounded immutable batches and
deduplicates verified files. The final cell attempts a consistent immediate
flush on a catchable Stop/error, respecting HF server delays. Forced process
death cannot flush; completed local cache/files remain reusable.

No prospective source upload or new model is triggered automatically. Register
actual data and freeze meaningful subject counts/selection criteria before a
later model fit. Similarity is review triage, not physical identity evidence.
'''
 cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(setup)]
 stages=[('Collect image, geometry, identity and training-exposure CSV records',"launch('run_gate.py','--intake',INTAKE)"),
  ('Render the interactive source explorer and prepare direct files',"launch('report_gate.py')\nlaunch('publish_gate.py','--prepare-only')"),
  ('Publish the completed data audit directly to HF',"from kaggle_secrets import UserSecretsClient\nos.environ['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    launch('publish_gate.py')\nfinally:\n    os.environ.pop('HF_TOKEN',None)"),
  ('Read results and preserve the executed notebook',"from IPython.display import display,FileLink\nimport pandas as pd\nbase=WORK/'data_gate_outputs'\nlatest=json.loads((base/'LATEST.json').read_text());folder=Path(latest['path'])\nprint((folder/'SUMMARY.json').read_text())\ndisplay(pd.read_csv(folder/'csv/historical_class_support.csv'))\ndisplay(pd.read_csv(folder/'csv/readiness_gaps.csv'))\ndisplay(FileLink(str(folder/'TREAD_ONE_Data_Gate.html')))\nprint((base/'PUBLICATION.json').read_text())\nprint('A false readiness flag is an expected audit finding when new intake/protocol is missing. No model was trained.')")]
 for title,code in stages:cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(code)])
 for i,c in enumerate(cells):
  c['id']=f'treadone-nb07-{i:02d}'
  if c.cell_type=='code':ast.parse(c.source)
 nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
 nbformat.validate(nb);target=ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb'
 if target.exists():assert all(not c.get('outputs') and c.get('execution_count') is None for c in nbformat.read(target,as_version=4).cells if c.cell_type=='code'),'Preserve executed NB07 before rebuilding'
 nbformat.write(nb,target);out=ROOT/'data_gate_validation';out.mkdir(exist_ok=True)
 result=dict(status='schema_syntax_payload_valid',notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),notebook_bytes=target.stat().st_size,embedded_files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},Kaggle_executed=False,live_publication_tested=False,new_training=False,new_inference=False)
 (out/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('Built CPU NB07:',target.stat().st_size,'bytes;',len(files),'embedded files')

if __name__=='__main__':main()
