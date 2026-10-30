"""Independently recount full gate and verify the packaged frozen worker."""
import ast,base64,csv,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];GATE=ROOT/'tread_two/appearance_gate';CHECK=ROOT/'tread_two/gate_validation'
sys.path[:0]=[str(ROOT/'tread_two/pilot'),str(GATE)]
from common import atomic,file_meta,read,sha
from gate_worker import SCHEMAS
work=CHECK/'local_full_v1';cfg=read(work/'CONFIG.json')
snapshot=Path(read(work/'LAST_SNAPSHOT.json')['path'])
manifest=read(snapshot/'MANIFEST.json');cursor=read(snapshot/'CURSOR.json')
assert cursor['phase']=='completed' and cursor['updates']==0 and manifest['contract']==cfg['protocol']
for name,meta in manifest['files'].items(): assert file_meta(snapshot/name)==meta,name
counts={};frames={}
expected=dict(source_train=386,patch_quality=12352,predictions=2702,patch_evidence=259392,affinities=1556352,feature_consistency=86464,execution=386,candidate_bank=18)
for family,fields in SCHEMAS.items():
    count=0;parts=[]
    for path in sorted((snapshot/'csv').glob('epoch_*/'+family+'.csv')):
        for chunk in pd.read_csv(path,chunksize=150000):
            assert list(chunk.columns)==fields
            count+=len(chunk)
            if family!='affinities': parts.append(chunk)
    counts[family]=count
    assert count==expected[family]==cursor['row_cursors'][family],family
    if parts: frames[family]=pd.concat(parts,ignore_index=True)
source=frames['source_train'].fillna('');p=frames['predictions'];patch=frames['patch_evidence'];bank=frames['candidate_bank']
assert set(source.image_id)==set(cfg['contract']['training_image_ids']) and source.image_index.is_unique
assert p.groupby('image_index').size().eq(7).all() and not p.duplicated(['image_index','variant']).any()
prob=p[[f'probability_{c}' for c in range(3)]].to_numpy();logits=p[[f'logit_{c}' for c in range(3)]].to_numpy()
assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
assert (prob.argmax(1)==p.prediction).all() and p.correct.eq(p.prediction.eq(p.class_index).astype(int)).all()
assert np.allclose(-np.log(prob[np.arange(len(p)),p.class_index.to_numpy(int)]),p.nll,atol=5e-7)
saved=torch.load(cfg['selected'],weights_only=True,map_location='cpu')['model']
bias=saved['bias'].numpy();labels=saved['bank_labels'].numpy()
w=np.where(np.arange(3)[:,None]==labels[None],1.,-1.)*2*saved['raw_weights'].sigmoid().numpy()/6
temperature=float(.05+.95*saved['raw_temperature'].sigmoid())
keys=list(zip(p.image_index,p.variant));index={key:i for i,key in enumerate(keys)}
patchsum=patch.groupby(['image_index','variant','class_index']).contribution.sum().unstack().reindex(pd.MultiIndex.from_tuples(keys))
assert np.allclose(patchsum.to_numpy()+bias,logits,atol=2e-6)
alpha=patch.query('class_index == 0').set_index(['image_index','variant','patch']).pooling_mass
calculated=np.tile(bias.astype(np.float64),(len(p),1))
for path in sorted((snapshot/'csv').glob('epoch_*/affinities.csv')):
    for chunk in pd.read_csv(path,chunksize=150000):
        assert np.allclose(np.exp((chunk.cosine.to_numpy()-1)/temperature),chunk.affinity,atol=2e-6,rtol=2e-5)
        mass=alpha.reindex(pd.MultiIndex.from_frame(chunk[['image_index','variant','patch']])).to_numpy()
        assert np.isfinite(mass).all()
        location=np.array([index[(i,v)] for i,v in zip(chunk.image_index,chunk.variant)])
        pooled=mass*chunk.affinity.to_numpy()
        for c in range(3): np.add.at(calculated[:,c],location,pooled*w[c,chunk.prototype.to_numpy(int)])
residual=float(np.abs(calculated-logits).max());assert residual<2e-6,residual
assert len(bank)==bank.image_id.nunique()==18 and bank.groupby('class_index').size().to_dict()=={0:6,1:6,2:6}
assert set(bank.role)=={'train'} and set(bank.status)=={'UNTRAINED_REFERENCE_PROPOSAL_DO_NOT_INSTALL'}
coverage=pd.read_csv(snapshot/'report/candidate_query_coverage.csv')
assert coverage.remaining_original_query_photos.ge(1).all()
assert frames['execution'].optimizer_updates.eq(0).all() and frames['execution'].parameters_with_grad.eq(0).all()
for i in range(386):
    native=torch.load(snapshot/f'artifacts/train/{i:04d}.pt',weights_only=False,map_location='cpu')
    assert native['role']=='train' and native['image_id']==source.set_index('image_index').loc[i,'image_id']
    assert native['image_sha256']==source.set_index('image_index').loc[i,'image_sha256']
    assert np.isfinite(native['descriptors']).all()

delivery=read(CHECK/'NOTEBOOK.json');notebook=Path(delivery['notebook'])
assert sha(notebook)==delivery['sha256']
nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
payload=None
for cell in nb.cells:
    if cell.cell_type!='code': continue
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant): payload=json.loads(base64.b64decode(node.args[0].value))
assert len(payload)==36
fixture=Path(tempfile.mkdtemp(prefix='fixture_notebook_',dir=CHECK))
for name,body in payload.items():
    path=(fixture/'source'/name).resolve();assert path.is_relative_to((fixture/'source').resolve())
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(body));assert sha(path)==delivery['sources'][name]
copied=dict(cfg,frozen_source=str(fixture/'source/frozen_source'),control_root=str(fixture))
atomic(fixture/'CONFIG.json',copied)
environment=dict(os.environ);environment.pop('HF_TOKEN',None)
with (fixture/'worker.log').open('w',encoding='utf-8') as stream:
    child=subprocess.run([sys.executable,'-B',str(fixture/'source/appearance_gate/gate_worker.py'),'--config',str(fixture/'CONFIG.json'),'--folder',str(fixture/'run'),'--limit-images','1'],stdout=stream,stderr=subprocess.STDOUT,env=environment)
assert child.returncode==91,'Extracted notebook worker failed; inspect worker.log'
from store import Store
actual=Store(fixture/'run',cfg['contract'],SCHEMAS)
omit={'compute_seconds','cuda_peak_allocated_dense_bytes','cuda_peak_allocated_head_bytes'}
for family in SCHEMAS:
    if family=='candidate_bank': continue
    a=list(actual.rows(family))
    first=next((snapshot/'csv').glob('epoch_0000/'+family+'.csv'))
    b=[]
    with first.open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if family=='source_train' or int(row['image_index'])==0: b.append(row)
            elif int(row['image_index'])>0: break
    assert len(a)==len(b)
    for x,y in zip(a,b):
        for field in SCHEMAS[family]:
            if family=='execution' and field in omit: continue
            assert x[field]==y[field],(family,field)
report=read(snapshot/'report/GATE_RESULT.json');assert report['candidate_bank_structural_pass']
html=(snapshot/'report/Training_Gate_Explorer.html').read_text(encoding='utf-8')
data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',html,re.S)[1])
assert len(data['cases'])==386 and len(data['bank'])==18 and all(len(r['probes'])==7 for r in data['cases'])
script=re.search(r'<script>(.*?)</script>',html,re.S)[1]
node=subprocess.run(['node','--check'],input=script,text=True,encoding='utf-8',capture_output=True)
assert node.returncode==0,node.stderr
baseline=read(CHECK/'PRESERVATION.json')['notebooks'];assert all(sha(Path(r['path']))==r['sha256'] for r in baseline.values())
destination=ROOT/'tread_two/appearance_gate_results';destination.mkdir(exist_ok=True)
for path in (snapshot/'report').iterdir():
    if path.is_file(): shutil.copy2(path,destination/path.name)
result=dict(status='passed',protocol=cfg['protocol'],scope='Complete local GTX1650 gate; actual Kaggle dual-T4/live-HF gate publication pending',
    source_hashes=cfg['contract']['sources'],environment=cfg['contract']['environment'],snapshot=str(snapshot),
    manifest_files_verified=len(manifest['files']),snapshot_bytes=sum(v['bytes'] for v in manifest['files'].values()),
    scientific_rows=counts,total_scientific_rows=sum(counts.values()),images=386,predictions=2702,logits_reconstructed=8106,maximum_logit_residual=residual,
    all386_native_source_roles_and_hashes_verified=True,packaged_worker_one_image_matches_full_science=True,
    untrained_bank_proposal_photos=18,original_training_tyre_support=[1,3,4],remaining_query_photo_minimum=int(coverage.remaining_original_query_photos.min()),
    query_cohort_counts={'v1':378,'candidate_excluding18photos':368},training_updates=0,validation_or_test_forwards=0,new_independent_tyres=0,
    notebook_sha256=delivery['sha256'],embedded_files=36,HTML_payload_and_JS_syntax_checked=True,browser_render_tested=False,
    local_HF_credentials_read=False,remote_writes=0,actual_Kaggle_dual_T4_tested=False,live_gate_HF_publication_tested=False)
atomic(CHECK/'FULL_VALIDATION.json',result)
print(json.dumps(result,indent=2))
