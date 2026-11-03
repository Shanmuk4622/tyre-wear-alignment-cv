"""Actual-data numeric, heldout noninterference, package/viewer and publish checks."""
import ast,base64,hashlib,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import nbformat
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];GATE=ROOT/'tread_two/readiness_gate';CHECK=ROOT/'tread_two/readiness_validation';OUT=ROOT/'tread_two/readiness_report';SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(GATE));from public_inputs import atomic,sha,Inputs
from run_gate import frames
from publish_gate import prepare,publish
fixture=Path(tempfile.mkdtemp(prefix='acceptance_',dir=CHECK));a=json.loads((OUT/'ANALYSIS.json').read_text());summary=a['summary'];checks=[]
assert summary['training_images']==386 and summary['conditional_folds']==6 and summary['folds_missing_fit_class']==2 and not summary['scientific_readiness']
assert [r['conservative_components'] for r in summary['class_support']]==[1,1,4] and [r['known_original_tyres'] for r in summary['class_support']]==[1,3,4]
for name,m in a['tables'].items():
    p=OUT/(name+'.csv');df=pd.read_csv(p);assert len(df)==m['rows'] and list(df.columns)==m['columns'] and sha(p)==m['sha256']
refs=pd.read_csv(OUT/'held_component_bank_support.csv');assert refs.loc[refs.unsupported.eq(1),'normalization_if_supported'].isna().all()
assert refs.loc[refs.unsupported.eq(0),'normalization_if_supported'].gt(0).all()
values=pd.read_csv(OUT/'online_component_counterfactuals.csv');assert len(values)==188160 and values.groupby('sample').size().eq(4).all()
assert values[values.variant.eq('frozen_prior')].delta_nll.eq(0).all()
checks.append('Actual support1/1/4,missing-class folds2/6,all table hashes/counts,unsupported normalization and188160 algebraic rows')
reader=Inputs(GATE/'INPUTS.json',CHECK/'cache');data={key:pd.read_csv(reader.file(key),keep_default_na=False) for key in reader.spec['inputs']};assert reader.calls==0
def run(data,out):return frames(data['review_source.csv'],data['identity_images.csv'],data['identity_components.csv'],data['identity_tyres.csv'],data['review_predictions.csv'],data['review_context.csv'],out)
# Scientific output must ignore every held-out numeric prediction/context/label.
changed={k:v.copy() for k,v in data.items()}
for name in ('review_predictions.csv','review_context.csv'):
    df=changed[name];mask=df.stage.ne('train')
    for c in df.select_dtypes(include='number'):df.loc[mask,c]=float('nan')
changed['review_source.csv'].loc[changed['review_source.csv'].role.ne('train'),'class_index']=-99
changed['identity_images.csv'].loc[changed['identity_images.csv'].historical_study_split.ne('train'),'class_index']=-88
tables,other=run(changed,fixture/'heldout_changed');assert tables==a['tables'] and other==summary
checks.append('Perturb every held-out numeric prediction/correction and registry label; every scientific CSV remains byte-identical')
broken={k:v.copy() for k,v in data.items()};index=broken['identity_images.csv'].index[broken['identity_images.csv'].historical_study_split.eq('train')][0];broken['identity_images.csv'].loc[index,'image_sha256']='bad'
try:run(broken,fixture/'bad_identity')
except AssertionError:pass
else:raise AssertionError('Changed training source identity must fail')
checks.append('Training identity mismatch rejected')
viewer=subprocess.run(['node',str(ROOT/'analysis/test_readiness_viewer.cjs'),str(OUT/'Evidence_Passport.html')],capture_output=True,text=True,encoding='utf-8');assert viewer.returncode==0,viewer.stderr;ui=json.loads(viewer.stdout);atomic(CHECK/'VIEWER.json',ui)
checks.append('Actual passport JavaScript filters/epoch bounds/variants/curves pass synthetic DOM')
delivery=json.loads((CHECK/'NOTEBOOK.json').read_text());nb=nbformat.read(delivery['notebook'],as_version=4);nbformat.validate(nb);assert sha(Path(delivery['notebook']))==delivery['sha256']
code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==4 and all(c.execution_count is None and not c.outputs for c in code)
work=fixture/'notebook';work.mkdir();cell=code[0].source.replace("Path('/kaggle/working')",f'Path({str(fixture)!r})').replace("Path('/kaggle/working/tread_two_nb15_readiness')",f'Path({str(work)!r})')
context={};exec(compile(cell,'NB15_setup','exec'),context)
for n,h in delivery['sources'].items():assert sha(work/n)==h
exec(compile(cell,'NB15_setup_rerun','exec'),context)
script=work/'gate/run_gate.py';script.write_bytes(script.read_bytes()+b'\n# corruption fixture\n')
try:exec(compile(cell,'NB15_setup_mismatch','exec'),context)
except AssertionError:pass
else:raise AssertionError('Changed extracted source must fail')
original=base64.b64decode(context['payload']['gate/run_gate.py']);script.write_bytes(original)
fresh=fixture/'fresh_report'
proc=subprocess.run([sys.executable,'-B',str(script),'--cache',str(CHECK/'cache'),'--out',str(fresh)],capture_output=True,text=True,encoding='utf-8')
assert proc.returncode==0,proc.stderr[-2500:];fresh_a=json.loads((fresh/'ANALYSIS.json').read_text());assert fresh_a['tables']==a['tables'] and fresh_a['public_read_requests']==0
checks.append('Actual40-file setup extracts/reruns/rejects drift; fresh packaged analysis reproduces every CSV without network')
sys.path.insert(0,str(SOURCE));import publisher
saved=list(sys.path);spec=importlib.util.spec_from_file_location('gate_fake_service',ROOT/'analysis/check_context_training.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.path[:]=saved
prepared=prepare(OUT,SOURCE,fixture/'publication');service=module.Service();hub=publisher.Publisher.__new__(publisher.Publisher);hub.destination=prepared['destination'];hub.owner='gate-fixture';hub.api=service;hub.session=service;hub.calls=0;hub.last_verified=None;hub.log=fixture/'REMOTE.jsonl';service.lost=True
with patch.object(publisher,'Publisher',return_value=hub):
    receipt=publish(prepared,SOURCE,fixture/'publication','fixture');before=service.commits;again=publish(prepared,SOURCE,fixture/'publication','fixture')
assert receipt['owner_released'] and again['status']=='already_published_same_manifest' and service.commits==before
assert all(n.startswith(prepared['destination']+'/') for n in service.files) and '/fit/' not in prepared['destination'] and '/review/' not in prepared['destination']
checks.append('Offline additive publication/lost response/owner release/identical rerun,without any original fit or review namespace writes')
result=dict(status='passed',checks=checks,notebook_sha256=delivery['sha256'],embedded_files=delivery['embedded_files'],CSV_tables=len(a['tables']),CSV_rows=sum(m['rows'] for m in a['tables'].values()),publication_files=prepared['files'],source_hashes={p.name:sha(p) for p in GATE.iterdir() if p.suffix in ('.py','.json')},
    heldout_noninterference=True,fresh_packaged_science_equal=True,actual_setup_executed_locally=True,private_SDK_install_actual_Kaggle_pending=True,viewer=ui,model_forwards=0,optimizer_updates=0,remote_writes=0,actual_NB15_Kaggle_HF_completion=False)
atomic(CHECK/'PACKAGE_VALIDATION.json',result);print(json.dumps(result,indent=2))
