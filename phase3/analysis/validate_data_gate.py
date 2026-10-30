"""Bind real exports, exact notebook sources, CPU replay and offline publication."""
from PIL import Image
import ast,base64,csv,gzip,io,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic,table,digest,locate
from publisher_r2 import Publisher
from test_training_evidence import Service
OUT=ROOT/'data_gate_validation'

def main():
 base=ROOT/'data_gate_outputs';latest=read(base/'LATEST.json');folder=Path(latest['path']);summary=read(folder/'SUMMARY.json');schema=read(folder/'CSV_SCHEMA.json')
 assert summary['source_integrity_passed'] and summary['historical_split_consistent'] and not summary['prospective_training_ready']
 assert summary['source_images']==570 and summary['confirmed_existing_tyres']==12 and summary['new_intake_images']==0
 assert len(schema)==22 and all(len(table(folder/'csv'/(n+'.csv')))==s['rows'] and sha(folder/'csv'/(n+'.csv'))==s['sha256'] for n,s in schema.items())
 assert summary['csv_rows']==sum(s['rows'] for s in schema.values())
 points=table(folder/'csv/point_geometry.csv');assert len(points)==1632 and sum(r['human_eligible']=='True' for r in points)==720
 q=table(folder/'csv/image_quality.csv');assert sum(int(r['tread_outside_tyre_native_pixels'])>0 for r in q)==45 and all(int(r['tread_outside_tyre_valid_pixels'])==0 for r in q)
 support=table(folder/'csv/historical_class_support.csv');assert next(r for r in support if r['historical_split']=='train' and r['proxy_class']=='low')['confirmed_original_tyres']=='1'
 assert all(r['images']=='0' for r in support if r['historical_split'] in ('validation','test') and r['proxy_class']=='mid')
 exposure=table(folder/'csv/training_image_exposure.csv');assert sum(int(r['sampled_augmented_occurrences']) for r in exposure)==47040
 assert all(int(r['sampled_augmented_occurrences'])==0 for r in exposure if r['historical_split']!='train')
 assert read(OUT/'UNIT_TESTS.json')['status']=='passed' and read(OUT/'ARTIFACT_CSV_VALIDATION.json')['CSV_rows']==summary['csv_rows']
 delivery=read(OUT/'DELIVERY.json');notebook=ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb';assert sha(notebook)==delivery['notebook_sha256']
 nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb);files={}
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  assert not cell.outputs and cell.execution_count is None
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    for n,v in json.loads(gzip.decompress(base64.b64decode(node.args[0].value))).items():files[n]=base64.b64decode(v);assert files[n]==(ROOT/n).read_bytes()
 assert len(files)==42
 fresh=Path(tempfile.mkdtemp(prefix='fresh_cpu_',dir=OUT)).resolve();assert fresh.is_relative_to(OUT.resolve())
 for n,v in files.items():p=fresh/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
 cache=fresh/'data_gate_outputs/cache';cache.mkdir(parents=True)
 shutil.copytree(base/'cache/features',cache/'features')
 # Force four real source recomputations while keeping all other completed units.
 for p in sorted((cache/'features').glob('*.json'))[:4]:p.rename(p.with_suffix('.retained'))
 pins=read(ROOT/'data_gate/exposure_sources.json')
 for condition,pin in pins['conditions'].items():
  for entry in pin['files']:
   src=ROOT/'revision2_validation/nb05_publication/remote'/condition/entry['name'];dest=cache/'exposure'/condition/Path(entry['name']).parent.name/'train_samples.csv';dest.parent.mkdir(parents=True,exist_ok=True)
   try:os.link(src,dest)
   except OSError:shutil.copy2(src,dest)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1');env.pop('HF_TOKEN',None)
 stages=[]
 for name,args in [('run_gate.py',['--data-root',str(locate())]),('report_gate.py',[]),('publish_gate.py',['--prepare-only'])]:
  harness=fresh/'harness.py';harness.write_text('import runpy,sys\nfrom unittest.mock import patch\n'+f'sys.path.insert(0,{str(fresh/"data_gate")!r})\n'+f'sys.argv={[name]+args!r}\n'+'import torch\nassert not torch.cuda.is_available()\n'+f'with patch("requests.sessions.Session.request",side_effect=AssertionError("Network disabled in fresh replay")):\n    runpy.run_path({str(fresh/"data_gate"/name)!r},run_name="__main__")\n',encoding='utf-8')
  with open(fresh/(name+'.log'),'w',encoding='utf-8') as log:result=subprocess.run([sys.executable,'-B',str(harness)],env=env,stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,(name,(fresh/(name+'.log')).read_text(encoding='utf-8')[-5000:]);stages.append(name);print('Fresh CPU gate stage passed:',name,flush=True)
 other=Path(read(fresh/'data_gate_outputs/LATEST.json')['path']);assert read(other/'CONTRACT.json')['run_id']==latest['run_id']
 assert all(sha(folder/'csv'/(n+'.csv'))==sha(other/'csv'/(n+'.csv')) for n in schema)
 assert sha(folder/'TREAD_ONE_Data_Gate.html')==sha(other/'TREAD_ONE_Data_Gate.html')
 html=(folder/'TREAD_ONE_Data_Gate.html').read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',html,re.S).group(1);p=OUT/'report_syntax.js';p.write_text(script,encoding='utf-8')
 checked=subprocess.run([shutil.which('node'),'--check',str(p)],capture_output=True,text=True);assert checked.returncode==0,checked.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0];assert len(data['images'])==570
 for r in data['images']:
  with Image.open(io.BytesIO(base64.b64decode(r['preview'].split(',',1)[1]))) as image:assert image.size==(144,192);image.verify()
 publication=read(base/'PUBLICATION_DELIVERY.json');snapshot=Path(publication['path']);assert read(fresh/'data_gate_outputs/PUBLICATION_DELIVERY.json')['contract']==publication['contract']
 service=Service();hub=Publisher.__new__(Publisher);hub.owner='offline-data-gate';hub.destination='offline-data-gate';hub.log=OUT/'publication_fixture.jsonl';hub.api=service;hub.session=service;hub.calls=0;hub.last_verified=None
 hub.claim();service.lost=True;receipt=hub.publish(snapshot,release=True);assert receipt['files']==46 and receipt['owner_released'] and receipt['zip_uploads']==0
 for name,h in read(ROOT/'revision2_validation/DELIVERY.json')['sources'].items():assert sha(ROOT/name)==h
 for name,key in [('phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb','executed_notebook_sha256'),('phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb','executed_notebook_sha256')]:
  location=ROOT/'revision2_validation'/('nb05_publication/NOTEBOOK_INSPECTION.json' if 'NB05_' in name else 'nb06_publication/VERIFICATION.json')
  key='sha256' if 'NB05_' in name else key;assert sha(ROOT/'notebooks'/name)==read(location)[key]
 protected=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
 result=dict(status='passed',run_id=latest['run_id'],CSV_tables=22,CSV_rows=summary['csv_rows'],source_images=570,existing_tyres=12,human_point_records=720,weak_point_records=912,
  saved_training_occurrences=47040,raw_disagreement_images=45,unignored_disagreement_images=0,prospective_training_ready=False,unit_checks=read(OUT/'UNIT_TESTS.json')['check_count'],
  NB07_sha256=sha(notebook),embedded_files=42,fresh_CPU_stages=stages,fresh_recomputed_images=4,all_CSV_and_HTML_byte_equal=True,CUDA_disabled=True,network_disabled=True,
  JS_syntax_valid=True,embedded_previews_verified=570,browser_render_verified=False,fixture_publication_files=46,live_HF_publication_tested=False,Kaggle_executed=False,
  original_Markdown_preserved=len(protected),frozen_sources_and_executed_notebooks_preserved=True,new_training=False,new_inference=False,new_independent_tyres=0,fresh_workspace=str(fresh))
 atomic(OUT/'VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
