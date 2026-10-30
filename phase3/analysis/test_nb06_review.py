"""Fresh CPU-only/no-network review from the actual NB06 payload and pinned cache."""
from PIL import Image
import ast,base64,gzip,io,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from transactions import read,sha,atomic
from publisher_r2 import Publisher
from test_training_evidence import Service
PUB=ROOT/'revision2_validation/nb05_publication'

def main():
 delivery=read(ROOT/'revision2_validation/nb06_delivery/DELIVERY.json');notebook=ROOT/'notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb'
 assert sha(notebook)==delivery['notebook_sha256'];nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb);files={}
 for c in nb.cells:
  if c.cell_type!='code':continue
  assert not c.outputs and c.execution_count is None
  for node in ast.walk(ast.parse(c.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(node.args[0].value)));files.update({n:base64.b64decode(v) for n,v in payload.items()})
 assert len(files)==36 and {n:__import__('hashlib').sha256(b).hexdigest() for n,b in files.items()}==delivery['embedded_files']
 assert all((ROOT/n).read_bytes()==b for n,b in files.items())
 validation=ROOT/'revision2_validation/nb06_validation';validation.mkdir(exist_ok=True);work=Path(tempfile.mkdtemp(prefix='fresh_cpu_',dir=validation)).resolve()
 assert work.is_relative_to(validation.resolve())
 for n,b in files.items():
  p=work/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 cache=work/'revision2_validation/nb05_publication';cache.mkdir(parents=True)
 def hardlink(src,dst):
  try:os.link(src,dst)
  except OSError:shutil.copy2(src,dst)
 for n in ('remote','objects'):shutil.copytree(PUB/n,cache/n,copy_function=hardlink)
 for n in ('INVENTORY.json','PREFLIGHT_VERIFICATION.json'):shutil.copy2(PUB/n,cache/n)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1');env.pop('HF_TOKEN',None)
 stages=[]
 for script,args in [('verify_nb05_publication.py',[]),('audit_nb05_results.py',[]),('render_nb05_review.py',[]),('publish_nb05_review.py',['--prepare-only'])]:
  harness=work/'harness.py'
  harness.write_text('import sys,runpy\nfrom pathlib import Path\nfrom unittest.mock import patch\n'
   +f'sys.path.insert(0,{str(work/"analysis")!r})\nsys.path.insert(0,{str(work/"revision2")!r})\n'
   +'import spec,torch\nassert not torch.cuda.is_available()\n'
   +f'spec.locate=lambda value="":Path({str(spec.locate())!r})\nsys.argv={[script]+args!r}\n'
   +f'with patch("requests.sessions.Session.request",side_effect=AssertionError("Network disabled in fresh CPU cache test")):\n    runpy.run_path({str(work/"analysis"/script)!r},run_name="__main__")\n',encoding='utf-8')
  with open(work/(script+'.log'),'w',encoding='utf-8') as log:
   process=subprocess.run([sys.executable,'-B',str(harness)],env=env,stdout=log,stderr=subprocess.STDOUT)
  assert process.returncode==0,(script,(work/(script+'.log')).read_text(encoding='utf-8')[-6000:])
  stages.append(script);print('Fresh CPU stage passed:',script,flush=True)
 assert all(sha(p)==sha(cache/'analysis'/p.name) for p in (PUB/'analysis').glob('*.csv'))
 assert sha(PUB/'analysis/TREAD_ONE_NB05_Comparison_Explorer.html')==sha(cache/'analysis/TREAD_ONE_NB05_Comparison_Explorer.html')
 assert all(sha(p)==sha(cache/'analysis/figures'/p.name) for p in (PUB/'analysis/figures').glob('*.png'))
 html=(PUB/'analysis/TREAD_ONE_NB05_Comparison_Explorer.html').read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
 syntax=validation/'explorer_syntax.js';syntax.write_text(script,encoding='utf-8')
 checked=subprocess.run([shutil.which('node'),'--check',str(syntax)],capture_output=True,text=True);assert checked.returncode==0,checked.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0];assert len(data['records'])==412 and len(data['images'])==103
 count=0
 for source in data['images'].values():
  for name in ('image','truth_masks'):
   with Image.open(io.BytesIO(base64.b64decode(source[name].split(',',1)[1]))) as image:assert image.size==(384,512);image.verify();count+=1
 for record in data['records']:
  for uri,shape in [(record['masks'],(384,512))]+[(h,(96,128)) for h in record['heats']]:
   with Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))) as image:assert image.size==shape;image.verify();count+=1
 assert count==1854
 bundle=read(PUB/'REVIEW_DELIVERY.json');snapshot=Path(bundle['path']);fresh_bundle=read(cache/'REVIEW_DELIVERY.json')
 assert fresh_bundle['contract']==bundle['contract'] and fresh_bundle['files']==35
 for n,v in read(snapshot/'MANIFEST.json')['files'].items():assert sha(snapshot/n)==v['sha256']
 service=Service();hub=Publisher.__new__(Publisher);hub.destination='offline-nb06-review';hub.owner='review-writer';hub.api=service;hub.session=service;hub.calls=0;hub.last_verified=None;hub.log=work/'publication_fixture.jsonl'
 hub.claim();service.lost=True;receipt=hub.publish(snapshot,release=True)
 assert receipt['files']==35 and receipt['owner_released'] and receipt['zip_uploads']==0
 manifest=json.loads(service.files[hub.destination+'/MANIFEST.json']);assert len(manifest['files'])==34
 assert sum(n.endswith('.csv') for n in manifest['files'])==10
 result=dict(status='passed',notebook_sha256=sha(notebook),embedded_files=36,fresh_CPU_stages=stages,
  CUDA_disabled=True,network_disabled_for_fresh_stages=True,dataset_locator='Explicit existing read-only prepared data substituted for Kaggle input discovery in the local harness',
  derived_CSV_tables_exact=10,figures_and_explorer_byte_identical=True,JS_syntax_valid=True,embedded_images_verified=count,
  fixture_publication_files=35,fixture_ambiguous_commit_verified=True,live_review_publication_tested=False,Kaggle_executed=False,
  source_NB05_preserved=sha(ROOT/'notebooks/phase3_NB05_TREAD_ONE_Shared_Gradient_Paired_Pilot.ipynb')==delivery['source_NB05_sha256'],
  optimizer_updates=0,new_inference=False,fresh_workspace=str(work),browser_render_verified=False)
 atomic(validation/'VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
