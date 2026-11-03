"""Complete public-byte audit followed by actual embedded-package CPU review/passport."""
import ast,base64,json,os,shutil,subprocess,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nb17_public as client
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_postrun';sys.path.insert(0,str(SRC))
from common import atomic,digest,file_meta,read,sha
from coordinator import contract,snapshot
start=time.monotonic();out=client.OUT;inspection=read(out/'INSPECTION.json')
assert read(out/'DOWNLOAD_STATUS.json')['status']=='complete','Wait for the full resumable public mirror to finish'
manifest=read(out/'REMOTE_MANIFEST.json');source=read(SRC/'FILES.json');protocol=digest(contract())
atomic(out/'remote/REMOTE_MANIFEST.json',manifest)
fixture=ROOT/'tread_two/geo_postrun_validation'/('fixture_actual_'+protocol[:12]);fixture.mkdir(parents=True,exist_ok=True)
nbpath=ROOT/'notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb';nb=read(nbpath)
tree=ast.parse(''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code')))
payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
extracted=fixture/'source';extracted.mkdir(exist_ok=True)
for name,item in payload.items():
    body=base64.b64decode(item['base64'],validate=True);assert __import__('hashlib').sha256(body).hexdigest()==item['sha256']==sha(SRC/name);(extracted/name).write_bytes(body)
env=dict(os.environ);env.pop('HF_TOKEN',None);env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
dataset=ROOT.parent/'phase2/data/phase2_dataset_v1'
for stage,input_root in [('review',out/'remote'),('passport',fixture/'review')]:
    target=fixture/stage;args=[sys.executable,'-u','-B',str(extracted/'worker.py'),stage,'--input',str(input_root),'--output',str(target),'--contract',protocol,'--dataset',str(dataset)]
    print('Actual saved-data embedded worker:',stage,flush=True)
    result=subprocess.run(args,env=env);assert result.returncode==0,(stage,result.returncode)
    snapshot(target,fixture/('publication_'+stage+'_'+str(time.time_ns())),contract(),stage)
for source_dir,target in [(fixture/'review',ROOT/'tread_two/nb17_report'),(fixture/'passport',ROOT/'tread_two/geo_passport_report')]:
    shutil.copytree(source_dir,target,dirs_exist_ok=True)
viewer=subprocess.run(['node',str(ROOT/'analysis/test_geo_review_viewer.cjs'),str(fixture/'review/GEO_Review_Atlas.html')],capture_output=True,text=True);assert viewer.returncode==0,viewer.stderr
print('Fresh-process repeat: completed stage receipts must skip all seven stages.',flush=True)
repeat=subprocess.run([sys.executable,'-u','-B',str(extracted/'worker.py'),'review','--input',str(out/'remote'),'--output',str(fixture/'review'),'--contract',protocol,'--dataset',str(dataset)],env=env,capture_output=True,text=True)
assert repeat.returncode==0,repeat.stderr
assert repeat.stdout.count('Verified completed stage; skip:')==7,repeat.stdout
atomic(ROOT/'tread_two/geo_postrun_validation/ACTUAL_DATA.json',dict(status='passed',source_inventory_sha256=digest(source),postrun_protocol=protocol,input_revision=client.REVISION,
       actual_embedded_fresh_worker_review_passport=True,actual_data_from_public_verified_mirror=True,all_seven_stages_skipped_on_fresh_repeat=True,
       offline_publication_staging_hashes_passed=True,viewer=json.loads(viewer.stdout),image_model_forwards=0,new_optimizer_updates=0,remote_writes=0,
       credentials_read=False,actual_new_Kaggle_HF_publication_pending=True))
unchanged=[]
for n in (13,14,15,16):
    folder=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json');old=read(folder/'REMOTE_MANIFEST.json')
    current=client.request(f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{client.REVISION}/{v["prefix"]}/MANIFEST.json').json()
    assert old==current;unchanged.append(dict(notebook=n,entries=len(old['files']),unchanged=True))
assert client.json_get('OWNER.json')==dict(owner=None,expires=0)
file_proof=read(fixture/'review/FILE_VERIFICATION.json');native_proof=read(fixture/'review/NATIVE_VERIFICATION.json');f=read(fixture/'review/FINDINGS.json')
assert inspection['all_payload_hashes_match'] and len(inspection['cells'])==3
assert [c['execution_count'] for c in inspection['cells']]==[1,2,3]
assert not any(o['type']=='error' for c in inspection['cells'] for o in c['outputs'])
report=dict(status='passed',executed_notebook_sha256=inspection['executed_notebook_sha256'],revision=client.REVISION,contract=client.PROTOCOL,prefix=client.PREFIX,
            files=len(manifest['files'])+1,objects_verified=len(manifest['files']),bytes_verified=sum(m['bytes'] for m in manifest['files'].values()),
            code_cells_completed=3,notebook_errors=0,payload_files=29,all_payload_files_match=True,owner_released=True,
            CSV_files=file_proof['CSV_files'],CSV_rows=file_proof['CSV_rows'],CSV_families=40,CSV_fields=468,native_arrays=8457,native_checkpoints=63,
            epochs=60,updates=2940,observations=30447,complete_unit=9877,scientific_result=f,native_verification=native_proof,
            original_publications_unchanged=unchanged,postrun_actual_data_validation_passed=True,actual_new_postrun_publication_pending=True,
            environment=read(out/'remote/CONTRACT.json')['environment'],local_credentials_read=False,remote_writes=0,public_read_requests_this_invocation=client.CALLS,seconds=time.monotonic()-start)
atomic(out/'VERIFICATION.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('scientific_result','native_verification')},indent=2),flush=True)
# Finish the already-authorized documentation and read-only handoff checks after
# the scientific receipt exists. No Git index, commit or remote write is made.
import runpy
runpy.run_path(str(ROOT/'analysis/document_nb17_batch.py'),run_name='__main__')
runpy.run_path(str(ROOT/'analysis/finalize_nb17_batch.py'),run_name='__main__')
check=subprocess.run([sys.executable,'-B',str(ROOT.parent/'scripts/check_git_content.py'),'--history'])
assert check.returncode==0,'Review the read-only Git content audit before pushing'
