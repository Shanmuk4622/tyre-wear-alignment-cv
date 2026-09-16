"""Self-contained NB26-29; archived executed files are never discarded."""
import ast,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
setup='''import sys, subprocess, importlib.util, base64, json, os
from pathlib import Path
WORK=Path('/kaggle/working/hrnet_geometry');WORK.mkdir(exist_ok=True)
if importlib.util.find_spec('huggingface_hub') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','huggingface_hub>=0.36,<2'])
'''
for n in ['hrnet_protocol.py','hrnet_runtime.py','s9_expansion.py','s9_pilot.py']:
    setup+=f"(WORK/{n!r}).write_bytes(base64.b64decode({base64.b64encode((ROOT/'tyrelib'/n).read_bytes()).decode()!r}))\n"
setup+='''sys.path.insert(0,str(WORK))
import importlib, hrnet_protocol as hp
hp=importlib.reload(hp)
from kaggle_secrets import UserSecretsClient
TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
if not TOKEN: raise RuntimeError('Enable HF_TOKEN in Kaggle secrets')
from IPython.display import FileLink, display
'''
pre='''DATA_ROOT = ''  # Auto-detect ONE Tire Dataset Prepared dataset, or supply its FINAL path.
ROOT_DATA=hp.root_data(DATA_ROOT)
FOLDER=hp.preflight(WORK,ROOT_DATA,TOKEN)
display(FileLink(str(FOLDER/'QA.json')))
display(FileLink(str(FOLDER/'CONTRACT.json')))
print('120 original-image overlays are in this folder. Next: NB27 on T4; no more annotation requested.')
'''
def gpu(mode):
    return '''DATA_ROOT = ''  # Attach the same prepared dataset. No manual label upload required.
ROOT_DATA=hp.root_data(DATA_ROOT)
# Install only timm into a task-specific folder; never replace Kaggle torch/torchvision.
DEPS=WORK/'deps'
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps','--target',str(DEPS),'timm==1.0.15'])
import signal
env=os.environ.copy();env['HF_TOKEN']=TOKEN
env['PYTHONPATH']=str(DEPS)+os.pathsep+str(WORK)
env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['OMP_NUM_THREADS']='2'
child=subprocess.Popen([sys.executable,'-u',str(WORK/'hrnet_runtime.py'),'''+repr(mode)+''',
    '--work',str(WORK),'--root',str(ROOT_DATA)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
del env
try:
    for line in child.stdout:print(line,end='',flush=True)
    code=child.wait()
except KeyboardInterrupt:
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        print('Stop requested. Waiting for safe checkpoint/upload; do not force-kill the kernel.')
        for line in child.stdout:print(line,end='',flush=True)
        child.wait()
    raise
if code:raise RuntimeError('Run did not complete. Keep output and send the error; durable checkpoints are retained.')
print('Check the printed status: resumable means rerun this notebook; completed means the stage finished.')
'''
report='''# No dataset attachment required for reporting. Reads only small status/metric files.
from huggingface_hub import HfApi,hf_hub_download
cfg=hp.contract(WORK);protocol=hp.sha(hp.canonical(cfg));prefix=hp.prefix(cfg)
rev=HfApi(token=TOKEN).repo_info(hp.REPO,repo_type='dataset').sha
OUT=WORK/'report'/protocol;OUT.mkdir(parents=True,exist_ok=True)
def fetch(path):
    f=hp.retry(lambda:hf_hub_download(hp.REPO,path,repo_type='dataset',revision=rev,token=TOKEN))
    if Path(f).stat().st_size>10*1024**2:raise ValueError('Unexpectedly large report artifact')
    return hp.read_json(f)
train=[r for r in cfg['rows'] if r['role']=='train'];test={r['pilot_id']:r for r in cfg['rows'] if r['role']=='test'}
mean=[sum(r['x'][i] for r in train)/len(train) for i in range(6)]
baseline=sum(abs(mean[i]-r['x'][i]) for r in test.values() for i in range(6))/(6*len(test))
results=[]
for seed in cfg['seeds']:
    remote=prefix+f'/runs/seed{seed}'
    st=fetch(remote+'/STATUS.json')
    assert st['status']=='completed' and st['completed_epochs']==60 and st['protocol']==protocol and st['seed']==seed
    metric=fetch(remote+'/TEST_FINAL.json');rows=metric['records']
    assert len(rows)==len(test)*6 and {(r['pilot_id'],r['point']) for r in rows}=={(pid,n) for pid in test for n in hp.POINTS}
    errors=[]
    for r in rows:
        target=test[r['pilot_id']]['x'][hp.POINTS.index(r['point'])]
        assert r['label']==target and r['tyre']==test[r['pilot_id']]['session']
        error=abs(r['prediction']-target);assert abs(error-r['error_width_fraction'])<1e-9
        errors.append(error)
    score=sum(errors)/len(errors);assert abs(score-metric['mean_width_error'])<1e-9
    repair=fetch(remote+'/RUNTIME_REPAIR.json')
    results.append(dict(seed=seed,mean_width_error=score,train_mean_baseline=baseline,
        runtime_repair_revision=repair['revision'],runtime_repair_sha256=repair['source_sha256'],
        per_tyre={t:sum(abs(r['prediction']-r['label']) for r in rows if r['tyre']==t)/sum(r['tyre']==t for r in rows) for t in cfg['groups']['test']}))
report=dict(source_revision=rev,protocol=protocol,results=results,
    claim='HRNet versus train-only constant-coordinate baseline; NOT a matched SegFormer comparison',
    full_s9_complete=False,limitations=cfg['limitations'])
hp.write(OUT/'REPORT.json',report);hp.write(OUT/'CONTRACT.json',cfg)
hp.publish(OUT,prefix+'/report',TOKEN)
display(FileLink(str(OUT/'REPORT.json')));print(json.dumps(report,indent=2))
'''
common='''
User confirms 12 sessions are 12 different tyres. Identity source is recorded as
user-confirmed, not independently verified. No new labels requested. All 720
submitted points are visible: this is coordinate-only HRNet-W18, not learned
visibility, healthy-reference detection, or physical-angle estimation.
Dataset images come from attached Kaggle input; annotation JSON is pinned on HF.
No four-worker execution: use ONE copy. Existing notebooks/labels are untouched.
'''
items=[('NB26_HRNet_Preflight.ipynb','''# NB26 — HRNet preflight and frozen tyre split
**CPU · ONE copy · Internet ON · HF_TOKEN · attach Tire Dataset Prepared · Run All.**
Verifies 120 image hashes and annotations, records user-confirmed tyre identities,
locks 8/2/2 tyre groups using image-count balance and fixed hash tie-break only.
Produces all 120 point overlays and QA flags. Mechanical checks do not certify
label accuracy. No GPU/model download. Publishes one preflight commit. Next NB27.
''',pre,False),('NB27_HRNet_Smoke_Resume.ipynb','''# NB27 — HRNet-W18 GPU and resume smoke test
**T4 GPU · ONE copy · Internet ON · HF_TOKEN · prepared dataset · Run All.**
Run NB26 first. A dual-T4 session is fine: only cuda:0 is used to keep memory bounded.
Downloads only the pinned HRNet-W18 pretrained weights, never the dataset again.
Tests four training steps on two TRAIN images and compares uninterrupted versus
checkpoint-reloaded continuation, with 1e-5 tolerance. Publishes pass/fail, hardware,
model identity and loss evidence. This is not an overfit/convergence study.
Only a passing smoke test unlocks NB28. No long training or test-set use here.
''',gpu('smoke'),True),('NB28_HRNet_Training.ipynb','''# NB28 — resumable HRNet-W18 geometry training
**T4 GPU · ONE copy · Internet ON · HF_TOKEN · prepared dataset · Run All.**
Run NB26 then NB27 first. Uses only cuda:0 even when two T4s are available.
Three seeds run sequentially, 60 epochs each, batch 2, AMP, frozen BatchNorm
statistics, no augmentation. Fixed final-epoch endpoint; development metrics
are logged but do not select checkpoints. Test labels are used only for evaluation.
Local checkpoint after EVERY completed optimizer step includes model, AdamW,
scaler, scheduler, random states, deterministic sampler cursor and full history.
Uploads synchronous atomic batches every 30 minutes, each seed completion and
catchable Stop/error. No per-epoch/claim commits. Stop waits for the safe step.
Forced kills can lose work since the last HF snapshot. Rerun to restore durable
state; seeds already trained resume at epoch 60 and only re-evaluate/re-publish.
Gracefully stops after about 9 hours; rerun the same notebook for remaining work.
Keep one writer per seed. Uses bounded checkpoints, never the whole HF dataset.
After all three seeds finish, NB29 audits/report results. Matched segmentation
comparison requires a separate same-split baseline; this does not complete S9.
''',gpu('train'),True),('NB29_HRNet_Report.ipynb','''# NB29 — HRNet result audit and report
**CPU · ONE copy · Internet ON · HF_TOKEN · attach nothing · Run All.**
Run after NB28 completes all three seeds. Verifies epoch-60 statuses, test-image
coverage and recalculates point errors. Reports per-tyre/seed metrics and a simple
constant-coordinate baseline fitted only on training labels. Downloads no weights.
This is NOT a matched HRNet-versus-SegFormer result or proof of full S9 completion.
Publishes one small result report to HF. Send executed notebooks for verification.
''',report,False)]
for name,title,run,isgpu in items:
    if '--only' in sys.argv and name!=sys.argv[sys.argv.index('--only')+1]:continue
    stage_setup=setup
    if name=='NB28_HRNet_Training.ipynb':
        repair=(ROOT/'tyrelib/hrnet_amp_repair.py').read_bytes()
        stage_setup+=f"(WORK/'hrnet_amp_repair.py').write_bytes(base64.b64decode({base64.b64encode(repair).decode()!r}))\n"
        run=run.replace("str(WORK/'hrnet_runtime.py'),'train',","str(WORK/'hrnet_amp_repair.py'),")
        title+='''
**Numerical repair:** resumes the existing seed-1 checkpoint (18 completed batches
at the last verified audit), with unchanged model/split/checkpoint namespace.
Runs an injected-overflow GPU regression first, then retries nonfinite AMP batches
at lower loss scales, with bounded FP32 fallback. Never skips a batch or advances
the cursor without a successful optimizer update. Persistent FP32 nonfinites still
stop safely. Repair source/hash/events are published alongside every snapshot.
No NB26/NB27 rerun is needed; their original smoke scope is preserved explicitly.
'''
    cells=[]
    for i,(kind,text) in enumerate([('markdown',title+common),('code',stage_setup),('code',run)]):
        c=dict(cell_type=kind,id=f'hrnet-{i}',metadata={},source=text.splitlines(True))
        if kind=='code':ast.parse(text);c.update(execution_count=None,outputs=[])
        cells.append(c)
    dest=ROOT/'notebooks'/name
    if dest.exists():
        raw=dest.read_bytes()
        if any(c.get('outputs') for c in json.loads(raw)['cells']):
            archive=dest.parent/'execution_archives';archive.mkdir(exist_ok=True)
            saved=archive/(dest.stem+'_'+hashlib.sha256(raw).hexdigest()[:12]+'.ipynb')
            if not saved.exists():saved.write_bytes(raw)
    dest.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python'),kaggle=dict(isGpuEnabled=isgpu,isInternetEnabled=True))),indent=1),encoding='utf-8')
    print(name)
