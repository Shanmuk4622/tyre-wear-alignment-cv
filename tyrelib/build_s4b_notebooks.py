"""Build only the S4b training/report pair; preserve completed notebooks."""
from pathlib import Path
import build_notebooks as b
import build_closure_notebooks as c

COMMON = c.PULL[:c.PULL.index('REVISION = hf_read')] + r'''
import yaml
SOURCE_ROOT = Path(sess.stage_dir)/'s4b_source'
def source_file(rel, revision=SOURCE_REVISION):
    return Path(hf_read(rel,lambda:hf_hub_download(REPO,rel,repo_type='dataset',revision=revision,
        token=READ_TOKEN,local_dir=str(SOURCE_ROOT/revision))))
def source_csv(rel,revision=SOURCE_REVISION):
    return pd.read_csv(source_file(rel,revision))
selection=source_csv('tables/stage_b_selection.csv')
effects=source_csv('tables/stage_b_effects.csv')
PLAN=make_plan(selection,effects)
HASH=plan_hash(PLAN)
PREFIX=f'confirmations/{REVISION}/{HASH}'
OUT=Path(sess.stage_dir)/'s4b'/HASH
OUT.mkdir(parents=True,exist_ok=True)
cfgs=configs(sess,tl,PLAN)
run_ids=[cfg['run_id'] for cfg in cfgs]
base_ids=[f'a-{arch}-base-f1-s{seed}' for arch in PLAN['architectures'] for seed in PLAN['seeds']]
baseline=pd.concat([source_csv(f'runs/{rid}/metrics/final.csv') for rid in base_ids],ignore_index=True)
assert len(baseline)==6 and baseline.run_id.is_unique
assert baseline.status.eq('completed').all() and baseline.epochs_trained.eq(60).all()
assert baseline.epochs_planned.eq(60).all()
for rid in base_ids:
    old=yaml.safe_load(source_file(f'runs/{rid}/config.yaml').read_text())
    expected=sess.config(old['arch'],1,int(old['seed']),stage='a')
    differences={k:(old.get(k),expected[k]) for k in tl.RECIPE if k!='num_workers' and old.get(k)!=expected[k]}
    assert not differences, f'Baseline recipe changed: {rid} {differences}; do not train a mismatched comparison'
print('Frozen S4b plan:',PLAN)
print('New runs:',len(run_ids),'reused baselines:',len(base_ids))
print('No masks or new annotations required; fold 1 only.')
'''

SMOKE = r'''import sys,time,json,statistics,torch,tyrelib as tl
arch,bs,expected=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
assert torch.cuda.device_count()>=2
fmt=torch.channels_last
torch.manual_seed(4622)
model=tl.build_model(arch,3,pretrained=False,head='coral',img_size=384)
assert sum(p.numel() for p in model.parameters())==expected,'Architecture parameter count differs from published baseline'
model=model.cuda().to(memory_format=fmt).train()
single=bool(int(sys.argv[4]))  # passed from the actual training config, never a separate model list
if not single:
    model=torch.nn.DataParallel(model)
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
scaler=tl._grad_scaler(torch.device('cuda'))
x=torch.randn(bs,3,384,384,device='cuda').to(memory_format=fmt)
y=torch.arange(bs,device='cuda')%3
durations=[]
for step in range(8):
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); started=time.perf_counter()
    with tl._autocast(torch.device('cuda')):
        loss=tl.CoralHead.loss(model(x),y)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    scaler.step(opt);scaler.update();torch.cuda.synchronize()
    durations.append(time.perf_counter()-started)
median=statistics.median(durations[3:])
print(json.dumps(dict(arch=arch,batch=bs,parameters=expected,gpus_used=1 if single else 2,
    step_seconds=durations,warm_median_seconds=median,torch=torch.__version__)),flush=True)
assert median < 4.0, 'Runtime too slow (>4s/step): stop before spending hours, inspect preflight log'
print('PASS',arch,'timed training preflight')
'''

TRAIN = r'''import torch,sys,subprocess
assert torch.cuda.device_count()>=2, 'Select Kaggle GPU T4 x2'
assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)), 'This notebook is validated for dual T4; do not silently change the runtime'
# Publish the selection rule BEFORE claiming any experiment. Never use new results to re-rank.
plan_path=OUT/f'protocol_{ACCOUNT}.json'
plan_path.write_text(json.dumps(PLAN,indent=2))
sess.uploader.enqueue(plan_path,f'{PREFIX}/workers/{plan_path.name}')
assert sess.push_now('S4b frozen protocol'), 'HF publication failed; do not start unpublished experiments'
print(sess.reconcile(run_ids).to_string(index=False))
for arch in PLAN['architectures']:
    mine=[rid for rid in run_ids if f'-{arch}-' in rid and sess.inventory.state(rid)!='completed']
    if not mine:
        continue
    expected=int(baseline.loc[baseline.arch.eq(arch),'n_params_total'].iloc[0])
    runtime=next(cfg for cfg in cfgs if cfg['arch']==arch)
    p=subprocess.run([sys.executable,'-c',SMOKE,arch,str(runtime['batch_size']),str(expected),str(int(runtime['_single_gpu']))],
        capture_output=True,text=True,timeout=900,cwd=str(Path.cwd()))
    log=OUT/f'preflight_{arch}_{ACCOUNT}.txt'
    log.write_text(p.stdout+'\n'+p.stderr)
    sess.uploader.enqueue(log,f'{PREFIX}/workers/{log.name}')
    assert sess.push_now(f'S4b dual-T4 preflight {arch}'), 'Preflight upload failed'
    print(log.read_text())
    assert p.returncode==0, 'Preflight failed before claims; no model/batch reduction was made'
try:
    summaries=sess.run_all(cfgs,title='S4b two-architecture confirmation',isolate_runs=True,
        steal_stale=False,takeover_when_idle=True)
finally:
    sess.push_now('S4b training stopped or major cell complete')
assert sess.finish(), 'Pending HF upload; retry final flush before leaving'
sess.confirm_on_hf(run_ids)
print('When all 18 runs are FINISHED, run NB12R once for the joint report. One worker finishing is not all-team completion.')
'''

REPORT = r'''assert NUM_WORKERS==1,'Run exactly one reporting copy'
live=hf_read('current HF revision',lambda:HfApi().repo_info(REPO,repo_type='dataset',token=READ_TOKEN)).sha
files=set(hf_read('completion inventory',lambda:HfApi().list_repo_files(REPO,repo_type='dataset',revision=live,token=READ_TOKEN)))
rows,coverage=[],[]
for cfg in cfgs:
    rid=cfg['run_id']; p=f'runs/{rid}'
    required=[p+'/STATUS.json',p+'/metrics/final.csv',p+'/metrics/epochs.csv',
              p+'/checkpoints/ckpt_best.pt',p+'/checkpoints/ckpt_last.pt',p+'/config.yaml']
    if not set(required)<=files:
        coverage.append(dict(run_id=rid,verified=False,reason='missing required artifacts'))
        continue
    status=json.loads(source_file(p+'/STATUS.json',live).read_text())
    final=source_csv(p+'/metrics/final.csv',live)
    history=source_csv(p+'/metrics/epochs.csv',live)
    saved=yaml.safe_load(source_file(p+'/config.yaml',live).read_text())
    recipe_ok=all(saved.get(k)==cfg[k] for k in tl.RECIPE if k!='num_workers')
    identity_ok=all(saved.get(k)==cfg[k] for k in ('arch','fold','seed','stage','technique','run_id'))
    valid=(len(final)==1 and status.get('status')=='completed' and status.get('epoch')==60
           and list(history.epoch)==list(range(1,61)) and recipe_ok and identity_ok)
    if valid:
        row=final.iloc[0]
        valid=(row.status=='completed' and row.epochs_trained==60 and row.epochs_planned==60
            and all(row[k]==cfg[k] for k in ('arch','fold','seed','stage','technique','run_id'))
            and np.isclose(row.final_val_f1_macro,history.iloc[-1].val_f1_macro)
            and np.isfinite([row.best_val_f1_macro,row.final_val_f1_macro]).all())
        if valid:
            # The primary is F1 at the recorded best-QWK epoch, NOT max F1 over epochs.
            best=history.loc[history.epoch.eq(int(row.best_epoch))]
            valid=len(best)==1 and np.isclose(best.iloc[0].val_f1_macro,row.best_val_f1_macro)
    coverage.append(dict(run_id=rid,verified=bool(valid),reason='complete' if valid else 'status/history/config/metric disagreement'))
    if valid: rows.append(final.iloc[0].to_dict())
pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
result=dict(revision=REVISION,plan_hash=HASH,verified_runs=len(rows),expected_runs=18,
    verification_revision=live,baseline_revision=SOURCE_REVISION,
    status='complete' if len(rows)==18 else 'incomplete',
    limitation=PLAN['claim'],checkpoint_verification='inventory presence, not tensor loading')
publish_names=['coverage.csv','STATUS.json','protocol.json']
if len(rows)==18:
    paired,summary=analyse(PLAN,pd.DataFrame(rows),baseline)
    paired.to_csv(OUT/'paired_effects.csv',index=False)
    summary.to_csv(OUT/'architecture_effects.csv',index=False)
    decisions=summary.groupby('factor').agg(architectures=('arch','nunique'),same_direction_architectures=('same_direction','sum'))
    decisions['description']=np.where(decisions.same_direction_architectures.eq(2),
        'same direction in both additional architectures; descriptive only',
        'does not reproduce the discovery direction in both architectures')
    decisions.to_csv(OUT/'factor_decisions.csv')
    publish_names += ['paired_effects.csv','architecture_effects.csv','factor_decisions.csv']
    print(summary.to_string(index=False));print(decisions.to_string())
else:
    print(pd.DataFrame(coverage).to_string(index=False))
    print('Not complete: no partial averages or positive confirmation claims are published.')
(OUT/'protocol.json').write_text(json.dumps(PLAN,indent=2))
(OUT/'STATUS.json').write_text(json.dumps(result,indent=2))
for name in publish_names:
    sess.uploader.enqueue(OUT/name,f'{PREFIX}/{name}')
assert sess.finish(),'Final upload failed; rerun report'
verified=hf_read('verify report',lambda:HfApi().repo_info(REPO,repo_type='dataset',token=READ_TOKEN)).sha
remote_result=json.loads(source_file(f'{PREFIX}/STATUS.json',verified).read_text())
assert remote_result==result
print('HF-VERIFIED REPORT',verified,result)
'''

def build():
    helper=Path(__file__).with_name('s4b_confirmation.py').read_text()
    training=c.start('''# NB12 — S4b: confirmation on two additional architectures

18 NEW jobs: ConvNeXt-V2 Tiny and MobileNetV4 × 3 factors × seeds 1/2/3,
fold 1 only, 60 epochs each. Six existing Stage-A baselines are reused.
No annotation work, no NB11, no SAM2, no change to existing models.

**Runtime repair 2026-09-10-r2:** BOTH architectures use GPU 0 only, matching
their faster Stage-A execution: ConvNeXt batch 32, MobileNet batch 64. The four existing
checkpoints resume under their original IDs; scientific configs/hashes unchanged.
Preflight now times eight training steps and refuses a warmed median >=4s.
An actual epoch over 10 minutes also saves/pauses for investigation, not eight hours of repeated slow epochs.
Stop all old copies before uploading this repair and keep the same account labels.

Factors are the three highest signed mean effects in the original NB06 report:
class-weighted sampling, random initialisation, uniform sampling. Only the first
was positive in discovery. Confirm the directions honestly, not three claimed gains.
Selection uses the frozen HF revision, never the new confirmation results.
This is a declared extension because NB06 already covered three architectures;
it is not a blind test on new data or proof of significance.

Run instructions: Internet ON; HF_TOKEN secret; attach Tire Dataset Prepared;
GPU T4 x2. Run All. For FOUR accounts, set the same ACTIVE_KAGGLE_ACCOUNTS
tuple to ('acct1','acct2','acct3','acct4') in every copy and set ACCOUNT per copy.
Default is one copy. Run all copies with this identical notebook/protocol.
Do not change model, batch, seed, epoch count or factor settings.

Models run in isolated child processes to release GPU/host memory. Scratch and
checkpoints use /kaggle/temp; monitor actual disk, do not assume 1 TB is available.
HF batches ordinary results every 30 minutes, plus major completion and catchable
Stop; work-takeover coordination can require a separate small commit. Hard OS
kills cannot flush. Fresh sessions resume the last published COMPLETE EPOCH,
not the exact interrupted batch. Strict resume refuses corrupt/mismatched checkpoints.
Do not run NB12R until all 18 jobs are FINISHED. GPU preflight runs before claims.
''','c',True)
    training += [b.code(helper),b.code(COMMON),b.code('SMOKE = '+repr(SMOKE)),b.code(TRAIN)]
    c.save('NB12_S4B_Confirmation.ipynb',training)
    report=c.start('''# NB12R — S4b completion audit and paired report

Run ONCE after NB12's 18 jobs finish. CPU sufficient; Internet and HF_TOKEN.
No dataset attachment or training required. Reads live per-run HF records,
checks all 60 epochs/configs/metrics and checkpoint paths, then reports effects
against the six frozen Stage-A baselines. Incomplete runs produce an incomplete
status, not confirmation claims. Primary matches NB06's best-QWK-selected F1;
final-epoch differences are separately reported. A negative effect is not a gain.
Same-direction results are descriptive, not statistical significance/new-tyre evidence.
''','creport',False)
    report += [b.code(helper),b.code(COMMON),b.code(REPORT)]
    c.save('NB12R_S4B_Report.ipynb',report)

if __name__=='__main__':
    build()
