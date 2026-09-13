"""CPU-only evidence packaging and descriptive reporting; no new model training."""
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import time
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO='Shanmuk4622/tyre-wear-study'
SOURCE='35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b'
S5='s5/s5-manual-2026-09-10-r1/1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df/report'
S9='s9/exploratory-fusion-r1/20dca3a2d91884230d555a7a36dabdbab8a075776487f3d147f1d4f9ddd21b87'
S4B='confirmations/s4b-2026-09-09-r1/9f2e32c8c372c1c985d7827a893c6be0bd7c6801019cd4b5456f3f2913bd0e72'
LEGACY_FIGURES=['fig01_accuracy_vs_ter','fig02_per_fold','fig03_saliency_panels','fig04_stress_matrix',
 'fig05_h1_stability','fig06_ofat_effects','fig07_faithfulness','fig08_best_epoch','fig09_energy','fig10_per_session']
GAPS=[('Full S9 HRNet/PatchCore','Deferred by user; not executed'),
 ('SAM2 comparison / blind consistency','Deferred; existing masks are not independent repeat labels'),
 ('H2','Inconclusive/undefined in the implemented intervention evidence'),
 ('H3','Untested; preregistered fine-grained arms absent'),
 ('Video temporal consistency','Not measured; no suitable video evidence'),
 ('Resolution x ROI factorial interaction','Not established by separate OFAT arms'),
 ('Independent tyre generalisation','Not established; folds0/2 leak-flagged and fold1 has few tyres'),
 ('Broader Tier5/6 and XAI extensions','Not completed by the implemented tracks'),
 ('Manuscript submission','Requires human review, reference checking and venue formatting')]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def sources():
    files={}
    for name in ['master_architectures','baselines_by_fold','baseline_summary','hypothesis_outcomes',
                 'stage_a_quarantined','saliency_panel_manifest','intervention_region_coverage']:
        files['tables/classification_'+name+'.csv']='tables/closure_2026-09-09/'+name+'.csv'
    for name in ['stage_b_effects','stage_b_selection','stress_tests','calibration','conformal','ensemble_metrics','tta']:
        files['tables/'+name+'.csv']='tables/'+name+'.csv'
    for name in ['coverage','architecture_effects','factor_decisions','paired_effects']:
        files['tables/s4b_'+name+'.csv']=S4B+'/'+name+'.csv'
    for name in ['inventory','roi_by_run','roi_summary','localisation_by_run','localisation_summary']:
        files['tables/s5_'+name+'.csv']=S5+'/'+name+'.csv'
    for name in ['fusion_by_run','fusion_summary']:
        files['tables/s9_'+name+'.csv']=S9+'/'+name+'.csv'
    for name in LEGACY_FIGURES:
        files['figures/legacy_'+name+'.png']='analysis/closure_2026-09-09/'+name+'.png'
    for name,path in [('s5',S5),('s9',S9),('s4b',S4B)]:files['provenance/'+name+'_STATUS.json']=path+'/STATUS.json'
    return files


def retry(fn):
    for attempt in range(8):
        try:return fn()
        except Exception as exc:
            response=getattr(exc,'response',None)
            if getattr(response,'status_code',None) not in (429,500,502,503,504) or attempt==7:raise
            delay=min(300,5*2**attempt);hint=response.headers.get('Retry-After','')
            if hint:
                try:delay=max(delay,float(hint))
                except ValueError:
                    try:delay=max(delay,parsedate_to_datetime(hint).timestamp()-time.time())
                    except (ValueError,TypeError,OverflowError):pass
            print(f'HF temporarily unavailable; retry in {delay:.0f}s',flush=True)
            until=time.monotonic()+delay+2
            while time.monotonic()<until:time.sleep(min(5,max(0,until-time.monotonic())))


class Context:
    def __init__(self,sess):
        self.sess=sess;self.token=sess.uploader.token
        assert self.token and sess.uploader.enabled, 'Enable HF_TOKEN and Internet'
        assert sess.num_workers==1, 'Use one CPU notebook copy'
        self.code=sha(Path(__file__).read_bytes());self.prefix='s10/reporting-r1/'+self.code
        self.root=Path(sess.stage_dir)/'s10_reporting'/self.code;self.root.mkdir(parents=True,exist_ok=True)
        assert shutil.disk_usage(self.root).free>2*2**30,'Need2GiB free scratch; no datasets/checkpoints required'
        self.last=time.monotonic()
    def read(self,path,rev=SOURCE):
        return Path(retry(lambda:hf_hub_download(REPO,path,repo_type='dataset',revision=rev,
                    token=self.token,cache_dir=str(self.root/'cache')))).read_bytes()
    def write(self,path,raw):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(p);return p
    def json(self,path,value):return self.write(path,json.dumps(value,indent=2,allow_nan=False).encode())
    def flush(self,reason):
        assert self.sess.push_now(reason),'Upload failed; retain local scratch and rerun'
        self.last=time.monotonic()
    def enqueue(self,paths):
        self.sess.uploader.enqueue_batch([(self.root/p,self.prefix+'/'+p) for p in paths])


def validate(root):
    def read(n):return pd.read_csv(root/'tables'/n)
    s5=json.loads((root/'provenance/s5_STATUS.json').read_text())
    s9=json.loads((root/'provenance/s9_STATUS.json').read_text())
    assert s5['status']=='complete' and s5['verified_runs']==81
    assert s9['status']=='exploratory_analysis_complete' and s9['processed_runs']==81 and not s9['full_s9_complete']
    for name,value in s9['artifact_sha256'].items():assert sha((root/'tables'/('s9_'+name)).read_bytes())==value
    inv=read('s5_inventory.csv');assert len(inv)==81 and inv.run_id.is_unique and inv.epoch.eq(60).all() and inv.status.eq('completed_verified').all()
    roi=read('s5_roi_by_run.csv');fusion=read('s9_fusion_by_run.csv')
    assert len(roi)==len(fusion)==405 and not roi.duplicated(['run_id','mode']).any() and not fusion.duplicated(['run_id','arm']).any()
    assert set(roi.run_id)==set(fusion.run_id)==set(inv.run_id)
    for frame,key,score,delta in [(roi,'mode','macro_f1','delta_macro_f1_vs_full'),(fusion,'arm','macro_f1','delta_vs_full')]:
        base=frame[frame[key].eq('full')].set_index('run_id')[score]
        assert np.allclose(frame[delta],frame[score]-frame.run_id.map(base))
    assert len(read('classification_master_architectures.csv'))==17
    assert len(read('classification_stage_a_quarantined.csv'))==9
    return {'s5_runs':81,'s5_epoch_records_previously_audited':4860,'s9_source_runs':81,
            'roi_rows':len(roi),'fusion_rows':len(fusion),'retained_architectures':17,'quarantined_runs':9}


def evidence(sess):
    ctx=Context(sess);manifest=[];pending=[]
    try:
        for local,remote in sources().items():
            raw=ctx.read(remote);ctx.write(local,raw);pending.append(local)
            manifest.append(dict(local=local,repo=REPO,revision=SOURCE,path=remote,sha256=sha(raw),bytes=len(raw)))
            print(f'{len(manifest)}/{len(sources())} evidence files verified',flush=True)
            if time.monotonic()-ctx.last>=1800:ctx.enqueue(pending);ctx.flush('S10 evidence30min');pending=[]
        counts=validate(ctx.root)
        ctx.json('evidence_manifest.json',manifest);pending.append('evidence_manifest.json')
        ctx.json('EVIDENCE_STATUS.json',dict(status='evidence_bundle_ready',source_revision=SOURCE,code_sha256=ctx.code,
            manifest_sha256=sha((ctx.root/'evidence_manifest.json').read_bytes()),counts=counts,
            full_project_complete=False,remaining=GAPS));pending.append('EVIDENCE_STATUS.json')
        ctx.enqueue(pending);ctx.flush('NB19 evidence package complete')
    except BaseException:
        ctx.enqueue(pending);ctx.flush('NB19 interrupted: preserve downloaded evidence');raise
    print('NB19 complete. Run NB20 on CPU next. Published:',ctx.prefix)
    return ctx.prefix


def plot_panel(frame,value,group,filename,title,root,zero=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharex=True)
    for fold,ax in enumerate(axes):
        sub=frame[frame.fold.eq(fold)].groupby(group)[value].agg(['mean','std','count']).sort_index()
        assert sub['count'].eq(3).all(), 'Expect three seeds per model/fold, not pooled images'
        ax.errorbar(sub['mean'],np.arange(len(sub)),xerr=sub['std'].fillna(0),fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(sub)),sub.index);ax.set_title(f'Fold {fold}'+(' — leakage flagged' if fold!=1 else ' — few tyres'))
        ax.grid(axis='x',alpha=.25)
        if zero:ax.axvline(0,color='black',linewidth=.8)
    fig.suptitle(title+'\nMean ± seed SD; descriptive, not confidence intervals',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.90));path=root/'figures'/filename
    fig.savefig(path,dpi=150);plt.close(fig)


def render(root):
    counts=validate(root)
    dense=pd.read_csv(root/'tables/s5_localisation_by_run.csv')
    roi=pd.read_csv(root/'tables/s5_roi_by_run.csv');fusion=pd.read_csv(root/'tables/s9_fusion_by_run.csv')
    plots=[('s10_11_box_ap.png','S5 box AP50:95 by model/fold','s5_localisation_by_run.csv'),
           ('s10_12_predicted_roi.png','Predicted-tread ROI change versus full image','s5_roi_by_run.csv'),
           ('s10_13_fixed_fusion.png','Equal full+tyre+tread fusion change versus full image','s9_fusion_by_run.csv'),
           ('s10_14_manual_mask_iou.png','Tread mask IoU against existing manual masks','s5_localisation_by_run.csv')]
    plot_panel(dense,'bbox_map_50_95','model',plots[0][0],plots[0][1],root)
    plot_panel(roi[roi['mode'].eq('pred_tread')],'delta_macro_f1_vs_full','model',plots[1][0],plots[1][1],root,True)
    plot_panel(fusion[fusion.arm.eq('full_tyre_tread_equal')],'delta_vs_full','model',plots[2][0],plots[2][1],root,True)
    plot_panel(dense[dense.tread_iou.notna()],'tread_iou','model',plots[3][0],plots[3][1],root)
    manifest=[dict(file='figures/legacy_'+n+'.png',kind='inherited NB10R',caption=n,
        caveat='Inherited endpoint and limitations; figures8–10 are best-epoch/energy/session, not missing original experiments') for n in LEGACY_FIGURES]
    manifest += [dict(file='figures/'+n,kind='new descriptive figure',caption=t,source='tables/'+src,
        caveat='Existing folds, three seed SD; not new-tyre validation') for n,t,src in plots]
    summary=fusion.groupby('arm').delta_vs_full.mean()
    hypothesis=pd.read_csv(root/'tables/classification_hypothesis_outcomes.csv')
    conformal=pd.read_csv(root/'tables/conformal.csv')
    text='''# Results and limitations — review draft

This is a reproducible reporting package, not a completed manuscript or a claim
that the entire original proposal was executed. Labels represent mileage-proxy
classes, not measured tread depth, verified roadworthiness or alignment.

## Scope and endpoints

The retained architecture sweep contains17 architectures and153 valid runs;
nine mislabeled substitutions remain quarantined. Implemented StageB has108
runs; S4b has18 confirmation runs. S5 has81 evaluated runs at60epochs each.
NB18 analyses81 saved S5 prediction sets with fixed fusion rules; it is not the
full HRNet/PatchCore pipeline, which is deferred by user.

Classification source tables retain both selected-epoch and final-epoch fields.
Do not substitute selected-epoch scores for fixed-budget results. Inherited H1
is explicitly a legacy selected-epoch analysis. S5/S9 use the fixed final
classifier endpoint and preserve CORAL versus softmax decisions.

## Key supported results

S5 localisation and full/predicted/oracle ROI tables are bundled with their
source revision. Localisation quality alone does not demonstrate improved
classification. The fixed fusion comparisons below did not improve the mean
endpoint; averages are descriptive, equally weighted by source run, not
independent observations or significance tests.

'''
    text+='| Fusion arm | Mean macro-F1 delta versus full |\n|---|---:|\n'
    for arm,value in summary.items():text+=f'| {arm} | {value:.6f} |\n'
    text+='\n## Hypotheses and calibration\n\nSee `tables/classification_hypothesis_outcomes.csv` and `tables/conformal.csv`.\n'
    text+='H2 is inconclusive/undefined; H3 lacks its planned model arms. Calibration coverage must be reported as achieved, not guaranteed.\n\n'
    text+='## Figures\n\nThe10 inherited panels retain their original definitions; four new panels extend dense-task/fusion reporting. Figure14 is model-versus-manual agreement, NOT annotator self-consistency.\n\n'
    for item in manifest:text+=f'### {item["caption"]}\n\n{item["caveat"]}\n\n![{item["caption"]}]({item["file"]})\n\n'
    text+='## Limitations and uncompleted work\n\n'
    for name,state in GAPS:text+=f'- {name}: {state}.\n'
    text+='\n## Review checklist\n\n- Check terminology, author contributions and source references.\n- Discuss fold leakage and sample independence explicitly.\n- Choose the submission scope; do not imply full original-plan completion.\n- Review all figures and result tables before submission.\n'
    (root/'RESULTS_DRAFT.md').write_text(text,encoding='utf-8')
    body='<h1>Results package — review draft</h1><p>Not full-project completion. See RESULTS_DRAFT.md for scope and limitations.</p>'
    body+='<h2>Fusion mean deltas (descriptive)</h2>'+summary.to_frame('mean_delta').to_html()
    body+='<h2>Hypotheses (legacy definitions retained)</h2>'+hypothesis.to_html(index=False)
    body+='<h2>Achieved conformal coverage</h2>'+conformal.to_html(index=False)
    for item in manifest:body+=f'<h2>{html.escape(item["caption"])}</h2><p>{html.escape(item["caveat"])}</p><img src="{item["file"]}" alt="{html.escape(item["caption"])}">'
    body+='<h2>Uncompleted/deferred work</h2>'+pd.DataFrame(GAPS,columns=['Item','Status']).to_html(index=False)
    (root/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>Tyre study results</title><style>body{font:16px Arial;margin:32px;max-width:1500px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>'+body,encoding='utf-8')
    (root/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return counts,manifest


def report(sess):
    ctx=Context(sess)
    rev=retry(lambda:HfApi(token=ctx.token).repo_info(REPO,repo_type='dataset').sha)
    try:status=json.loads(ctx.read(ctx.prefix+'/EVIDENCE_STATUS.json',rev))
    except Exception as exc:raise RuntimeError('Run the matching NB19 successfully before NB20; evidence package unavailable') from exc
    raw=ctx.read(ctx.prefix+'/evidence_manifest.json',rev)
    assert sha(raw)==status['manifest_sha256'] and status['status']=='evidence_bundle_ready' and status['source_revision']==SOURCE and status['code_sha256']==ctx.code
    ctx.write('evidence_manifest.json',raw)
    for item in json.loads(raw):
        data=ctx.read(ctx.prefix+'/'+item['local'],rev);assert sha(data)==item['sha256']
        ctx.write(item['local'],data)
    pending=[]
    try:
        counts,figures=render(ctx.root)
        pending=['RESULTS_DRAFT.md','REPORT.html','figure_manifest.json']+[x['file'] for x in figures if x['kind']=='new descriptive figure']
        ctx.json('REPORT_STATUS.json',dict(status='report_package_ready_for_review',source_revision=SOURCE,
            evidence_revision=rev,code_sha256=ctx.code,figure_count=len(figures),counts=counts,
            full_project_complete=False,remaining=GAPS,
            artifact_sha256={p:sha((ctx.root/p).read_bytes()) for p in pending}))
        pending.append('REPORT_STATUS.json');ctx.enqueue(pending);ctx.flush('NB20 report package complete')
    except BaseException:
        ctx.enqueue(pending);ctx.flush('NB20 interrupted');raise
    print('Report ready for review, not submission certification. Published:',ctx.prefix)
    return ctx.prefix
