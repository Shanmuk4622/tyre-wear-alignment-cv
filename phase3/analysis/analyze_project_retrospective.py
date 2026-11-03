"""Cross-experiment synthesis and new TRACE component audit from saved evidence.

No model forwards, fitting, network calls or old source/evidence changes.
"""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'retrospective_20261102';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import read,atomic,file_meta,sha
from review import Saved
torch.set_num_threads(4)
nb24=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path']);saved=Saved(nb24/'remote/fit')
cache=read(nb24/'remote/fit/inputs/cache/CACHE.json');source={r['image_id']:r for r in cache['rows']}
consumed=[]
for family in ('sample_observations','image_losses','control_losses','local_token_statistics'):
    for path in saved.paths(family):
        n=path.relative_to(saved.root).as_posix();meta=file_meta(path)
        assert meta==saved.state['closed'][n]
        consumed.append(dict(family=family,file=path.relative_to(ROOT).as_posix(),**meta))
pd.DataFrame(consumed).to_csv(OUT/'CONSUMED_SCIENCE_INPUTS.csv',index=False)
obs=saved.frame('sample_observations');loss=saved.frame('image_losses');control=saved.frame('control_losses')
tokens=saved.frame('local_token_statistics');tokens=tokens[(tokens.stage=='endpoint_final')&(tokens.epoch==60)]
tiles=tokens.groupby(['role','layer'],as_index=False).agg(tiles=('tile','count'),rank_min=('effective_rank','min'),rank_median=('effective_rank','median'),rank_max=('effective_rank','max'),std_median=('std_median','median'))
tiles.to_csv(OUT/'TRACE_TILE_SUMMARY.csv',index=False)
end=obs[obs.stage=='endpoint_final'].copy();assert len(end)==570
controls=control[(control.stage=='endpoint_final')&(control.view==0)]
means=loss[(loss.stage=='endpoint_final')&(loss.view==0)].pivot(index='image_id',columns='objective',values='image_mean').astype(float)
rows=[]
for row in end.itertuples():
    r=source[row.image_id];c=controls[controls.image_id==row.image_id].set_index('control');local=means.loc[row.image_id]
    a=dict(image_id=row.image_id,role=r['role'],component=r['component'],proxy_class=r['class_index'],source_sha256=r['image_sha256'],
        observation=int(row.observation),epoch=60,parameter_update=2940,eligible=bool(row.eligible),masked_model=float(c.iloc[0].model_mean),local_model=float(local['local']),local_position_control=float(local['local_position_control']))
    for name in ('channel_mean','position_mean','visible_band_mean'):
        a['control_'+name]=float(c.loc[name,'control_mean']);a['gain_'+name]=1-a['masked_model']/a['control_'+name]
    a['local_gain']=1-a['local_model']/a['local_position_control'];a['beats_all_controls']=all(a['gain_'+n]>0 for n in ('channel_mean','position_mean','visible_band_mean'))
    a['passes_5pct_all_controls']=all(a['gain_'+n]>=.05 for n in ('channel_mean','position_mean','visible_band_mean'))
    rows.append(a)
images=pd.DataFrame(rows);images.to_csv(OUT/'TRACE_IMAGE_UTILITY.csv',index=False)
groups=[]
for keys,frame in images.groupby(['role','component','proxy_class']):
    r=dict(role=keys[0],component=keys[1],proxy_class=int(keys[2]),images=len(frame),masked_model=float(frame.masked_model.mean()),local_model=float(frame.local_model.mean()),local_position_control=float(frame.local_position_control.mean()),
        images_beating_all_controls=int(frame.beats_all_controls.sum()),images_passing_5pct_all_controls=int(frame.passes_5pct_all_controls.sum()))
    for name in ('channel_mean','position_mean','visible_band_mean'):
        r['control_'+name]=float(frame['control_'+name].mean());r['gain_'+name]=1-r['masked_model']/r['control_'+name]
    r['local_gain']=1-r['local_model']/r['local_position_control'];groups.append(r)
groups=pd.DataFrame(groups);groups.to_csv(OUT/'TRACE_COMPONENT_UTILITY.csv',index=False)
roles=[]
for role,g in images.groupby('role'):
    controls_mean={n:float(g['control_'+n].mean()) for n in ('channel_mean','position_mean','visible_band_mean')}
    roles.append(dict(role=role,images=len(g),masked_model=float(g.masked_model.mean()),strongest_aggregate_control=min(controls_mean,key=controls_mean.get),
        gain_vs_strongest_aggregate_control=1-float(g.masked_model.mean())/min(controls_mean.values()),
        images_beating_all_controls=int(g.beats_all_controls.sum()),images_passing_5pct_all_controls=int(g.passes_5pct_all_controls.sum()),
        local_gain=1-float(g.local_model.mean())/float(g.local_position_control.mean())))
pd.DataFrame(roles).to_csv(OUT/'TRACE_ROLE_UTILITY.csv',index=False)
# Reconcile to original final endpoint cohort values; this is a diagnostic
# partition of the same observations, not new checkpoint selection or a test.
previous=pd.read_csv(ROOT/'tread_two/trace_completion_report/ENDPOINT_COHORTS.csv').set_index('role')
for r in roles:
    assert np.isclose(r['masked_model'],previous.loc[r['role'],'masked_loss'],atol=1e-12)
    assert np.isclose(r['gain_vs_strongest_aggregate_control'],min(previous.loc[r['role'],'masked_gain_'+n] for n in ('channel_mean','position_mean','visible_band_mean')),atol=1e-12)
classification=[]
def add(label,seed,endpoint,epoch,val_ba,test_ba,dice=None,point=None,nll=None,scope='same 103-image development endpoint'):
    classification.append(dict(model=label,seed=seed,endpoint=endpoint,epoch=epoch,validation_BA=val_ba,test_observed_BA=test_ba,mean_Dice=dice,human_point_MAE_width=point,nll=nll,scope=scope))
r1=pd.read_csv(ROOT/'completion_20261028/csv/run_summary.csv')
validation_ba={1:.55,2:.825,3:1.}
for r in r1.itertuples():
    selected=r.endpoint=='selected';add('TREAD-ONE r1',int(r.seed),'selected' if selected else 'final',int(r.selected_epoch) if selected else 60,validation_ba[int(r.seed)] if selected else None,float(r.balanced_accuracy_observed),float(r.mean_two_region_dice),float(r.human_point_mae_width_fraction),float(r.nll))
p2=read(ROOT.parent/'phase2/completion_seed3/verification.json');selected=[]
for model,seed in [('hrnet',3),('mobilenetv4',1),('resnet50',2),('segformer',2),('yolo26m',1)]:
    r=next(x for x in p2['verified_runs'] if x['run']==f'{model}-combined-seed{seed}')
    selected.append(dict(model=model,seed=seed,epoch=r['best_epoch'],metric='human_point_MAE_width_fraction' if model=='hrnet' else r['metric'],source_metric=r['metric'],higher_is_better=model!='hrnet',validation=-r['validation'] if model=='hrnet' else r['validation'],test=-r['test'] if model=='hrnet' else r['test'],test_images=r['test_count'],selection='original validation winner; first-seed tie rule'))
    if model in ('mobilenetv4','resnet50'):add('Phase2 '+model,seed,'selected',r['best_epoch'],r['validation'],r['test'],scope='validation-selected installed seed; no test-best seed substitution')
pd.DataFrame(selected).to_csv(OUT/'PHASE2_SELECTIONS.csv',index=False)
paired=pd.read_csv(ROOT/'revision2_validation/nb05_publication/analysis/endpoint_metrics.csv')
for r in paired.itertuples():
    add('TREAD-ONE paired baseline' if r.condition=='baseline' else 'TREAD-ONE shared projection',1,r.endpoint,int(r.epoch),(.55 if r.condition=='baseline' else .8375) if r.endpoint=='selected' else None,
        float(r.balanced_accuracy_observed),float(r.mean_two_region_dice),float(r.human_point_mae),float(r.nll))
atlas=pd.read_csv(ROOT/'tread_two/research_release/endpoint_metrics.csv')
for r in atlas.itertuples():
    add(r.label,1,r.endpoint,int(r.epoch),dict(v1=.5,lc=.5625,context=.5)[r.model] if r.endpoint=='selected' else None,float(r.observed_class_BA),nll=float(r.nll))
pd.DataFrame(classification).to_csv(OUT/'CLASSIFICATION_ENDPOINTS.csv',index=False)
experiments=[
 dict(family='TREAD-ONE r1',fit_notebook='NB01',fits=3,seed_policy='1,2,3',epochs_per_fit=60,updates_per_fit=2940,task='classification + masks + points + additive evidence',selected='seed3 epoch20; original validation rule',result='classification transfer weak; internal dense outputs useful',source='phase3CompletionAudit.md'),
 dict(family='TREAD-ONE paired r2',fit_notebook='NB05',fits=2,seed_policy='matched seed1 baseline + projection',epochs_per_fit=60,updates_per_fit=2940,task='shared-gradient intervention',selected='baseline23 / projection17',result='projection selected BA17.073% vs baseline50%; point error worse',source='phase3NB05Completion.md'),
 dict(family='ATLAS v1',fit_notebook='NB08',fits=1,seed_policy='1',epochs_per_fit=60,updates_per_fit=2940,task='reference-based proxy classifier + frozen dense anchor',selected='epoch18',result='low recall0/62; appearance-sensitive',source='phase3TreadTwoFindings.md'),
 dict(family='ATLAS-LC',fit_notebook='NB11',fits=1,seed_policy='1',epochs_per_fit=60,updates_per_fit=2940,task='local-contrast paired reference classifier',selected='epoch7',result='appearance stability and selected NLL improve; low recall unchanged',source='phase3NB12Completion.md'),
 dict(family='ATLAS-CONTEXT',fit_notebook='NB13 + NB13A same fit',fits=1,seed_policy='1',epochs_per_fit=60,updates_per_fit=2940,task='bounded positional residual over frozen LC prior',selected='epoch7',result='layout-sensitive; NLL worse; low recall unchanged',source='phase3NB13Completion.md'),
 dict(family='GEO',fit_notebook='NB17',fits=1,seed_policy='1',epochs_per_fit=60,updates_per_fit=2940,task='isolated geometry + texture + annotation-error heads',selected='unavailable under original joint gate',result='geometry maintained; representation0/60 eligible; learned error heads lose to controls',source='phase3NB17Completion.md'),
 dict(family='TRACE',fit_notebook='NB24 wraps NB22',fits=1,seed_policy='1',epochs_per_fit=60,updates_per_fit=2940,task='fixed-geometry local texture reconstruction',selected='unavailable under original gate',result='masked gain21.63%; local gain36.44%; broad diversity0/60',source='phase3NB24Completion.md')]
pd.DataFrame(experiments).to_csv(OUT/'PHASE3_FIT_LEDGER.csv',index=False)
assert sum(r['fits'] for r in experiments)==10
# Evidence consumed by this synthesis, including existing verified reports and
# the new bounded component analysis. Hash before any documentation updates.
sources=[
 ROOT.parent/'docs/18_STAGE_A_RESULTS.md',ROOT.parent/'docs/23_S4B_CONFIRMATION.md',ROOT.parent/'docs/24_S5_MANUAL_DENSE_TASKS.md',ROOT.parent/'docs/25_S9_FUSION_AND_REMAINING_GAPS.md',ROOT.parent/'docs/report/CLAIMS_AND_LIMITATIONS.md',
 ROOT.parent/'ACCESS_latex_template_20260513/README.md',ROOT.parent/'ACCESS_latex_template_20260513/SUBMISSION_REVISION_20260924.md',
 ROOT.parent/'phase2/phase2Seed3Completion.md',ROOT.parent/'phase2/phase2CompletionAudit.md',ROOT.parent/'phase2/phase2DatasetReadme.md',ROOT.parent/'phase2/completion_seed3/verification.json',
 *[ROOT/n for n in ['phase3CompletionAudit.md','phase3NB02Completion.md','phase3NB03Completion.md','phase3NB04Completion.md','phase3NB05Completion.md','phase3NB06Completion.md','phase3NB07RepeatCheck.md','phase3TreadOneWorkstation.md','phase3TreadTwoFindings.md','phase3NB11Completion.md','phase3NB12Completion.md','phase3NB13Completion.md','phase3NB14Completion.md','phase3NB15Completion.md','phase3NB16Completion.md','phase3NB17Completion.md','phase3NB20Completion.md','phase3NB24Completion.md','phase3TreadTwoGEOLessons.md','phase3TreadTwoTRACEFindings.md']],
 ROOT/'completion_20261028/csv/run_summary.csv',ROOT/'revision2_validation/nb05_publication/analysis/endpoint_metrics.csv',ROOT/'tread_two/research_release/endpoint_metrics.csv',ROOT/'tread_two/nb17_report/validation_reconstruction.csv',ROOT/'tread_two/nb17_report/FINDINGS.json',
 nb24/'VERIFICATION.json',nb24/'SCIENTIFIC_VERIFICATION.json',nb24/'remote/fit/POINTER.json',nb24/'remote/fit/state_1.pt']
pd.DataFrame([dict(file=p.relative_to(ROOT.parent).as_posix(),**file_meta(p)) for p in sources]).to_csv(OUT/'SOURCE_EVIDENCE.csv',index=False)
summary=dict(status='passed',scope='Retrospective synthesis of preserved evidence; new saved-only TRACE component/image diagnostic; no fresh remote re-verification',
    phase3_measured_fit_executions=10,phase3_measured_epochs=600,phase3_measured_optimizer_updates=29400,disposable_engineering_updates_excluded=True,
    new_TRACE_image_diagnostics=570,new_TRACE_component_summaries=len(groups),TRACE_roles=roles,
    source_files=len(sources),consumed_scientific_files=len(consumed),consumed_scientific_bytes=sum(x['bytes'] for x in consumed),measured_fit_families=7,new_optimizer_updates=0,image_model_forwards=0,new_photos=0,remote_writes=0,network_calls=0)
atomic(OUT/'ANALYSIS.json',summary)
print(json.dumps(summary,indent=2));print(groups[['role','component','images','gain_visible_band_mean','local_gain','images_beating_all_controls']].to_string(index=False))
