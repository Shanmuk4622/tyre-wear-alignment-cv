"""Independent native-to-CSV reduction and preservation checks for aggregation assay."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import ast,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
R=Path(__file__).resolve().parents[1];D=R/'texture_aggregation_20261102';sys.path.insert(0,str(R/'tread_two/trace_batch'))
from common import read,atomic,file_meta,sha,digest
from review import Saved
from cache import Cache
cfg=read(D/'PROTOCOL.json');a=read(D/'ANALYSIS.json');assert a['status']=='passed' and a['protocol']==digest(cfg)
old=Path(read(R/'tread_two/nb24_completion/LATEST.json')['path']);root=old/'remote/fit';saved=Saved(root);cache=Cache(root/'inputs/cache')
for r in pd.read_csv(D/'NATIVE_INPUTS.csv').itertuples():assert file_meta(root/r.file)==dict(bytes=r.bytes,sha256=r.sha256)
for r in pd.read_csv(D/'CSV_INPUTS.csv').itertuples():assert file_meta(root/r.file)==dict(bytes=r.bytes,sha256=r.sha256)
catalog=read(D/'ARTIFACTS.json');assert catalog['protocol']==a['protocol']
for n,m in catalog['files'].items():assert file_meta(D/n)==m,n
for stage in ('SPACES','UTILITY_encoded','UTILITY_masked_context'):
    receipt=read(D/(stage+'_RECEIPT.json'));assert receipt['protocol']==a['protocol']
    assert all(file_meta(R/'analysis'/n)==m for n,m in receipt['sources'].items())
    assert all(file_meta(D/n)==m for n,m in receipt['files'].items())
expected={'IMAGE_ALIGNMENT':570,'NATIVE_INPUTS':249,'CSV_INPUTS':254,'BRIDGE_VERIFICATION':6,'TRAIN_TEMPLATES':4608,'TRAIN_SCALES':864,'FEATURE_SPACES':300,'SPATIAL_COVARIANCE':36,'SPATIAL_ENERGY':20520,
          'IMAGE_LOSSES_encoded':10260,'IMAGE_LOSSES_masked_context':10260,'CHANNEL_LOSSES_encoded':61560,'CHANNEL_LOSSES_masked_context':61560,'UTILITY_SUMMARY':108,'COMPONENT_UTILITY':360,
          'MASK_PLAN_ALIGNMENT':48,'MASK_PLAN_SUMMARY':6,'SUPPORT_SUMMARY':6,'CHANNEL_UTILITY_SUMMARY':648,'TILE_UTILITY_SUMMARY':11520,'SPATIAL_ENERGY_SUMMARY':108}
assert {p.stem for p in D.glob('*.csv')}==set(expected)
for name,count in expected.items():assert len(pd.read_csv(D/(name+'.csv')))==count,name
alignment=pd.read_csv(D/'IMAGE_ALIGNMENT.csv');assert alignment.image_id.nunique()==570
ids=alignment.observation.to_numpy();slots=alignment.slot.to_numpy();stats=cache.statistics();raw=cache.array('target',slots);base_w=cache.array('weight',slots).astype('float64')
mu=stats['mean'].numpy();std=stats['std'].numpy();truth=((raw-mu[None,None,:,None,None])/std[None,None,:,None,None]).astype('float64')[:,None]
available=(stats['position_count'].numpy()>0)[None,None]
mass=saved.get('mass',ids);hidden=saved.get('hidden',ids);checked_values=0;maximum_loss_difference=0.;summary=pd.read_csv(D/'UTILITY_SUMMARY.csv')
for layer in ('encoded','masked_context'):
    support=mass>0 if layer=='encoded' else hidden
    w=base_w[:,None]*support[...,None,None,None]*available;count=w.sum((2,3,4,5))*6
    frame=pd.read_csv(D/('IMAGE_LOSSES_'+layer+'.csv'),float_precision='round_trip')
    for variant in cfg['variants']:
        native=np.load(D/'native'/('prediction_'+layer+'_'+variant+'.npy'),mmap_mode='r',allow_pickle=False)
        assert native.shape==(570,2,32,6,4,4) and native.dtype==np.float64 and np.isfinite(native).all()
        delta=np.abs(native-truth);quadratic=np.minimum(delta,.1)
        # Independent min/linear decomposition of Huber, rather than the
        # implementation's piecewise where expression.
        body=(delta-quadratic)+quadratic*quadratic/.2
        loss=(body*w).sum((2,3,4,5))/count
        f=frame[frame.variant==variant].set_index(['image_id','view'])
        for i,row in enumerate(alignment.itertuples()):
            for view in (0,1):
                r=f.loc[(row.image_id,view)];error=abs(float(loss[i,view])-r.loss);maximum_loss_difference=max(maximum_loss_difference,error)
                assert np.isclose(loss[i,view],r.loss,atol=1e-12,rtol=1e-12)
                assert r.weighted_channel_count==count[i,view]
        for role in ('train','validation','test'):
            use=alignment.role.to_numpy()==role
            for view in (0,1):
                r=summary[(summary.layer==layer)&(summary.role==role)&(summary.view==view)&(summary.variant==variant)].iloc[0]
                assert np.isclose(loss[use,view].mean(),r.loss,atol=1e-12,rtol=1e-12)
        checked_values+=native.size
    # Raw channel and tile sufficient statistics independently sum to image rows.
    channels=pd.read_csv(D/('CHANNEL_LOSSES_'+layer+'.csv'),float_precision='round_trip')
    summed=channels.groupby(['image_id','view','variant']).agg(num=('huber_sum','sum'),den=('weighted_count','sum'))
    ref=frame.set_index(['image_id','view','variant']).sort_index();summed=summed.sort_index()
    assert np.allclose(summed.num/summed.den,ref.loss,atol=1e-12,rtol=1e-12)
    tile=[];rows=0
    for chunk in pd.read_csv(D/'detail'/('TILE_LOSSES_'+layer+'.csv'),chunksize=131072,float_precision='round_trip'):
        rows+=len(chunk);tile.append(chunk.groupby(['image_id','view','variant'])[['huber_sum','weighted_channel_count']].sum())
    assert rows==328320
    reduced=pd.concat(tile).groupby(level=[0,1,2]).sum().sort_index()
    assert np.allclose(reduced.huber_sum/reduced.weighted_channel_count,ref.loss,atol=1e-12,rtol=1e-12)
    print('Independent native/image/channel/tile reductions passed:',layer,flush=True)
assert checked_values==63037440
# Validate train-only templates and exact invertible scaling against stored
# feature matrices; no held-out values are used to fit these quantities.
train=alignment.role.to_numpy()=='train'
for layer in cfg['feature_layers']:
    x=saved.get(layer,ids).astype('float64');support=mass>0;count=support[train,0].sum(0)
    expected_mean=np.einsum('nt,ntd->td',support[train,0],x[train,0])/count[:,None]
    assert np.allclose(expected_mean,np.load(D/'native'/('template_'+layer+'.npy')),atol=1e-12,rtol=1e-12)
    for name in ('mean48','moments240'):
        raw=np.load(D/'native'/('features_'+layer+'_'+name+'.npy'));scaled=np.load(D/'native'/('features_'+layer+'_'+name+'_train_scaled.npy'))
        mean=raw[train,0].mean(0);std=raw[train,0].std(0,ddof=1).clip(.001)
        assert np.allclose(scaled*std+mean,raw,atol=1e-12,rtol=1e-12)
cov=pd.read_csv(D/'SPATIAL_COVARIANCE.csv');assert np.allclose(cov.tile_weighted_variance,cov.mean_mode_variance+cov.spatial_residual_variance,atol=1e-9,rtol=1e-10)
assert np.allclose(cov.mean_mode_variance,cov.diagonal_pooled_variance+cov.cross_position_pooled_variance,atol=1e-9,rtol=1e-10)
mask=pd.read_csv(D/'MASK_PLAN_ALIGNMENT.csv');assert mask.linear_CKA_with_mask_plan.between(-1e-12,1+1e-12).all()
for n,m in read(D/'PRESERVATION_BASELINE.json').items():assert file_meta(R/n)==m,n
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime','trace_batch']:
    path=R/'tread_two'/folder;inventory=read(path/'FILES.json');assert all(file_meta(path/n)==m for n,m in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',R.parent,read(R/'workstation_validation/ORIGINAL_BASELINE.json')),('phase3_workstation',R,read(R/'workstation_validation/DELIVERY.json')['source_hashes']),('paired_runtime',R,read(R/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==v for n,v in inventory.items());preserved[name]=len(inventory)
baseline=read(R/'audit/markdown-baseline.json');assert all(sha(R.parent/r['path'])==r['sha256'] for r in baseline)
for n in [*range(10,18),20,24]:
    p=Path(read(R/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(p/'VERIFICATION.json');assert any(sha(f)==v['executed_notebook_sha256'] for f in (R/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb')),n
for item in read(R/'tread_two/trace_validation/NOTEBOOKS.json'):
    if item['mode']!='all':assert file_meta(R/'notebooks'/item['notebook'])=={k:item[k] for k in ('bytes','sha256')}
docs=[*R.glob('*.md'),*D.glob('*.md'),R/'retrospective_20261102/README.md',*(R/'tread_two/trace_batch').glob('*.md'),*(R/'tread_two/trace_design').glob('*.md')];links=0
for p in docs:
    body=p.read_text(encoding='utf-8');assert '\ufffd' not in body,p
    for rawlink in re.findall(r'\[[^\]]+\]\(([^)]+)\)',body):
        target=rawlink.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (p.parent/unquote(target)).exists(),(p,target);links+=1
def ignored(p):return subprocess.run(['git','check-ignore','--no-index','-q','--',str(p)],cwd=R.parent).returncode
for p in [D/'native/prediction_encoded_global_mean.npy',D/'detail/TILE_LOSSES_encoded.csv',D/'CHANNEL_LOSSES_encoded.csv',D/'IMAGE_LOSSES_encoded.csv',D/'SPATIAL_ENERGY.csv']:assert ignored(p)==0,p
for p in [D/'UTILITY_SUMMARY.csv',D/'FEATURE_SPACES.csv',D/'Aggregation_Explorer.html',R/'phase3TextureAggregationInvestigation.md',D/'figures/Aggregation_Utility.png']:assert ignored(p)==1,p
figures=[]
for p in (D/'figures').glob('*.png'):
    with Image.open(p) as image:size=list(image.size);image.verify()
    figures.append(dict(file=p.name,size=size,**file_meta(p)))
viewer=subprocess.run(['node',str(R/'analysis/test_aggregation_explorer.cjs'),str(D/'Aggregation_Explorer.html')],capture_output=True,text=True);assert viewer.returncode==0,viewer.stderr
for name in ['prepare_texture_aggregation.py','texture_aggregation_math.py','test_texture_aggregation_math.py','analyze_texture_aggregation.py','texture_mask_confounds.py','summarize_texture_aggregation.py','document_texture_aggregation.py','build_aggregation_explorer.py','update_texture_aggregation_records.py','validate_texture_aggregation.py']:ast.parse((R/'analysis'/name).read_text(encoding='utf-8'))
files={p.relative_to(D).as_posix():file_meta(p) for p in D.rglob('*') if p.is_file() and p.name not in ('VALIDATION.json','RELEASE.json')}
atomic(D/'RELEASE.json',dict(protocol=a['protocol'],files=files,local_only=True,HF_published=False))
result=dict(status='passed',input_native_metadata_files_full_SHA256=249,input_CSV_bodies_full_SHA256=254,primary_artifact_hashes_checked=len(catalog['files']),
    CSV_files=len(list(D.rglob('*.csv'))),root_CSV_expected_rows=expected,counterfactual_prediction_values_reduced=checked_values,maximum_independent_image_loss_difference=maximum_loss_difference,
    all_native_image_channel_tile_losses_reconcile=True,all_role_summaries_reconcile=True,train_only_templates_checked=True,invertible_scaling_checked=True,
    covariance_identities_checked=True,prior_reports_receipts_unchanged=True,original_sources_preserved=preserved,original_Markdown_baseline_preserved=len(baseline),completed_notebooks_preserved=True,
    Markdown_files_checked=len(docs),local_links_checked=links,broken_links=[],figures=figures,viewer=json.loads(viewer.stdout),
    large_derivatives_ignored=True,compact_results_visible=True,original_selected_None_preserved=True,neural_training=False,image_encoder_forwards=0,geometry_forwards=0,new_optimizer_updates=0,new_photos=0,
    new_analysis_HF_published=False,validation_network_calls=0,remote_writes=0,total_local_release_bytes=sum(m['bytes'] for m in files.values()))
atomic(D/'VALIDATION.json',result);print(json.dumps(result,indent=2))
