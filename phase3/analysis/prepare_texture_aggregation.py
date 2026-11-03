"""Freeze the saved-feature investigation and fetch only pinned native evidence."""
import json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
from nb24_public import *
from common import digest
from review import Saved
DEST=ROOT/'texture_aggregation_20261102';DEST.mkdir(exist_ok=True)
saved=Saved(OUT/'remote/fit');obs=saved.frame('sample_observations');final=obs[obs.stage=='endpoint_final'].sort_values('observation')
assert len(final)==570 and set(final.epoch)=={60} and set(final.parameter_update)=={2940}
protocol=dict(version='TRACE-saved-aggregation-v1',date='2026-11-02',fit_protocol=META['protocol'],fit_revision=META['stages']['fit']['revision'],
    original_selected=None,endpoint='final_epoch60_update2940',roles=dict(train=386,validation=81,test=103),views=[0,1],primary_view=0,
    feature_layers=['encoded','full_context','masked_context'],readouts=['local_pooled','context_mean','pooled','projected'],
    utility_branches=dict(encoded='local_prediction on supported cells',masked_context='masked_prediction on hidden cells'),
    variants=['identity','global_mean','band_means','column_means','moments5','random5','train_template','residual_mean','residual_moments5'],
    projection='coverage-mass weighted least squares, pseudoinverse rcond1e-10; original per-image saved masses',
    template_fit='unweighted train-view0 per-position feature mean over supported train images only; held-out never fitted',
    random_seed=462225,random_basis='constant plus four fixed QR-orthogonal zero-sum random position vectors',
    reconstruction='same frozen decoder; FP64 CPU matrix arithmetic; standardized fixed targets using original FP32 statistics',
    target_support='same cell intersection with train-position-control availability for all variants; coverage explicitly recorded',
    replay_tolerances=dict(prediction_atol=3e-5,prediction_rtol=2e-5,loss_atol=2e-6,bridge_atol=3e-5),
    spectra='role-centered covariance singular-value entropy rank; component centering descriptive only',
    covariance='fixed train-mean mass and same cohorts, diagonal/cross-position covariance and mean-mode energy',
    scale_control='train-only per-coordinate std with floor1e-3; invertible rescaling changes rank metric, adds no information',
    model_training=False,optimizer_updates=0,image_encoder_forwards=0,geometry_forwards=0,frozen_decoder_replay=True,
    neural_network_training=False,new_photos=0,workstation_changes=False,remote_writes=0,
    claim_scope='posthoc saved-feature/decoder counterfactuals; original historical roles already inspected; no gate relaxation or new checkpoint selection')
config=DEST/'PROTOCOL.json'
if config.exists():assert read(config)==protocol
else:atomic(config,protocol)
atomic(DEST/'PROTOCOL_ID.json',dict(sha256=digest(protocol)))
ids=final.observation.to_numpy();kinds=set(protocol['feature_layers']+protocol['readouts']+['mass','hidden','local_prediction','masked_prediction'])
wanted=set()
for r in saved.index.itertuples():
    if r.kind in kinds and ((ids>=r.first_observation)&(ids<=r.last_observation)).any():wanted.add(r.file)
cache=read(saved.root/'inputs/cache/CACHE.json')
for name in cache['files']:
    if name.startswith(('native/cache_target/','native/cache_weight/')):wanted.add('inputs/cache/'+name)
wanted.update(['inputs/cache/CACHE.json','inputs/cache/controls/target_statistics.pt','exports/final.pt','CONTRACT.json','POINTER.json',read(OUT/'FIT_POINTER.json')['state']])
print('Pinned saved-feature inputs:',len(wanted),'objects,',sum(MANIFESTS['fit']['files'][n]['bytes'] for n in wanted),'bytes; existing verified files reused.',flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,_ in enumerate(pool.map(lambda name:fetch('fit',name),sorted(wanted))):
        if (i+1)%40==0:print('Aggregation input bytes verified',i+1,'/',len(wanted),flush=True)
import pandas as pd
inventory=[dict(file=n,**file_meta(saved.root/n)) for n in sorted(wanted)]
pd.DataFrame(inventory).to_csv(DEST/'NATIVE_INPUTS.csv',index=False)
source={r['image_id']:r for r in cache['rows']};alignment=[]
for r in final.itertuples():
    s=source[r.image_id];alignment.append(dict(image_id=r.image_id,role=s['role'],component=s['component'],proxy_class=s['class_index'],source_sha256=s['image_sha256'],slot=s['slot'],observation=r.observation))
pd.DataFrame(alignment).to_csv(DEST/'IMAGE_ALIGNMENT.csv',index=False)
baseline=DEST/'PRESERVATION_BASELINE.json'
if not baseline.exists():
    names=['tread_two/nb24_completion/23e081e52454/VERIFICATION.json','tread_two/nb24_completion/23e081e52454/SCIENTIFIC_VERIFICATION.json','retrospective_20261102/ANALYSIS.json','retrospective_20261102/VALIDATION.json','phase3ProjectRetrospective.md','phase3TRACEComponentReview.md','phase3TreadTwoTRACEFindings.md','phase3NB24Completion.md']
    atomic(baseline,{n:file_meta(ROOT/n) for n in names})
receipt=dict(status='passed',protocol=digest(protocol),inputs=len(inventory),bytes=sum(x['bytes'] for x in inventory),all_full_body_SHA256_verified=True,
    public_read_requests_this_process=public.CALLS,local_credentials_read=False,remote_writes=0,new_photos=0)
atomic(DEST/'INPUT_VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
