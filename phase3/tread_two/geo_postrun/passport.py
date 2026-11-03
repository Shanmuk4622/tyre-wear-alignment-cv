"""Finite research handoff derived only from the completed verified review."""
import html,shutil
from pathlib import Path
import pandas as pd
from common import atomic,read,file_meta,digest
from inputs import PIN
from tables import Tables

def valid(out,contract):
    out=Path(out)
    if not (out/'RESULT.json').exists() or not (out/'OUTPUT_RECEIPT.json').exists():return False
    receipt=read(out/'OUTPUT_RECEIPT.json');result=read(out/'RESULT.json')
    return receipt['contract']==contract and result.get('review_contract')==contract and result.get('input')==PIN and all((out/name).exists() and file_meta(out/name)==meta for name,meta in receipt['files'].items())

def run(review,out,contract):
    review=Path(review);out=Path(out);tables=Tables(out);result=read(review/'RESULT.json')
    assert result['status']=='completed' and result['input']==PIN
    f=result['findings'];v=f['final_validation'];assert not f['selected_checkpoint_available']
    verification=read(review/'FILE_VERIFICATION.json');native=read(review/'NATIVE_VERIFICATION.json')
    capabilities=[dict(capability='Tyre / tread segmentation',status='Research geometry; baseline floor preserved',evidence='Validation Dice '+format(v['mean_dice'],'.9f'),limit='Small previously used development cohorts; final is unselected'),
                  dict(capability='Tread point localisation',status='Human-only retrospective improvement',evidence='Validation width-normalised MAE '+format(v['human_point_MAE'],'.9f'),limit='Weak/absent annotations excluded from human metrics'),
                  dict(capability='Texture representation',status='Rejected by original eligibility gate',evidence='0/60 eligible; final projector rank '+format(v['projector_rank'],'.6f'),limit='Near-rank-one held-out pooled vectors; test and validation are previously inspected'),
                  dict(capability='Annotation error estimation',status='Experimental only',evidence='quality_comparison.csv and quality_calibration.csv',limit='Annotation error estimate; no calibrated safety confidence'),
                  dict(capability='Explainability',status='Descriptive overlays, probes and retrieval',evidence='570 overlays; 1030 probes; cross-component source exclusions checked',limit='Model sensitivity / residuals are not causal defect or wear labels'),
                  dict(capability='Mileage classification',status='Not fitted in GEO',evidence='No classification objective/head fitted',limit='Earlier ATLAS mileage proxy failures remain unchanged'),
                  dict(capability='Physical depth / tyre safety',status='Not validated',evidence='No measured physical-depth or safety targets',limit='Do not make physical depth, serviceability or safety claims')]
    tables.write('capabilities',capabilities)
    metrics=[dict(metric=k,value=v[k],scope='Original validation / epoch 60 / unselected') for k in ['mean_dice','human_point_MAE','masked_huber','mean_huber','masked_gain','pair_cosine','projector_rank','global_rank','projector_std_median']]
    tables.write('validation_metric_passport',metrics)
    publication=read(review/'HF_PUBLICATION.json') if (review/'HF_PUBLICATION.json').exists() else dict(status='local_analysis_only_publication_pending',revision='UNPUBLISHED_LOCAL_ANALYSIS',destination='phase3/tread-two/geo-r1/review/'+contract)
    provenance=[dict(kind='frozen_fit',repository=PIN['repository'],revision=PIN['revision'],prefix=PIN['prefix'],protocol=PIN['protocol'],bytes=PIN['bytes'],objects=PIN['selected_objects'],status='published_and_byte_verified'),
                dict(kind='derived_review',repository=PIN['repository'],revision=publication['revision'],prefix=publication['destination'],protocol=contract,bytes=publication.get('bytes'),objects=publication.get('selected_objects'),status=publication['status'])]
    tables.write('provenance',provenance)
    reconstruction=[dict(item='Published file bytes / hashes',coverage=str(verification['objects'])+' objects; '+str(verification['bytes'])+' bytes',status='Verified',gap='No historical unsent filesystem bytes can be inferred'),
                    dict(item='CSV schemas / saved counters',coverage=str(verification['CSV_files'])+' CSVs; '+str(verification['CSV_rows'])+' rows',status='Verified',gap='Per-run retained telemetry does not add independent tyres'),
                    dict(item='Complete training objective',coverage='2940 updates from saved vectors and sufficient numerators/denominators',status='Verified',gap='Anchor drift uses logged sums; no new frozen-anchor forward'),
                    dict(item='Native measured bodies',coverage=str(native['native_shards'])+' shards + 2 controls; all 60 epoch weights',status='Verified',gap='Historical full optimizer states are rolling, not all permanently kept'),
                    dict(item='Final masks / confusion',coverage='14,008,320 dense CSV rows against existing PNG masks',status='Verified',gap='No new photographs or independent annotation audit'),
                    dict(item='Probe raw texture',coverage='1030 compact geometry/probe observations plus CSV vectors/statistics',status='Declared narrower coverage',gap='Probe raw texture bodies were not retained by original policy')]
    tables.write('reconstruction_coverage',reconstruction)
    decisions=[dict(action='Promote GEO selected model',decision='No',reason='No checkpoint eligible under original rule'),dict(action='Run another full fit automatically',decision='No',reason='Requires a distinct justified protocol; no automatic sweep'),dict(action='Collect new tyre photographs',decision='No',reason='User explicitly limits work to existing photos'),dict(action='Modify workstations',decision='Deferred',reason='User deferred application changes'),dict(action='Close this pilot as documented research',decision='Yes',reason='Complete evidence retained; accepted and failed criteria recorded')]
    tables.write('handoff_decisions',decisions)
    pointers=[dict(endpoint='final_epoch_60',status='Research only; not selected',revision=PIN['revision'],logical_file='exports/final.pt',protocol=PIN['protocol']),dict(endpoint='selected',status='Unavailable: original gate rejected all epochs',revision=PIN['revision'],logical_file='',protocol=PIN['protocol'])]
    tables.write('native_checkpoint_pointers',pointers)
    for name in ['endpoint_cohorts.csv','endpoint_latent_spaces.csv','intervention_summary.csv','quality_comparison.csv','source_family_inventory.csv','component_variance_decomposition.csv','endpoint_token_spaces.csv','endpoint_patch_latent.csv','resource_summary.csv','timing_summary.csv']:
        shutil.copyfile(review/name,out/name)
    report=f'''# TREAD-TWO / GEO — research evidence passport

The one-seed pilot completed **60 epochs / 2,940 updates**. All 60 validation
epochs preserve the geometry floor, and **zero** pass the representation gate.
The selected endpoint is unavailable. Keep the final epoch as a research artifact.

| Validation metric | G0 | Final epoch 60 |
|---|---:|---:|
| Mean tyre / tread Dice | {f['G0']['mean_dice']:.9f} | {v['mean_dice']:.9f} |
| Human point MAE / image width | {f['G0']['human_point_MAE']:.9f} | {v['human_point_MAE']:.9f} |
| Projector effective rank | {f['G0']['rank']:.6f} | {v['projector_rank']:.6f} |
| Masked prediction gain vs train-only mean | — | {100*v['masked_gain']:.4f}% |

```mermaid
flowchart LR
    A[Existing 570 photos] --> B[Completed frozen NB17]
    B --> C[Verified bytes, CSVs and native tensors]
    C --> D[Geometry floor preserved]
    C --> E[Representation gate failed]
    D --> F[Research evidence passport]
    E --> F
    F --> G[Unpromoted final artifact; no selected model]
```

The 96D pooled encoder also has validation rank **{v['global_rank']:.6f}**.
Rank from all 570 images is not a substitute for the original validation rule.
Paired consistency and a larger training signal do not establish useful held-out
representation diversity. Probe sensitivity is descriptive, not a defect label.

The small cohorts were used during development. No new independent generalisation,
physical tread depth, safety, or GEO classification claim is supported. Training
has 386 photos but only six conservative components (low/mid/high = 1/1/4).
No new photos, optimiser update, image/model forward, checkpoint reselection,
workstation change or silent future fit is part of this handoff.

Full technical provenance and capability limits are in the accompanying direct
CSV tables. Native scientific files remain at pinned HF revision `{PIN['revision']}`
under `{PIN['prefix']}`. Original manifest object paths identify the exact bytes.
'''
    (out/'MODEL_CARD.md').write_text(report,encoding='utf-8')
    rows=''.join('<tr><td>'+html.escape(c['capability'])+'</td><td>'+html.escape(c['status'])+'</td><td>'+html.escape(c['evidence'])+'</td><td>'+html.escape(c['limit'])+'</td></tr>' for c in capabilities)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GEO evidence passport</title><style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:1rem;background:#101924;color:#edf4fa;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526578;padding:.7rem;text-align:left}input{font:inherit;padding:.6rem}a{color:#9be7ed}</style><h1>TREAD-TWO / GEO · research passport</h1><p>60 epochs complete. Zero eligible selected checkpoints. Geometry remains useful as retrospective research evidence; the joint candidate remains unpromoted.</p><label>Filter capabilities <input id="filter"></label><table><thead><tr><th>Capability</th><th>Status</th><th>Evidence</th><th>Limit</th></tr></thead><tbody>ROWS</tbody></table><p>No physical depth or safety validation. Existing photos only; no further fit is launched.</p><script>document.getElementById('filter').addEventListener('input',function(){for(const row of document.querySelectorAll('tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(this.value.toLowerCase())});</script></html>'''.replace('ROWS',rows)
    (out/'GEO_Evidence_Passport.html').write_text(page,encoding='utf-8')
    result=dict(status='completed',promotion='research_only',selected_available=False,capabilities=len(capabilities),input=PIN,review_contract=contract,review_publication=publication,new_photos=0,new_optimizer_updates=0,image_model_forwards=0)
    inventory={p.relative_to(out).as_posix():file_meta(p) for p in out.rglob('*') if p.is_file() and p.name not in ('RESULT.json','OUTPUT_RECEIPT.json','ERROR.json','STATUS.json') and not p.name.endswith('.tmp')}
    atomic(out/'OUTPUT_RECEIPT.json',dict(contract=contract,files=inventory))
    atomic(out/'RESULT.json',result);return result
