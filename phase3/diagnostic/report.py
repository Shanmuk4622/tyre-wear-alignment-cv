"""Durable CSV summaries, transparent findings and a self-contained image explorer."""
import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from metrics import VARIANTS, classification
from storage import atomic, read, write_csv


def build_report(folder,root):
    folder=Path(folder);root=Path(root);contract=read(folder/'CONTRACT.json')
    with open(folder/'csv/predictions.csv',encoding='utf-8',newline='') as stream:
        predictions=list(csv.DictReader(stream))
    summary_rows=[];class_rows=[];confusion_rows=[];groups=defaultdict(list)
    for row in predictions:
        for level,group in (('cohort','all'),('group',row['group']),('domain',row['domain'])):
            groups[(row['job'],row['split'],row['variant'],level,group)].append(row)
    for (job,split,variant,level,group),rows in sorted(groups.items()):
        metrics,classes,matrix=classification(rows)
        identity=dict(job=job,split=split,variant=variant,level=level,group=group)
        summary_rows.append(dict(identity,**metrics))
        class_rows.extend(dict(identity,**r) for r in classes)
        confusion_rows.extend(dict(identity,label=c,prediction=p,count=int(matrix[c,p])) for c in range(3) for p in range(3))
    sf=['job','split','variant','level','group','images','accuracy','balanced_accuracy_observed','nll','brier','mean_two_region_dice','human_point_images','human_point_mae']
    cf=['job','split','variant','level','group','class_index','support','predicted','tp','fp','fn','recall','precision','f1']
    write_csv(folder/'summary/cohort_group_metrics.csv',summary_rows,sf)
    write_csv(folder/'summary/class_metrics.csv',class_rows,cf)
    write_csv(folder/'summary/confusion.csv',confusion_rows,['job','split','variant','level','group','label','prediction','count'])
    baseline=[r for r in summary_rows if r['level']=='cohort' and r['variant']=='normal']
    info=dict(status=read(folder/'CURSOR.json')['status'],family='TREAD-ONE',fixture=contract['fixture'],
       images_per_checkpoint=len(contract['rows']),jobs=len(contract['jobs']),prediction_rows=len(predictions),
       expected_prediction_rows=len(contract['rows'])*len(contract['jobs'])*len(VARIANTS),
       all_metrics_are_posthoc=True,training_updates=0,test_images_evaluated=0,baseline_cohorts=baseline,
       no_new_independent_tyres=True,gradient_scope='One fixed nominal training image per available class; no optimizer update; no historical gradient claim.')
    atomic(folder/'RESULTS.json',info)
    events=[]
    if (folder/'EVENTS.jsonl').exists():
        for line in (folder/'EVENTS.jsonl').read_text(encoding='utf-8').splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
    write_csv(folder/'summary/events.csv',events,['event_id','time_utc','kind','unit','job','seconds','message','last_verified_remote_revision'])
    markdown=['# TREAD-ONE — classifier diagnosis','',
      '**Post-hoc FP32 inference with fixed r1 weights. No training, new test selection or new independent tyres.**','',
      f"Status: **{info['status']}**. Fixture: **{info['fixture']}**. {len(predictions):,} / {info['expected_prediction_rows']:,} prediction/intervention rows retained.",'',
      '| Checkpoint | Cohort | Images | Observed-class balanced accuracy | Dice | Human point MAE / width |',
      '|---|---|---:|---:|---:|---:|']
    for r in baseline:
        point=f"{100*r['human_point_mae']:.3f}%" if r['human_point_mae'] is not None else 'unavailable'
        markdown.append(f"| {r['job']} | {r['split']} | {r['images']} | {100*r['balanced_accuracy_observed']:.3f}% | {100*r['mean_two_region_dice']:.3f}% | {point} |")
    markdown+=['','## Gate and input interventions','',
      '| Checkpoint | Cohort | Intervention | Balanced accuracy | NLL |',
      '|---|---|---|---:|---:|']
    for r in summary_rows:
        if r['level']=='cohort': markdown.append(f"| {r['job']} | {r['split']} | {r['variant']} | {100*r['balanced_accuracy_observed']:.3f}% | {r['nll']:.5f} |")
    markdown+=['','## Interpretation limits','',
      '- Nominal train-image performance is not historical augmented-batch training accuracy.',
      '- Validation contains low/high only; missing mid recall is unavailable, never zero evidence for an unseen class.',
      '- Uniform/oracle gates hold learned features and the classifier fixed; they are interventions, not refitted competing models.',
      '- Oracle masks define evaluation-only interventions. Background neutralization uses normalized zero (ImageNet mean), not black pixels.',
      '- No checkpoint is reselected. Later use of the observed test cohort would be development analysis.',
      '- Gradient alignment is a fixed present-batch diagnostic; it cannot reconstruct unlogged training history.',
      '- Images in the explorer are a fixed first example per split/tyre-or-clip/checkpoint; they are illustrative, not a random failure sample.',
      '- Direct CSV contains scalar records; gallery/arrays.npy preserves full FP32 masks, latent maps, signed evidence and gates for gallery images.',
      '- Additional image views and more telemetry are not additional independent tyres.','']
    (folder/'FINDINGS.md').write_text('\n'.join(markdown),encoding='utf-8')
    visuals=[]
    if (folder/'gallery/METADATA.json').exists():
        metadata=read(folder/'gallery/METADATA.json')['images']
        arrays=np.load(folder/'gallery/arrays.npy',mmap_mode='r',allow_pickle=False)
        for i,meta in enumerate(metadata):
            packed={}
            for variant in VARIANTS:
                index=0 if variant in ('normal','uniform_gate','oracle_tread_gate') else 1 if variant=='background_neutralized' else 2
                a=arrays[i,index];gate=a[17]
                if variant=='uniform_gate': gate=np.ones_like(gate)/gate.size
                if variant=='oracle_tread_gate': gate=a[18]
                if variant in ('uniform_gate','oracle_tread_gate'):
                    evidence=np.einsum('khw,ck->chw',a[2:14],np.array(meta['class_weights'],dtype=np.float32))*gate
                else: evidence=a[14:17]
                # Preview downsampling/quantization affects display only; full FP32 maps remain saved.
                layers=np.concatenate([a[:2],gate[None],evidence],axis=0).reshape(6,64,2,48,2).mean((2,4))
                scales=[1.,1.,float(np.max(layers[2]))]+[float(np.abs(layers[c]).max()) for c in range(3,6)]
                encoded=[]
                for c,layer in enumerate(layers):
                    if c<3: values=np.clip(layer/max(scales[c],1e-12)*255,0,255).astype(np.uint8)
                    else: values=np.clip(layer/max(scales[c],1e-12)*127+128,0,255).astype(np.uint8)
                    encoded.append(base64.b64encode(values.tobytes()).decode())
                packed[variant]=dict(layers=encoded,scales=scales,image=meta['images'][index])
            visuals.append({k:v for k,v in meta.items() if k not in ('images','class_weights','class_bias') }|dict(variants=packed,bias=meta['class_bias']))
    fields=['job','seed','endpoint','epoch','split','group','domain','image_id','label','variant','prediction','confidence','class_margin',
       'probability_0','probability_1','probability_2','mean_two_region_dice','human_point_mae','probability_l1_delta','evidence_residual']
    payload=dict(results=info,contract=contract,rows=[{k:r[k] for k in fields} for r in predictions],gallery=visuals)
    template=(root/'diagnostic/explorer.html').read_text(encoding='utf-8')
    document=template.replace('__TREAD_DATA__',json.dumps(payload,separators=(',',':')).replace('<','\\u003c'))
    target=folder/'TREAD_ONE_Classifier_Diagnostic.html';target.write_text(document,encoding='utf-8')
    assert target.stat().st_size<64*1024**2,'Explorer exceeds its documented 64 MiB output budget.'
