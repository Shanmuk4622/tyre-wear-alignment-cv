"""Full training-only image/probe report; no classifier selection or fitting."""
import base64
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image


def picture(rgb,size=(128,192)):
    image=Image.fromarray((np.asarray(rgb).transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8))
    image.thumbnail(size);stream=io.BytesIO();image.save(stream,format='JPEG',quality=80)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def build(snapshot,cfg):
    from common import atomic,csv_write
    snapshot=Path(snapshot);out=snapshot/'report';out.mkdir(exist_ok=True)
    def table(name):
        files=sorted((snapshot/'csv').glob('epoch_*/'+name+'.csv'))
        if not files:
            from gate_worker import SCHEMAS
            return pd.DataFrame(columns=SCHEMAS[name])
        return pd.concat([pd.read_csv(path) for path in files],ignore_index=True)
    source=table('source_train').fillna('');p=table('predictions');features=table('feature_consistency');bank=table('candidate_bank')
    quality=table('patch_quality');execution=table('execution')
    assert len(source)==386 and len(p)==2702 and len(features)==86464 and len(quality)==12352
    assert execution.optimizer_updates.eq(0).all() and execution.parameters_with_grad.eq(0).all()
    summary=[]
    for (cohort,variant),g in p.groupby(['cohort','variant']):
        recalls=g.groupby('class_index').correct.mean()
        summary.append(dict(cohort=cohort,variant=variant,n=len(g),accuracy=float(g.correct.mean()),balanced_accuracy=float(recalls.mean()),nll=float(g.nll.mean())))
    pd.DataFrame(summary).to_csv(out/'train_probe_metrics.csv',index=False)
    class_summary=p.groupby(['cohort','variant','class_index']).agg(n=('correct','size'),accuracy=('correct','mean'),nll=('nll','mean')).reset_index()
    class_summary.to_csv(out/'train_probe_class_metrics.csv',index=False)
    sourceclass=source[['image_index','class_index']]
    features=features.merge(sourceclass,on='image_index',validate='many_to_one')
    invariance=[]
    for (variant,cls),g in features.groupby(['variant','class_index']):
        unclipped=g[g.new_clip_fraction.eq(0)&g.valid_fraction.ge(.95)]
        invariance.append(dict(variant=variant,class_index=cls,patches=len(g),unclipped_patches=len(unclipped),
            rgb_median=float(unclipped.rgb_relative_l2.median()) if len(unclipped) else None,contrast_median=float(unclipped.local_contrast_relative_l2.median()) if len(unclipped) else None,
            rgb_p95=float(unclipped.rgb_relative_l2.quantile(.95)) if len(unclipped) else None,contrast_p95=float(unclipped.local_contrast_relative_l2.quantile(.95)) if len(unclipped) else None))
    pd.DataFrame(invariance).to_csv(out/'feature_invariance.csv',index=False)
    proposal=[]
    selected_ids=set(bank.image_id)
    originals=source[source.domain=='original_photo']
    for (cls,tyre),g in originals.groupby(['class_index','physical_tyre_id']):
        retained=g[~g.image_id.isin(selected_ids)]
        proposal.append(dict(class_index=cls,physical_tyre_id=tyre,original_photos=len(g),candidate_support_photos=int(g.image_id.isin(selected_ids).sum()),remaining_original_query_photos=len(retained)))
    pd.DataFrame(proposal).to_csv(out/'candidate_query_coverage.csv',index=False)
    valid_proposal=len(bank)==18 and bank.image_id.nunique()==18 and bank.groupby('class_index').size().to_dict()=={0:6,1:6,2:6} and all(r['remaining_original_query_photos']>=1 for r in proposal)
    assert len(p.image_index.unique())==386 and not source.image_id.isna().any()
    result=dict(status='completed_train_only_gate',images=386,v1_queries=378,support_images=8,variants=7,predictions=2702,
        candidate_bank_structural_pass=bool(valid_proposal),candidate_images=int(bank.image_id.nunique()),candidate_physical_tyres=int(bank.physical_tyre_id.nunique()),
        candidate_class_tyre_support={str(c):int(g.physical_tyre_id.nunique()) for c,g in bank.groupby('class_index')},
        optimizer_updates=0,validation_or_test_forwards=0,new_independent_tyres=0,measured_v2_classifier=False,
        scope='Frozen selected v1 probes and an untrained reference/feature proposal; not classifier promotion or physical wear/safety proof')
    atomic(out/'GATE_RESULT.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for cohort,color in [('v1_query','#16806e'),('support_photo','#985fa9')]:
        rows=pd.DataFrame(summary).query('cohort == @cohort').set_index('variant').reindex(cfg['contract']['rules']['variants'])
        axes[0].plot(range(7),rows.balanced_accuracy,marker='o',label=cohort,color=color)
    axes[0].set_xticks(range(7),cfg['contract']['rules']['variants'],rotation=35,ha='right');axes[0].set(ylabel='Training-only fixed-checkpoint BA',ylim=(0,1.05));axes[0].legend()
    x=features[features.variant.isin(['brightness_080','brightness_120'])&features.new_clip_fraction.eq(0)&features.valid_fraction.ge(.95)]
    axes[1].boxplot([x.rgb_relative_l2.to_numpy(),x.local_contrast_relative_l2.to_numpy()],tick_labels=['RGB relative L2','Local-contrast relative L2'],showfliers=False)
    axes[1].set(ylabel='Appearance change / original feature norm',title='Unclipped, supported brightness-probe patches\nFeature consistency is not classification accuracy')
    fig.suptitle('ATLAS train-only gate · frozen v1 · zero optimizer updates')
    fig.savefig(out/'appearance_gate.png',dpi=160);plt.close(fig)
    rgb=np.load(snapshot/'artifacts/candidate_rgb.npy',allow_pickle=False) if len(bank) else np.zeros((0,3,64,64),dtype=np.float32)
    lc=np.load(snapshot/'artifacts/candidate_local_contrast.npy',allow_pickle=False) if len(bank) else np.zeros((0,4,64,64),dtype=np.float32)
    fig,axes=plt.subplots(3,6,figsize=(13,7),layout='constrained')
    for ax,row,patch in zip(axes.flat,bank.itertuples(index=False),rgb):
        ax.imshow(patch.transpose(1,2,0).clip(0,1));ax.set_title(f"{['low','mid','high'][row.class_index]} · case {row.prototype}\nphoto {row.image_index} · patch {row.patch}",fontsize=9);ax.axis('off')
    fig.suptitle('UNTRAINED reference proposal · 18 distinct training photos · original pixels\nStill only 1 low / 3 mid / 4 high known training tyres; not depth or new data')
    fig.savefig(out/'candidate_bank.png',dpi=160);plt.close(fig)
    cases=[]
    for row in source.itertuples(index=False):
        packet=torch.load(snapshot/f'artifacts/train/{row.image_index:04d}.pt',map_location='cpu',weights_only=False)
        cases.append(dict(image_id=row.image_id,class_index=row.class_index,domain=row.domain,tyre=row.physical_tyre_id,
            cohort='support_photo' if row.is_v1_bank else 'v1_query',atlas=picture(packet['atlas']['rgb']),probes=packet['probes']))
    data=dict(cases=cases,bank=[dict(image_id=r.image_id,tyre=r.physical_tyre_id,class_index=int(r.class_index),rgb=picture(rgb[i],(64,64)),contrast=picture(np.repeat(((lc[i,:1]+1)/2),3,axis=0),(64,64))) for i,r in enumerate(bank.itertuples(index=False))],result=result,metrics=summary)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS training-only gate</title><style>body{font:16px system-ui;background:#102330;color:#e0edf2;max-width:1200px;margin:auto;padding:24px}p{line-height:1.5}.notice{background:#293844;border-left:4px solid #e7b166;padding:14px}select,input,button{font:inherit;padding:8px;margin:5px;background:#294654;color:inherit;border:1px solid #7597a5;border-radius:6px}.case{display:grid;grid-template-columns:120px 1fr;gap:20px;border:1px solid #416479;border-radius:10px;padding:14px;margin:14px 0}.case img{max-width:110px}.bank{display:flex;flex-wrap:wrap;gap:12px}.bank article{width:145px;font-size:12px;border:1px solid #537381;padding:8px}.bank img{width:64px}table{border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #456474}@media(max-width:650px){.case{grid-template-columns:1fr}}</style><h1>ATLAS · training-only appearance/reference gate</h1><p class="notice">Frozen selected v1 classifier. No fitting, validation/test forward passes or new tyres. Original and proposed support photos are part of training data; support-photo results are separated from v1 queries. The proposed 18-case bank is UNTRAINED and must not be installed in the old model. Chroma/glare flags are image heuristics, not human paint/wear labels.</p><h2>Train-only probe summary</h2><div id="summary"></div><h2>Distinct-view reference proposal</h2><p>Original RGB and the normalized-luminance preview are shown side-by-side. These are deterministic feature previews, not depth maps. More views of a tyre are not more independent tyres.</p><div id="bank" class="bank"></div><h2>All training images</h2><select id="cohort"><option value="all">All training images</option><option value="v1_query">V1 query pool</option><option value="support_photo">Original support photos</option></select><select id="class"><option value="all">All classes</option><option value="0">Low</option><option value="1">Mid</option><option value="2">High</option></select><select id="variant">__VARIANTS__</select><input id="search" placeholder="Find image ID" aria-label="Find image ID"><p id="count"></p><button id="prev">Previous</button><button id="next">Next</button><div id="cases"></div><p>Atlas thumbnails are the original predicted visible 2D atlas. Probe scores use the named deterministic transformation; transformed pixels are reproducible from retained native RGB, valid masks and source rules. No held-out endpoint is used here.</p><script type="application/json" id="data">__DATA__</script><script>'use strict';const d=JSON.parse(document.querySelector('#data').textContent),$=s=>document.querySelector(s);let page=0;let t=document.createElement('table');for(const r of d.metrics){let tr=document.createElement('tr');for(const value of [r.cohort,r.variant,r.n,Number(r.balanced_accuracy).toFixed(4),Number(r.nll).toFixed(4)]){let td=document.createElement('td');td.textContent=value;tr.append(td)}t.append(tr)}$('#summary').append(t);for(const b of d.bank){let a=document.createElement('article');for(const src of [b.rgb,b.contrast]){let im=document.createElement('img');im.src=src;im.alt=b.image_id;a.append(im)}let p=document.createElement('p');p.textContent='Class '+b.class_index+' · '+b.image_id+' · tyre '+b.tyre;a.append(p);$('#bank').append(a)}function render(){let rows=d.cases.filter(x=>($('#cohort').value==='all'||x.cohort===$('#cohort').value)&&($('#class').value==='all'||x.class_index===Number($('#class').value))&&x.image_id.toLowerCase().includes($('#search').value.toLowerCase()));page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(rows.length/8)-1)));$('#count').textContent=rows.length+' images · page '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/8));$('#cases').replaceChildren();for(const r of rows.slice(page*8,page*8+8)){let a=document.createElement('article');a.className='case';let im=document.createElement('img');im.src=r.atlas;im.alt=r.image_id;let text=document.createElement('div'),p=document.createElement('p');const q=r.probes.find(x=>x.variant===$('#variant').value);p.textContent=r.image_id+' · '+r.cohort+' · target '+r.class_index+' → '+q.prediction+' · probabilities '+[0,1,2].map(c=>Number(q['probability_'+c]).toFixed(4)).join(' / ')+' · supported '+q.supported_patches+'/32';text.append(p);a.append(im,text);$('#cases').append(a)}}for(const s of ['#cohort','#class','#variant','#search'])$(s).addEventListener('input',()=>{page=0;render()});$('#prev').onclick=()=>{page--;render()};$('#next').onclick=()=>{page++;render()};render();</script></html>'''
    html=html.replace('__VARIANTS__',''.join('<option>'+v+'</option>' for v in cfg['contract']['rules']['variants']))
    html=html.replace("let t=document.createElement('table');", "let t=document.createElement('table'),header=document.createElement('tr');for(const title of ['Training cohort','Probe','Images','Balanced accuracy','NLL']){let th=document.createElement('th');th.textContent=title;header.append(th)}t.append(header);")
    (out/'Training_Gate_Explorer.html').write_text(html.replace('__DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    from gate_worker import SCHEMAS
    dictionary=[]
    for family,fields in SCHEMAS.items():
        for field in fields:
            units='class-logit units' if field.startswith('logit') or field in ('contribution','evidence_residual') else 'seconds' if 'seconds' in field else 'bytes' if 'bytes' in field else 'dimensionless / identifier as named'
            dictionary.append(dict(table=family,column=field,units=units,missing='Blank means unavailable; not numeric zero',scope='Training-only frozen-model gate'))
    csv_write(out/'FIELD_DICTIONARY.csv',dictionary,['table','column','units','missing','scope'])
    (out/'README.md').write_text('''# Training-only ATLAS appearance/reference gate

All 386 source images retain role=train. V1 query/support photos are reported
separately. Seven fixed probes use the frozen selected epoch18 classifier;
there is no optimizer, new training, validation/test query or new independent
tyre. New local-contrast channels are a deterministic feature proposal, not a
trained replacement classifier or a depth map. The same frozen v1 image is
canonicalized once; transformations occur after its predicted atlas.

The proposed bank uses original training photos only, one case per distinct
photo, least-represented available physical tyre first, then fixed descriptor
diversity/coverage/glare score. It must retain at least one original query photo
per known training tyre. Low texture is flagged, not excluded as presumed worn
tread. Chroma/glare heuristics are not human annotations or causal attribution.
Candidate pixel arrays are direct .npy and explicitly UNTRAINED. Do not install
them in the v1 classifier whose coefficients were trained against a different bank.

Local contrast: channel-mean luminance; masked 9×9 local mean/std; std floor .02;
clip standardized luminance to ±3 and divide by3; Sobel x/y and gradient magnitude.
Relative L2 is change/original feature norm, not accuracy. Input clipping, the
std floor and flat regions limit scale equivariance. The unclipped supported
subset and denominators are explicit in feature_invariance.csv.

Raw CSVs and native original RGB/valid/grid, dense predictions and descriptors
are retained. Directory names epoch_0000… come from the reused cursor store:
they are blocks of50 input images, not training epochs. Every image is one
atomic recovery unit coupled to CSV byte/row cursors and native artifacts.
Candidate assembly is a separate final unit; interruption resumes without
changing the frozen source/environment/weights. Actual source hashes and role
registry are included. Native .pt/.npy are direct; no data ZIP/NPZ is uploaded.

The source fit and workstations remain untouched. Better consistency of these
fixed features does not establish unseen-tyre classification, calibration,
physical wear/depth/alignment or safety. Known original train tyre support is
still only 1/3/4 by class; video parent mapping remains unresolved. Any future
model/data/input/bank/objective/sampler change requires a new frozen protocol.
''',encoding='utf-8')
