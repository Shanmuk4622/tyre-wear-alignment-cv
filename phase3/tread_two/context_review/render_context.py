"""Static scientific figures and a self-contained review of saved ATLAS rows."""
import base64
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def render(out, inputs):
    out=Path(out)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    def csv(name): return pd.read_csv(out/(name+'.csv'))
    cohort=csv('cohort_metrics');classes=csv('class_metrics')
    identity=csv('prediction_audit').query("stage in ['selected_eval','final_eval']")
    probes=cohort.query("stage in ['selected_eval','final_eval','diagnostics']")
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for stage,color in [('train','#167c69'),('validate','#9352b0')]:
        rows=cohort[(cohort.stage==stage)&((cohort.endpoint=='paired_online_combined') if stage=='train' else True)]
        axes[0,0].plot(rows.epoch,rows.balanced_accuracy_observed,label=stage,color=color)
        axes[0,1].plot(rows.epoch,rows.nll,label=stage,color=color)
    axes[0,0].set(ylabel='Balanced accuracy (observed target classes)',xlabel='Epoch',ylim=(0,1.05))
    axes[0,1].set(ylabel='Three-class NLL',xlabel='Epoch')
    for ax in axes[0]: ax.axvline(7,color='#555',ls='--',lw=1);ax.legend();ax.grid(alpha=.2)
    for endpoint,ax in zip(['selected','final'],axes[1]):
        rows=identity[identity.endpoint==endpoint]
        cm=np.zeros((3,3),int)
        np.add.at(cm,(rows.label.to_numpy(int),rows.prediction.to_numpy(int)),1)
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=62)
        for i in range(3):
            for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>35 else 'black',fontsize=15)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['low','mid','high'],yticklabels=['low','mid','high'],xlabel='Prediction',ylabel='Target',title=f'{endpoint}: low correct {cm[0,0]}/62; high correct {cm[2,2]}/41')
    fig.suptitle('ATLAS-CONTEXT completed pilot · frozen LC prior plus learned spatial correction\nTraining scores are online sampled observations, not a fixed final training-set evaluation')
    fig.savefig(figures/'fit_and_confusion.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    variants=['identity','brightness_080','brightness_120','gamma_080','gamma_120','grayscale_both','occlude_top_patch','reverse_bands','reverse_columns']
    for endpoint,color,offset in [('selected','#158473',-.18),('final','#a34e9a',.18)]:
        rows=probes[probes.endpoint==endpoint].set_index('variant').loc[variants]
        axes[0].bar(np.arange(9)+offset,rows.balanced_accuracy_observed,width=.35,label=endpoint,color=color)
        axes[1].plot(range(9),rows.nll,marker='o',label=endpoint,color=color)
    for ax in axes[:2]: ax.set_xticks(range(9),['Original','Dim ×0.8','Bright ×1.2','Gamma0.8','Gamma1.2','Gray both','Top patch','Reverse bands','Reverse columns'],rotation=25,ha='right');ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Balanced accuracy',ylim=(0,1))
    axes[1].set(ylabel='Three-class NLL')
    aff=csv('reference_class_affinities').query("endpoint == 'selected' and variant == 'identity'")
    matrix=aff.pivot(index='label',columns='reference_class',values='mean_affinity').reindex(index=[0,2],columns=[0,1,2])
    image=axes[2].imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    for i in range(2):
        for j in range(3): axes[2].text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',color='white' if matrix.iloc[i,j]>.55 else 'black')
    axes[2].set(xticks=range(3),xticklabels=['low refs','mid refs','high refs'],yticks=range(2),yticklabels=['low target','high target'],title='Selected pooled reference affinity\nClass averages, six fixed cases each')
    fig.colorbar(image,ax=axes[2],shrink=.65)
    fig.suptitle('Saved interventions change appearance; they do not establish causal shortcut attribution')
    fig.savefig(figures/'probes_and_affinities.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    exposure=csv('query_exposure')
    for cls,color in [(0,'#3180af'),(1,'#e49c35'),(2,'#9566ad')]:
        rows=exposure[exposure.class_index==cls]
        axes[0].hist(rows.training_view_observations,bins=20,alpha=.65,label=['low','mid','high'][cls],color=color)
    axes[0].set(xlabel='Training view observations per image (2 per draw)',ylabel='Images');axes[0].legend()
    embed=csv('bank_embedding_cosines').query("endpoint == 'selected'").pivot(index='prototype_a',columns='prototype_b',values='cosine')
    im=axes[1].imshow(embed,cmap='coolwarm',vmin=-1,vmax=1);axes[1].set(xlabel='Literal reference index',ylabel='Literal reference index',title='Selected bank embedding cosines')
    for position in [5.5,11.5]: axes[1].axhline(position,color='k',lw=.8);axes[1].axvline(position,color='k',lw=.8)
    fig.colorbar(im,ax=axes[1],shrink=.7)
    calibration=csv('endpoint_calibration')
    axes[2].plot([0,1],[0,1],ls='--',color='#777')
    for endpoint,color in [('selected','#158473'),('final','#a34e9a')]:
        rows=calibration[(calibration.endpoint==endpoint)&calibration.n.gt(0)]
        axes[2].scatter(rows.mean_confidence,rows.accuracy,s=rows.n*4,label=endpoint,color=color,alpha=.7)
    axes[2].set(xlim=(0,1),ylim=(-.05,1.05),xlabel='Mean raw confidence',ylabel='Observed accuracy',title='Endpoint calibration · bubbles ∝ images');axes[2].legend()
    fig.suptitle('Existing cohort only · no new independent tyres; absent held-out mid targets')
    fig.savefig(figures/'exposure_embedding_calibration.png',dpi=160);plt.close(fig)

    original=inputs.file('report/ATLAS_Evidence_Explorer.html').read_text(encoding='utf-8')
    data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',original,re.S)[1])
    image_data={row['image_id']:dict(picture=row['picture'],atlas_picture=row['atlas_picture']) for row in data['cases']}
    predictions=csv('prediction_audit').query("stage in ['selected_eval','final_eval','diagnostics']")
    contributions=csv('reference_contributions')
    patch=csv('endpoint_patch_evidence')
    delta=csv('endpoint_context_evidence')
    summary=csv('context_observations').set_index('sample')
    attention=json.loads((out/'ATTENTION_VIEW.json').read_text())
    explanation={}
    for sample,rows in predictions.groupby('sample'):
        row=rows.iloc[0].to_dict()
        row.update(image_data[row['image_id']])
        row['patches']=patch[patch['sample']==sample].pivot(index='patch',columns='class_index',values='contribution').to_numpy().tolist()
        extra=delta[delta['sample']==sample].sort_values('patch')
        row['context_patches']=extra[['delta_0','delta_1','delta_2']].to_numpy().tolist()
        row['base_patches']=(np.array(row['patches'])-np.array(row['context_patches'])).tolist()
        row['source_patch']=extra.source_patch.astype(int).tolist()
        sr=summary.loc[sample]
        row['base_logits']=[float(sr[f'base_logit_{c}']) for c in range(3)]
        row['bias_delta']=[float(sr[f'bias_delta_{c}']) for c in range(3)]
        row['attention']=attention.get(str(int(sample)))
        row['references']=contributions[contributions['sample']==sample].pivot(index='class_index',columns='prototype',values='signed_contribution').to_numpy().tolist()
        explanation[str(int(sample))]=row
    payload=dict(cases=json.loads(pd.DataFrame(explanation.values()).to_json(orient='records')),bank=data['bank'],
        cohort=records(probes),pairs=records(csv('probe_pairs')),bank_sources=records(inputs.table('bank_cases')),
        source_revision=json.loads((out/'ANALYSIS.json').read_text())['source_revision'],source_protocol=inputs.manifest['contract'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS-CONTEXT completed pilot review</title>
<style>body{font:16px system-ui;background:#0c1b26;color:#e3edf3;max-width:1250px;margin:auto;padding:24px}h1,h2{line-height:1.2}p{line-height:1.6}.notice{border-left:4px solid #edb95f;background:#243844;padding:12px}.controls{display:flex;flex-wrap:wrap;gap:8px}select,input,button{font:inherit;background:#203d4f;color:inherit;border:1px solid #628196;padding:9px;border-radius:7px}.case{border:1px solid #416478;padding:16px;margin:16px 0;border-radius:12px;display:grid;grid-template-columns:210px 1fr;gap:20px}.photos img{max-width:100px;max-height:190px;margin:3px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;max-width:220px}.grid div{height:28px;border-radius:3px}.refs{display:flex;flex-wrap:wrap;gap:8px}.refs img{width:55px}.refs span{display:inline-flex;flex-direction:column;font-size:12px}table{border-collapse:collapse;width:100%;margin:15px 0;font-size:14px}th,td{border-bottom:1px solid #496271;text-align:right;padding:8px}th:first-child,td:first-child{text-align:left}a{color:#7adcc5}.wrong{color:#ffb2a2}.correct{color:#89e0bf}@media(max-width:750px){.case{grid-template-columns:1fr}body{padding:12px}table{font-size:12px}}</style>
<h1>TREAD-TWO / ATLAS-CONTEXT r3 · completed pilot review</h1><p class="notice">No retraining or new inference. The declared selected checkpoint is epoch 7, final is epoch 60. The original validation rule selected epoch 7. The tables show actual results; no new selection, fitting or inference is performed. Reference cards explain the frozen LC branch; combined tiles include cross-patch context correction. There are no held-out mid targets. Richer rows do not add independent tyres. Scores and signed reference evidence are not physical wear or safety truth.</p>
<h2>Saved interventions</h2><div id="summary"></div><p>LC uses channel-mean luminance, so grayscale stability is built into this representation. Brightness and gamma change the query only. Occlusion replaces the top <em>signed</em> patch for the base predicted class with its saved per-channel mean replacement. These interventions cannot isolate paint, illumination, tyre identity or camera effects.</p>
<h2>All endpoint and probe records</h2><div class="controls"><select id="endpoint"><option>selected</option><option>final</option></select><select id="variant"><option value="identity">Original</option><option value="grayscale_both">Grayscale query + bank</option><option value="brightness_080">Query brightness ×0.8</option><option value="brightness_120">Query brightness ×1.2</option><option value="gamma_080">Query gamma0.8</option><option value="gamma_120">Query gamma1.2</option><option value="occlude_top_patch">Top signed patch replaced</option><option value="reverse_bands">Reverse band order</option><option value="reverse_columns">Reverse columns</option></select><select id="component"><option value="combined">Combined evidence</option><option value="base">Frozen LC evidence</option><option value="context">Context correction</option></select><select id="layer"><option value="0">Attention layer 1</option><option value="1">Attention layer 2</option></select><select id="head"><option value="0">Head 1</option><option value="1">Head 2</option><option value="2">Head 3</option><option value="3">Head 4</option></select><select id="target"><option value="all">All target classes</option><option value="0">Low targets</option><option value="2">High targets</option></select><select id="filter"><option value="all">All records</option><option value="wrong">Wrong predictions</option><option value="flip">Prediction changes</option></select><select id="class"><option value="pred">Explain predicted class</option><option value="0">Explain low</option><option value="1">Explain mid</option><option value="2">Explain high</option></select><input id="search" placeholder="Find image ID" aria-label="Find image ID"></div><p id="count"></p><button id="previous">Previous</button> <button id="next">Next</button><div id="cases"></div>
<p>Page size is eight images. Green/red tiles show positive/negative class-logit contributions, scaled separately for each displayed image/class. Hover for exact values. Reference cards show the frozen LC branch only. Tile components separate base/context/combined evidence. Layout interventions rearrange tokens; thumbnails show the original image. Base/probe probability changes are paired by image and checkpoint, never by row order.</p>
<script type="application/json" id="data">__DATA__</script><script>'use strict';const d=JSON.parse(document.querySelector('#data').textContent),$=s=>document.querySelector(s),labels=['low','mid','high'];let page=0;const bases=new Map(d.cases.filter(x=>x.variant==='identity').map(x=>[x.endpoint+'|'+x.image_id,x]));function fmt(x){return Number(x).toFixed(4)}function text(parent,s){let p=document.createElement('p');p.textContent=s;parent.append(p)}let table=document.createElement('table'),head=document.createElement('tr');for(const title of ['Endpoint / variant','N','BA','NLL','Low predictions','Mid predictions','High predictions']){let th=document.createElement('th');th.textContent=title;head.append(th)}table.append(head);for(const r of d.cohort){let tr=document.createElement('tr');for(const v of [r.endpoint+' / '+r.variant,r.n,fmt(r.balanced_accuracy_observed),fmt(r.nll),r.predicted_low,r.predicted_mid,r.predicted_high]){let td=document.createElement('td');td.textContent=v;tr.append(td)}table.append(tr)}$('#summary').append(table);function render(){let rows=d.cases.filter(x=>x.endpoint===$('#endpoint').value&&x.variant===$('#variant').value&&($('#target').value==='all'||x.label===Number($('#target').value))&&x.image_id.toLowerCase().includes($('#search').value.toLowerCase()));rows=rows.filter(x=>{const b=bases.get(x.endpoint+'|'+x.image_id);return $('#filter').value==='all'||($('#filter').value==='wrong'?x.correct===0:x.prediction!==b.prediction)});page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(rows.length/8)-1)));$('#count').textContent=rows.length+' records · page '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/8));$('#cases').replaceChildren();for(const x of rows.slice(page*8,page*8+8)){const b=bases.get(x.endpoint+'|'+x.image_id),cls=$('#class').value==='pred'?x.prediction:Number($('#class').value);let article=document.createElement('article');article.className='case';let photos=document.createElement('div');photos.className='photos';for(const [src,alt] of [[x.picture,x.image_id],[x.atlas_picture,'Saved base atlas: '+x.image_id]]){let im=document.createElement('img');im.src=src;im.alt=alt;photos.append(im)}let grid=document.createElement('div');grid.className='grid';const tiles=$('#component').value==='base'?x.base_patches:($('#component').value==='context'?x.context_patches:x.patches);const scale=Math.max(...tiles.map(v=>Math.abs(v[cls])),1e-12);tiles.forEach((v,i)=>{let tile=document.createElement('div');tile.style.background=v[cls]>=0?'rgba(63,223,170,'+Math.abs(v[cls])/scale+')':'rgba(249,117,104,'+Math.abs(v[cls])/scale+')';tile.title='patch '+i+' / original patch '+x.source_patch[i]+' / '+labels[cls]+' / '+v[cls];grid.append(tile)});photos.append(grid);let detail=document.createElement('div');text(detail,x.image_id);text(detail,'Target '+labels[x.label]+' → '+labels[x.prediction]+'; original prediction '+labels[b.prediction]);text(detail,'Frozen LC logits: '+x.base_logits.map(fmt).join(' / '));text(detail,'Context bias correction: '+x.bias_delta.map(fmt).join(' / '));text(detail,'Scores low / mid / high: '+[0,1,2].map(c=>fmt(x['probability_'+c])).join(' / '));text(detail,'True probability change '+fmt(x.true_probability-b.true_probability)+'; NLL change '+fmt(x.nll-b.nll));text(detail,'Supported patches '+x.supported_patches+'/32; clipping '+fmt(x.clipped_fraction)+'; signed accounting residual '+Number(x.evidence_residual).toExponential(2));text(detail,'Showing evidence for '+labels[cls]+'. Saved atlas thumbnail is the original; patch/reference numbers reflect the selected intervention.');if(x.variant==='occlude_top_patch')text(detail,'Replaced patch '+x.occluded_patch+'; RGB '+[x.replacement_r,x.replacement_g,x.replacement_b].map(fmt).join(' / '));let refs=document.createElement('div');refs.className='refs';let ranked=x.references[cls].map((value,index)=>({value,index}));ranked.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));for(const r of ranked.slice(0,6)){let span=document.createElement('span'),im=document.createElement('img'),caption=document.createElement('span');im.src=d.bank[r.index];im.alt='Actual training reference '+r.index;const src=d.bank_sources.find(z=>z.prototype_index===r.index);im.title=src.source_image_id+' / tyre '+src.source_tyre_id;caption.textContent='case '+r.index+' '+(r.value>=0?'+':'')+fmt(r.value);span.append(im,caption);refs.append(span)}text(detail,'Reference cards account for the frozen case branch only; context is shown separately.');if(x.variant==='identity'&&x.attention){const layer=Number($('#layer').value),head=Number($('#head').value);let ag=document.createElement('div');ag.style.display='grid';ag.style.gridTemplateColumns='repeat(32,4px)';for(let q=0;q<32;q++)for(let k=0;k<32;k++){let cell=document.createElement('div'),v=x.attention[layer][head][q][k];cell.style.height='4px';cell.style.background='rgba(95,180,250,'+Math.min(1,v*8)+')';cell.title='query '+q+' key '+k+' probability '+v;ag.append(cell)}text(photos,'Saved attention: rows query, columns key. Hover for probabilities; not causal wear truth.');photos.append(ag)}else{text(photos,'Full attention matrices were retained for original endpoints only.')}detail.append(refs);article.append(photos,detail);$('#cases').append(article)}}for(const selector of ['#endpoint','#variant','#target','#filter','#class','#search','#component','#layer','#head'])$(selector).addEventListener('input',()=>{page=0;render()});$('#previous').onclick=()=>{page--;render()};$('#next').onclick=()=>{page++;render()};render();</script></html>'''
    (out/'ATLAS_Failure_Explorer.html').write_text(page.replace('__DATA__',json.dumps(payload,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    methods='''# ATLAS-CONTEXT completed pilot — post-hoc methods

All new CSVs are derived from immutable saved NB13 records. No Torch/model
loading, optimizer, new inference, new checkpoint selection or new tyre is used.
Source revision, input hashes and exact row counts are retained in ANALYSIS.json
and input_inventory.csv. Missing values mean unavailable, including absent
mid-class targets. Counts of repeated training draws are not unique image counts.

## Calculations and units

| Quantity | Definition / units |
|---|---|
| BA observed | Mean recall over target classes actually present; fraction |
| NLL | Negative natural log of saved true-class probability, three classes |
| Brier | Sum of squared differences from one-hot target, three classes |
| ECE | Ten equal-width maximum-confidence bins, weighted absolute accuracy gap |
| True margin | True-class logit minus largest other-class logit |
| Pooled reference affinity | Sum over 32 patches of saved geometric mass × exp((cosine−1)/temperature) |
| Reference contribution | Pooled affinity × the saved signed class/reference coefficient |
| Reconstructed logit | Sum of 18 frozen reference contributions plus weighted context correction and combined class bias |
| Patch concentration | Largest absolute patch contribution / total absolute mass, for predicted class |
| Probe delta | Probe minus identity, same image and checkpoint; never a different epoch/model |
| Training accuracy | Online pre-update draws with replacement and augmentation, mixing model states; not a fixed final training-set test |
| Dense error | Fraction of image width; only comparable guide rows and actual target kind |

The grayscale probe modifies both queries and bank, brightness modifies query
only, and the occlusion probe replaces a single top-signed patch with its saved
per-channel mean. None uniquely isolates paint, identity, light or capture
devices. Exact additive accounting is not causal explanation. No confidence
threshold is selected for deployment. No image-wise confidence interval is
reported as independent-tyre evidence. Existing low/high validation and test
roles are preserved; test is an already-inspected development endpoint.

The figure/model-selection and probability checks are independent calculations
from CSV. The original frozen anchor remains the control on original-frame RGB,
not the transformed query. No comparisons suggest transformed-anchor forwards.
All original native states and CSVs remain in their original HF namespace.
'''
    (out/'METHODS.md').write_text(methods,encoding='utf-8')
    audit=json.loads((out/'ANALYSIS.json').read_text())
    lines=['# Derived CSV inventory','','All tables below are new post-hoc derivatives of saved NB13 evidence. The original raw tables remain in their original fit namespace. Repeated draws and probes do not increase independent tyre count.','','| Table | Rows | Columns |','|---|---:|---:|']
    for name,meta in audit['csv_tables'].items():
        lines.append(f"| [{name}.csv]({name}.csv) | {meta['rows']:,} | {len(meta['columns'])} |")
    lines+=['','[Field dictionary](table_dictionary.csv) · [Calculation methods](METHODS.md) · [Interactive image/probe explorer](ATLAS_Failure_Explorer.html).','', 'All source hashes are retained in input_inventory.csv. Empty fields are unavailable/not applicable, especially absent mid-class target recall. Source masks/points are frozen predictions, not new physical wear measurements.']
    (out/'CSV_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_context_notes(out)



def write_context_notes(out):
    from context_checks import write_notes
    write_notes(Path(out))
