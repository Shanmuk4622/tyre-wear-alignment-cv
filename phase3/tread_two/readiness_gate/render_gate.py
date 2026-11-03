"""Durable scientific figures and an offline interactive evidence passport."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def render(out):
    out=Path(out);figures=out/'figures';figures.mkdir(exist_ok=True)
    support=pd.read_csv(out/'training_class_support.csv');folds=pd.read_csv(out/'conditional_fold_class_support.csv');variants=pd.read_csv(out/'online_class_variants.csv');groups=pd.read_csv(out/'online_component_variants.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained');x=np.arange(3)
    axes[0].bar(x,support.images,color=['#257c9b','#c28f31','#8862a7']);axes[0].set(xticks=x,xticklabels=['low','mid','high'],ylabel='Existing training images')
    axes[1].bar(x-.18,support.known_original_tyres,width=.35,label='Known original tyres');axes[1].bar(x+.18,support.conservative_components,width=.35,label='Conservative components');axes[1].set(xticks=x,xticklabels=['low','mid','high'],ylabel='Groups; mid co-assignment is not identity');axes[1].legend(fontsize=8)
    matrix=folds.pivot(index='fold',columns='class_index',values='fit_components').to_numpy();axes[2].imshow(matrix,cmap='YlGnBu',vmin=0,vmax=4)
    for (r,c),v in np.ndenumerate(matrix):axes[2].text(c,r,str(v),ha='center',va='center',color='white' if v>2 else '#222')
    axes[2].set(xticks=x,xticklabels=['low','mid','high'],yticks=range(6),ylabel='Conditional withheld-component fold',title='Remaining fit components')
    fig.suptitle('Existing-data support audit · conditional folds, no held-component model fitting');fig.savefig(figures/'independent_support.png',dpi=160);plt.close(fig)
    colors={'frozen_prior':'#647f9b','bias_only':'#b08c31','spatial_only':'#8758a1','full_context':'#118577'}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for c,ax in enumerate(axes):
        for variant,color in colors.items():
            g=variants[variants.label.eq(c)&variants.variant.eq(variant)];ax.plot(g.epoch,g.online_nll,label=variant.replace('_',' '),color=color)
        ax.set(title=['low','mid','high'][c],xlabel='Epoch of saved online state',ylabel='NLL on sampled paired views');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Algebraic correction parts on identical saved training views · no independently fitted ablations');fig.savefig(figures/'correction_parts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    full=groups[groups.variant.eq('full_context')]
    matrix=full.pivot(index='leakage_component',columns='epoch',values='mean_delta_nll');limit=max(1e-8,np.abs(matrix.to_numpy()).max());image=axes[0].imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
    names=['mid co-assigned' if 'unresolved' in v else v.replace('tyre:','').replace('mileage_100000_plus__','high ').replace('new_tire__','low ') for v in matrix.index]
    axes[0].set(yticks=range(len(names)),yticklabels=names,xticks=[0,19,39,59],xticklabels=[1,20,40,60],xlabel='Epoch',title='Full minus frozen-prior online NLL');fig.colorbar(image,ax=axes[0],label='Negative = lower loss on same saved views')
    for field,color in [('rescued_prior_errors','#118577'),('spoiled_prior_corrects','#c4674d')]:
        sums=full.groupby('epoch')[field].sum();axes[1].plot(sums.index,sums,label=field.replace('_',' '),color=color)
    axes[1].set(xlabel='Epoch',ylabel='View observations; two per paired draw');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Training-only utility · repeated changing model states, not unseen-tyre performance');fig.savefig(figures/'utility_by_component.png',dpi=160);plt.close(fig)
    analysis=json.loads((out/'ANALYSIS.json').read_text());summary=analysis['summary'];caps=json.loads((out/'CAPABILITIES.json').read_text())
    data=json.dumps(dict(summary=summary,capabilities=caps,components=json.loads(groups.to_json(orient='records')),support=support.to_dict('records')),separators=(',',':'),allow_nan=False).replace('<',chr(92)+'u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TREAD-TWO evidence passport</title>
<style>body{font:16px system-ui;background:#f2f5f7;color:#203343;margin:0}main{max-width:1200px;margin:auto;padding:28px}h1{margin:0 0 10px}section{background:white;padding:20px;margin:16px 0;border-radius:12px}label{display:inline-block;margin:8px}select,input{padding:7px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}svg{width:100%;height:260px}.status{background:#fff0d8;padding:16px;border-left:5px solid #ba7c25}small{display:block;color:#516575;margin-top:10px}</style>
<main><h1>TREAD-TWO · Evidence passport</h1><p>Existing photos only. No fitting, inference, new tyres or workstation installation.</p><div class="status" id="status"></div>
<section><h2>What this model can support</h2><div id="caps"></div><small>Proxy mileage classes are not physical wear, tread depth or safety measurements. Geometry is retained from the earlier frozen branch.</small></section>
<section><h2>Saved training correction components</h2><label>Epoch <input id="epoch" type="number" value="60" min="1" max="60"></label><label>Class <select id="klass"><option value="all">All</option><option value="0">Low</option><option value="1">Mid</option><option value="2">High</option></select></label><label>Algebraic variant <select id="variant"><option>full_context</option><option>frozen_prior</option><option>bias_only</option><option>spatial_only</option></select></label><div id="rows"></div><svg id="chart" viewBox="0 0 1000 260" role="img" aria-label="Online sampled NLL by epoch"></svg><small>Blue: chosen variant, orange: frozen prior, averaged over saved view observations. These are changing online states; repeated views are correlated. Removing an additive term is not an independently trained ablation.</small></section>
<section><h2>Conditional holdout feasibility</h2><div id="support"></div><small>Unresolved mid video-parent co-assignment is a leakage safeguard; it does not assert that the known mid tyres are identical. These folds were enumerated, not trained.</small></section></main>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent);const el=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
el('status').textContent='Classifier: research only, unpromoted. '+D.summary.folds_missing_fit_class+' of '+D.summary.conditional_folds+' conditional holdouts remove an entire fit class. No next full classifier fit is justified by this gate.';
el('caps').innerHTML=Object.entries(D.capabilities).map(([k,v])=>'<p><b>'+esc(k.replaceAll('_',' '))+'</b>: '+esc(v)+'</p>').join('');
el('support').innerHTML='<table><tr><th>Class</th><th>Photos/frames</th><th>Known original tyres</th><th>Conservative groups</th></tr>'+D.support.map(r=>'<tr><td>'+esc(r.class_name)+'</td><td>'+r.images+'</td><td>'+r.known_original_tyres+'</td><td>'+r.conservative_components+'</td></tr>').join('')+'</table>';
function render(){const e=Math.max(1,Math.min(60,Number(el('epoch').value)||60)),c=el('klass').value,v=el('variant').value;el('epoch').value=e;const chosen=D.components.filter(r=>r.epoch===e&&r.variant===v&&(c==='all'||r.label===Number(c)));
el('rows').innerHTML='<table><tr><th>Conservative component</th><th>Views</th><th>Accuracy</th><th>NLL</th><th>ΔNLL vs prior</th><th>Rescued / spoiled</th></tr>'+chosen.map(r=>'<tr><td>'+esc(r.leakage_component)+'</td><td>'+r.view_observations+'</td><td>'+r.online_accuracy.toFixed(3)+'</td><td>'+r.online_nll.toFixed(4)+'</td><td>'+r.mean_delta_nll.toFixed(4)+'</td><td>'+r.rescued_prior_errors+' / '+r.spoiled_prior_corrects+'</td></tr>').join('')+'</table>';
const lines=[v,'frozen_prior'].map(variant=>Array.from({length:60},(_,i)=>{const rows=D.components.filter(r=>r.epoch===i+1&&r.variant===variant&&(c==='all'||r.label===Number(c)));return rows.reduce((a,r)=>a+r.online_nll*r.view_observations,0)/rows.reduce((a,r)=>a+r.view_observations,0);}));const max=Math.max(.001,...lines.flat());el('chart').innerHTML='<line x1="40" y1="225" x2="980" y2="225" stroke="#999"/><text x="40" y="250">epoch 1</text><text x="900" y="250">epoch 60</text><text x="40" y="15">NLL max '+max.toFixed(3)+'</text>'+lines.map((a,j)=>'<polyline fill="none" stroke="'+['#257c9b','#c28f31'][j]+'" stroke-width="3" points="'+a.map((y,i)=>(40+i*940/59)+','+(225-y/max*200)).join(' ')+'"/>').join('');}
['epoch','klass','variant'].forEach(id=>el(id).addEventListener('change',render));render();
</script></html>'''.replace('__DATA__',data)
    (out/'Evidence_Passport.html').write_text(html,encoding='utf-8')
    lines=['# NB15 · existing-data readiness and evidence passport','','**Completed analysis, scientific readiness false; no new fit or inference.**','', '| Class | Training images | Known original tyres | Conservative components |','|---|---:|---:|---:|']
    for r in support.itertuples():lines.append(f'| {r.class_name} | {r.images} | {r.known_original_tyres} | {r.conservative_components} |')
    lines+=['',f"{summary['folds_missing_fit_class']} of six conditional holdouts remove an entire training class. This is a support constraint, not proof that every possible model must fail.",'', '[Interactive passport](Evidence_Passport.html) · [Methods](METHODS.md) · [CSV inventory](CSV_INDEX.md).','', 'Context-r3 stays research-only. Existing photos suffice for this audit. Both workstations remain unchanged; another full classification fit is not supported by this gate.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['# Direct CSV inventory','','| Table | Rows | Columns |','|---|---:|---:|']+[f'| [{n}.csv]({n}.csv) | {m["rows"]:,} | {len(m["columns"])} |' for n,m in analysis['tables'].items()]
    (out/'CSV_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'METHODS.md').write_text('''# Readiness and correction utility methods

All numeric analysis uses the 386 original historical training images and 47,040 saved training-view observations only. Source registries include all historical roles for provenance; held-out rows are filtered before labels/logits/losses/corrections enter calculations. Frozen LC itself was selected historically on validation: this is not an untouched end-to-end validation protocol.

## Identity and conditional folds

Join NB14 and independently published NB07 by image ID and exact source SHA. NB07 conservative leakage components remain authoritative. Three possible mid video parents and their clips remain co-assigned, without asserting physical identity equivalence. Known original tyres and conservative components are counted separately. Every fold conditionally withholds one of six training components; all other historical roles are unchanged. No fold is trained, used to select a checkpoint, or installed. All folds must retain all three classes to support a complete group-held-out classification comparison across this training cohort. Low and conservatively co-assigned mid each have only one training component; their withheld folds have no remaining fit examples for that class. Other grouping assumptions are not silently substituted.

The 18-entry bank support table removes all references from the held component and all identical query source hashes. It then counts allowed references per class. 6/n normalization is reported only for positive n; missing support is blank, never infinity or a fabricated zero score. No bank is retrained or applied to held-out queries. These conditional counts do not make an existing globally trained encoder component-held-out.

## Algebraic online variants

For each saved training observation at its actual pre-update parameter cursor, reconstruct the three logits for: frozen prior b; bias-only b+g; spatial-only b+s; and full context b+g+s. Global and spatial context terms come from saved CSV; no image/model forward or optimizer operation occurs. Full logits and true-class NLL must match published rows within 2e-6. Softmax is calculated after subtracting the row maximum; NLL clips true-class probability at 1e-12. Mean NLL is a mean over view observations; delta NLL is variant minus prior on the same view. Accuracy is argmax agreement. Confidence is maximum probability. A rescued error means prior wrong and variant correct; spoiled correct means prior correct and variant wrong. These counts are view observations, not independent tyres or paired draws. Both views are retained. Epoch/group/image aggregates mix changing online model states and replacement draws. They are not fixed final-model cohort evaluation or independently trained ablations. No gain, threshold, blend or checkpoint is fitted/chosen from these calculations.

## Keys, units and missingness

`sample` joins immutable NB14 observation records; training sample IDs have validation gaps and are not renumbered. `parameter_update` identifies already-applied updates for the online state; `epoch` is one-based. `image_id` and `image_sha256` identify original source bytes. `fold` is a zero-based conditional metadata fold, never a completed fit. `class_index`, `label`, `query_class` and `reference_class` encode low=0, mid=1, high=2. Logits are unnormalized class scores; probabilities, accuracy and confidence are dimensionless fractions; NLL and delta NLL use natural logs. Counts distinguish images, known original tyres, conservative components and repeated views. Missing reference normalization means zero remaining support. Booleans describe feasibility/integrity, not predictive accuracy. The machine dictionary lists every table/column; source pins, hashes, full counts and versions remain in ANALYSIS.json and INPUTS.json.

## Capabilities and limits

CAPABILITIES.json records the measured family status, not a newly trained model: classifier research-only/unpromoted; geometry retained from frozen TREAD-ONE with existing scope; signed evidence/attention are internal accounting rather than causal physical-wear explanations. Physical depth and safety assessment remain unavailable. The gate does not require new photography, change either workstation, certify deployment or prove classification impossible under every alternative task/protocol. It prevents claiming a complete independent three-class tyre-held-out check from the available training grouping.
''',encoding='utf-8')
