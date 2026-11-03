"""Publication figures and a self-contained, filterable saved-evidence atlas."""
import base64,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import atomic,read,file_meta

def render(saved,core):
    out=core['tables'].root;figures=out/'figures';figures.mkdir(exist_ok=True)
    epochs=pd.read_csv(out/'validation_reconstruction.csv');f=core['findings'];g0=f['G0']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].plot(epochs.epoch,epochs.mean_dice,color='#167878');axes[0,0].axhline(g0['mean_dice'],color='#777',ls='--',label='G0');axes[0,0].axhline(g0['mean_dice']-.002,color='#c85236',ls=':',label='Eligibility floor');axes[0,0].set(title='Geometry preserved on validation',ylabel='Mean tyre/tread Dice',xlabel='Epoch');axes[0,0].legend()
    axes[0,1].plot(epochs.epoch,100*epochs.human_point_MAE,color='#167878');axes[0,1].axhline(100*g0['human_point_MAE'],color='#777',ls='--',label='G0');axes[0,1].set(title='Human point error',ylabel='% of image width (lower is better)',xlabel='Epoch');axes[0,1].legend()
    axes[1,0].plot(epochs.epoch,100*epochs.masked_gain,color='#4956a8');axes[1,0].axhline(5,color='#c85236',ls='--',label='5% utility requirement');axes[1,0].set(title='Masked prediction vs train-only mean',ylabel='Relative loss reduction (%)',xlabel='Epoch');axes[1,0].legend()
    axes[1,1].plot(epochs.epoch,epochs.projector_rank,label='64D projector',color='#4956a8');axes[1,1].plot(epochs.epoch,epochs.global_rank,label='96D pooled encoder',color='#167878');axes[1,1].axhline(4,color='#c85236',ls='--',label='Projector eligibility rank');axes[1,1].set(title='Held-out representation degeneracy',ylabel='Effective rank',xlabel='Epoch');axes[1,1].legend()
    fig.suptitle('NB17 / GEO: completed execution, zero eligible checkpoints',fontsize=16);fig.savefig(figures/'GEO_Validation.png',dpi=160);plt.close(fig)
    spectra=pd.read_csv(out/'validation_spectra.csv');latent=pd.read_csv(out/'endpoint_latent_spaces.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);axes=axes.ravel()
    for name,group in spectra[spectra.epoch==60].groupby('space'):axes[0].plot(group.dimension+1,100*group.fraction,label=name)
    axes[0].set(title='Final validation variance spectrum',ylabel='Share of total covariance (%)',xlabel='Dimension');axes[0].set_xlim(1,12);axes[0].legend()
    subset=latent[(latent.stage=='endpoint')&(latent.role.isin(['train','validation','test','all']))]
    roles=['train','validation','test','all'];x=np.arange(4)
    for k,space in enumerate(['projected64','global96']):
        a=subset[subset.space==space].set_index('role');axes[1].bar(x+(k-.5)*.36,[a.loc[r,'effective_rank'] for r in roles],width=.36,label=space)
    axes[1].set(xticks=x,xticklabels=roles,title='Role matters: pooled rank can mislead',ylabel='Effective rank');axes[1].legend()
    inter=pd.read_csv(out/'intervention_summary.csv').sort_values('embedding_cosine_change');axes[2].barh(inter.variant,inter.embedding_cosine_change,color='#4956a8');axes[2].set(title='Previously inspected test-role probes',xlabel='Mean embedding cosine change')
    component_path=out/'component_variance_decomposition.csv'
    if component_path.exists():
        comp=pd.read_csv(component_path)
        for k,space in enumerate(['projected64','global96']):
            a=comp[comp.space==space].set_index('role');axes[3].bar(x+(k-.5)*.36,[100*a.loc[r,'between_fraction'] for r in roles],width=.36,label=space)
        axes[3].set(xticks=x,xticklabels=roles,title='Variation between conservative components',ylabel='Share of total scatter (%)',ylim=(0,100));axes[3].legend()
    else:axes[3].axis('off')
    fig.suptitle('Saved-vector analysis; descriptive, no checkpoint reselection',fontsize=15);fig.savefig(figures/'GEO_Representation_And_Probes.png',dpi=160);plt.close(fig)
    q=pd.read_csv(out/'quality_comparison.csv');q=q[(q.stage=='endpoint')];grad=pd.read_csv(out/'training_task_gradients.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True);x=np.arange(len(q));axes[0].bar(x-.18,q.MAE,.36,label='Experimental estimate');axes[0].bar(x+.18,q.median_MAE,.36,label='Train-only constant median');axes[0].set(xticks=x,xticklabels=['Tyre error','Tread error','Human point error'],ylabel='Mean absolute annotation error',title='Error estimates: all 570 development images');axes[0].legend()
    for name,group in grad.groupby('group'):
        series=group.groupby('epoch').pre_clip_norm.mean() if 'epoch' in group else group.groupby((group['update']-1)//49+1).pre_clip_norm.mean()
        axes[1].plot(series.index,series.values,label=name)
    axes[1].axhline(1,color='#777',ls='--');axes[1].set(yscale='log',xlabel='Epoch',ylabel='Mean pre-clip norm',title='Independent task gradient clipping');axes[1].legend();fig.savefig(figures/'GEO_Quality_And_Gradients.png',dpi=160);plt.close(fig)
    # Parse data only from the published document, without executing its script.
    html=(saved.root/'reports/final_Explorer.html').read_text(encoding='utf-8');mark='const cases=';start=html.index(mark)+len(mark)
    cases,end=json.JSONDecoder().raw_decode(html[start:]);assert len(cases)==570
    metrics=core['case'].query("stage=='endpoint'").set_index('sample')
    guides=saved.table('annotation_targets').set_index(['image_id','guide','side']).y_height_fraction
    endpoint_points=core['points'][core['points']['sample'].isin(metrics.index)]
    point_lookup={int(sample):frame for sample,frame in endpoint_points.groupby('sample')}
    for c in cases:
        row=metrics.loc[c['sample']];c['mean_dice']=float(row.mean_dice);c['masked_ratio']=float(row.masked_ratio)
        c['human_point_MAE']=float(row.human_point_MAE) if c['human'] else None
        c.pop('point_error',None);c.pop('quality_target',None) # never present absent/weak point placeholders as GT
        c['points']=[dict(prediction=float(r.prediction),target=float(r.target) if c['human'] else None,y=float(guides.loc[(c['image_id'],r.guide,r.side)])) for r in point_lookup[c['sample']].itertuples()]
        assert len(c['points'])==6
        assert c['image'] and c['role'] in ('train','validation','test')
    payload=json.dumps(cases,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GEO review atlas</title>
<style>body{font:16px system-ui;background:#101924;color:#e9f2fa;margin:2rem;max-width:1500px}h1{font-size:2rem}label{display:inline-block;margin:.4rem}select,button,input{font:inherit;padding:.5rem;border-radius:6px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:1rem}article,.panel{background:#1d3041;border:1px solid #36546b;padding:1rem;border-radius:12px;overflow-wrap:anywhere}img{width:100%;border-radius:6px}a{color:#8ed7ea}.badge{color:#ffd497}#count{font-weight:bold}p{line-height:1.5}</style>
<h1>TREAD-TWO / GEO Ã‚Â· evidence atlas</h1><p class="badge">Completed research pilot Ã‚Â· no eligible selected checkpoint Ã‚Â· no production promotion</p><p>Saved final-epoch overlays only. Blue/gold show predicted tyre/tread regions. Point errors use human annotations only. Masked texture residuals are experimental; no physical wear, depth, damage or safety inference is established. Test images have already been inspected and remain descriptive.</p>
<section class="panel"><label>Role <select id="role"><option value="all">All roles</option><option>train</option><option>validation</option><option>test</option></select></label><label>Component <select id="component"><option value="all">All components</option></select></label><label>Order <select id="sort"><option value="dice">Lowest mask Dice</option><option value="point">Largest human point error</option><option value="masked">Largest masked/control ratio</option><option value="id">Image name</option></select></label><label>Find <input id="search" placeholder="Image name"></label><button id="csv">Download filtered CSV</button><p id="count"></p><button id="previous">Previous</button> <button id="next">Next</button><span id="page"></span></section><p></p><div id="grid"></div>
<script>const cases=PAYLOAD;let page=0;const size=24;const el=id=>document.getElementById(id);const number=x=>x===null?'Unavailable':x.toFixed(6);
function filtered(){let rows=cases.filter(c=>(el('role').value==='all'||c.role===el('role').value)&&(el('component').value==='all'||c.component===el('component').value)&&c.image_id.toLowerCase().includes(el('search').value.toLowerCase()));const order=el('sort').value;if(order==='point')rows=rows.filter(c=>c.human);return rows.sort((a,b)=>order==='point'?b.human_point_MAE-a.human_point_MAE:order==='masked'?b.masked_ratio-a.masked_ratio:order==='id'?a.image_id.localeCompare(b.image_id):a.mean_dice-b.mean_dice)}
function render(){const rows=filtered();page=Math.max(0,Math.min(page,Math.ceil(rows.length/size)-1));el('count').textContent=rows.length+' source images Ã‚Â· human errors never include weak/absent annotations';el('page').textContent=' Page '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/size));el('grid').replaceChildren();for(const c of rows.slice(page*size,(page+1)*size)){const a=document.createElement('article'),title=document.createElement('strong'),img=document.createElement('img'),note=document.createElement('p');title.textContent=c.image_id;img.src='data:image/jpeg;base64,'+c.image;img.alt='Saved final tyre and tread overlay';img.loading='lazy';note.textContent=c.role+' Ã‚Â· '+c.component+' | Dice '+number(c.mean_dice)+' | Human point MAE '+number(c.human_point_MAE)+' | Masked/control '+number(c.masked_ratio);a.append(title,img,note);el('grid').append(a)}}
for(const component of [...new Set(cases.map(c=>c.component))].sort()){const o=document.createElement('option');o.textContent=component;o.value=component;el('component').append(o)}for(const id of ['role','component','sort','search'])el(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()});el('previous').addEventListener('click',()=>{page--;render()});el('next').addEventListener('click',()=>{page++;render()});el('csv').addEventListener('click',()=>{const fields=['image_id','role','component','mean_dice','human_point_MAE','masked_ratio'];const content=[fields.join(','),...filtered().map(c=>fields.map(k=>JSON.stringify(c[k]??'')).join(','))].join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:'text/csv'}));a.download='GEO_filtered_cases.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});render();</script></html>'''.replace('PAYLOAD',payload)
    page=page.replace('</style>','.photo{position:relative}.photo img{display:block}.photo svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}</style>')
    page=page.replace('Saved final-epoch overlays only.','Saved final-epoch overlays only. Pink circles: predicted tread points; green rings: human targets when available.')
    photo="""function photo(c,img){const p=document.createElement('div');p.className='photo';const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 384 512');svg.setAttribute('aria-label','Predicted points and available human targets');for(const q of c.points){for(const [x,color,r] of [[q.prediction,'#ff668e',3],[q.target,'#6aff99',5]]){if(x===null)continue;const mark=document.createElementNS('http://www.w3.org/2000/svg','circle');for(const [key,value] of Object.entries({cx:x*384,cy:q.y*512,r:r,fill:'none',stroke:color,'stroke-width':2}))mark.setAttribute(key,value);svg.append(mark)}}p.append(img,svg);return p} """
    page=page.replace('function render(){',photo+'function render(){').replace('a.append(title,img,note)','a.append(title,photo(c,img),note)')
    (out/'GEO_Review_Atlas.html').write_text(page,encoding='utf-8')
    result=dict(figures=3,interactive_images=570,page_size=24,filters=['role','component','human_point_error','masked_ratio','image_name'],predicted_point_markers=3420,human_target_markers=720,weak_or_absent_point_errors_hidden=True,html_bytes=(out/'GEO_Review_Atlas.html').stat().st_size)
    atomic(out/'VISUALS.json',result);return result
