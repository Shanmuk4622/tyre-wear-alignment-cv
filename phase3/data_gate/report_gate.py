"""Measured CSV inventory, source-image explorer and static scientific summaries."""
from PIL import Image
import argparse,csv,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_gate import ROOT,read,atomic,sha

def rows(folder,name):
 with open(folder/'csv'/(name+'.csv'),encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def report(folder):
 folder=Path(folder);summary=read(folder/'SUMMARY.json');images=rows(folder,'source_images');quality={r['image_id']:r for r in rows(folder,'image_quality')};support=rows(folder,'historical_class_support');gallery=read(folder/'GALLERY.json')
 for name,h in read(folder/'CONTRACT.json')['source_hashes'].items():assert sha(ROOT/name)==h,'Source changed; rerun collection before rendering'
 figures=folder/'figures';figures.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 labels=['low','mid','high'];colors=['#2780ac','#d28242','#54895f']
 for ax,field,title in zip(axes,('confirmed_original_tyres','images'),('Recorded original physical tyres','Images / frames')):
  for j,role in enumerate(('train','validation','test')):
   values=[int(next(r for r in support if r['historical_split']==role and r['proxy_class']==c)[field]) for c in labels]
   x=np.arange(3)+(j-1)*.25;ax.bar(x,values,width=.24,label=role,color=colors[j])
   for xx,v in zip(x,values):ax.text(xx,v+.02*max(max(values),1),str(v),ha='center',fontsize=9)
  ax.set_xticks(range(3),labels);ax.set_title(title);ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.grid(axis='y',alpha=.15)
 axes[0].legend();fig.suptitle('Existing development dataset · 570 images / 12 original tyres; no new subjects')
 fig.savefig(figures/'NB07_class_and_subject_support.png',dpi=165);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
 for domain,color in (('original_photo','#2780ac'),('new_video_frame','#d28242')):
  q=[r for r in quality.values() if r['domain']==domain];axes[0].scatter([float(r['brightness_0_255']) for r in q],[float(r['focus_laplacian_variance']) for r in q],s=12,alpha=.55,label=domain,color=color)
 axes[0].set(xlabel='Brightness (0–255), fixed 512×384 grid',ylabel='Laplacian variance (intensity²)',yscale='symlog',title='Descriptive capture differences');axes[0].legend(fontsize=8)
 count=[sum(int(r['tread_outside_tyre_native_pixels'])>0 for r in quality.values()),sum(int(r['tread_outside_tyre_valid_pixels'])>0 for r in quality.values())]
 axes[1].bar(['Raw source disagreements','After source ignore mask'],count,color=['#d28242','#2780ac'])
 for i,v in enumerate(count):axes[1].text(i,v+.8,str(v),ha='center')
 axes[1].set(title='Images with tread outside tyre',ylabel='Image count',ylim=(0,max(count+[1])*1.2));fig.suptitle('Quality descriptors and annotation policy · no labels repaired or models fitted')
 fig.savefig(figures/'NB07_quality_and_annotation_review.png',dpi=165);plt.close(fig)
 entries=[dict(**r,quality=quality[r['image_id']],preview=gallery[r['image_id']]) for r in images]
 data=dict(summary=summary,images=entries,issues=rows(folder,'intake_issues'),gaps=rows(folder,'readiness_gaps'),parents=rows(folder,'video_parent_review'),support=support)
 payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
 html=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'unsafe-inline';style-src 'unsafe-inline';img-src data:;connect-src 'none'"><title>TREAD-ONE data and protocol gate</title><style>body{font:16px/1.5 system-ui;max-width:1180px;margin:26px auto;padding:0 22px;background:#12212e;color:#edf5fa}aside{color:#bad0df}table{width:100%;border-collapse:collapse}th,td{padding:8px;text-align:left;border-bottom:1px solid #365167}select,input{font:inherit;padding:7px;margin:5px;border-radius:5px;border:0}#gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px}.card{background:#203746;padding:12px;border-radius:8px}.card img{width:144px;height:192px;object-fit:contain}.small{font-size:13px;color:#cadce7}#gaps{background:#263c4c;padding:16px}pre{white-space:pre-wrap}button{padding:8px;cursor:pointer}</style></head><body><h1>TREAD-ONE data and protocol gate</h1><p id="status"></p><aside>Current photos, video frames and derived rows are not additional independent tyres. Similarity is a review aid, not identity or class ground truth. Source masks and original study splits are preserved.</aside><h2>Prospective training readiness</h2><div id="gaps"></div><h2>Historical class / subject support</h2><table><thead><tr><th>Split</th><th>Class</th><th>Images</th><th>Original tyres</th><th>Unmapped video frames</th><th>Human geometry images</th></tr></thead><tbody id="support"></tbody></table><h2>Source image and quality explorer</h2><label>Class<select id="cls"><option value="all">All</option><option value="0">low</option><option value="1">mid</option><option value="2">high</option></select></label><label>Split<select id="split"><option>all</option><option>train</option><option>validation</option><option>test</option></select></label><label>Domain<select id="domain"><option>all</option><option>original_photo</option><option>new_video_frame</option></select></label><label>Review<select id="review"><option value="all">All</option><option value="conflict">Raw annotation disagreement</option><option value="unmapped">Unresolved video identity</option></select></label><input id="search" placeholder="Find source image ID"><p id="count"></p><div id="gallery"></div><button id="more">Show next 60 images</button><h2>Video parent review</h2><p>All three clips are known existing mid tyres and are distinct from each other. Exact original-session mappings remain unconfirmed unless supplied in the intake resolution file. Hash distance does not confirm a mapping.</p><table><thead><tr><th>Clip</th><th>Possible parent</th><th>Minimum dHash distance</th><th>Confirmed in intake</th></tr></thead><tbody id="parents"></tbody></table><p class="small">Brightness and focus are fixed-grid descriptors, not physical wear/depth/alignment measurements or a classifier shortcut proof. Current prospective inputs and the draft policy must be completed before readiness can pass.</p><script>const DATA=__DATA__;const $=id=>document.getElementById(id);let limit=60;
$('status').textContent=`${DATA.summary.source_images} existing images · ${DATA.summary.confirmed_existing_tyres} original tyres · ${DATA.summary.new_intake_images} intake images · ${DATA.summary.csv_tables} CSV tables`;
for(const gap of DATA.gaps){const p=document.createElement('p');p.textContent=gap.code+': '+gap.detail;$('gaps').append(p)}if(!DATA.gaps.length)$('gaps').textContent=DATA.summary.prospective_training_ready?'All declared prospective checks passed. No model is launched by this report.':'Review the intake validation results.';
for(const r of DATA.support){const tr=document.createElement('tr');for(const v of [r.historical_split,r.proxy_class,r.images,r.confirmed_original_tyres,r.unmapped_existing_video_frames,r.human_geometry_images]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('support').append(tr)}
for(const r of DATA.parents){const tr=document.createElement('tr');for(const v of [r.video_id,r.possible_original_tyre_id,r.min_dhash_hamming_to_original_photos,r.selected_by_confirmed_intake]){const td=document.createElement('td');td.textContent=v;tr.append(td)}$('parents').append(tr)}
function show(reset=true){if(reset)limit=60;const rs=DATA.images.filter(r=>($('cls').value==='all'||r.class_index===$('cls').value)&&($('split').value==='all'||r.historical_study_split===$('split').value)&&($('domain').value==='all'||r.domain===$('domain').value)&&r.image_id.toLowerCase().includes($('search').value.toLowerCase())&&($('review').value!=='conflict'||Number(r.quality.tread_outside_tyre_native_pixels)>0)&&($('review').value!=='unmapped'||r.domain==='new_video_frame'&&!r.confirmed_original_tyre_id));$('count').textContent=rs.length+' matching images · showing '+Math.min(limit,rs.length);$('gallery').replaceChildren();for(const r of rs.slice(0,limit)){const card=document.createElement('div');card.className='card';const img=document.createElement('img');img.src=r.preview;img.alt=r.image_id;const title=document.createElement('p');title.textContent=r.image_id;title.className='small';const detail=document.createElement('pre');detail.className='small';detail.textContent=['low','mid','high'][Number(r.class_index)]+' · '+r.historical_study_split+'\n'+(r.confirmed_original_tyre_id||'Existing video tyre: parent unresolved')+'\nBrightness '+Number(r.quality.brightness_0_255).toFixed(1)+' / 255\nFocus variance '+Number(r.quality.focus_laplacian_variance).toFixed(1)+'\nRaw disagreement '+r.quality.tread_outside_tyre_native_pixels+' px';card.append(img,title,detail);$('gallery').append(card)}$('more').disabled=limit>=rs.length}
for(const id of ['cls','split','domain','review'])$(id).addEventListener('change',()=>show());$('search').addEventListener('input',()=>show());$('more').addEventListener('click',()=>{limit+=60;show(false)});show();</script></body></html>'''.replace('__DATA__',payload)
 (folder/'TREAD_ONE_Data_Gate.html').write_text(html,encoding='utf-8')
 atomic(folder/'VISUALS.json',dict(source_previews=570,quality_grid=[512,384],preview_grid=[192,144],JS_syntax_verified=False,browser_render_verified=False,new_model_inference=False))
 print('Rendered data gate report, 570 source previews and two scientific figures',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='');args=parser.parse_args();base=Path(args.output or ROOT/'data_gate_outputs');report(read(base/'LATEST.json')['path'])

if __name__=='__main__':main()
