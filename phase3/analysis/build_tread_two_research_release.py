"""Consolidate completed public studies into a local research handoff, without fitting."""
import hashlib,json,os,sys
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/research_release';OUT.mkdir(exist_ok=True)
SOURCES={};TABLES={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source(path):
    path=Path(path);SOURCES[path.relative_to(ROOT).as_posix()]=sha(path);return path
def read(path):return json.loads(source(path).read_text(encoding='utf-8'))
def latest(n):return Path(read(ROOT/f'tread_two/nb{n:02}_completion/LATEST.json')['path'])
def csv(name,value):
    frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value);path=OUT/(name+'.csv');temp=path.with_suffix('.tmp')
    frame.to_csv(temp,index=False,float_format='%.12g',lineterminator='\n');os.replace(temp,path)
    TABLES[name]=dict(rows=len(frame),columns=list(frame.columns),sha256=sha(path));return frame
def href(revision,prefix,name='MANIFEST.json'):
    return 'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/'+revision+'/'+quote(prefix+'/'+name,safe='/')
def local(path):return '../../'+Path(path).relative_to(ROOT).as_posix()
publications=[];store={}
def publication(key,title,kind,notebook,folder,manifest_name='REMOTE_MANIFEST.json',prefix=None,rows=None,row_scope='independently checked scientific rows'):
    v=read(folder/'VERIFICATION.json');assert v['status']=='passed'
    m=read(folder/manifest_name);revision=v['revision'];prefix=prefix or v.get('prefix',v.get('destination'))
    assert prefix and m['format']=='immutable-direct-files-v1'
    row=dict(publication=key,title=title,kind=kind,notebook=notebook,revision=revision,prefix=prefix,contract=m['contract'],files_with_manifest=len(m['files'])+1,
        payload_bytes=sum(x['bytes'] for x in m['files'].values()),scientific_rows=rows if rows is not None else v.get('CSV_rows',v.get('csv_rows',v.get('total_scientific_rows'))),row_count_scope=row_scope,
        verification_status='passed',owner_released=v['owner_released'],manifest_url=href(revision,prefix),HF_tree='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/'+revision+'/'+prefix,
        receipt=local(folder/'VERIFICATION.json'),receipt_sha256=sha(folder/'VERIFICATION.json'))
    publications.append(row);store[key]=dict(verification=v,manifest=m,row=row,folder=folder);return row
old=ROOT/'tread_two/pilot_validation/nb08_followup/82e7e1c5774f'
publication('atlas_v1_fit','ATLAS v1 fit','fit','NB08',old,'fit_MANIFEST.json',prefix='phase3/tread-two/atlas-v1/fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd',rows=35502696,row_scope='saved cursor declaration; selected families checked; not every large CSV independently recounted')
for n,key,title,kind in [(9,'atlas_v1_review','ATLAS v1 evidence review','review'),(10,'appearance_gate','Training appearance/reference gate','gate'),(11,'atlas_lc_fit','ATLAS-LC r2 fit','fit'),(12,'atlas_lc_review','ATLAS-LC evidence review','review'),(13,'atlas_context_fit','ATLAS-CONTEXT r3 fit and recovery','fit'),(14,'atlas_context_review','ATLAS-CONTEXT evidence review','review'),(15,'readiness_gate','Existing-data readiness and passport','gate')]:
    publication(key,title,kind,'NB13/NB13A' if n==13 else f'NB{n:02}',latest(n))
identity=ROOT/'data_gate_validation/nb07_publication';identity_v=read(identity/'VERIFICATION.json')
publication('identity_gate','Source identity and leakage constraints','gate','NB07',identity,'remote/REMOTE_MANIFEST.json',rows=identity_v['structured_rows'],row_scope='structured CSV rows; four raw/control CSV inputs are additional')
csv('publications',publications)
gate=store['readiness_gate'];gate_remote=gate['folder']/'remote';caps=read(gate_remote/'CAPABILITIES.json');summary=read(gate_remote/'SUMMARY.json')
assert summary['scientific_readiness'] is False and caps['classification']=='research_only_unpromoted'
models=[];metrics=[];artifacts=[]
def artifact(publication,name,label,kind):
    entry=store[publication];meta=entry['manifest']['files'][name]
    row=dict(publication=publication,label=label,kind=kind,logical_name=name,bytes=meta['bytes'],sha256=meta['sha256'],revision=entry['row']['revision'],url=href(entry['row']['revision'],entry['row']['prefix'],meta['object']))
    artifacts.append(row);return row
for model,label,fit,review,selected,folder,architecture,protocol_doc in [
    ('v1','ATLAS v1','atlas_v1_fit','atlas_v1_review',18,'pilot','Shared local reference evidence on RGB','phase3TreadTwoFindings.md'),
    ('lc','ATLAS-LC r2','atlas_lc_fit','atlas_lc_review',7,'lc_pilot','Paired local-contrast reference evidence','phase3TreadTwoLCArchitecture.md'),
    ('context','ATLAS-CONTEXT r3','atlas_context_fit','atlas_context_review',7,'context_pilot','Frozen selected LC prior plus bounded positional correction','phase3TreadTwoContextArchitecture.md')]:
    fit_entry=store[fit];mf=fit_entry['manifest'];v=fit_entry['verification'];assert v['epochs']==60 and v['optimizer_updates']==2940 and v['selected_epoch']==selected
    source_inventory=read(ROOT/'tread_two'/folder/'FILES.json')
    assert all(sha(ROOT/'tread_two'/folder/n)==meta['sha256'] for n,meta in source_inventory.items())
    path=source(store[review]['folder']/'remote/cohort_metrics.csv');assert sha(path)==store[review]['manifest']['files']['cohort_metrics.csv']['sha256']
    cohorts=pd.read_csv(path);endpoints=cohorts[cohorts.stage.isin(['selected_eval','final_eval'])];assert len(endpoints)==2
    for row in endpoints.itertuples():
        assert row.n==103 and row.balanced_accuracy_observed==.5 and row.low_to_mid==62 and row.predicted_high==41
        metrics.append(dict(model=model,label=label,endpoint=row.endpoint,epoch=selected if row.endpoint=='selected' else 60,images=row.n,
            accuracy=row.accuracy,observed_class_BA=row.balanced_accuracy_observed,low_recall=0,high_recall=1,nll=row.nll,brier=row.brier,ece_10_equal_width=row.ece_10_equal_width,
            scope='Already inspected historical low/high endpoint; no mid targets; descriptive, not independent confirmation'))
    checkpoints={}
    for endpoint,epoch in [('selected',selected),('final',60)]:
        candidates=[n for n in mf['files'] if n.endswith(f'/epoch_{epoch:04}.pt')];assert len(candidates)==1,(model,epoch,candidates)
        checkpoints[endpoint]=artifact(fit,candidates[0],label+' '+endpoint+' checkpoint','native_checkpoint')
    models.append(dict(model=model,label=label,status='research_only_unpromoted',architecture=architecture,seed=1,epochs=60,optimizer_updates=2940,
        selected_epoch=selected,final_epoch=60,selection='Original validation BA then NLL then earliest epoch',source_entries=len(source_inventory),source_inventory_sha256=sha(ROOT/'tread_two'/folder/'FILES.json'),
        fit_publication=fit,review_publication=review,selected_checkpoint_sha256=checkpoints['selected']['sha256'],final_checkpoint_sha256=checkpoints['final']['sha256'],
        selected_checkpoint_url=checkpoints['selected']['url'],final_checkpoint_url=checkpoints['final']['url'],architecture_doc=local(ROOT/protocol_doc)))
    artifact(review,'ATLAS_Failure_Explorer.html',label+' interactive failure evidence','interactive_evidence')
    artifact(review,'figures/fit_and_confusion.png',label+' fit and confusion figure','figure')
    md=[f'# {label} — research model card','', '**Status: measured fit completed; research-only and unpromoted.**','',architecture+'.', '', '| Item | Recorded value |','|---|---|',f'| Seed / fit | 1 / 60 epochs / 2,940 updates |',f'| Validation-selected epoch | {selected}; original selection retained |',f'| Frozen source inventory | {len(source_inventory)} entries / `{sha(ROOT/"tread_two"/folder/"FILES.json")}` |',f'| Selected native checkpoint SHA-256 | `{checkpoints["selected"]["sha256"]}` |',f'| Final native checkpoint SHA-256 | `{checkpoints["final"]["sha256"]}` |','', '| Endpoint | Observed-class BA | Low / high recall | NLL | Brier | ECE |','|---|---:|---|---:|---:|---:|']
    for item in metrics:
        if item['model']==model:md.append(f'| {item["endpoint"]} | {item["observed_class_BA"]:.2%} | 0/62 / 41/41 | {item["nll"]:.6f} | {item["brier"]:.6f} | {item["ece_10_equal_width"]:.6f} |')
    md+=['',f'[Selected native artifact]({checkpoints["selected"]["url"]}) · [Final artifact]({checkpoints["final"]["url"]}) · [Fit receipt]({fit_entry["row"]["receipt"]}) · [Architecture / findings]({local(ROOT/protocol_doc)}).','',
        'Classes are mileage proxies, not physical wear, tread depth or safety. The 103-image historical endpoint is already inspected and has low/high targets only. Scores do not establish unseen-tyre reliability. Training repeats correlated views of the same existing tyres. Retained dense geometry comes from the frozen TREAD-ONE branch; no new dense improvement is attributed to this classifier. Signed evidence/attention records internal calculations and is not a causal physical explanation.', '',
        '[Source and capability limits](README.md) · [Complete artifact registry](artifacts.csv). Weights are referenced by immutable HF revision/object SHA; this handoff does not deserialize or republish them.']
    (OUT/f'MODEL_CARD_{model.upper()}.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
csv('models',models);metrics=csv('endpoint_metrics',metrics)
artifact('readiness_gate','CAPABILITIES.json','Published capability passport','capability_record');artifact('readiness_gate','Evidence_Passport.html','Training support and correction passport','interactive_evidence')
csv('artifacts',artifacts)
csv('capabilities',[
    dict(capability='Mileage-proxy classification',status=caps['classification'],evidence='All three candidate selected/final endpoints: low recall 0/62, observed-class BA 50%',limit='Not promoted to either workstation'),
    dict(capability='Segmentation and localization',status='retained_frozen_TREAD_ONE',evidence='Geometry outputs and native weights retained from earlier study',limit='No new geometry fit or deployment certification'),
    dict(capability='Signed evidence and attention',status='internal_accounting',evidence='Published reference/context conservation and saved endpoint attention',limit='Not causal physical-wear truth'),
    dict(capability='Independent class-supported group holdouts',status='not_supported_for_complete_three_class_training_comparison',evidence='Training conservative components low/mid/high = 1/1/4; two conditional folds remove a fit class',limit='Conditional support enumeration; no held-component model was fitted'),
    dict(capability='Physical depth in millimetres',status='unavailable',evidence='No measured physical target supplied',limit='Do not convert mileage proxies to depth'),
    dict(capability='Safety assessment',status='unavailable',evidence='No safety validation supplied',limit='No safety decision is established')])
csv('provenance',[dict(path=path,sha256=value) for path,value in sorted(SOURCES.items())])
fields=[]
definitions={'observed_class_BA':'Mean recall over target classes present: low/high only; no absent mid recall is invented','low_recall':'Correct low predictions / all62historical low targets','high_recall':'Correct high predictions / all41historical high targets','nll':'Mean negative natural log of true-class probability; three output classes','brier':'Mean sum of three squared probability-minus-one-hot target errors','ece_10_equal_width':'Ten equal-width max-confidence bins; weighted absolute accuracy-confidence gap','scientific_rows':'Row count in this publication under the explicit row_count_scope; repeated views/derived rows are not independent tyres','row_count_scope':'States whether counts are exhaustively checked, saved cursor declarations or structured-table counts','files_with_manifest':'Logical direct scientific/control files selected by the manifest plus the manifest itself','receipt_sha256':'SHA-256 of the safe local independent verification receipt','source_inventory_sha256':'SHA-256 of the unchanged frozen FILES.json for this candidate','selected_epoch':'Original validation-selected checkpoint; never chosen by the historical test comparison'}
for table,meta in TABLES.items():
    for column in meta['columns']:
        units='natural-log units' if column=='nll' else 'dimensionless fraction' if column in ('accuracy','observed_class_BA','low_recall','high_recall','ece_10_equal_width') else 'dimensionless squared probability error' if column=='brier' else 'bytes' if column in ('bytes','payload_bytes') else 'integer count/index' if column in ('epochs','epoch','images','scientific_rows','files_with_manifest','optimizer_updates','seed','source_entries','selected_epoch','final_epoch') else 'text/hash/URL/category or boolean as named'
        fields.append(dict(table=table,column=column,definition=definitions.get(column,column.replace('_',' ')+' for this pinned research registry; see README.md for scope'),units=units,missing='Blank means not applicable/unavailable; no new measurement inferred'))
csv('table_dictionary',fields)
figures=OUT/'figures';figures.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
for ax,field,title in zip(axes,['observed_class_BA','low_recall','nll','ece_10_equal_width'],['Observed low/high BA','Low recall','Three-class NLL','ECE: ten confidence bins']):
    for j,endpoint in enumerate(['selected','final']):
        group=metrics[metrics.endpoint.eq(endpoint)].set_index('model').loc[['v1','lc','context']]
        ax.bar(np.arange(3)+(j-.5)*.36,group[field],width=.36,label=endpoint,color=['#237e8e','#b77b35'][j])
    ax.set(xticks=range(3),xticklabels=['v1','LC','Context'],title=title);ax.grid(axis='y',alpha=.2)
    if field in ('observed_class_BA','low_recall','ece_10_equal_width'):ax.set_ylim(0,1)
axes[0].legend();fig.suptitle('Completed ATLAS candidates · same already inspected 103-image endpoint · no mid targets')
fig.savefig(figures/'candidate_comparison.png',dpi=160);plt.close(fig)
(OUT/'CAPABILITIES.json').write_text(json.dumps(caps,sort_keys=True,indent=2)+'\n',encoding='utf-8')
data=json.dumps(dict(models=models,metrics=metrics.to_dict('records'),publications=publications,artifacts=artifacts,capabilities=pd.read_csv(OUT/'capabilities.csv').to_dict('records')),separators=(',',':'),allow_nan=False).replace('<',chr(92)+'u003c')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TREAD-TWO research evidence</title><style>body{font:16px system-ui;background:#f2f5f7;color:#233b4a;margin:0}main{max-width:1250px;margin:auto;padding:28px}section{background:white;padding:22px;border-radius:12px;margin:18px 0}h1{margin-bottom:5px}select{padding:8px;margin:10px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}a{color:#146d86}.status{border-left:5px solid #b77b35;padding:16px;background:#fff2db}.scroll{overflow:auto}img{width:100%}small{color:#526a78}</style><main><h1>TREAD-TWO · Research evidence handoff</h1><p>Three completed candidate fits, their evidence reviews and the existing-data readiness findings.</p><p class="status">Research only. All candidates miss the62historical low images. Geometry is retained from TREAD-ONE. Mileage classes do not measure physical depth or safety.</p><section><h2>Candidate results</h2><label>Model <select id="model"><option value="all">All candidates</option><option value="v1">ATLAS v1</option><option value="lc">ATLAS-LC r2</option><option value="context">ATLAS-CONTEXT r3</option></select></label><div id="models" class="scroll"></div><img src="figures/candidate_comparison.png" alt="Selected and final candidates share50%observed-class balanced accuracy and zero low recall; context NLL and ECE are worse"><small>The same already inspected103-image low/high endpoint is used. These descriptive comparisons are not independent confirmation. Original validation checkpoint selection is retained.</small></section><section><h2>Verified evidence</h2><label>Record <select id="kind"><option value="all">All records</option><option value="fit">Model fits</option><option value="review">Evidence reviews</option><option value="gate">Data and readiness gates</option></select></label><div id="records" class="scroll"></div><small>All scientific evidence remains at its original pinned HF revisions. This local handoff creates no new model results and has not itself been published to HF.</small></section><section><h2>Capabilities and limits</h2><div id="caps"></div></section><section><h2>Checkpoint and explorer artifacts</h2><div id="artifacts"></div><p><a href="README.md">Handoff notes</a> · <a href="models.csv">Model registry CSV</a> · <a href="endpoint_metrics.csv">Metrics CSV</a> · <a href="publications.csv">Publication registry CSV</a> · <a href="RELEASE.json">Provenance manifest</a></p></section></main><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const chosen=el('model').value;el('models').innerHTML='<table><tr><th>Candidate</th><th>Endpoint / epoch</th><th>BA</th><th>Low recall</th><th>NLL</th><th>ECE</th><th>Card</th></tr>'+D.metrics.filter(r=>chosen==='all'||r.model===chosen).map(r=>'<tr><td>'+esc(r.label)+'</td><td>'+r.endpoint+' / '+r.epoch+'</td><td>'+r.observed_class_BA.toFixed(3)+'</td><td>'+r.low_recall.toFixed(3)+'</td><td>'+r.nll.toFixed(6)+'</td><td>'+r.ece_10_equal_width.toFixed(6)+'</td><td><a href="MODEL_CARD_'+r.model.toUpperCase()+'.md">model card</a></td></tr>').join('');const kind=el('kind').value;el('records').innerHTML='<table><tr><th>Study</th><th>Kind</th><th>Files</th><th>Rows / scope</th><th>Evidence</th></tr>'+D.publications.filter(r=>kind==='all'||r.kind===kind).map(r=>'<tr><td>'+esc(r.title)+' ('+esc(r.notebook)+')</td><td>'+esc(r.kind)+'</td><td>'+r.files_with_manifest+'</td><td>'+r.scientific_rows.toLocaleString()+'<br><small>'+esc(r.row_count_scope)+'</small></td><td><a href="'+esc(r.HF_tree)+'">HF snapshot</a> · <a href="'+esc(r.receipt)+'">audit</a></td></tr>').join('');}
el('caps').innerHTML=D.capabilities.map(r=>'<p><b>'+esc(r.capability)+'</b>: '+esc(r.status.replaceAll('_',' '))+'<br><small>'+esc(r.evidence)+'; '+esc(r.limit)+'</small></p>').join('');el('artifacts').innerHTML=D.artifacts.map(r=>'<p><a href="'+esc(r.url)+'">'+esc(r.label)+'</a> <small>'+esc(r.kind.replaceAll('_',' '))+' · SHA '+r.sha256.slice(0,12)+'…</small></p>').join('');['model','kind'].forEach(id=>el(id).addEventListener('change',render));render();
</script></html>'''.replace('__DATA__',data)
html=html.replace('the62historical','the 62 historical').replace('share50%observed-class','share 50% observed-class').replace('inspected103-image','inspected 103-image')
(OUT/'Evidence_Hub.html').write_text(html,encoding='utf-8')
(OUT/'README.md').write_text('''# TREAD-TWO completed research handoff

**Current v1 / LC r2 / context r3 experiments, reviews and readiness checks are complete and HF-verified. Classification remains research-only and unpromoted.** No additional training/publication notebook is needed to close these unchanged studies. This handoff is a local metadata/report artifact; its indexed scientific evidence is already published at the original HF revisions. The handoff itself has not been uploaded.

[Interactive evidence hub](Evidence_Hub.html) · [Model registry](models.csv) · [Endpoint metrics](endpoint_metrics.csv) · [Publication registry](publications.csv) · [Native checkpoint/explorer URLs](artifacts.csv) · [Capabilities](capabilities.csv) · [Provenance](provenance.csv) · [Release manifest](RELEASE.json).

## Scope and preserved identities

Three distinct candidate fits each completed seed 1, 60 epochs and 2,940 updates. NB13A recovers the same context fit; it is not a fourth experiment. Each original validation-selected checkpoint and final endpoint is retained by exact SHA and pinned HF object URL. Original validation selection is retained. Model cards explain the architecture, results, sources and limits: [v1](MODEL_CARD_V1.md), [LC](MODEL_CARD_LC.md), [context](MODEL_CARD_CONTEXT.md).

The nine-publication registry includes three fits, three evidence reviews and three data/appearance/readiness gates. Row totals are scoped per publication: NB08's 35,502,696 rows are the original saved cursor declaration, not an exhaustive recount; NB11/NB13 raw rows and later derived tables have their documented independent checks. NB07's structured rows exclude additional raw/control CSV inputs. Repeated scientific families and derived tables do not count as independent observations, and audit depth differs between studies. Receipt SHA values and source CSV hashes are retained.

## Actual findings and limits

All three candidates' selected/final endpoints have observed-class BA 50%, low recall 0/62 and high recall 41/41. LC improves selected probability loss versus v1 but does not recover low recognition; context raises confidence and worsens historical NLL/calibration. These are the same already inspected 103 low/high images, with no mid targets. The data has 386 training images and 8 known original training tyres; conservative components are low/mid/high = 1/1/4. Withholding low or mid removes a fit class under the unchanged leakage constraints. Mid co-assignment does not mean the three confirmed mid tyres are physically identical. More augmentations/CSV rows do not supply independent tyres.

Full context rescues 0 prior errors on 47,040 saved training views; the frozen prior already classifies all those sampled views correctly. Algebraic correction parts are not independently trained ablations or fixed final-model training-cohort measurements. The readiness finding is a support constraint, not proof that every possible alternative model must fail. Physical tread depth, physical wear truth and safety assessment remain unavailable. Dense geometry is retained from frozen TREAD-ONE; no new dense improvement or deployment certification is claimed. Signed evidence/attention are internal accounting, not causal physical explanations.

## Next engineering boundary

Use this handoff as the evidence/source contract when application work resumes. Both existing workstations remain untouched because their changes were deferred. The research browser is a report, not a new tyre-inference application. Existing photos suffice for this completed closure; no new acquisition prerequisite or automatic fourth full classifier fit is introduced. A later experiment requires a distinct supported objective/protocol. Do not rerun completed notebooks merely to change a false readiness flag.

## Rebuild and retained source

Generated by [build_tread_two_research_release.py](../../analysis/build_tread_two_research_release.py) under cv_conda. It consumes pinned verification receipts/manifests and saved CSV metrics; no model weights are loaded, no model forward/optimizer update occurs, and no HF write or credential lookup occurs. The release records every consumed local source hash and all output CSV/figure/report hashes. Compact metadata, cards and curated comparison figure are versioned; original full HF mirrors/native models/large telemetry stay outside Git.
''',encoding='utf-8')
artifacts_out={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='RELEASE.json'}
release=dict(status='local_research_handoff_complete',version=1,classification='research_only_unpromoted',scientific_readiness=False,completed_candidate_fits=3,indexed_publications=9,HF_publication_of_this_index=False,
    new_model_forwards=0,optimizer_updates=0,new_independent_tyres=0,remote_writes=0,local_credentials_read=False,workstation_changes=False,
    builder_sha256=sha(Path(__file__)),source_hashes=SOURCES,tables=TABLES,artifacts=artifacts_out)
(OUT/'RELEASE.json').write_text(json.dumps(release,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in release.items() if k not in ('source_hashes','tables','artifacts')},indent=2))
