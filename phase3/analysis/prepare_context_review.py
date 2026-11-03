"""Create an isolated r3 saved-evidence review from preserved generic audit code."""
import ast,hashlib,json,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'tread_two/lc_review';NEW=ROOT/'tread_two/context_review';NEW.mkdir(exist_ok=True)
assert not (NEW/'review_context.py').exists(),'Initial scaffold only; preserve a delivered review'
def change(s,a,b):
    assert a in s,a[:100]
    return s.replace(a,b)
s=(OLD/'review_lc.py').read_text(encoding='utf-8')
for a,b in [('3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8','a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e'),('ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28','36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'),('phase3/tread-two/atlas-lc-r2/fit/','phase3/tread-two/atlas-context-r3/fit/'),('53342','53754'),('1236','1648'),('1442','1854'),('64680','144060'),('.eq(22)','.eq(49)'),('from paired_review import extend','from paired_context import extend'),('from render_lc_review import render','from render_context import render')]:s=s.replace(a,b)
s=change(s,"    assert math.isclose(selection.validation_ba_observed.max(),.5625)\n    assert validation[validation.label==0].correct.any()", "    assert math.isclose(float(best.validation_ba_observed),float(selection.validation_ba_observed.max()))")
s=change(s,"    assert identity[identity.label.eq(0)].prediction.eq(1).all()\n    assert identity[identity.label.eq(2)].prediction.eq(2).all()", "    assert identity.groupby('endpoint').size().eq(103).all()")
s=change(s,"    embeddings=inputs.table('reference_embeddings')", "    embeddings=inputs.table('reference_embeddings')\n    from context_checks import extend as context_audit\n    ctx=context_audit(inputs,out,p,coef,save,checks)")
s=change(s,'        calculated=selected@weights.T+bias', "        context_delta=ctx['deltas'].groupby('sample')[[f'delta_{c}' for c in range(3)]].sum().loc[sample_ids].to_numpy()\n        calculated=selected@weights.T+bias+context_delta")
s=change(s,'        for row,calc in zip(end.itertuples(index=False),calculated):','        for row,calc,delta in zip(end.itertuples(index=False),calculated,context_delta):')
s=change(s,'reference_sum=float(calc[c]-bias[c]),reconstructed_logit', 'reference_sum=float(calc[c]-bias[c]-delta[c]),context_spatial_sum=float(delta[c]),reconstructed_logit')
s=change(s,"logits reconstructed from saved576 affinities, geometric pooling,18 reference weights andbias", "logits reconstructed from saved576 affinities, geometric pooling,18 frozen reference weights,context correction andcombined bias")
s=s.replace('Inputs are immutable NB11','Inputs are immutable NB13').replace('never alter NB11','never alter NB13')
ast.parse(s);(NEW/'review_context.py').write_text(s,encoding='utf-8')
s=(OLD/'paired_review.py').read_text(encoding='utf-8').replace('960156','967572')
s=change(s,'total=.5*(ce_a+ce_b)+.1*js+.05*texture', 'residual=inputs.table("context_summary").set_index("sample").correction_l2\n    regularizer=.5*(residual.loc[paired.sample_a].to_numpy()+residual.loc[paired.sample_b].to_numpy())\n    assert np.allclose(regularizer,paired.residual_l2,atol=1e-6,rtol=2e-6)\n    total=.5*(ce_a+ce_b)+.1*js+.05*texture+.02*regularizer')
s=s.replace("['ce_a','ce_b','js','texture','total']","['ce_a','ce_b','js','texture','residual_l2','total']").replace("('ce_a','ce_b','js','texture')","('ce_a','ce_b','js','texture','residual_l2')")
s=s.replace("texture=('texture','mean'),total=", "texture=('texture','mean'),residual_l2=('residual_l2','mean'),total=")
s=s.replace("suffixes=('_v1','_r2')","suffixes=('_v1','_r3')").replace("+'_r2'","+'_r3'").replace("save('v1_r2_comparison'","save('v1_r3_comparison'").replace('ATLAS-LC','ATLAS-CONTEXT')
ast.parse(s);(NEW/'paired_context.py').write_text(s,encoding='utf-8')
s=(OLD/'render_lc_review.py').read_text(encoding='utf-8')
s=s[:s.index('\n\ndef write_r2_notes')]
s=s.replace('ATLAS-LC','ATLAS-CONTEXT').replace('r2 ·','r3 ·').replace('NB11','NB13').replace('write_r2_notes(out)','write_context_notes(out)').replace('1,442','1,854')
s=change(s,"title=f'{endpoint}: 62 low → mid; 41 high → high'", "title=f'{endpoint}: low correct {cm[0,0]}/62; high correct {cm[2,2]}/41'")
s=s.replace('training fit does not transfer to the low tyre','frozen LC prior plus learned spatial correction')
s=s.replace("'grayscale_both','occlude_top_patch']","'grayscale_both','occlude_top_patch','reverse_bands','reverse_columns']").replace('np.arange(7)','np.arange(9)').replace('range(7)','range(9)')
s=change(s,"['Original','Dim ×0.8','Bright ×1.2','Gamma0.8','Gamma1.2','Gray both','Top patch']", "['Original','Dim ×0.8','Bright ×1.2','Gamma0.8','Gamma1.2','Gray both','Top patch','Reverse bands','Reverse columns']")
s=change(s,"    patch=csv('endpoint_patch_evidence')", "    patch=csv('endpoint_patch_evidence')\n    delta=csv('endpoint_context_evidence')\n    summary=csv('context_observations').set_index('sample')\n    attention=json.loads((out/'ATTENTION_VIEW.json').read_text())")
s=change(s,"        row['references']=", "        extra=delta[delta['sample']==sample].sort_values('patch')\n        row['context_patches']=extra[['delta_0','delta_1','delta_2']].to_numpy().tolist()\n        row['base_patches']=(np.array(row['patches'])-np.array(row['context_patches'])).tolist()\n        row['source_patch']=extra.source_patch.astype(int).tolist()\n        sr=summary.loc[sample]\n        row['base_logits']=[float(sr[f'base_logit_{c}']) for c in range(3)]\n        row['bias_delta']=[float(sr[f'bias_delta_{c}']) for c in range(3)]\n        row['attention']=attention.get(str(int(sample)))\n        row['references']=")
s=change(s,'In the original endpoint, all 62 low targets are predicted mid; all 41 high targets high. Both have 50% balanced accuracy.', 'The original validation rule selected epoch 7. The tables show actual results; no new selection, fitting or inference is performed. Reference cards explain the frozen LC branch; combined tiles include cross-patch context correction.')
s=change(s,'<select id="target">','<select id="component"><option value="combined">Combined evidence</option><option value="base">Frozen LC evidence</option><option value="context">Context correction</option></select><select id="layer"><option value="0">Attention layer 1</option><option value="1">Attention layer 2</option></select><select id="head"><option value="0">Head 1</option><option value="1">Head 2</option><option value="2">Head 3</option><option value="3">Head 4</option></select><select id="target">')
s=change(s,'<select id="target">','<select id="target">')
s=change(s,'<option value="occlude_top_patch">Top signed patch replaced</option>', '<option value="occlude_top_patch">Top signed patch replaced</option><option value="reverse_bands">Reverse band order</option><option value="reverse_columns">Reverse columns</option>')
s=change(s,"const scale=Math.max(...x.patches.map(v=>Math.abs(v[cls])),1e-12);x.patches.forEach", "const tiles=$('#component').value==='base'?x.base_patches:($('#component').value==='context'?x.context_patches:x.patches);const scale=Math.max(...tiles.map(v=>Math.abs(v[cls])),1e-12);tiles.forEach")
s=change(s,"tile.title='patch '+i+' / '+labels[cls]+' / '+v[cls]", "tile.title='patch '+i+' / original patch '+x.source_patch[i]+' / '+labels[cls]+' / '+v[cls]")
s=change(s,"text(detail,'Scores low / mid / high: '", "text(detail,'Frozen LC logits: '+x.base_logits.map(fmt).join(' / '));text(detail,'Context bias correction: '+x.bias_delta.map(fmt).join(' / '));text(detail,'Scores low / mid / high: '")
s=change(s,"detail.append(refs);article.append", "text(detail,'Reference cards account for the frozen case branch only; context is shown separately.');if(x.variant==='identity'&&x.attention){const layer=Number($('#layer').value),head=Number($('#head').value);let ag=document.createElement('div');ag.style.display='grid';ag.style.gridTemplateColumns='repeat(32,4px)';for(let q=0;q<32;q++)for(let k=0;k<32;k++){let cell=document.createElement('div'),v=x.attention[layer][head][q][k];cell.style.height='4px';cell.style.background='rgba(95,180,250,'+Math.min(1,v*8)+')';cell.title='query '+q+' key '+k+' probability '+v;ag.append(cell)}text(photos,'Saved attention: rows query, columns key. Hover for probabilities; not causal wear truth.');photos.append(ag)}else{text(photos,'Full attention matrices were retained for original endpoints only.')}detail.append(refs);article.append")
s=change(s,"'#class','#search']", "'#class','#search','#component','#layer','#head']")
s=s.replace('Reference cards show actual source patches and their signed aggregate contributions.','Reference cards show the frozen LC branch only. Tile components separate base/context/combined evidence. Layout interventions rearrange tokens; thumbnails show the original image.').replace('Reconstructed logit | Sum of 18 reference contributions plus class bias','Reconstructed logit | Sum of 18 frozen reference contributions plus weighted context correction and combined class bias')
s += '''\n\ndef write_context_notes(out):
    from context_checks import write_notes
    write_notes(Path(out))
'''
ast.parse(s);(NEW/'render_context.py').write_text(s,encoding='utf-8')
s=(OLD/'publish_lc_review.py').read_text(encoding='utf-8').replace('from review_lc import','from review_context import').replace('ATLAS-LC r2 completed-fit evidence and paired-training review','ATLAS-CONTEXT r3 completed-fit evidence, correction and attention review').replace('phase3/tread-two/atlas-lc-r2/review/','phase3/tread-two/atlas-context-r3/review/').replace('frozen r2 transport','frozen r3 transport')
ast.parse(s);(NEW/'publish_context_review.py').write_text(s,encoding='utf-8')
assets=NEW/'assets';assets.mkdir(exist_ok=True)
for name in ['v1_cohort_metrics.csv','PEER.json']:(assets/name).write_bytes((OLD/'assets'/name).read_bytes())
lc=ROOT/'tread_two/lc_analysis_report'
(assets/'lc_cohort_metrics.csv').write_bytes((lc/'cohort_metrics.csv').read_bytes())
pred=pd.read_csv(lc/'prediction_audit.csv');selected=pred[(pred.stage=='selected_eval')&(pred.variant=='identity')]
selected[['image_id','label','logit_0','logit_1','logit_2']].to_csv(assets/'lc_selected_identity.csv',index=False,float_format='%.17g',lineterminator='\n')
peer=json.loads((assets/'PEER.json').read_text());peer.update(lc_fit_revision='3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8',lc_review_revision='77d03303a9d42f89bfd3fbc567faedf86b09a855',lc_selected_epoch=7,
    lc_metrics_sha256=hashlib.sha256((assets/'lc_cohort_metrics.csv').read_bytes()).hexdigest(),lc_selected_identity_sha256=hashlib.sha256((assets/'lc_selected_identity.csv').read_bytes()).hexdigest(),comparison_scope='Already-inspected cohort, changed models; descriptive comparison, not independent confirmation')
(assets/'PEER.json').write_text(json.dumps(peer,indent=2)+'\n',encoding='utf-8')
print('Isolated context review scaffold and pinned peer evidence prepared. Completed LC/v1 sources unchanged.')
