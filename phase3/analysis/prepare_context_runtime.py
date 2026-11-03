"""Create a separate ATLAS-CONTEXT r3; never edit completed LC source/assets."""
import ast,json,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'tread_two/lc_pilot';NEW=ROOT/'tread_two/context_pilot'
assert not (NEW/'FILES.json').exists(),'Preserve frozen/active context revision'
NEW.mkdir(exist_ok=True)
def write(n,s):p=NEW/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def change(s,a,b):assert a in s,a[:100];return s.replace(a,b)
for n in json.loads((OLD/'FILES.json').read_text()):
    p=NEW/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/n,p)
s=(OLD/'model_atlas.py').read_text(encoding='utf-8')
s=change(s,'reference_allowed=allowed, reference_scale=scale, allowed_counts=counts, effective_coefficients=effective)', 'reference_allowed=allowed, reference_scale=scale, allowed_counts=counts, effective_coefficients=effective,\n                    local_contrast_tiles=query_lc.reshape(batch,32,4,64,64))')
write('frozen_lc.py',s)
done=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
m=json.loads((done/'REMOTE_MANIFEST.json').read_text());meta=m['files']['artifacts/epochs/epoch_0007.pt']
base=dict(protocol=m['contract'],revision='3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8',epoch=7,bytes=meta['bytes'],sha256=meta['sha256'],
 url='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8/phase3/tread-two/atlas-lc-r2/fit/'+m['contract']+'/'+meta['object'],
 source='Original validation-selected LC model; no test-based substitution')
write('BASE_INPUT.json',json.dumps(base,indent=2)+'\n')
s=(OLD/'spec.py').read_text(encoding='utf-8')
s=change(s,"engineering=False):","engineering=False, base_weights=None):")
s=change(s,"recipe = read(HERE / 'RECIPE.json')", "recipe = read(HERE / 'RECIPE.json')\n    base_weights=Path(base_weights) if base_weights else Path(weights).parent/'atlas_lc_selected_epoch7.pt'\n    assert sha(base_weights)==read(HERE/'BASE_INPUT.json')['sha256'],'Wrong frozen LC base'")
s=change(s,"weights=str(Path(weights).resolve()))", "weights=str(Path(weights).resolve()),base_weights=str(base_weights.resolve()))")
write('spec.py',s)
s=(OLD/'worker.py').read_text(encoding='utf-8')
s=s.replace("TreadTwoAtlas(bank, labels, cfg['weights'])", "TreadTwoAtlas(bank, labels, cfg['weights'],cfg['base_weights'])").replace("TreadTwoAtlas(bank, labels, self.cfg['weights'])", "TreadTwoAtlas(bank, labels, self.cfg['weights'],self.cfg['base_weights'])")
s=s.replace('81274',"self.recipe['trainable_parameters']")
s=s.replace("self.state['model']['bank_rgb']","self.state['model']['base.bank_rgb']").replace("self.state['model']['bank_labels']","self.state['model']['base.bank_labels']")
s=s.replace('self.model.anchor.state_dict()', 'self.model.base.state_dict()')
s=change(s,"assert not self.model.anchor.training and all(not p.requires_grad and p.grad is None for p in self.model.anchor.parameters())", "assert not self.model.base.training and all(not p.requires_grad and p.grad is None for p in self.model.base.parameters())")
s=change(s,"('ce_a','ce_b','js','texture','total')", "('ce_a','ce_b','js','texture','residual_l2','total')")
s=change(s,"('ce_a','ce_b','js','texture')", "('ce_a','ce_b','js','texture','residual_l2')")
s=change(s,"reference_evidence=((base['affinity'][slot] * base['patch_weights'][slot,:,None]).sum(0)[None] * base['effective_coefficients'][slot]).cpu().numpy()))", "reference_evidence=((base['affinity'][slot] * base['patch_weights'][slot,:,None]).sum(0)[None] * base['effective_coefficients'][slot]).cpu().numpy(),\n                    base_logits=base['base_logits'][slot].cpu().numpy(),context_evidence=base['context_evidence'][slot].cpu().numpy(),\n                    context_bias=base['context_bias'].cpu().numpy(),attention=base['attention'][slot].cpu().numpy(),\n                    context_embeddings=base['query_embeddings'][slot].cpu().numpy(),base_query_embeddings=base['base_query_embeddings'][slot].cpu().numpy()))\n                for layer in range(2):\n                    for head in range(4):\n                        for q in range(32):\n                            tables.setdefault('endpoint_attention',[]).extend(dict(run=self.cfg['run'],sample=self.state['sample_cursor']+slot,layer=layer,head=head,query=q,key=k,value=float(base['attention'][slot,layer,head,q,k])) for k in range(32))")
write('worker.py',s)
s=(OLD/'telemetry.py').read_text(encoding='utf-8')
s=change(s,'    return tables, internal', '''    correction=out['context_evidence'].detach().cpu().numpy()
    base_logits=out['base_logits'].detach().cpu().numpy();bias_delta=out['context_bias'].detach().cpu().numpy()
    attention=out['attention'].detach().cpu().numpy();alpha=values['patch_weights']
    tables['context_evidence']=[];tables['context_summary']=[]
    for i,row in enumerate(rows):
        for patch in range(32):
            tables['context_evidence'].append(dict(run=cfg['run'],sample=start_sample+i,patch=patch,source_patch=int(out['source_patch'][patch]),**{f'delta_{c}':float(correction[i,patch,c]) for c in range(3)}))
        total=correction[i].sum(0)+bias_delta
        base_prob=np.exp(base_logits[i]-base_logits[i].max());base_prob/=base_prob.sum()
        record=dict(run=cfg['run'],sample=start_sample+i,base_prediction=int(base_prob.argmax()),base_nll=float(-np.log(max(base_prob[row['class_index']],1e-12))),correction_l2=float(np.square(total).mean()))
        record.update({f'base_logit_{c}':float(base_logits[i,c]) for c in range(3)})
        record.update({f'bias_delta_{c}':float(bias_delta[c]) for c in range(3)})
        for layer in range(2):
            for head in range(4):
                a=attention[i,layer,head]
                entropy=-(a*np.log(a.clip(1e-12))).sum(-1)/np.log(max(2,int((values['patch_coverage'][i]>0).sum())))
                record[f'entropy_{layer}_{head}']=float((entropy*alpha[i]).sum())
                record[f'max_attention_{layer}_{head}']=float((a.max(-1)*alpha[i]).sum())
        tables['context_summary'].append(record)
    return tables, internal''')
write('telemetry.py',s)
s=(OLD/'schemas.py').read_text(encoding='utf-8')
s+='''
SCHEMAS['context_evidence']=['run','sample','patch','source_patch','delta_0','delta_1','delta_2']
SCHEMAS['context_summary']=['run','sample','base_prediction','base_nll','correction_l2']+[f'base_logit_{c}' for c in range(3)]+[f'bias_delta_{c}' for c in range(3)]+[field for layer in range(2) for head in range(4) for field in (f'entropy_{layer}_{head}',f'max_attention_{layer}_{head}')]
SCHEMAS['endpoint_attention']=['run','sample','layer','head','query','key','value']
SCHEMAS['view_pairs'] += ['residual_l2']
SCHEMAS['train_updates'] += ['residual_l2']
'''
write('schemas.py',s)
s=(OLD/'objective.py').read_text(encoding='utf-8')
s=change(s,"total=.5*(ca+cb)+.1*js+.05*texture", "residual=.5*(a['context_total'].square().mean(-1)+b['context_total'].square().mean(-1))\n    total=.5*(ca+cb)+.1*js+.05*texture+.02*residual")
s=change(s,'total=total,patch_distance=distance)', 'total=total,patch_distance=distance,residual_l2=residual)')
write('objective.py',s)
s=(OLD/'coordinator.py').read_text(encoding='utf-8')
s=change(s,'cfg = configuration(root, weights)', '''base_weights=assets/'atlas_lc_selected_epoch7.pt'
    base_meta=read(HERE/'BASE_INPUT.json')
    if not base_weights.exists() or sha(base_weights)!=base_meta['sha256']:
        hub=Publisher('phase3/tread-two/context-asset-read',token,assets/'BASE_REMOTE_HISTORY.jsonl')
        def transfer_base():
            part=base_weights.with_suffix('.part')
            with hub.session.get(base_meta['url'],timeout=(20,150),stream=True) as response:
                response.raise_for_status()
                with part.open('wb') as stream:
                    for block in response.iter_content(1024**2):stream.write(block)
                    stream.flush();os.fsync(stream.fileno())
            assert part.stat().st_size==base_meta['bytes'] and sha(part)==base_meta['sha256']
            os.replace(part,base_weights)
        hub.call(transfer_base,'pinned_LC_base_download')
    cfg = configuration(root, weights,base_weights=base_weights)''')
s=change(s,"configuration(measured['data_root'], measured['weights'], engineering=True)","configuration(measured['data_root'], measured['weights'], engineering=True,base_weights=measured['base_weights'])")
s=change(s,"configuration(cfg['data_root'], cfg['weights'])", "configuration(cfg['data_root'], cfg['weights'],base_weights=cfg['base_weights'])")
s=s.replace('One ATLAS-LC r2 fit complete.','One ATLAS-CONTEXT r3 fit complete.')
write('coordinator.py',s)
s=(OLD/'acceptance.py').read_text(encoding='utf-8').replace('2 * 103 * 7','2 * 103 * 9')
s=change(s,"'patch_evidence','predictions','case_rules'):","'patch_evidence','predictions','case_rules','context_evidence','context_summary'):")
s=change(s,"        else:\n            planned = size * 60 / 2 * 16", "        elif family=='endpoint_attention':\n            planned=size*103/2\n        else:\n            planned = size * 60 / 2 * 16")
a=s.index("    native_bytes = sum(");b=s.index('    # Working copy',a)
s=s[:a]+'''    # Extrapolate each native artifact category by its actual observation count.
    # Do not multiply epoch weights by the endpoint-image ratio.
    categories={kind:sum(v['bytes'] for n,v in expected.state['artifacts'].items() if n.startswith('artifacts/'+kind+'/') and n.endswith('.pt')) for kind in ('epochs','endpoint_maps','dense')}
    projected_native=categories['epochs']*60/2+categories['endpoint_maps']*103/2+categories['dense']*570/max(observed_dense,1)+100_000_000
'''+s[b:]
write('acceptance.py',s)
recipe=json.loads((OLD/'RECIPE.json').read_text())
recipe.update(family='TREAD-TWO/ATLAS-CONTEXT',revision='atlas-context-r3-existing-cohort-seed1',HF_prefix='phase3/tread-two/atlas-context-r3',
 objective='paired CE + .1 probability JS + .05 contextual-embedding distance + .02 squared bounded total correction',
 architecture='Frozen validation-selected LC epoch7; warm-start trainable patch encoder; two position-aware 96D four-head attention blocks; zero-initialized bounded spatial correction',
 frozen_base_sha256=base['sha256'],frozen_base_protocol=base['protocol'],trainable_parameters=0,
 diagnostics=recipe['diagnostics']+['reverse_bands','reverse_columns'],
 context_bound=2.,context_bias_bound=.5,context_attention='Full selected/final identity matrices CSV/native; per-head summaries for every observation; no full per-update matrix retention')
write('RECIPE.json',json.dumps(recipe,indent=2)+'\n')
vendored=json.loads((NEW/'VENDORED.json').read_text());vendored['frozen_lc.py']=dict(source='tread_two/lc_pilot/model_atlas.py',adaptation='Expose the exact computed LC input tiles only; classification math unchanged',source_sha256=json.loads((OLD/'FILES.json').read_text())['model_atlas.py']['sha256'])
write('VENDORED.json',json.dumps(vendored,indent=2)+'\n')
for p in NEW.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Separate context runtime scaffold created; previous30source/assets preserved')
