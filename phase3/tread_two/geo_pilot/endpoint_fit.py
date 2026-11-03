"""Train-reference diagnostics, native exports and interactive endpoint evidence."""
import base64,html,io,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from common import atomic,cpu,native,read
from geometry import band_weights
from telemetry_fit import add,array

def thumbnail(root,endpoint,row,batch,out):
    rgb=(array(batch['rgb'][0]).transpose(1,2,0)*255).clip(0,255).astype('uint8')
    mask=array(out['refined']['masks'][0])>=.5
    color=np.zeros_like(rgb);color[np.repeat(np.repeat(mask[0],4,0),4,1)]=[60,140,245];color[np.repeat(np.repeat(mask[1],4,0),4,1)]=[245,187,60]
    display=np.where((color.sum(-1)>0)[...,None],(.65*rgb+.35*color).astype('uint8'),rgb)
    path=Path(root)/'gallery_cache'/endpoint/(row['image_id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(display).resize((192,256)).save(path,format='JPEG',quality=72)
    return path.relative_to(root).as_posix()

def finish_endpoint(engine,records,value):
    c=engine.cursor;items=c['endpoint_info'];bank=[r for r in items if r['role']=='train' and r['SSL_valid']]
    missing=0
    for query in items:
        refs=[r for r in bank if r['sha256']!=query['sha256'] and r['component']!=query['component']]
        if not query['SSL_valid'] or not refs:missing+=1;continue
        z=np.asarray(query['embedding']);z=z/max(np.linalg.norm(z),1e-12)
        scored=[]
        for ref in refs:
            x=np.asarray(ref['embedding']);similarity=float(z@x/max(np.linalg.norm(x),1e-12));scored.append((similarity,ref['image_id'],ref))
        for rank,(similarity,_,ref) in enumerate(sorted(scored,key=lambda v:(-v[0],v[1]))[:5],1):
            add(records,'endpoint_neighbours',sample=query['sample'],reference_id=ref['image_id'],reference_sha256=ref['sha256'],reference_component=ref['component'],reference_stage='train_identity_same_checkpoint',cosine_similarity=similarity,rank=rank,
                same_source_blocked=True,same_component_blocked=True,proxy_label_context_only=True)
    add(records,'gate_checks',check='retrieval_availability',stage=c['endpoint'],passed=missing==0,observed=str(missing)+' queries unavailable',required='all valid queries have cross-component training references',evidence_path='csv/endpoint_neighbours',fixture_or_measured='fixture' if engine.fixture else 'measured')
    target=engine.store.root/f'exports/{c["endpoint"]}.pt'
    native(target,dict(anchor=torch.load(engine.store.root/'anchor.pt',map_location='cpu',weights_only=True),trainable=cpu(engine.eval_model.state_dict()),
                       protocol=engine.protocol,endpoint=c['endpoint'],selected_epoch=c['best']['epoch'] if c['endpoint']=='selected' else engine.cfg['epochs'],
                       preprocessing=dict(hw=[512,384],mean=[.485,.456,.406],std=[.229,.224,.225]),capabilities='research_geometry_and_texture_no_classification_depth_or_safety',source_inventory=engine.cfg['sources']))
    engine.store.register(target)
    data=engine.store.root/f'reports/{c["endpoint"]}_cases.json';atomic(data,items);engine.store.register(data)
    aggregate=engine.store.root/f'reports/{c["endpoint"]}_metrics.json';atomic(aggregate,value);engine.store.register(aggregate)
    cases=[]
    for item in items:
        case={k:v for k,v in item.items() if k not in ('embedding','projected')};case['endpoint']=c['endpoint'];case['image']=base64.b64encode((engine.root/item['thumbnail']).read_bytes()).decode();cases.append(case)
    payload=json.dumps(cases,separators=(',',':')).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GEO endpoint evidence</title><style>body{font:16px system-ui;margin:2rem;background:#101722;color:#edf2fa}select{font:inherit;padding:.5rem}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem}article{background:#24334a;padding:1rem;overflow-wrap:anywhere}img{max-width:100%}</style><h1>TREAD-TWO / GEO · ENDPOINT</h1><p>Blue/gold: predicted tyre/tread. Experimental error estimates and masked residuals are not physical wear, depth, damage labels or safety probabilities. These are development cohorts.</p><label>Cohort <select id="role"><option value="all">All</option><option>train</option><option>validation</option><option>test</option></select></label><p id="count"></p><div id="grid"></div><script>const cases=PAYLOAD;function render(){const role=document.getElementById('role').value,rows=cases.filter(c=>role==='all'||c.role===role),g=document.getElementById('grid');g.replaceChildren();document.getElementById('count').textContent=rows.length+' source images';for(const c of rows){const a=document.createElement('article'),title=document.createElement('p'),img=document.createElement('img'),note=document.createElement('p');title.textContent=c.image_id;img.src='data:image/jpeg;base64,'+c.image;img.alt='Predicted tyre/tread overlay';note.textContent='Dice '+c.dice.map(x=>x.toFixed(4)).join(' / ')+' · texture support '+c.SSL_valid;a.append(title,img,note);g.append(a)}}document.getElementById('role').addEventListener('change',render);render();</script></html>'''.replace('ENDPOINT',c['endpoint']).replace('PAYLOAD',payload)
    path=engine.store.root/f'reports/{c["endpoint"]}_Explorer.html';path.write_text(page,encoding='utf-8');engine.store.register(path)

@torch.no_grad()
def probe(engine,batch,base,out,seeds,variant):
    model=engine.eval_model
    meta=dict(available=True,source_patch=None,mapping=list(range(32)),reason='')
    if variant.startswith('brightness'):
        batch['rgb']=(batch['rgb']*float(variant.split('_')[-1])).clamp(0,1)
    elif variant.startswith('gamma'):batch['rgb']=batch['rgb'].clamp_min(1e-8).pow(float(variant.split('_')[-1]))
    elif variant=='grayscale':batch['rgb']=batch['rgb'].mean(1,keepdim=True).expand(-1,3,-1,-1).clone()
    elif variant=='blur':batch['rgb']=F.avg_pool2d(F.pad(batch['rgb'],(2,2,2,2),mode='reflect'),5,1)
    elif variant=='background_neutral':
        known=(batch['masks'][:,:1]>.5)|(batch['masks'][:,1:2]>.5)|(batch['masks'][:,2:3]>.5)
        batch['rgb']=torch.where(known,batch['rgb'],torch.full_like(batch['rgb'],.5))
    elif variant in ('reverse_bands','reverse_columns','top_residual_tile'):
        atlas=dict(out['atlas'])
        if variant=='top_residual_tile':
            v=out['views'][0];mass=v['weight'].sum((-3,-2,-1))*4
            eligible=v['selected']&(mass>0);error=((v['prediction']-v['target']).abs()*v['weight']).sum((-3,-2,-1))/mass.clamp_min(1)
            if eligible.any():
                tile=int(torch.where(eligible,error,torch.full_like(error,-1))[0].argmax());row,col=divmod(tile,4);atlas['rgb']=atlas['rgb'].clone();atlas['rgb'][:,:,row*64:(row+1)*64,col*64:(col+1)*64]=0;meta['source_patch']=tile
            else:meta.update(available=False,reason='no_valid_observable_masked_tile')
        else:
            for key in ('rgb','valid','ignored'):
                value=atlas[key];n,ch,h,w=value.shape
                block=value.reshape(n,ch,8,64,4,64)
                atlas[key]=block.flip(2 if variant=='reverse_bands' else 4).reshape(n,ch,h,w).contiguous()
            if variant=='reverse_bands':
                meta['mapping']=[(7-r)*4+c for r in range(8) for c in range(4)]
                atlas['row_parameters']=atlas['row_parameters'].reshape(-1,8,64,3).flip(1).reshape(-1,512,3)
                atlas['row_valid']=atlas['row_valid'].reshape(-1,8,64).flip(1).reshape(-1,512)
            else:meta['mapping']=[r*4+(3-c) for r in range(8) for c in range(4)]
            # Column reversal is a tile permutation, not a new monotonic warp.
        atlas['mass'],atlas['coverage']=band_weights(atlas['valid'])
        out=dict(out,atlas=atlas,views=[model.texture_view(atlas,[int(s)*2+i for s in seeds]) for i in (0,1)],probe_meta=meta)
        return batch,base,out
    else:raise ValueError(variant)
    base=engine.anchor(batch['rgb'],batch['guides'],engine.eval_device);out=model(base,batch,seeds);out['probe_meta']=meta
    return batch,base,out

def final_report(engine,records):
    c=engine.cursor;coverage=c['training_SSL_valid']/max(c['training_draws'],1);selected=c['best']
    value=None
    if selected is not None:value=next(v for v in c['history'] if v['epoch']==selected['epoch'])
    useful=bool(value and value['mean_huber'] and value['masked_huber']<=.95*value['mean_huber'] and value['pair_cosine']<=.05 and coverage>=.9)
    result=dict(status='completed',protocol=engine.protocol,scope='engineering_fixture' if engine.fixture else 'one_measured_development_candidate',epochs=engine.cfg['epochs'],updates=c['update'],selected=selected,
                training_SSL_draw_coverage=coverage,predeclared_representation_utility_passed=useful,utility_evaluated_on='selected_original_validation_only_plus_training_coverage',classification_trained=False,physical_depth_validated=False,safety_validated=False,promotion='research_only',history=c['history'])
    target=engine.store.root/'RESULT.json';atomic(target,result);engine.store.register(target)
    for name,passed,observed,required in [('optimizer_update_budget',c['update']==engine.cfg['epochs']*engine.cfg['batches_per_epoch'],c['update'],engine.cfg['epochs']*engine.cfg['batches_per_epoch']),
                                         ('representation_utility',useful,useful,'all_predeclared_dense_rank_prediction_consistency_coverage_requirements'),
                                         ('selected_checkpoint_available',selected is not None,selected['epoch'] if selected else 'unavailable','original_validation_eligibility_rule')]:
        add(records,'gate_checks',check=name,stage='final',passed=passed,observed=str(observed),required=str(required),evidence_path='RESULT.json',fixture_or_measured='fixture' if engine.fixture else 'measured')
