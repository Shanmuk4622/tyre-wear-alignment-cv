"""Build isolated r2 sources; never rewrite a completed v1 experiment."""
import ast,csv,hashlib,json,shutil,sys,textwrap
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tread_two/pilot';NEW=ROOT/'tread_two/lc_pilot'
NEW.mkdir(exist_ok=True);(NEW/'assets').mkdir(exist_ok=True)
assert not (NEW/'FILES.json').exists(), 'Initial scaffolding only: preserve the frozen delivered r2; review editable runtime for a separate revision.'
def write(name,value):
    path=NEW/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value,encoding='utf-8')
def change(s,a,b):
    assert a in s,a[:100]
    return s.replace(a,b)
for p in OLD.glob('*.py'):shutil.copy2(p,NEW/p.name)
for n in ('VENDORED.json',):shutil.copy2(OLD/n,NEW/n)
# Exact train-only representation, vendored as a separately frozen r2 dependency.
source=(ROOT/'tread_two/appearance_gate/gate_features.py').read_text()
write('lc_features.py',source[:source.index('\ndef descriptor(')])
write('LC_RULES.json',(ROOT/'tread_two/appearance_gate/RULES.json').read_text())
s=(NEW/'data_atlas.py').read_text()
s=change(s,"group = str(generator.choice(sorted(pools[cls][domain])))","groups = sorted(pools[cls][domain])\n        weights = np.sqrt([len(pools[cls][domain][g]) for g in groups]); weights /= weights.sum()\n        group = str(generator.choice(groups, p=weights))")
s=change(s,"1 / len(pools) / len(pools[cls]) / len(pools[cls][domain]) / len(pools[cls][domain][group])","1 / len(pools) / len(pools[cls]) * float(weights[groups.index(group)]) / len(pools[cls][domain][group])")
s=change(s,"and r['image_id'] not in bank_ids","")
s=change(s,'assert len(queries) == 378','assert len(queries) == 386')
s=change(s,"uniform(.9, 1.1, 3)","uniform(.8, 1.2, (2,3))")
s += '''\n\ndef source_hashes(rows,device):
    return torch.tensor([list(bytes.fromhex(r['image_sha256'])) for r in rows],dtype=torch.uint8,device=device)
'''
write('data_atlas.py',s)
s=(OLD/'model_atlas.py').read_text()
s=change(s,'from common import sha','from common import HERE, read, sha\nimport numpy as np\nfrom lc_features import contrast')
start=s.index('        self.register_buffer(\'sobel_x\'');end=s.index('\n\nclass TreadTwoAtlas',start)
s=s[:start]+'''\n    def forward(self, features):
        return F.normalize(self.body(features).flatten(1).float(),dim=1,eps=1e-6)
'''+s[end:]
s=change(s,"self.register_buffer('bank_labels', bank_labels.long().clone())","self.register_buffer('bank_labels', bank_labels.long().clone())\n        self.register_buffer('bank_valid',torch.from_numpy(np.load(HERE/'assets/bank_valid.npy',allow_pickle=False)))\n        self.register_buffer('bank_sha256',torch.from_numpy(np.load(HERE/'assets/bank_sha256.npy',allow_pickle=False)))\n        self.lc_rules=read(HERE/'LC_RULES.json')")
s=change(s,"def classify(self, atlas, augmentation=None, variant='identity', occluded_patch=None):","def classify(self, atlas, source_hash, augmentation=None, variant='identity', occluded_patch=None):")
s=change(s,'brightness, contrast, saturation = factors[:, 0], factors[:, 1], factors[:, 2]','brightness, contrast_factor, gamma = factors[:, 0], factors[:, 1], factors[:, 2]')
s=change(s,'tiles = (tiles - center) * contrast + center\n            gray = tiles.mean(2, keepdim=True)\n            tiles = gray + (tiles - gray) * saturation','tiles = ((tiles - center) * contrast_factor + center).clamp(0,1).pow(gamma)')
s=change(s,"elif variant not in ('identity', 'occlude_top_patch'):","elif variant == 'gamma_080':\n            tiles = tiles.clamp(0,1).pow(.8)\n        elif variant == 'gamma_120':\n            tiles = tiles.clamp(0,1).pow(1.2)\n        elif variant not in ('identity', 'occlude_top_patch'):")
s=change(s,'features = self.encoder(torch.cat((tiles.reshape(batch * 32, 3, 64, 64), bank), 0))',"query_lc=contrast(tiles.reshape(batch*32,3,64,64),valid.reshape(batch*32,1,64,64),self.lc_rules)\n        bank_lc=contrast(bank,self.bank_valid,self.lc_rules)\n        features = self.encoder(torch.cat((query_lc, bank_lc), 0))")
s=change(s,'pair = affinity[:, :, None, :] * coefficients[None, None] * alpha[:, :, None, None]', '''assert source_hash.dtype==torch.uint8 and source_hash.shape==(batch,32), 'Full source SHA required; no label input'
        allowed=~(source_hash[:,None,:]==self.bank_sha256[None]).all(-1)
        counts=torch.stack([allowed[:,self.bank_labels==c].sum(1) for c in range(3)],1)
        assert (counts>0).all(), 'No remaining reference support'
        scale=6/counts[:,self.bank_labels].float()
        effective=coefficients[None]*allowed[:,None,:]*scale[:,None,:]
        pair = affinity[:, :, None, :] * effective[:, None] * alpha[:, :, None, None]''')
s=change(s,"bias=self.bias.clone(), temperature=temperature, replacement=replacement)","bias=self.bias.clone(), temperature=temperature, replacement=replacement,\n                    reference_allowed=allowed, reference_scale=scale, allowed_counts=counts, effective_coefficients=effective)")
s=change(s,"def forward(self, rgb, augmentation=None, variant='identity', occluded_patch=None):","def forward(self, rgb, augmentation=None, variant='identity', occluded_patch=None, *, source_hash):")
s=change(s,"output = self.classify(atlas, augmentation, variant, occluded_patch)\n        return dict(output, masks=dense['masks'], points=dense['points'], guides=dense['guides'],\n                    tread_one_probabilities=dense['probabilities'], tread_one_logits=dense['logits'], atlas=atlas)",'''shared=dict(masks=dense['masks'],points=dense['points'],guides=dense['guides'],
                        tread_one_probabilities=dense['probabilities'],tread_one_logits=dense['logits'],atlas=atlas)
            if augmentation is None:
                return dict(self.classify(atlas,source_hash,None,variant,occluded_patch),**shared)
            assert augmentation.shape==(len(rgb),2,3)
            first=dict(self.classify(atlas,source_hash,augmentation[:,0]),**shared)
            second=dict(self.classify(atlas,source_hash,augmentation[:,1]),**shared)
            first['_paired']=second
            return first''')
s=s.replace('ATLAS v1:','ATLAS-LC r2:').replace('# v1 uses FP32','# r2 uses FP32')
write('model_atlas.py',s)
write('objective.py','''"""Fixed symmetric appearance objective; no held-out/reference labels enter pairs."""
import torch
from torch.nn import functional as F
def objective(a,b,labels):
    ca=F.cross_entropy(a['logits'],labels,reduction='none')
    cb=F.cross_entropy(b['logits'],labels,reduction='none')
    p,q=a['probabilities'],b['probabilities'];m=(p+q)/2
    js=.5*((p*(p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1)+(q*(q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1))
    distance=1-F.cosine_similarity(a['query_embeddings'],b['query_embeddings'],dim=-1)
    texture=(distance*a['patch_weights']).sum(-1)
    total=.5*(ca+cb)+.1*js+.05*texture
    return total.mean(),dict(ce_a=ca,ce_b=cb,js=js,texture=texture,total=total,patch_distance=distance)
''')
s=(OLD/'schemas.py').read_text()
s += '''
SCHEMAS['case_rules']=['run','sample','prototype','reference_class','allowed','same_source_sha256','normalization_scale']
SCHEMAS['view_pairs']=['run','epoch','update','image_index','sample_a','sample_b','ce_a','ce_b','js','texture','total','label','prediction_a','prediction_b','disagree']
SCHEMAS['view_patch_consistency']=['run','sample_a','sample_b','patch','pooling_mass','cosine_distance']
SCHEMAS['predictions'] += ['source_sha256','allowed_low','allowed_mid','allowed_high']
SCHEMAS['sample_exposures']=['gamma' if x=='saturation' else x for x in SCHEMAS['sample_exposures']]+['view']
SCHEMAS['train_updates'] += ['ce_a','ce_b','js','texture','prediction_views']
'''
write('schemas.py',s)
s=(OLD/'telemetry.py').read_text()
s=change(s,"pred.update({f'logit_{c}'", "pred.update(source_sha256=row['image_sha256'],**dict(zip(('allowed_low','allowed_mid','allowed_high'),out['allowed_counts'][slot].detach().cpu().tolist())))\n        pred.update({f'logit_{c}'")
s=change(s,'    return tables, internal', '''    allowed=out['reference_allowed'].detach().cpu().numpy();scales=out['reference_scale'].detach().cpu().numpy()
    tables['case_rules']=[dict(run=cfg['run'],sample=start_sample+i,prototype=j,reference_class=j//6,allowed=int(allowed[i,j]),same_source_sha256=int(not allowed[i,j]),normalization_scale=float(scales[i,j])) for i in range(len(rows)) for j in range(18)]
    return tables, internal''')
write('telemetry.py',s)
s=(OLD/'worker.py').read_text()
s=change(s,'from data_atlas import bank_cases, batch, dataset, load_bank, schedule','from data_atlas import bank_cases, batch, dataset, load_bank, schedule, source_hashes\nfrom objective import objective')
s=change(s,"'deterministic training-only bank retained without appearance filtering'","'NB10 training-only diversity proposal; source-SHA self exclusion; all386 queries'")
s=change(s,"out = self.model(images, factors)\n        loss = torch.nn.functional.cross_entropy(out['logits'], labels)","out = self.model(images, factors, source_hash=source_hashes(rows,images.device))\n        paired=out['_paired']\n        loss,parts=objective(out,paired,labels)")
s=change(s,"tables['parameter_gradients'] = gradients",'''second,records_b=observations(paired,rows,self.registry,self.cfg,'train','online_before_update_view_b',epoch0+1,
            update,self.state['sample_cursor']+len(rows),update-1,before,variant='appearance_view_b')
        merge(tables,second);records.extend(records_b)
        tables['view_pairs']=[];tables['view_patch_consistency']=[]
        values={k:v.detach().cpu().numpy() for k,v in parts.items()}
        for i,row in enumerate(rows):
            sa=self.state['sample_cursor']+i;sb=sa+len(rows)
            pa=int(out['probabilities'][i].argmax());pb=int(paired['probabilities'][i].argmax())
            tables['view_pairs'].append(dict(run=self.cfg['run'],epoch=epoch0+1,update=update,image_index=self.registry[row['image_id']],sample_a=sa,sample_b=sb,label=row['class_index'],prediction_a=pa,prediction_b=pb,disagree=int(pa!=pb),**{k:float(values[k][i]) for k in ('ce_a','ce_b','js','texture','total')}))
            for p in range(32): tables['view_patch_consistency'].append(dict(run=self.cfg['run'],sample_a=sa,sample_b=sb,patch=p,pooling_mass=float(out['patch_weights'][i,p]),cosine_distance=float(values['patch_distance'][i,p])))
        tables['parameter_gradients'] = gradients''')
start=s.index('        for slot, (draw, row) in enumerate(zip(draws, rows)):');end=s.index('        dense, packed',start)
s=s[:start]+'''        for view in range(2):
            for slot,(draw,row) in enumerate(zip(draws,rows)):
                tables['sample_exposures'].append(dict(run=self.cfg['run'],sample=self.state['sample_cursor']+view*len(rows)+slot,
                    epoch=epoch0+1,update=update,slot=slot,occurrence=draw['occurrence'],image_index=self.registry[row['image_id']],
                    class_draw=draw['class_draw'],domain_draw=draw['domain_draw'],source_group=draw['source_group'],draw_probability=draw['probability'],
                    augmentation_seed=draw['augmentation_seed'],view=view,brightness=float(factor_values[slot,view,0]),contrast=float(factor_values[slot,view,1]),gamma=float(factor_values[slot,view,2])))
'''+s[end:]
s=change(s,"self.state['sample_cursor'] += len(rows)","self.state['sample_cursor'] += 2*len(rows)")
s=change(s,"self.state['sample_cursor'] += 2*len(rows)\n        self.state['eval_records'].extend(records)","self.state['sample_cursor'] += len(rows)\n        self.state['eval_records'].extend(records)")
s=change(s,"update, self.state['sample_cursor'], update - 1, before)","update, self.state['sample_cursor'], update - 1, before, variant='appearance_view_a')")
s=change(s,"loss=float(loss.detach()), lr=lr", "loss=float(loss.detach()),prediction_views=2,**{k:float(v.detach().mean()) for k,v in parts.items() if k in ('ce_a','ce_b','js','texture')}, lr=lr")
s=change(s,'base = self.eval_model(images)', 'base = self.eval_model(images,source_hash=source_hashes(rows,images.device))')
s=change(s,'self.eval_model(images, variant=variant, occluded_patch=occluded)','self.eval_model(images, variant=variant, occluded_patch=occluded,source_hash=source_hashes(rows,images.device))')
# Endpoint reference mass must use the per-query effective coefficients.
s=change(s,"base['coefficients']", "base['effective_coefficients'][slot]")
s=s.replace('online_before_update\'', 'online_before_update_view_a\'')
write('worker.py',s)
recipe=json.loads((OLD/'RECIPE.json').read_text())
recipe.update(family='TREAD-TWO/ATLAS-LC',revision='atlas-lc-r2-existing-cohort-seed1',
 objective='mean of two CE + .1 probability JS + .05 band-weighted paired query-embedding cosine distance',
 sampler='uniform class/domain; group probability proportional sqrt(n); uniform image within group; replacement; 386 training queries',
 augmentation='two query-only brightness/contrast/gamma views, independent uniform [.8,1.2]; seed addressed; one frozen dense pass',
 bank_policy='18 unique training photos from pinned NB10; full source SHA self-reference exclusion and 6/n_allowed reference-class scale; all386 queries',
 bank_risks='paint/glare/identity shortcuts remain; reference diversity does not create independent tyres',
 diagnostics=['grayscale_both','brightness_080','brightness_120','gamma_080','gamma_120','occlude_top_patch'],
 HF_prefix='phase3/tread-two/atlas-lc-r2',representation='masked local9x9 z/std floor.02 clipped3, Sobel x/y/magnitude; four channels')
recipe.pop('bank_rgb_sha256')
write('RECIPE.json',json.dumps(recipe,indent=2)+'\n')
# Native corrected transport, identical policy to the audited external v1 repair.
s=(OLD/'rate_budget.py').read_text().replace('import re\n','import re\nimport math\n')
launcher=(ROOT/'analysis/nb08_transport_launcher.py').read_text()
a=launcher.index('    def reserve(');b=launcher.index('    original_send',a)
methods=launcher[a:b].replace('instant = now()','instant = time.time()').replace('headers, instant = response.headers, now()','headers, instant = response.headers, time.time()').replace('event(log,','event(self.root / \'RATE_EVENTS.jsonl\',').replace('pause(min(30., delay))','time.sleep(min(30., delay))')
a=s.index('    def reserve(');b=s.index('\n\nclass BudgetSession',a)
s=s[:a]+methods+s[b:]
s=s.replace('class Budget:',"POLICY={'api':(300,96),'resolver':(300,400),'transfer':(300,1000),'repo_commit':(3600,96)}\n\nclass Budget:")
s=change(s,'category = self.budget.category(request.url)',"parsed=urlparse(request.url)\n        if request.method.upper()=='POST' and parsed.hostname=='huggingface.co' and '/commit/' in parsed.path:\n            self.budget.reserve('repo_commit')\n        category = self.budget.category(request.url)")
write('rate_budget.py',s)
# Remove only previously verified publication staging before rebuilding a larger snapshot.
s=(OLD/'coordinator.py').read_text()
s=change(s,"base.mkdir(exist_ok=True)\n    target", "base.mkdir(exist_ok=True)\n    for previous in base.glob('snapshot-*'):\n        marker=previous/'VERIFIED_PUBLICATION.json'\n        if marker.exists():\n            resolved=previous.resolve();assert resolved.parent==base.resolve()\n            shutil.rmtree(resolved)\n    target")
s=change(s,"receipt = hub.publish(target, release=release_owner)", "receipt = hub.publish(target, release=release_owner)\n    atomic(target/'VERIFIED_PUBLICATION.json',receipt)")
s=s.replace('One ATLAS fit complete.', 'One ATLAS-LC r2 fit complete.')
write('coordinator.py',s)
s=(OLD/'acceptance.py').read_text()
s=change(s,'60 * 49 * 8 + 60 * 81 + 2 * 103 * 5','2 * 60 * 49 * 8 + 60 * 81 + 2 * 103 * 7')
s=change(s,"'sample_exposures'):","'sample_exposures','view_pairs','view_patch_consistency'):")
s=change(s,"'patch_evidence','predictions'):","'patch_evidence','predictions','case_rules'):")
write('acceptance.py',s)
for p in NEW.glob('*.py'):ast.parse(p.read_text())
print('r2 source skeleton built:',NEW)
