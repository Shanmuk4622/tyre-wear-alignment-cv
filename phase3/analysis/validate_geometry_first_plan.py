"""Validate design consistency/source preservation and schedule feasibility, not a model."""
import ast,hashlib,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'tread_two/geometry_first/plan';OUT=ROOT/'tread_two/geometry_first/validation';OUT.mkdir(exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=read(PLAN/'PLAN.json');s=read(PLAN/'SCHEMAS.json');b=read(PLAN/'BUDGET.json');sources=read(PLAN/'SOURCES.json')
assert p['status']=='design_complete_implementation_pending' and not p['scope']['measured_training_started']
assert not p['architecture']['classification_loss'] and not p['architecture']['shared_trainable_geometry_texture_parameters']
assert p['fit']['candidate_fits']==1 and p['fit']['epochs']*p['fit']['batches_per_epoch']==p['fit']['updates']==2940
assert p['fit']['draws']==2940*8 and p['fit']['texture_views']==2*2940*8
assert s['CSV_families']==40 and s['declared_columns']==sum(len(t['columns']) for t in s['families'].values())==443
for name,meta in s['families'].items():
    names=[c['name'] for c in meta['columns']];assert len(names)==len(set(names)) and set(meta['key']).issubset(names),name
    assert all(c['dtype'] in ['utf8','bool','int64','uint64','float32','float64'] for c in meta['columns'])
assert {c['name'] for c in s['families']['pair_losses']['columns']}.isdisjoint({'variance_a','variance_b','covariance_a','covariance_b'})
assert {'variance_a','variance_b','covariance_a','covariance_b'}.issubset({c['name'] for c in s['families']['batch_objectives']['columns']})
assert all(sha(ROOT/name)==h for name,h in sources['local_sources'].items())
anchor=read(ROOT/'workstation/models/MODEL.json');assert p['anchor']['weights_sha256']==anchor['weights_sha256'] and p['anchor']['parameters']==anchor['parameters'] and p['anchor']['epoch']==20 and p['anchor']['seed']==3
history=pd.read_csv(ROOT/'data_gate_validation/nb07_publication/remote/csv/historical_class_support.csv').query("historical_split=='train'")
assert history.images.sum()==386 and history.human_geometry_images.sum()==70 and history.weak_geometry_images.sum()==152 and 386-70-152==164
assert b['projected_logical_bytes']<=p['resources']['logical_artifact_budget_bytes'] and b['projected_same_filesystem_peak_bytes']<=p['resources']['maximum_projected_peak_bytes']
assert p['resources']['maximum_projected_peak_bytes']+p['resources']['free_reserve_bytes']<=p['resources']['assumed_output_bytes']
assert b['sample_upper_bound']==23520+81*60+570*2 and b['texture_view_upper_bound']==2*b['sample_upper_bound']
assert b['native_exact_shape_bytes']['masked_predictions_and_targets_FP32']==b['texture_view_upper_bound']*8*2*4*16*16*4
# Prototype only the proposed packing to establish that the stated invariants are feasible.
roster=pd.read_csv(ROOT/'tread_two/readiness_report/training_roster.csv');ids=sorted(roster.image_id.tolist());lookup=roster.set_index('image_id').leakage_component.to_dict()
largest=roster.leakage_component.value_counts().index[0];a=[x for x in ids if lookup[x]==largest];c=[x for x in ids if lookup[x]!=largest];assert (len(a),len(c))==(249,137)
def pack(seed,epoch):
    rng=np.random.default_rng(seed*1000003+epoch);aa=list(rng.permutation(a));cc=list(rng.permutation(c))
    batches=[[(aa[i],False),(cc[i],False)] for i in range(49)]
    remaining=[(x,False) for x in [*aa[49:],*cc[49:]]];padding=list(rng.permutation(ids)[:6]);remaining += [(x,True) for x in padding];remaining=[remaining[i] for i in rng.permutation(len(remaining))]
    for bi,batch in enumerate(batches):
        while len(batch)<8:
            present={x for x,pad in batch};found=next((j for j,item in enumerate(remaining) if item[0] not in present),None)
            if found is not None:batch.append(remaining.pop(found));continue
            repaired=False
            for bj in range(bi):
                prior=batches[bj]
                for slot in range(2,len(prior)):
                    candidate=prior[slot]
                    if candidate[0] in present:continue
                    pick=next((j for j,item in enumerate(remaining) if item[0] not in {x for x,pad in prior}),None)
                    if pick is not None:prior[slot]=remaining.pop(pick);batch.append(candidate);repaired=True;break
                if repaired:break
            assert repaired,'Plan packing prototype needs a revised deterministic repair rule'
    assert not remaining
    flat=[item for batch in batches for item in batch];assert Counter(x for x,pad in flat if not pad)==Counter(ids) and sum(pad for x,pad in flat)==6
    assert all(len(set(x for x,pad in batch))==8 and len({lookup[x] for x,pad in batch})>=2 for batch in batches)
    return flat
for seed in range(1,9):
    for epoch in range(1,61):pack(seed,epoch)
assert pack(1,17)==pack(1,17)
# Numeric serialization planning check; this is not a training precision test.
values=np.random.default_rng(41).standard_normal(10000).astype(np.float32);recovered=np.array([float(format(float(x),'.9g')) for x in values],dtype=np.float32);assert np.array_equal(values.view(np.uint32),recovered.view(np.uint32))
rng=np.random.default_rng(13);truth=rng.integers(0,2,(100,16));valid=rng.integers(0,2,(100,16));pred=rng.integers(0,2,100)
foreground=(truth*valid).sum(1);available=valid.sum(1)
assert int((pred*foreground).sum())==int((pred[:,None]*truth*valid).sum())
assert int((pred*(available-foreground)).sum())==int((pred[:,None]*(1-truth)*valid).sum())
assert int(((1-pred)*foreground).sum())==int(((1-pred[:,None])*truth*valid).sum())
for folder,count in [('pilot',24),('lc_pilot',30),('context_pilot',32)]:
    base=ROOT/'tread_two'/folder;inventory=read(base/'FILES.json');assert len(inventory)==count and all(sha(base/n)==m['sha256'] for n,m in inventory.items())
for n in (10,11,12,13,14,15):
    done=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(done/'VERIFICATION.json')
    filename={10:'phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb',11:'phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb',12:'phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb',13:'phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb',14:'phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb',15:'phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb'}[n]
    assert sha(ROOT/'notebooks'/filename)==v['executed_notebook_sha256']
for item in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(item['path'])==item['sha256']
def protected(base,items):assert all(sha(base/name)==h for name,h in items.items());return len(items)
preservation=dict(original_workstation=protected(ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),phase3_workstation=protected(ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),paired_runtime=protected(ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources']))
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/item['path'])==item['sha256'] for item in baseline)
docs=[*ROOT.glob('*.md'),*(ROOT/'tread_two').glob('*.md'),*PLAN.glob('*.md')]
links=0
for doc in docs:
    text=doc.read_text(encoding='utf-8');assert '\ufffd' not in text
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=target.strip('<>')
        if target.startswith(('https://','http://','mailto:','#')):continue
        assert (doc.parent/unquote(target.split('#')[0])).resolve().exists(),(doc,target);links+=1
notebook=ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb';assert not notebook.exists(),'Do not claim this design-only check covers a delivered notebook'
report=dict(status='passed',scope='planning/source/shape arithmetic/schedule feasibility only; no model implementation acceptance',model_forward_calls=0,optimizer_updates=0,HF_writes=0,new_independent_tyres=0,implementation_pending=True,actual_T4_and_storage_measurements_pending=True,
    CSV_families=40,declared_columns=443,planned_parameter_counts_not_measured=True,schedule_prototypes=480,schedule_replay_identical=True,FP32_decimal_roundtrip_fixture=10000,coarse_cell_TP_FP_FN_reconstruction=True,
    arithmetic_projected_logical_bytes=b['projected_logical_bytes'],arithmetic_projected_peak_bytes=b['projected_same_filesystem_peak_bytes'],actual_CSV_widths_unmeasured=True,
    original_Markdown_preserved=len(baseline),frozen_source_counts=[24,30,32],preservation_counts=preservation,executed_notebooks_preserved=True,Markdown_documents_checked=len(docs),local_links_checked=links,broken_links=[],
    plan_artifact_sha256={path.name:sha(path) for path in PLAN.iterdir() if path.is_file()},master_plan_sha256=sha(ROOT/'phase3TreadTwoGeometryPlan.md'))
(OUT/'PLAN_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='plan_artifact_sha256'},indent=2))
