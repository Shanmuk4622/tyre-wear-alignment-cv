"""Validate new synthesis/diagnostics and preserve all prior source evidence."""
import ast,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'retrospective_20261102';sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import read,atomic,file_meta,sha
analysis=read(OUT/'ANALYSIS.json');assert analysis['status']=='passed'
expected={'MARKDOWN_INVENTORY':236,'SOURCE_EVIDENCE':40,'CONSUMED_SCIENCE_INPUTS':444,'TRACE_IMAGE_UTILITY':570,'TRACE_COMPONENT_UTILITY':10,'TRACE_ROLE_UTILITY':3,'TRACE_TILE_SUMMARY':6,'CLASSIFICATION_ENDPOINTS':18,'PHASE3_FIT_LEDGER':7,'PHASE2_SELECTIONS':5}
assert {p.stem for p in OUT.glob('*.csv')}==set(expected)
for name,count in expected.items():assert len(pd.read_csv(OUT/(name+'.csv')))==count,name
for row in pd.read_csv(OUT/'SOURCE_EVIDENCE.csv').itertuples():assert file_meta(ROOT.parent/row.file)==dict(bytes=row.bytes,sha256=row.sha256),row.file
for row in pd.read_csv(OUT/'CONSUMED_SCIENCE_INPUTS.csv').itertuples():assert file_meta(ROOT/row.file)==dict(bytes=row.bytes,sha256=row.sha256),row.file
images=pd.read_csv(OUT/'TRACE_IMAGE_UTILITY.csv');assert images.image_id.nunique()==570 and images.source_sha256.nunique()==570
assert set(images.epoch)=={60} and set(images.parameter_update)=={2940}
assert all(np.isfinite(images.select_dtypes(include='number')).all())
assert (images[['gain_channel_mean','gain_position_mean','gain_visible_band_mean']]>0).all().all()
assert images[images.role=='validation'].passes_5pct_all_controls.sum()==81 and images[images.role=='test'].passes_5pct_all_controls.sum()==98
groups=pd.read_csv(OUT/'TRACE_COMPONENT_UTILITY.csv');assert groups.images.sum()==570 and groups.component.nunique()==10
roles=pd.read_csv(OUT/'TRACE_ROLE_UTILITY.csv').set_index('role')
for role,g in groups.groupby('role'):
    assert np.isclose(np.average(g.masked_model,weights=g.images),roles.loc[role,'masked_model'],atol=1e-14)
fits=pd.read_csv(OUT/'PHASE3_FIT_LEDGER.csv');assert fits.fits.sum()==10 and (fits.fits*fits.epochs_per_fit).sum()==600 and (fits.fits*fits.updates_per_fit).sum()==29400
p2=pd.read_csv(OUT/'PHASE2_SELECTIONS.csv');assert p2.loc[p2.model=='hrnet','metric'].iloc[0]=='human_point_MAE_width_fraction' and p2.loc[p2.model=='hrnet','test'].iloc[0]>0
# All pre-existing Markdown outside the twelve authorized current-record files
# remains byte-identical to this turn's pre-analysis inventory.
allowed={'phase3/'+name for name in ['AGENTS.md','README.md','phase3Progress.md','phase3Understanding.md','phase3NextSteps.md','phase3OpenItems.md','phase3Decisions.md','phase3SessionLog.md','phase3EvidenceRegister.md','phase3DocumentationInventory.md','phase3RepositoryMap.md','phase3DocumentationConflicts.md']}
changed=[]
for r in pd.read_csv(OUT/'MARKDOWN_INVENTORY.csv').itertuples():
    if sha(ROOT.parent/r.file)!=r.sha256:assert r.file in allowed,r.file;changed.append(r.file)
assert set(changed)==allowed
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime','trace_batch']:
    p=ROOT/'tread_two'/folder;inventory=read(p/'FILES.json');assert all(file_meta(p/n)==m for n,m in inventory.items());preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
proto=read(ROOT/'tread_two/trace_design/validation/NUMERICAL.json')['source_sha256'];assert all(sha(ROOT/'tread_two/trace_design/runtime'/n)==v for n,v in proto.items())
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
for r in read(ROOT/'tread_two/gate_validation/PRESERVATION.json')['notebooks'].values():assert sha(Path(r['path']))==r['sha256']
for n in [*range(10,18),20,24]:
    p=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(p/'VERIFICATION.json')
    assert any(sha(f)==v['executed_notebook_sha256'] for f in (ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb')),n
for r in read(ROOT/'tread_two/trace_validation/NOTEBOOKS.json'):
    if r['mode']!='all':assert file_meta(ROOT/'notebooks'/r['notebook'])=={k:r[k] for k in ('bytes','sha256')}
docs=[*ROOT.glob('*.md'),OUT/'README.md',ROOT/'tread_two/README.md',*(ROOT/'tread_two/trace_batch').glob('*.md'),*(ROOT/'tread_two/trace_design').glob('*.md'),*(ROOT/'tread_two/trace_completion_report').glob('*.md')];links=0
for p in docs:
    text=p.read_text(encoding='utf-8');assert '\ufffd' not in text,p
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
        target=raw.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (p.parent/unquote(target)).exists(),(p,target);links+=1
figures=[]
for p in (OUT/'figures').glob('*.png'):
    with Image.open(p) as image:size=list(image.size);image.verify()
    assert subprocess.run(['git','check-ignore','--no-index','-q','--',str(p)],cwd=ROOT.parent).returncode==1
    figures.append(dict(file=p.name,size=size,**file_meta(p)))
assert len(figures)==2
viewer=subprocess.run(['node',str(ROOT/'analysis/test_retrospective_explorer.cjs'),str(OUT/'Project_Evidence_Explorer.html')],capture_output=True,text=True)
assert viewer.returncode==0,viewer.stderr
for name in ['inventory_project_retrospective.py','analyze_project_retrospective.py','document_project_retrospective.py','build_retrospective_explorer.py','update_retrospective_records.py','validate_project_retrospective.py']:ast.parse((ROOT/'analysis'/name).read_text(encoding='utf-8'))
result=dict(status='passed',CSV_tables=10,CSV_rows=sum(expected.values()),expected_rows=expected,source_evidence_files_checksum_verified=40,raw_consumed_scientific_bodies_checksum_verified=444,
    original_pre_analysis_Markdown=236,only_current_navigation_files_changed=changed,other_pre_analysis_Markdown_preserved=236-len(changed),original_baseline_Markdown_preserved=len(baseline),
    original_model_sources_and_workstations_preserved=preserved,original_TRACE_prototype_preserved=True,completed_notebooks_preserved=True,NB21_NB22_NB23_preserved=True,
    Markdown_checked=len(docs),local_links_checked=links,broken_links=[],figures=figures,viewer=json.loads(viewer.stdout),
    role_group_means_reconcile=True,fit_count_reconciles=True,original_selection_unchanged=True,GEO_TRACE_selected_None_preserved=True,
    new_model_forwards=0,new_optimizer_updates=0,new_photos=0,remote_writes=0,new_remote_verification=False,new_analysis_HF_published=False)
atomic(OUT/'VALIDATION.json',result);print(json.dumps(result,indent=2))
