"""Set up an isolated manuscript and copied compact source evidence; no training."""
import hashlib,json,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
P=Path(__file__).resolve().parent;R=P.parent;OLD=R.parent/'ACCESS_latex_template_20260513'
for name in ('sections','figures','generated','evidence','audit','.build','preview'):(P/name).mkdir(exist_ok=True)
def meta(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf-8'))
vendor=['ieeeaccess.cls','IEEEtran.bst','engine_compat.tex','spotcolor.sty','logo.png','notaglinelogo.png','bullet.png']
vendor+=sorted(p.name for p in OLD.iterdir() if p.suffix in ('.map','.pfb','.tfm','.fd'))
for name in vendor:
    target=P/name
    if not target.exists():shutil.copyfile(OLD/name,target)
    assert meta(target)==meta(OLD/name)
(P/'audit/TEMPLATE_FILES.json').write_text(json.dumps({n:meta(P/n) for n in vendor},indent=2)+'\n',encoding='utf-8')
evidence={
 'phase3_fits.csv':R/'retrospective_20261102/PHASE3_FIT_LEDGER.csv',
 'classification_endpoints.csv':R/'retrospective_20261102/CLASSIFICATION_ENDPOINTS.csv',
 'phase2_selections.csv':R/'retrospective_20261102/PHASE2_SELECTIONS.csv',
 'atlas_endpoints.csv':R/'tread_two/research_release/endpoint_metrics.csv',
 'tread_one_summary.csv':R/'completion_20261028/csv/run_summary.csv',
 'paired_endpoints.csv':R/'revision2_validation/nb05_publication/analysis/endpoint_metrics.csv',
 'geo_validation.csv':R/'tread_two/nb17_report/validation_reconstruction.csv',
 'geo_findings.json':R/'tread_two/nb17_report/FINDINGS.json',
 'trace_roles.csv':R/'retrospective_20261102/TRACE_ROLE_UTILITY.csv',
 'trace_component_utility.csv':R/'retrospective_20261102/TRACE_COMPONENT_UTILITY.csv',
 'trace_feature_cohorts.csv':R/'tread_two/trace_completion_report/ENDPOINT_COHORTS.csv'}
for name in ['UTILITY_SUMMARY','COMPONENT_UTILITY','CHANNEL_UTILITY_SUMMARY','FEATURE_SPACES','SPATIAL_COVARIANCE','MASK_PLAN_ALIGNMENT','MASK_PLAN_SUMMARY','SUPPORT_SUMMARY','SPATIAL_ENERGY_SUMMARY','BRIDGE_VERIFICATION','IMAGE_ALIGNMENT']:
    evidence['aggregation_'+name.lower()+'.csv']=R/'texture_aggregation_20261102'/(name+'.csv')
for name in ['PROTOCOL','ANALYSIS','VALIDATION','MASK_ADDENDUM_PROTOCOL','INPUT_VERIFICATION']:
    evidence['aggregation_'+name.lower()+'.json']=R/'texture_aggregation_20261102'/(name+'.json')
nb24=Path(read(R/'tread_two/nb24_completion/LATEST.json')['path'])
evidence['nb24_verification.json']=nb24/'VERIFICATION.json';evidence['nb24_science.json']=nb24/'SCIENTIFIC_VERIFICATION.json'
receipts=[]
for name,source in evidence.items():
    target=P/'evidence'/name
    if not target.exists():shutil.copyfile(source,target)
    assert meta(target)==meta(source)
    receipts.append(dict(file=name,source=source.relative_to(R.parent).as_posix(),**meta(target)))
(P/'audit/EVIDENCE_FILES.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
baseline=P/'audit/PRESERVATION_BASELINE.json'
if not baseline.exists():
    files=[*OLD.glob('*.tex'),*OLD.glob('*.bib'),*OLD.glob('*.cls'),OLD/'main.pdf',*OLD.glob('sections/*.tex')]
    files+=[R/n for n in ['phase3NB24Completion.md','phase3ProjectRetrospective.md','phase3TRACEComponentReview.md','phase3TextureAggregationInvestigation.md','texture_aggregation_20261102/VALIDATION.json','retrospective_20261102/VALIDATION.json']]
    baseline.write_text(json.dumps({p.relative_to(R.parent).as_posix():meta(p) for p in files},indent=2)+'\n',encoding='utf-8')
ignore='''# LaTeX scratch/cache/previews; final PDF and curated figures remain visible.
.build/
preview/
*.aux
*.log
*.out
*.blg
*.fls
*.fdb_latexmk
*.synctex.gz
*.bbl
*.xdv
*.run.xml
*.bcf
!main.pdf
!figures/*.pdf
!figures/*.png
!evidence/qualitative/*.png
'''
(P/'.gitignore').write_text(ignore,encoding='utf-8')
print('Isolated paper directory ready;',len(vendor),'unchanged template files;',len(receipts),'copied scientific inputs.')
