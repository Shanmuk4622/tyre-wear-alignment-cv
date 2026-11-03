"""Read-only preservation, document-link and Git visibility checks for NB24."""
import ast,re,subprocess
from urllib.parse import unquote
from PIL import Image
from nb24_public import *
v=read(OUT/'VERIFICATION.json');assert v['status']=='passed'
assert sha(ROOT/'notebooks/phase3_NB24_TREAD_TWO_TRACE_All_In_One_Batch.ipynb')==v['executed_notebook_sha256']==sha(OUT/'executed.ipynb')
import nbformat
nbformat.validate(nbformat.read(ROOT/'notebooks/phase3_NB24_TREAD_TWO_TRACE_All_In_One_Batch.ipynb',as_version=4))
preserved={}
for folder in ['pilot','lc_pilot','context_pilot','geo_pilot','geo_postrun','geometry_first/runtime','trace_batch']:
    path=ROOT/'tread_two'/folder;inventory=read(path/'FILES.json');assert all(file_meta(path/n)==m for n,m in inventory.items()),folder;preserved[folder]=len(inventory)
for name,base,inventory in [('original_workstation',ROOT.parent,read(ROOT/'workstation_validation/ORIGINAL_BASELINE.json')),
                          ('phase3_workstation',ROOT,read(ROOT/'workstation_validation/DELIVERY.json')['source_hashes']),
                          ('paired_runtime',ROOT,read(ROOT/'revision2_validation/DELIVERY.json')['sources'])]:
    assert all(sha(base/n)==value for n,value in inventory.items());preserved[name]=len(inventory)
proto=read(ROOT/'tread_two/trace_design/validation/NUMERICAL.json')['source_sha256']
assert all(sha(ROOT/'tread_two/trace_design/runtime'/n)==value for n,value in proto.items())
baseline=read(ROOT/'audit/markdown-baseline.json');assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in baseline)
notebooks=[]
for n in [*range(10,18),20]:
    p=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);identity=read(p/'VERIFICATION.json')['executed_notebook_sha256']
    assert any(sha(f)==identity for f in (ROOT/'notebooks').glob(f'phase3_NB{n:02d}*.ipynb')),n;notebooks.append(n)
for item in read(ROOT/'tread_two/trace_validation/NOTEBOOKS.json'):
    if item['mode']=='all':continue
    assert file_meta(ROOT/'notebooks'/item['notebook'])=={k:item[k] for k in ('bytes','sha256')}
for name in ['inspect_nb24_completion.py','audit_nb24_metadata.py','nb24_public.py','verify_nb24_objects.py','audit_nb24_science.py','finalize_nb24_verification.py','fetch_nb24_passport.py','document_nb24_completion.py','check_nb24_handoff.py']:
    ast.parse((ROOT/'analysis'/name).read_text(encoding='utf-8'))
docs=[*ROOT.glob('*.md'),ROOT/'tread_two/README.md',*(ROOT/'tread_two/trace_batch').glob('*.md'),*(ROOT/'tread_two/trace_design').glob('*.md'),*(ROOT/'tread_two/trace_completion_report').glob('*.md')]
links=0
for path in docs:
    body=path.read_text(encoding='utf-8');assert '\ufffd' not in body,path
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)',body):
        target=raw.strip('<>').split('#',1)[0]
        if not target or re.match(r'^[a-zA-Z]+:',target):continue
        assert (path.parent/unquote(target)).exists(),(path,target);links+=1
figures=[]
for p in (ROOT/'tread_two/trace_completion_report/figures').glob('*.png'):
    with Image.open(p) as image:size=list(image.size);image.verify()
    figures.append(dict(file=p.name,size=size,**file_meta(p)))
assert len(figures)==2
assert file_meta(ROOT/'tread_two/trace_completion_report/figures/TRACE_Validation.png')==file_meta(OUT/'remote/review/TRACE_Validation.png')
assert file_meta(ROOT/'tread_two/trace_completion_report/MODEL_CARD.md')==file_meta(OUT/'remote/review/MODEL_CARD.md')
def ignored(p):return subprocess.run(['git','check-ignore','--no-index','-q','--',str(p)],cwd=ROOT.parent).returncode
for p in [OUT/'remote/fit/RESULT.json',OUT/'executed.ipynb']:assert ignored(p)==0,p
for p in [ROOT/'phase3NB24Completion.md',ROOT/'phase3TreadTwoTRACEFindings.md',ROOT/'tread_two/trace_batch/model.py',OUT/'VERIFICATION.json',ROOT/'tread_two/trace_completion_report/figures/TRACE_Completion_Insights.png']:
    assert ignored(p)==1,p
report=dict(status='passed',executed_NB24_preserved=True,NB21_NB22_NB23_clean_notebooks_preserved=True,prior_sources_preserved=preserved,prototype_modules_preserved=len(proto),
    prior_executed_notebooks_preserved=notebooks,original_root_Markdown_preserved=len(baseline),Markdown_files_checked=len(docs),local_links_checked=links,broken_links=[],
    figures=figures,heavy_remote_and_execution_archive_ignored=True,reports_and_source_visible=True,model_card_and_original_figure_byte_identical=True,
    remote_writes=0,local_credentials_read=False,new_optimizer_updates=0,new_photos=0)
atomic(OUT/'HANDOFF.json',report);print(report)
