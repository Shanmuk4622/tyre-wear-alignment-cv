"""Seal local review artifacts and audit Git visibility without changing Git state."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import fitz
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):
    return subprocess.run(['git','-C',str(REPO),*args],capture_output=True,text=True,check=False)
validation=json.loads((ROOT/'audit/VALIDATION.json').read_text())
assert validation['status']=='passed'
pdfsha=digest(ROOT/'main.pdf')
assert validation['main_pdf_sha256']==pdfsha
visual=(ROOT/'audit/VISUAL_REVIEW.md').read_text(encoding='utf-8')
assert pdfsha in visual and 'completed' in visual
prefix=ROOT.relative_to(REPO).as_posix()
visible=git('ls-files','-co','--exclude-standard','--',prefix)
assert visible.returncode==0,visible.stderr
names=sorted(set(visible.stdout.splitlines()))
expected=['main.pdf','main.tex','references.bib','figures/qualitative.pdf',
          'figures/qualitative.png','evidence/qualitative/low_atlas.png','ieeeaccess.cls','t1-times.pfb',
          'figures/spatial_budget.pdf','figures/spatial_budget.png','evidence/spatial_budget_summary.csv',
          'generated/spatial_budget_rows.tex','figures/spatial_budget_differences.pdf',
          'a1.jpg','a2.jpg','logo.png','notaglinelogo.png','bullet.png','REVIEW_RESPONSE.md',
          'figures/architecture.pdf','figures/architecture.svg','figures/architecture.png']
for name in expected:
    assert git('check-ignore','-q','--no-index',prefix+'/'+name).returncode==1,name
scratch=['main.aux','main.bbl','main.blg','main.log','main.out','main.xdv','preview/page_01.png','.build/cache/check.tmp',
         '.build/spatial_budget/IMAGE_RESULTS.csv','.build/spatial_budget/BASIS_VECTORS.csv',
         '.build/reviewer/AFFINE_IMAGE_LOSSES.csv','.build/revision_2_reference.pdf']
for name in scratch:
    assert git('check-ignore','-q','--no-index',prefix+'/'+name).returncode==0,name
files=[]
for name in names:
    p=REPO/name
    assert p.stat().st_size<5_000_000,(name,p.stat().st_size)
    assert not any(x in p.relative_to(ROOT).parts for x in ['.build','preview'])
    assert p.suffix.lower() not in ['.pt','.pth','.npy','.npz','.zip','.exe','.dll']
    files.append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
fonts={}
with fitz.open(ROOT/'main.pdf') as doc:
    for page in doc:
        for font in page.get_fonts():
            xref=font[0]
            if xref not in fonts:
                base,ext,kind,data=doc.extract_font(xref)
                fonts[xref]={'name':base,'format':ext,'type':kind,'embedded_bytes':len(data)}
assert all(f['embedded_bytes']>0 for f in fonts.values()),fonts
receipt={'status':'passed','Git_state_mutated':False,'paper_visible_files':len(files),
    'paper_visible_bytes':sum(x['bytes'] for x in files),'max_file_bytes':max(x['bytes'] for x in files),
    'curated_outputs_visible':expected,'scratch_ignored':scratch,'files':files,
    'PDF_fonts_embedded':list(fonts.values()),'PDF_SHA256':pdfsha}
(ROOT/'audit/DELIVERY_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
manifest={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT).as_posix()
    if rel=='audit/DELIVERY_MANIFEST.json':continue
    if any(x in p.relative_to(ROOT).parts for x in ['.build','preview','__pycache__']):continue
    if p.suffix in ['.aux','.bbl','.blg','.log','.out','.xdv','.bcf']:continue
    manifest[rel]={'bytes':p.stat().st_size,'sha256':digest(p)}
(ROOT/'audit/DELIVERY_MANIFEST.json').write_text(json.dumps({'status':'sealed_local_author_review','files':manifest},indent=2)+'\n')
print(json.dumps({'status':'passed','pages':validation['pages'],'sealed_files':len(manifest),
    'all_fonts_embedded':len(fonts),'curated_outputs_visible':True,'scratch_ignored':True,'PDF_SHA256':pdfsha}))
