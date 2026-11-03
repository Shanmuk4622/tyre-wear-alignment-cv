"""Bounded manuscript QA and full-page rendering; does not re-run study analyses."""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
import fitz
from PIL import Image,ImageOps,ImageDraw
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
preserved=json.loads((ROOT/'audit/PRESERVATION_BASELINE.json').read_text())
for name,meta in preserved.items():
    p=REPO/name
    assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],name
checks['original_manuscript_and_reports_unchanged']=len(preserved)
inputs=json.loads((ROOT/'audit/EVIDENCE_FILES.json').read_text())
for meta in inputs:
    p=ROOT/'evidence'/meta['file']
    assert sha(p)==meta['sha256'] and sha(REPO/meta['source'])==meta['sha256'],p
checks['copied_scientific_inputs_byte_identical']=len(inputs)
for name,meta in json.loads((ROOT/'audit/TEMPLATE_FILES.json').read_text()).items():
    p=ROOT/name
    assert sha(p)==meta['sha256'],p
checks['template_assets_unchanged']=True
extra=json.loads((ROOT/'audit/ADDITIONAL_INPUTS.json').read_text())
for meta in extra['outputs']:
    assert sha(ROOT/meta['file'])==meta['sha256'],meta['file']
checks['additional_evidence_outputs_verified']=len(extra['outputs'])
epochs=pd.read_csv(ROOT/'evidence/trace_validation_epochs.csv')
assert list(epochs.epoch)==list(range(1,61)) and not epochs.eligible.any()
assert epochs.masked_utility.sum()==60 and epochs.local_utility.sum()==60
assert epochs.encoder_diversity.sum()==0 and epochs.projector_diversity.sum()==0
checks['all_60_original_epoch_gate_outcomes_preserved']=True
u=pd.read_csv(ROOT/'evidence/aggregation_utility_summary.csv')
t=pd.read_csv(ROOT/'generated/table_aggregation.csv')
for r in t.itertuples():
    for col,layer,role in [('local_val','encoded','validation'),('local_test','encoded','test'),('masked_val','masked_context','validation'),('masked_test','masked_context','test')]:
        s=u[(u.variant==r.variant)&(u.layer==layer)&(u.role==role)&(u.view==0)]
        assert len(s)==1
        rec=s.iloc[0]
        assert np.isclose(getattr(r,col),100*rec.retained_gain_fraction,rtol=1e-12)
        assert np.isclose(rec.retained_gain_fraction,(rec.position_control_loss-rec.loss)/(rec.position_control_loss-rec.native_original_loss),rtol=1e-12)
checks['all_36_aggregation_table_values_reconcile']=True
ledger=pd.read_csv(ROOT/'evidence/phase3_fits.csv')
assert ledger.fits.sum()==10
assert (ledger.fits*ledger.epochs_per_fit).sum()==600
assert (ledger.fits*ledger.updates_per_fit).sum()==29400
checks['fit_execution_ledger_reconciles']=True
science=json.loads((ROOT/'evidence/nb24_science.json').read_text())
assert science['selected'] is None and science['eligible_epochs']==0
assert f"{science['final_validation']['local_gain']*100:.2f}"=='36.44'
assert f"{science['final_validation']['masked_gain_visible_band_mean']*100:.2f}"=='21.63'
checks['headline_utility_and_unselected_state_verified']=True
source='\n'.join(p.read_text(encoding='utf-8') for p in [ROOT/'main.tex',*sorted((ROOT/'sections').glob('*.tex'))])
cited=set(k.strip() for m in re.findall(r'\\cite\{([^}]+)\}',source) for k in m.split(','))
defined=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'references.bib').read_text(encoding='utf-8')))
assert cited==defined,(cited-defined,defined-cited)
assert 'codex' in cited
checks['defined_and_cited_references']=len(cited)
checks['primary_research_references']=len(cited)-1
checks['disclosed_AI_system_references']=1
notes=sum(p.read_text(encoding='utf-8').count('\\AIdraftingnote') for p in (ROOT/'sections').glob('*.tex'))
assert notes==9
checks['section_specific_AI_citation_notes']=notes
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S).group(1)
abstractwords=len(re.findall(r"\b[\w'-]+\b",abstract))
assert abstractwords<=250 and '\\cite' not in abstract and '\\footnote' not in abstract
checks['abstract_words']=abstractwords
assert not re.search(r'\b(TODO|TBD|INSERT HERE|PLACEHOLDER)\b',source)
broken=[];local_links=0
for p in ROOT.rglob('*.md'):
    if any(x in p.relative_to(ROOT).parts for x in ['.build','preview']):continue
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if target.startswith(('http://','https://','#','mailto:')):continue
        target=target.split('#')[0].strip('<>')
        local_links+=1
        if not (p.parent/target).exists():broken.append({'file':str(p.relative_to(ROOT)),'target':target})
assert not broken,broken
checks['local_documentation_links_checked']=local_links
log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log,'Fix overfull boxes before delivery'
assert not re.search(r'(Citation .* undefined|Reference .* undefined|There were undefined)',log)
checks['no_overfull_boxes_or_unresolved_references']=True
doc=fitz.open(ROOT/'main.pdf')
assert len(doc)>=8 and len(doc)<=24
text='\n'.join(page.get_text() for page in doc)
assert '??' not in text
assert 'VOLUME 4, 2016' not in text and 'Digital Object Identifier' not in text
normalized_text=re.sub(r'\s+',' ',text).lower()
assert 'sreenivasa reddy edara' in normalized_text and 'shanmukesh bonala' in normalized_text
(ROOT/'preview').mkdir(exist_ok=True)
(ROOT/'preview/EXTRACTED_TEXT.txt').write_text(text,encoding='utf-8')
pages=[];thumbs=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
    p=ROOT/f'preview/page_{i+1:02d}.png';pix.save(p)
    chars=len(page.get_text().strip())
    assert chars>100,(i+1,chars)
    pages.append({'page':i+1,'characters':chars,'size_pt':list(page.rect),'image':p.name})
    img=Image.open(p).convert('RGB');img.thumbnail((306,396))
    tile=Image.new('RGB',(326,426),'#e5e7eb');tile.paste(img,((326-img.width)//2,8))
    ImageDraw.Draw(tile).text((10,405),f'Page {i+1}',fill='black')
    thumbs.append(tile)
for start in range(0,len(thumbs),6):
    subset=thumbs[start:start+6]
    sheet=Image.new('RGB',(326*3,426*((len(subset)+2)//3)),'white')
    for j,tile in enumerate(subset):sheet.paste(tile,((j%3)*326,(j//3)*426))
    sheet.save(ROOT/f'preview/contact_{start//6+1:02d}.png')
checks.update({'status':'passed','pages':len(doc),'page_details':pages,
    'main_pdf_bytes':(ROOT/'main.pdf').stat().st_size,'main_pdf_sha256':sha(ROOT/'main.pdf'),
    'figures':len(re.findall(r'\\begin\{figure\*?\}',source)),
    'tables':len(re.findall(r'\\begin\{table\*?\}',source)),
    'visual_review':'Latest rendered pages require human/agent visual inspection; see VISUAL_REVIEW.md',
    'new_training':False,'new_photos':False,'workstation_changes':False,'HF_writes':False})
(ROOT/'audit/VALIDATION.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:checks[k] for k in ['status','pages','abstract_words','figures','tables','defined_and_cited_references','main_pdf_sha256']}))
