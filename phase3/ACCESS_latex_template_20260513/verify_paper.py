"""Bounded manuscript QA and full-page rendering; does not re-run study analyses."""
import hashlib,json,re,sys
from pathlib import Path
from xml.etree import ElementTree
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
budget=json.loads((ROOT/'audit/SPATIAL_BUDGET.json').read_text())
assert budget['protocol_sha256']==sha(ROOT/'evidence/spatial_budget_protocol.json')
assert budget['script_sha256']==sha(ROOT/'run_spatial_budget.py')
for name,expected in budget['outputs'].items():
    path=ROOT/name
    if not name.startswith('.build/') or path.exists():
        assert sha(path)==expected,name
summary=pd.read_csv(ROOT/'evidence/spatial_budget_summary.csv')
assert len(summary)==budget['summary_rows']
assert np.allclose(summary.delta_loss,summary.output_space_delta+summary.order_delta,atol=1e-14)
assert np.allclose(summary.retained,(summary.position_loss-summary.loss)/
                   (summary.position_loss-summary.native_loss),atol=1e-14)
for role,expected in [('validation','89.47'),('test','89.01')]:
    row=summary[(summary.role==role)&(summary.branch=='local')&(summary.view==0)
                &(summary.fit=='main')&(summary.method=='token_PCA')&(summary.k==5)]
    assert f'{100*row.retained.item():.2f}'==expected
detail=ROOT/'.build/spatial_budget/IMAGE_RESULTS.csv'
if detail.exists():
    records=pd.read_csv(detail)
    assert len(records)==budget['image_results_rows']
    assert ((records.loss-records.native_loss).abs()<=records.decoder_drift+1e-12).all()
    assert ((records.loss-records.output_first_loss).abs()<=records.order_gap+1e-12).all()
    keys=['branch','fit','method','k','role','view']
    values=['native_loss','position_loss','loss','decoder_drift','order_gap','feature_mse']
    reduced=records.groupby(keys)[values+['output_first_loss']].mean()
    reference=summary.set_index(keys).sort_index()
    assert reference.index.equals(reduced.index)
    assert np.allclose(reference[values],reduced[values],atol=1e-13,rtol=1e-12)
    assert np.allclose(reference.output_space_delta,reduced.output_first_loss-reduced.native_loss,atol=1e-13)
    assert np.allclose(reference.order_delta,reduced.loss-reduced.output_first_loss,atol=1e-13)
checks['matched_budget']={'summary_rows':len(summary),'image_rows_independently_reduced':len(records) if detail.exists() else 0,
                          'full_budget_prediction_difference':budget['max_full_budget_prediction_difference']}
architecture=json.loads((ROOT/'audit/TRACE_ARCHITECTURE.json').read_text())
assert sha(REPO/'phase3/tread_two/trace_batch/model.py')==architecture['source_sha256']
assert sum(architecture['parameters'].values())==71022
assert {k:len(architecture['calls'][k]) for k in ('encoder','metadata','context','decoder','projector')}=={
    'encoder':2,'metadata':1,'context':2,'decoder':2,'projector':1}
assert architecture['hidden_rgb_invariance_fixed_support']
review=json.loads((ROOT/'audit/REVIEWER_ANALYSIS.json').read_text())
assert review['status']=='passed' and review['checkpoint_epoch']==60
assert review['network_optimizer_updates']==review['image_or_geometry_forwards']==0
for name,expected in review['output_sha256'].items():
    path=ROOT/name
    if not name.startswith('.build/') or path.exists():
        assert sha(path)==expected,name
angles=pd.read_csv(ROOT/'evidence/review_subspace_angles.csv')
fitted=angles[(angles.left=='token_PCA') & angles.right.isin(['Jacobian_PCA','decoded_PCA'])]
assert len(fitted)==4 and fitted.residual_overlap.between(.977, .990).all()
affine=pd.read_csv(ROOT/'evidence/review_affine_readout.csv')
pca=affine[(affine.method=='token_PCA') & affine.role.isin(['validation','test'])]
for branch,role,expected in [('local','validation','90.21'),('local','test','89.91'),
                             ('masked','validation','18.07'),('masked','test','10.62')]:
    value=pca[(pca.branch==branch)&(pca.role==role)].retained.item()
    assert f'{100*value:.2f}'==expected
affine_detail=ROOT/'.build/reviewer/AFFINE_IMAGE_LOSSES.csv'
affine_image_rows=0
if affine_detail.exists():
    image_losses=pd.read_csv(affine_detail)
    affine_image_rows=len(image_losses)
    assert affine_image_rows==review['image_rows']==25080
    assert not image_losses.duplicated(['branch','method','role','image_id']).any()
    keys=['branch','method','role']
    values=['loss','native_loss','position_loss']
    groups=image_losses.groupby(keys)
    reduced=groups[values].mean()
    ref=affine.set_index(keys).sort_index()
    assert ref.index.equals(reduced.index)
    assert np.array_equal(ref.images,groups.size())
    assert np.allclose(ref[values],reduced[values],atol=1e-13,rtol=1e-12)
    assert np.allclose(ref.retained,(reduced.position_loss-reduced.loss)/
                       (reduced.position_loss-reduced.native_loss),atol=1e-13)
checks['reviewer_diagnostics']={'architecture_parameters':71022,'affine_fits':review['probe_fits'],
                                'affine_image_rows_independently_reduced':affine_image_rows}
asset_receipt=json.loads((ROOT/'audit/ASSET_BUILD.json').read_text(encoding='utf-8'))
figure_outputs={item['file'].replace('\\','/'):item['sha256'] for item in asset_receipt['outputs']}
for ext in ('pdf','png','svg'):
    name=f'figures/architecture.{ext}'
    assert sha(ROOT/name)==figure_outputs[name]
diagram_labels=('TRACE forward graph','Tile encoder E','Axial residual block')
with fitz.open(ROOT/'figures/architecture.pdf') as diagram:
    assert len(diagram)==1 and not diagram[0].get_images()
    diagram_text=re.sub(r'\s+',' ',diagram[0].get_text())
    assert all(label in diagram_text for label in diagram_labels)
svg=ElementTree.parse(ROOT/'figures/architecture.svg')
svg_text=' '.join(node.text or '' for node in svg.iter('{http://www.w3.org/2000/svg}text'))
assert all(label in svg_text for label in diagram_labels)
checks['figure_1_vector_and_editable_exports_verified']=True
source='\n'.join(p.read_text(encoding='utf-8') for p in [ROOT/'main.tex',*sorted((ROOT/'sections').glob('*.tex'))])
cited=set(k.strip() for m in re.findall(r'\\cite\{([^}]+)\}',source) for k in m.split(','))
defined=set(re.findall(r'@\w+\{([^,]+),',(ROOT/'references.bib').read_text(encoding='utf-8')))
assert cited==defined,(cited-defined,defined-cited)
assert 'codex' not in cited and len(cited)==23
checks['defined_and_cited_references']=len(cited)
assert source.count('https://openai.com/codex/')==1 and '\\AIdraftingnote' not in source
checks['linked_AI_disclosure_in_acknowledgment']=True
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
assert any(all(label in re.sub(r'\s+',' ',page.get_text()) for label in diagram_labels) for page in doc)
assert 'VOLUME 4, 2016' not in text and 'Digital Object Identifier' not in text
normalized_text=re.sub(r'\s+',' ',text).lower()
for value in ('edara sreenivasa reddy','shanmukesh bonala','sreenivasareddy.e@vitap.ac.in',
              'shanmukesh.23bce20070@vitapstudent.ac.in'):
    assert value in normalized_text,value
bios=(ROOT/'sections/biographies.tex').read_text(encoding='utf-8')
original_bios=(REPO/'ACCESS_latex_template_20260513/sections/biographies.tex').read_text(encoding='utf-8')
assert bios.strip()==original_bios.replace('sections/a1.jpg','a1.jpg').replace('sections/a2.jpg','a2.jpg').strip()
portrait_sizes={Image.open(ROOT/name).size for name in ('a1.jpg','a2.jpg')}
embedded_sizes={(im[2],im[3]) for page in doc for im in page.get_images(full=True)}
assert portrait_sizes<=embedded_sizes,(portrait_sizes,embedded_sizes)
checks['supplied_author_biographies_and_portraits']=True
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
