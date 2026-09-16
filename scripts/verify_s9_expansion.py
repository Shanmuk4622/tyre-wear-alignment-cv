"""Local contract tests; never publishes or fabricates human labels."""
import ast,csv,json,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_expansion as e
pilot=json.loads((ROOT/'PILOT_12_IMAGES/MANIFEST.json').read_text())
data=Path('D:/Dataset Download/Tire Dataset Prepared/FINAL')
rows=list(csv.DictReader((data/'manifests/clean_manifest.csv').open(encoding='utf-8-sig')))
selected=e.selection(rows,pilot)
assert len(selected)==120 and len({r['image_id'] for r in selected})==120
assert not {r['image_id'] for r in selected}&{r['image_id'] for r in pilot['images']}
assert e.selection(rows[::-1],pilot)==selected
dest=e.prepare(data,pilot,ROOT/'tyrelib/pilot_template.html',ROOT/'outputs/s9_expansion_validation')
manifest=json.loads((dest/'MANIFEST.json').read_text())
assert len(manifest['images'])==120
assert (dest/'ALL_120_IMAGES.zip').stat().st_size<e.MAX_PACKAGE_BYTES
html=(dest/'ANNOTATE.html').read_text(encoding='utf-8')
assert '__PILOT_DATA__' not in html and 'NB25' in html
fixture=dict(version=e.p.VERSION,package_id=manifest['package_id'],annotations=[])
assert e.validate(fixture,manifest)==[]
for r in manifest['images']:
    fixture['annotations'].append(dict(pilot_id=r['pilot_id'],image_sha256=r['image_sha256'],
        points={n:dict(state='uncertain',x=None,y=None) for n in e.p.POINTS},
        visual_review='unassessable',note='SOFTWARE TEST ONLY',independent_record='unknown'))
assert len(e.validate(fixture,manifest))==120 and all(r['complete'] for r in e.validate(fixture,manifest))
fixture['annotations'][0]['visual_review']='visible_issue';fixture['annotations'][0]['note']=''
assert not e.validate(fixture,manifest)[0]['complete']
fixture['annotations'][119]=fixture['annotations'][0]
try:e.validate(fixture,manifest);raise AssertionError('Duplicate accepted')
except ValueError:pass
assert 'Math.min(DATA.images.length-1,index+1)' in html
assert 'v.annotations.length>DATA.images.length' in html
assert "n+'/'+DATA.images.length+' reviewed'" in html
import zipfile
with zipfile.ZipFile(dest/'ALL_120_IMAGES.zip') as z:
    assert len([n for n in z.namelist() if n.startswith('images/')])==120
    for r in manifest['images']:assert e.p.digest(z.read('images/'+r['pilot_id']+'.jpg'))==r['image_sha256']
for name in ['NB24_S9_Geometry_Annotation_Batches.ipynb','NB25_S9_Geometry_Annotation_Intake.ipynb']:
    for c in json.loads((ROOT/'notebooks'/name).read_text())['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
print('PASS: deterministic 120-image selection, pilot exclusion, native package cap, visibility/partial validation, notebook syntax')
print('All 120 ZIP bytes:',(dest/'ALL_120_IMAGES.zip').stat().st_size)
