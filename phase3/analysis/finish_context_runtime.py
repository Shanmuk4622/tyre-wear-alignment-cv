"""Complete r3 saved-report attribution and freeze its isolated source package."""
import ast,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
def change(s,a,b):assert a in s,a[:100];return s.replace(a,b)
p=SOURCE/'report.py';s=p.read_text(encoding='utf-8')
s=change(s,"details[str(capture['sample'])] = dict(atlas_picture=image_data(Image.fromarray(atlas_rgb), (96,192)), references=scores)","details[str(capture['sample'])] = dict(atlas_picture=image_data(Image.fromarray(atlas_rgb), (96,192)), references=scores,\n                base_logits=capture['base_logits'].astype(float).tolist(),context_delta=(capture['context_evidence'].sum(0)+capture['context_bias']).astype(float).tolist())")
s=change(s,"'Logit accounting residual '+Number(x.evidence_residual).toExponential(2)","'Logit accounting residual '+Number(x.evidence_residual).toExponential(2),'Frozen LC logits: '+x.base_logits.map(v=>Number(v).toFixed(4)).join(' / '),'Context correction: '+x.context_delta.map(v=>Number(v).toFixed(4)).join(' / '),'Reference cards describe the frozen case branch; spatial tiles include its context correction.'")
s=s.replace('Supporting / opposing actual cases (predicted class)','Frozen case-branch support / opposition (predicted class)')
s=s.replace('TREAD-TWO / ATLAS — retained evidence','TREAD-TWO / ATLAS-CONTEXT r3 — retained evidence')
s=s.replace("fig.suptitle('ATLAS · '","fig.suptitle('ATLAS-CONTEXT r3 · '")
s=change(s,"atomic(output / 'REPORT.json', dict(mode=cfg['contract']['mode'], endpoint_images=len(cases),", "atomic(output / 'REPORT.json', dict(mode=cfg['contract']['mode'], endpoint_images=len(cases),\n        reference_cards_scope='Frozen LC branch only; context contribution reported separately',")
p.write_text(s,encoding='utf-8')
# Guard the category-aware native projection against accidental zero estimates.
p=SOURCE/'acceptance.py';s=p.read_text(encoding='utf-8')
s=change(s,'    projected_native=categories[',"    assert all(value>0 for value in categories.values()),'Missing measured native category'\n    projected_native=categories[")
s=change(s,'projected_native_bytes=int(projected_native),','projected_native_bytes=int(projected_native),native_fixture_bytes_by_category=categories,')
p.write_text(s,encoding='utf-8')
files={}
for p in sorted(SOURCE.rglob('*')):
    if p.is_file() and (p.suffix in ('.py','.npy','.csv') or p.name in ('RECIPE.json','VENDORED.json','LC_RULES.json','UPSTREAM.json','BASE_INPUT.json')):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        files[p.relative_to(SOURCE).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(SOURCE/'FILES.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
print('Context source/asset files frozen:',len(files))
