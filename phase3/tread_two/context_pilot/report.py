"""ATLAS-LC r2 curves and complete endpoint explorer; saved evidence only."""
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from common import HERE, atomic, file_meta, read


def rows(snapshot, family):
    for path in sorted((Path(snapshot) / 'csv').glob(f'epoch_*/{family}.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            yield from csv.DictReader(stream)


def image_data(image, size=(144, 192)):
    image = image.copy()
    image.thumbnail(size)
    stream = io.BytesIO()
    image.convert('RGB').save(stream, format='JPEG', quality=78)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


def build(snapshot, cfg):
    snapshot = Path(snapshot)
    output = snapshot / 'report'
    output.mkdir(exist_ok=True)
    metrics = [r for r in rows(snapshot, 'metrics') if r['level'] == 'cohort']
    epochs = [r for r in metrics if r['stage'] in ('train','validate') and r['metric'] in ('balanced_accuracy_observed','nll')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout='constrained')
    for stage, color in (('train','#248574'),('validate','#9460bd')):
        for axis, metric in zip(axes, ('balanced_accuracy_observed','nll')):
            values = sorted([r for r in epochs if r['stage'] == stage and r['metric'] == metric], key=lambda r: int(r['epoch']))
            if values:
                axis.plot([int(r['epoch']) for r in values], [float(r['value']) for r in values], marker='.', color=color, label=stage)
            axis.set(xlabel='Epoch', ylabel=metric.replace('_',' '))
            axis.grid(alpha=.2)
            if values:
                axis.legend()
    fig.suptitle('ATLAS-CONTEXT r3 · ' + cfg['contract']['mode'] + ' · training scores mix online update states')
    fig.savefig(output / 'training_curves.png', dpi=160)
    plt.close(fig)
    sources = {r['image_id']: r for r in rows(snapshot, 'source_images')}
    predictions = [r for r in rows(snapshot, 'predictions') if r['stage'] in ('selected_eval','final_eval')]
    details = {}
    reference_rows = []
    for path in sorted((snapshot / 'artifacts/endpoint_maps').glob('*/chunk_*.pt')):
        captures = torch.load(path, map_location='cpu', weights_only=False)
        for capture in captures:
            atlas_rgb = (capture['atlas']['rgb'].transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8)
            scores = capture['reference_evidence'].astype(float).tolist()
            details[str(capture['sample'])] = dict(atlas_picture=image_data(Image.fromarray(atlas_rgb), (96,192)), references=scores,
                base_logits=capture['base_logits'].astype(float).tolist(),context_delta=(capture['context_evidence'].sum(0)+capture['context_bias']).astype(float).tolist())
            for cls in range(3):
                for case in range(18):
                    reference_rows.append(dict(sample=capture['sample'], endpoint=path.parent.name, image_id=capture['image_id'],
                                               class_index=cls, prototype=case, contribution=scores[cls][case]))
    from common import csv_write
    csv_write(output / 'REFERENCE_CASES.csv', reference_rows, ['sample','endpoint','image_id','class_index','prototype','contribution'])
    patch = {}
    needed = {r['sample'] for r in predictions}
    for row in rows(snapshot, 'patch_evidence'):
        if row['sample'] in needed:
            patch.setdefault(row['sample'], [[0.] * 3 for _ in range(32)])[int(row['patch'])][int(row['class_index'])] = float(row['contribution'])
    thumbs = {}
    cases = []
    for row in predictions:
        identifier = row['image_id']
        if identifier not in thumbs:
            with Image.open(Path(cfg['data_root']) / sources[identifier]['image_path']) as picture:
                thumbs[identifier] = image_data(picture)
        cases.append(dict(row, picture=thumbs[identifier], patches=patch[row['sample']], **details[row['sample']]))
    bank = np.load(HERE / 'assets/bank_rgb.npy', allow_pickle=False)
    reference = [image_data(Image.fromarray((value.transpose(1,2,0).clip(0,1) * 255).round().astype(np.uint8)), (64,64)) for value in bank]
    curve = 'data:image/png;base64,' + base64.b64encode((output / 'training_curves.png').read_bytes()).decode()
    payload = json.dumps(dict(cases=cases, bank=reference, curve=curve, mode=cfg['contract']['mode'])).replace('</','<\\/')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATLAS evidence explorer</title>
<style>body{font:16px system-ui;background:#0c1b26;color:#e0eaf0;max-width:1200px;margin:auto;padding:25px}p{line-height:1.6;color:#b4cbd7}select,input,button{font:inherit;background:#213c4e;color:inherit;padding:9px;border:1px solid #597180;border-radius:7px;margin:6px}.case{display:grid;grid-template-columns:150px 1fr 200px;gap:20px;padding:18px;border:1px solid #29495c;border-radius:12px;margin:12px 0}.case img{max-width:144px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:2px}.grid div{height:20px}.bank img{width:64px;margin:4px}.warning{border-left:4px solid #f9bd6c;padding:12px;background:#283039}@media(max-width:700px){.case{grid-template-columns:1fr}.grid{max-width:200px}}</style>
<h1>TREAD-TWO / ATLAS-CONTEXT r3 — retained evidence</h1><p class="warning" id="scope"></p><img id="curves" style="max-width:100%" alt="Training and validation curves"><p>Every selected/final endpoint image is available below. Signed patch tiles explain logit contributions; exact source pixels, affinities, coefficients, native projections and numerical residuals are retained separately. Mileage proxies are not physical wear/depth/safety labels.</p>
<select id="endpoint"><option value="all">Both endpoints</option><option>selected</option><option>final</option></select><select id="errors"><option value="all">All predictions</option><option value="wrong">Wrong predictions</option><option value="empty">Unsupported atlas</option></select><input id="search" placeholder="Find image ID" aria-label="Find image ID"><p id="count"></p><div id="cases"></div><h2>Actual fixed reference patches</h2><p>Six per class; independent source tyres remain 1 / 3 / 4. Literal cases may contain paint, glare or identity shortcuts.</p><div id="bank" class="bank"></div>
<script type="application/json" id="data">__DATA__</script><script>'use strict';const d=JSON.parse(document.querySelector('#data').textContent),$=s=>document.querySelector(s);$('#scope').textContent=d.mode==='engineering_fixture'?'ENGINEERING FIXTURE ONLY. Validation examples, four optimizer updates; no measured development result.':'Existing-cohort development pilot. Historical low/high endpoint previously inspected; no held-out mid targets or independent confirmation.';$('#curves').src=d.curve;for(let i=0;i<d.bank.length;i++){let im=document.createElement('img');im.src=d.bank[i];im.alt='Actual reference '+i;im.title='Case '+i;$('#bank').append(im)}function render(){let all=d.cases.filter(x=>($('#endpoint').value==='all'||x.endpoint===$('#endpoint').value)&&($('#errors').value==='all'||($('#errors').value==='wrong'?x.correct==='0':x.empty_support==='1'))&&x.image_id.toLowerCase().includes($('#search').value.toLowerCase()));$('#count').textContent=all.length+' / '+d.cases.length+' endpoint records';$('#cases').replaceChildren();for(const x of all){let box=document.createElement('article');box.className='case';let im=document.createElement('img');im.src=x.picture;im.alt=x.image_id;let text=document.createElement('div');for(const s of [x.image_id,x.endpoint+' · checkpoint epoch '+x.epoch,'Proxy target '+x.label+' · prediction '+x.prediction+' · '+(x.correct==='1'?'correct':'wrong'),'Scores: '+[0,1,2].map(c=>Number(x['probability_'+c]).toFixed(4)).join(' / '),'Coverage: '+x.supported_patches+'/32 patches · clipping '+Number(x.clipped_fraction).toFixed(3),'Logit accounting residual '+Number(x.evidence_residual).toExponential(2),'Frozen LC logits: '+x.base_logits.map(v=>Number(v).toFixed(4)).join(' / '),'Context correction: '+x.context_delta.map(v=>Number(v).toFixed(4)).join(' / '),'Reference cards describe the frozen case branch; spatial tiles include its context correction.']){let p=document.createElement('p');p.textContent=s;text.append(p)}let grid=document.createElement('div');grid.className='grid';let cls=Number(x.prediction),scale=Math.max(...x.patches.map(v=>Math.abs(v[cls])),1e-9);for(let i=0;i<32;i++){let v=x.patches[i][cls],p=document.createElement('div');p.style.background=v>=0?'rgba(80,225,174,'+Math.abs(v)/scale+')':'rgba(244,112,111,'+Math.abs(v)/scale+')';p.title='Patch '+i+' class '+cls+' contribution '+v;grid.append(p)}box.append(im,text,grid);$('#cases').append(box)}}for(const s of ['#endpoint','#errors','#search'])$(s).addEventListener('input',render);render();</script></html>'''
    # Add literal positive/opposing case panels without hiding the complete
    # 3x18 aggregate in the direct derived CSV.
    html = html.replace("box.append(im,text,grid);", """let pictures=document.createElement('div');pictures.append(im);let atlas=document.createElement('img');atlas.src=x.atlas_picture;atlas.alt='Predicted visible tread atlas';pictures.append(atlas);let references=document.createElement('div');let heading=document.createElement('p');heading.textContent='Frozen case-branch support / opposition (predicted class)';references.append(heading);const ranked=x.references[cls].map((v,i)=>({v,i}));const order=[...ranked.filter(z=>z.v>0).sort((a,b)=>b.v-a.v).slice(0,3),...ranked.filter(z=>z.v<0).sort((a,b)=>a.v-b.v).slice(0,3)];for(const z of order){let item=document.createElement('span'),ref=document.createElement('img');ref.src=d.bank[z.i];ref.style.width='40px';ref.title='Actual reference '+z.i+' contribution '+z.v;let label=document.createElement('small');label.textContent='case '+z.i+' '+(z.v>=0?'+':'')+z.v.toFixed(4)+' ';item.append(ref,label);references.append(item)}text.append(references);box.append(pictures,text,grid);""")
    (output / 'ATLAS_Evidence_Explorer.html').write_text(html.replace('__DATA__', payload), encoding='utf-8')
    atomic(output / 'REPORT.json', dict(mode=cfg['contract']['mode'], endpoint_images=len(cases),
        reference_cards_scope='Frozen LC branch only; context contribution reported separately',
        derived_from_saved_CSV=True, model_forward_calls=0, trained_safety_claim=False, local_interactive_browser_tested=False))
    manifest = read(snapshot / 'MANIFEST.json')
    manifest['files'] = {p.relative_to(snapshot).as_posix(): file_meta(p) for p in snapshot.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
    atomic(snapshot / 'MANIFEST.json', manifest)
