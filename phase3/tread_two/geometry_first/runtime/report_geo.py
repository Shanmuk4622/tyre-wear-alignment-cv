"""Engineering evidence gallery; labelled predictions, not measured tyre condition."""
import base64
import csv
import html
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import atomic, csv_write, read


def render(output):
    output = Path(output)
    proof = read(output/'ENGINEERING_PROOF.json')
    rows = []
    for path in sorted((output/'direct/units').glob('*/objectives.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            rows.extend(csv.DictReader(stream))
    images = []
    for unit in sorted((output/'direct/units').iterdir()):
        values = np.load(unit/'geometry_logits.npy', allow_pickle=False)
        with (unit/'samples.csv').open(encoding='utf-8', newline='') as stream:
            samples = list(csv.DictReader(stream))
        for slot, logits in enumerate(values):
            tile = np.zeros((128, 96, 3), dtype=np.uint8)
            tile[logits[0]>=0] = (72, 139, 235)
            tile[logits[1]>=0] = (248, 191, 60)
            image = Image.fromarray(tile).resize((192, 256), Image.Resampling.NEAREST)
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            images.append(dict(update=int(unit.name), image_id=samples[slot]['image_id'], component=samples[slot]['component'],
                        png=base64.b64encode(stream.getvalue()).decode()))
    payload = json.dumps(images, separators=(',', ':')).replace('<', '\\u003c')
    table = '<tr><th>Update</th><th>Geometry loss</th><th>Masked Huber</th><th>SSL images</th></tr>'
    for row in rows:
        table += '<tr>'+''.join('<td>'+html.escape(row[k])+'</td>' for k in ('update', 'geometry_geometry', 'texture_masked_huber', 'texture_eligible_images'))+'</tr>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TREAD-TWO GEO engineering passport</title><style>body{font:16px system-ui;background:#101722;color:#edf2fa;margin:2rem;max-width:1300px}h1{color:#ffc85a}table{border-collapse:collapse}td,th{border:1px solid #506078;padding:.5rem}select,button{font:inherit;padding:.5rem}#gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:1rem;margin-top:1rem}article{background:#223047;padding:1rem;overflow-wrap:anywhere}img{display:block;margin:auto;max-width:100%}.muted{color:#b2c2d6}</style>
<h1>TREAD-TWO / GEO · engineering passport</h1><p>Disposable train-only updates. Blue: predicted tyre; gold: predicted tread. These masks are engineering observations, not physical wear, depth, safety or measured-fit results.</p>
<p id="summary"></p><table>TABLE</table><h2>Interactive predicted-mask gallery</h2><label>Update <select id="filter"><option value="all">All</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label><p id="count" class="muted"></p><div id="gallery"></div>
<script>const cases=PAYLOAD;function draw(){const filter=document.getElementById('filter').value;const chosen=cases.filter(c=>filter==='all'||String(c.update)===filter);document.getElementById('count').textContent=chosen.length+' engineering cases';const g=document.getElementById('gallery');g.replaceChildren();for(const c of chosen){const a=document.createElement('article'),title=document.createElement('p'),img=document.createElement('img'),note=document.createElement('p');title.textContent='Update '+c.update+' · '+c.image_id;note.textContent=c.component;img.src='data:image/png;base64,'+c.png;img.alt='Predicted tyre and tread masks';a.append(title,img,note);g.append(a)}}document.getElementById('filter').addEventListener('change',draw);draw();</script></html>'''.replace('TABLE', table).replace('PAYLOAD', payload)
    (output/'Engineering_Passport.html').write_text(page, encoding='utf-8')
    atomic(output/'GALLERY.json', dict(cases=len(images), scope='engineering_train_only', interactive_filter='update'))
    return dict(interactive_cases=len(images), html='Engineering_Passport.html')
