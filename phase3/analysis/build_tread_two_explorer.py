"""Build a self-contained geometry/case illustration, not model results or an app."""
import base64
import csv
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
HERE = Path(__file__).resolve().parents[1] / 'tread_two'
REPORT = HERE / 'prototype_report'


def records(name):
    with (REPORT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def picture(name):
    return 'data:image/png;base64,' + base64.b64encode((REPORT / name).read_bytes()).decode('ascii')


payload = {
    'queries': [dict(row, original=picture(f'query_{i}_input.png'), atlas=picture(f'query_{i}_atlas.png'))
                for i, row in enumerate(records('query_manifest.csv'))],
    'bank': [dict(row, image=picture(f"bank_{int(row['prototype_index']):02d}.png"))
             for row in records('prototype_bank.csv')],
    'rows': records('atlas_rows.csv'),
}
html = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TREAD-TWO / ATLAS — geometry and real reference cases</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#09171f;color:#e7eff4}*{box-sizing:border-box}
body{max-width:1280px;margin:auto;padding:30px}h1{font-size:clamp(28px,4vw,48px);margin:10px 0}p{line-height:1.65;color:#b8cbd6}
.tag{color:#7ae8c2;letter-spacing:.15em;font-size:12px;font-weight:700}.notice{border-left:3px solid #ffc978;padding:10px 18px;background:#202324}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:22px 0}button,select{font:inherit;padding:9px 14px;background:#18313f;border:1px solid #385765;border-radius:8px;color:inherit;cursor:pointer}
button[aria-pressed=true]{background:#256551;border-color:#7ae8c2}main{display:grid;grid-template-columns:1.1fr 1fr;gap:20px}.panel{padding:20px;border:1px solid #28404e;background:#10232e;border-radius:15px}
.images{display:flex;gap:20px;justify-content:center;align-items:start}.photo{max-width:52%;height:380px;object-fit:contain}.map{position:relative;width:190px;height:380px;flex-shrink:0}.map img{width:100%;height:100%}.grid{display:grid;grid-template:repeat(8,1fr)/repeat(4,1fr);position:absolute;inset:0}.grid button{padding:0;border-radius:0;border:1px solid #a4ffe440;background:transparent;color:transparent}.grid button:focus,.grid button:hover,.grid button[aria-pressed=true]{background:#52ffc335;outline:2px solid #b4ffdf;z-index:1}
.bank{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.case{padding:10px;text-align:left;display:flex;align-items:center;gap:10px}.case img{width:64px;height:64px;image-rendering:auto}.case span{font-size:12px;overflow-wrap:anywhere}.detail{margin-top:18px;padding:15px;background:#091922;border-radius:10px;min-height:120px;font-size:14px;line-height:1.7;overflow-wrap:anywhere}.detail img{width:96px;float:right;margin:0 0 12px 12px}small{color:#9bb9c9}a{color:#7ae8c2}footer{margin:24px 0;color:#9bb9c9;font-size:13px}@media(max-width:1000px){main{grid-template-columns:1fr}.photo,.map{height:320px}.map{width:160px}}@media(max-width:600px){body{padding:15px}.bank{grid-template-columns:repeat(2,1fr)}.case{display:block}.photo,.map{height:260px}.map{width:130px}}
</style>
<div class="tag">PROJECT CONCEPT · EXISTING TRAINING IMAGES · 29 OCTOBER 2026</div>
<h1>TREAD-TWO / ATLAS</h1><p>Read a consistent visible-tread map. Compare real local cases. Keep the dense model intact.</p>
<p class="notice"><b>Untrained architecture illustration.</b> This page shows the existing frozen dense branch's geometry and actual reference pixels. It shows no TREAD-TWO accuracy, learned similarity ranking or physical wear measurement.</p>
<div class="toolbar"><label for="query">Training example </label><select id="query"><option value="0">Low proxy</option><option value="1">Mid proxy</option><option value="2">High proxy</option></select><button id="lines" aria-pressed="true">Patch grid</button><small>Click an atlas patch to inspect its source coordinates.</small></div>
<main><section class="panel"><h2>Photo → predicted 2D atlas</h2><div class="images"><img class="photo" id="original" alt="Existing training photo"><div class="map"><img id="atlas" alt="Predicted visible tread atlas"><div class="grid" id="grid"></div></div></div><div class="detail" id="mapping" aria-live="polite"></div><p><small>Atlas 512×256 · 32 patches · unsupported pixels remain invalid. This row map is not 3D unrolling.</small></p></section>
<section class="panel"><h2>18 literal reference patches</h2><p>Six patches per class, from eight source photos. Independent training tyres: <b>1 low / 3 mid / 4 high.</b></p><div class="toolbar" id="filters"></div><div class="bank" id="bank"></div><div class="detail" id="case" aria-live="polite">Select a real patch to inspect its source identity. Paint, glare and weak detail are visible risks; this bank is provisional pending a training-only review.</div></section></main>
<footer>15 CPU architecture checks · 81,274 new trainable parameters · zero optimizer updates · workstation changes deferred.<br><a href="README.md">Design overview</a> · <a href="DESIGN.md">Architecture and evidence mathematics</a> · <a href="EXPERIMENT_PLAN.md">One-candidate pilot</a></footer>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';const data=JSON.parse(document.querySelector('#payload').textContent);const $=s=>document.querySelector(s);let query=0,patch=0,filter='all';
function addText(parent,text){const p=document.createElement('div');p.textContent=text;parent.append(p)}
function mapping(){const q=data.queries[query],r=Math.floor(patch/4),c=patch%4;const rows=data.rows.filter(x=>x.image_id===q.image_id&&Number(x.atlas_row)>=r*64&&Number(x.atlas_row)<(r+1)*64);const valid=rows.filter(x=>x.source_y_input_pixels!=='');const box=$('#mapping');box.replaceChildren();addText(box,'Source: '+q.image_id+' · training only');addText(box,'Patch '+patch+' · atlas row '+r+', column '+c+' · 64×64 pixels');if(valid.length){const a=valid[0],z=valid[valid.length-1];addText(box,'Supported source-row span: '+Number(a.source_y_input_pixels).toFixed(2)+'–'+Number(z.source_y_input_pixels).toFixed(2)+' pixels in the resized input.');addText(box,'Full-band support fraction: '+(100*rows.reduce((sum,x)=>sum+Number(x.valid_fraction),0)/rows.length).toFixed(1)+'%. This is a band statistic, not patch quality.')}else addText(box,'No supported rows in this band.');document.querySelectorAll('#grid button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===patch)))}
function renderQuery(){const q=data.queries[query];$('#original').src=q.original;$('#atlas').src=q.atlas;mapping()}
for(let i=0;i<32;i++){const b=document.createElement('button');b.title='Inspect patch '+i;b.setAttribute('aria-label','Inspect atlas patch '+i);b.onclick=()=>{patch=i;mapping()};$('#grid').append(b)}
function showCase(item){const box=$('#case');box.replaceChildren();const img=document.createElement('img');img.src=item.image;img.alt='Reference patch '+item.prototype_index;box.append(img);addText(box,'Case '+item.prototype_index+' · '+item.proxy_class+' mileage proxy');addText(box,'Source image: '+item.source_image_id);addText(box,'Known training tyre: '+item.source_tyre_id);addText(box,'Atlas patch '+item.atlas_patch_index+' · supported fraction '+(100*Number(item.valid_fraction)).toFixed(1)+'%');addText(box,'Clipped source-row fraction: '+(100*Number(item.clipped_source_rows_fraction)).toFixed(1)+'%');addText(box,'Source SHA-256: '+item.source_image_sha256);addText(box,'Literal reference pixels; classifier untrained. No similarity ranking is shown.')}
function renderBank(){const box=$('#bank');box.replaceChildren();for(const item of data.bank.filter(x=>filter==='all'||x.proxy_class===filter)){const b=document.createElement('button');b.className='case';const img=document.createElement('img');img.src=item.image;img.alt='Actual '+item.proxy_class+' reference '+item.prototype_index;const text=document.createElement('span');text.textContent=item.proxy_class+' · case '+item.prototype_index;b.append(img,text);b.onclick=()=>showCase(item);box.append(b)}document.querySelectorAll('#filters button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===filter)))}
for(const value of ['all','low','mid','high']){const b=document.createElement('button');b.textContent=value;b.dataset.value=value;b.onclick=()=>{filter=value;renderBank()};$('#filters').append(b)}
$('#query').onchange=e=>{query=Number(e.target.value);renderQuery()};$('#lines').onclick=()=>{const show=$('#lines').getAttribute('aria-pressed')!=='true';$('#lines').setAttribute('aria-pressed',String(show));$('#grid').style.visibility=show?'visible':'hidden'};renderQuery();renderBank();
</script></html>'''
(HERE / 'ATLAS_EXPLORER.html').write_text(html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/')), encoding='utf-8')
print(json.dumps({'status': 'built', 'queries': 3, 'reference_cases': 18,
                  'bytes': (HERE / 'ATLAS_EXPLORER.html').stat().st_size,
                  'scope': 'Interactive geometry/provenance illustration; no trained classifier results'}))
