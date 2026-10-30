import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
a=Path(json.loads((root/'data_gate_outputs/LATEST.json').read_text())['path'])
fresh=max((root/'data_gate_validation').glob('fresh_cpu_*'),key=lambda p:p.stat().st_mtime)
b=Path(json.loads((fresh/'data_gate_outputs/LATEST.json').read_text())['path'])
for p in (a/'csv').glob('*.csv'):
 q=b/'csv'/p.name
 if p.read_bytes()==q.read_bytes():continue
 with p.open(newline='',encoding='utf-8') as f, q.open(newline='',encoding='utf-8') as g:
  rr,ss=csv.DictReader(f),csv.DictReader(g);rows,other=list(rr),list(ss)
  print(p.name,'headers_equal',rr.fieldnames==ss.fieldnames,'records_equal',rows==other)
  if rows!=other:
   for r,s in zip(rows,other):
    if r!=s:
     print({k:(r[k],s[k]) for k in r if r[k]!=s.get(k)})
     break
