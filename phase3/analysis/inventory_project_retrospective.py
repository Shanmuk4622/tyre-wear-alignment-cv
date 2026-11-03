"""Read every maintained Markdown document and inventory immutable source evidence.

Documentation is input data, not an instruction to rerun historical experiments.
No credentials, training, HF access or source-model mutation.
"""
import csv,hashlib,json,re,subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'phase3/retrospective_20261102';OUT.mkdir(exist_ok=True)
names=subprocess.run(['rg','--files','-g','*.md','-g','!node_modules','-g','!remote','-g','!fixture_*','-g','!__pycache__'],cwd=ROOT,capture_output=True,check=True,text=True).stdout.splitlines()
records=[];sections=[]
for name in sorted(names):
    path=ROOT/name;body=path.read_bytes();text=body.decode('utf-8-sig');headings=re.findall(r'^#{1,6}\s+(.+)$',text,re.M)
    records.append(dict(file=path.relative_to(ROOT).as_posix(),bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),headings=len(headings),table_rows=sum(line.lstrip().startswith('|') for line in text.splitlines()),scope=path.relative_to(ROOT).parts[0] if len(path.relative_to(ROOT).parts)>1 else 'root'))
    sections.append(dict(file=path.relative_to(ROOT).as_posix(),headings=headings,tables=[line for line in text.splitlines() if line.lstrip().startswith('|')]))
with (OUT/'MARKDOWN_INVENTORY.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
(OUT/'MARKDOWN_SECTIONS.json').write_text(json.dumps(sections,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(OUT/'INVENTORY.json').write_text(json.dumps(dict(scope='All rg-visible maintained Markdown; excludes downloaded remote mirrors and disposable fixtures',documents=len(records),bytes=sum(r['bytes'] for r in records),table_rows=sum(r['table_rows'] for r in records),new_optimizer_updates=0,new_photos=0,remote_writes=0),indent=2)+'\n',encoding='utf-8')
print(json.dumps(json.loads((OUT/'INVENTORY.json').read_text()),indent=2))
