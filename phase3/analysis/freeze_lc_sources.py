"""Freeze r2 only; execution and notebook packaging verify this inventory."""
import ast,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot'
files={}
for p in sorted(SOURCE.rglob('*')):
    if p.is_file() and (p.suffix in ('.py','.npy','.csv') or p.name in ('RECIPE.json','VENDORED.json','LC_RULES.json','UPSTREAM.json')):
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
        files[p.relative_to(SOURCE).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(SOURCE/'FILES.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
print('r2 frozen files:',len(files))
