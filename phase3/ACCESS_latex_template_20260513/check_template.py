"""Read the current official template download in memory and compare core assets."""
import hashlib,io,json,zipfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
url='https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip'
response=requests.get(url,timeout=60)
response.raise_for_status()
z=zipfile.ZipFile(io.BytesIO(response.content))
receipt={'checked':'2026-11-03','official_url':url,'download_bytes':len(response.content),
    'download_sha256':hashlib.sha256(response.content).hexdigest(),'saved_zip':False,'core':{}}
for name in ['ieeeaccess.cls','IEEEtran.bst','spotcolor.sty']:
    members=[n for n in z.namelist() if Path(n).name==name and not n.startswith('__MACOSX/')]
    assert len(members)==1,(name,members)
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    official=hashlib.sha256(z.read(members[0])).hexdigest()
    assert actual==official,(name,actual,official)
    receipt['core'][name]={'official_member':members[0],'sha256':actual,'matches':True}
receipt['status']='passed'
(ROOT/'audit/CURRENT_TEMPLATE_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'passed','current_official_core_matches':len(receipt['core']),'saved_zip':False}))
