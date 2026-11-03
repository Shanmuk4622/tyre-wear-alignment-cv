"""One bounded public range probe for a slow scientific input; no credentials."""
import json,sys,time
from pathlib import Path
import requests
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
m=json.loads((done/'REMOTE_MANIFEST.json').read_text());name='csv/epoch_0060/affinities.csv';meta=m['files'][name]
url='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e/phase3/tread-two/atlas-context-r3/fit/'+m['contract']+'/'+meta['object']
s=requests.Session();s.trust_env=False;begin=time.monotonic()
with s.get(url+f'?download=true&probe={int(time.time())}',headers={'Range':'bytes=0-65535'},timeout=(10,30)) as r:
    print(json.dumps(dict(status=r.status_code,content_range=r.headers.get('Content-Range'),bytes=len(r.content),seconds=time.monotonic()-begin)),flush=True)
