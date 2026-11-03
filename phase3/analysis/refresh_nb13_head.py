"""Bounded final public head check; no credential cache or remote mutation."""
import datetime,json,sys,time
from pathlib import Path
import requests
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];out=Path(json.loads((ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json').read_text())['path'])
prior=json.loads((out/'PUBLIC_STATUS.json').read_text());fit=next(r['prefix'] for r in prior['runs'] if '/fit/' in r['prefix'])
s=requests.Session();s.trust_env=False
def get(url):
    r=s.get(url,timeout=(20,60),headers={'Cache-Control':'no-cache'});r.raise_for_status();return r.json()
base='https://huggingface.co';repo='Shanmuk4622/tyre-wear-study'
commit=get(f'{base}/api/datasets/{repo}/commits/main?limit=1&nb13_check={int(time.time())}')[0]
manifest=get(f'{base}/datasets/{repo}/resolve/{commit["id"]}/{fit}/MANIFEST.json')
previous=json.loads((out/'fit_36f473b0886d_MANIFEST.json').read_text())
result=dict(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_revision=commit['id'],latest_commit_time=commit['date'],
    same_repository_head=commit['id']==prior['revision'],same_fit_manifest=manifest==previous,remote_writes=0,credentials_read=False)
(out/'LATEST_HEAD_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
assert result['same_fit_manifest'],'Fit publication advanced; audit the new checkpoint before giving recovery advice.'
