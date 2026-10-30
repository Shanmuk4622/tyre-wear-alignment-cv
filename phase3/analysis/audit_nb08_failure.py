"""Read public pinned HF failure evidence only; never looks up a credential."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tread_two/pilot_validation/nb08_repair/5bb076e30cd8'
REPO = 'Shanmuk4622/tyre-wear-study'
ENGINEERING = 'phase3/tread-two/atlas-v1/engineering/0bb5f3d6cd71255011a8b4ca841a6d487076336273a64ad0a482f231cf5407d5'
FIT = 'phase3/tread-two/atlas-v1/fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd'
session = requests.Session()
session.trust_env = False
calls = 0


def fetch(url):
    global calls
    calls += 1
    response = session.get(url, timeout=(20,120))
    response.raise_for_status()
    return response


info = fetch(f'https://huggingface.co/api/datasets/{REPO}').json()
revision = info['sha']
base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
manifest = fetch(base + ENGINEERING + '/MANIFEST.json').json()
(OUT / 'REMOTE_ENGINEERING_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
selected = {name:meta for name,meta in manifest['files'].items() if name in ('PREFLIGHT_PROOF.json','CONTRACT.json','CURSOR.json','PREPARATION.json') or name.endswith(('HTTP_REQUESTS.jsonl','REMOTE_HISTORY.jsonl'))}
summary = {}
for name, meta in selected.items():
    body = fetch(base + ENGINEERING + '/' + meta['object']).content
    assert len(body) == meta['bytes'] and hashlib.sha256(body).hexdigest() == meta['sha256']
    path = OUT / 'remote' / name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(body)
    if name.endswith('.jsonl'):
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]
        summary[name] = dict(rows=len(rows), statuses={str(code):sum(row.get('status')==code for row in rows) for code in {row.get('status') for row in rows}},
                            last_rate_headers=[dict(category=row.get('category'), rate=row.get('rate_limit'),policy=row.get('rate_policy')) for row in rows if row.get('rate_limit')][-3:])
    else:
        summary[name] = json.loads(body)
owner = fetch(base + FIT + '/OWNER.json').json()
response = session.get(base + FIT + '/MANIFEST.json',timeout=(20,120));calls += 1
assert response.status_code in (200,404)
fit_manifest = response.json() if response.status_code == 200 else None
proof = summary['PREFLIGHT_PROOF.json']
contract = summary['CONTRACT.json']
result = dict(status='public_pinned_evidence_read',revision=revision,engineering_destination=ENGINEERING,fit_destination=FIT,
              engineering_proof_sha256=selected['PREFLIGHT_PROOF.json']['sha256'],
              engineering_status=proof['status'],live_restore_reported=proof['actual_live_HF_restore'],
              devices=proof.get('devices'),environment=contract['environment'],
              measured_manifest_present=fit_manifest is not None,owner=owner,
              owner_expires_UTC=datetime.fromtimestamp(owner['expires'],timezone.utc).isoformat(),
              HTTP_history={name:value for name,value in summary.items() if name.endswith('.jsonl')},
              public_read_requests=calls,local_credentials_read=False,remote_writes=0)
(OUT / 'REMOTE_FAILURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
