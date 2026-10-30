"""Pinned public NB08 follow-up audit; no credentials or remote mutations."""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
REPO = 'Shanmuk4622/tyre-wear-study'
PREFIX = 'phase3/tread-two/atlas-v1/'
FIT = PREFIX + 'fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd'
ENGINEERING = PREFIX + 'engineering/0bb5f3d6cd71255011a8b4ca841a6d487076336273a64ad0a482f231cf5407d5'
session = requests.Session()
session.trust_env = False
calls = 0


def get(url, missing=False):
    global calls
    calls += 1
    response = session.get(url, timeout=(20, 120))
    if missing and response.status_code == 404:
        return None
    response.raise_for_status()
    return response


def safe(value):
    text = json.dumps(value)
    text = re.sub(r'hf_[A-Za-z0-9_]+', '[HF_TOKEN_REDACTED]', text)
    text = re.sub(r'(?i)Bearer\s+[^\s\"\\]+', 'Bearer [REDACTED]', text)
    return json.loads(text)


info = get(f'https://huggingface.co/api/datasets/{REPO}').json()
revision = info['sha']
out = ROOT / 'tread_two/pilot_validation/nb08_followup' / revision[:12]
out.mkdir(parents=True, exist_ok=True)
base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
result = dict(revision=revision, checked_at_UTC=datetime.now(timezone.utc).isoformat(),
              notebook_sha256=hashlib.sha256((ROOT / 'notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb').read_bytes()).hexdigest(),
              local_credentials_read=False, remote_writes=0, destinations={})
for kind, prefix in [('fit', FIT), ('engineering', ENGINEERING)]:
    record = dict(prefix=prefix)
    manifest_response = get(base + prefix + '/MANIFEST.json', missing=True)
    owner_response = get(base + prefix + '/OWNER.json', missing=True)
    record['owner'] = owner_response.json() if owner_response else None
    if record['owner']:
        record['owner_expires_UTC'] = datetime.fromtimestamp(record['owner']['expires'], timezone.utc).isoformat()
    record['manifest_present'] = manifest_response is not None
    if manifest_response is not None:
        manifest = manifest_response.json()
        (out / (kind + '_MANIFEST.json')).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        record['manifest_summary'] = {key:value for key,value in manifest.items() if key != 'files'}
        record['files'] = len(manifest['files'])
        record['file_names'] = list(manifest['files'])
        selected = {name:meta for name,meta in manifest['files'].items()
                    if name in ('CURSOR.json', 'PREFLIGHT_PROOF.json', 'CONTRACT.json')
                    or name.endswith(('LIFECYCLE.jsonl','NUMERICAL_EVENTS.jsonl', 'EVENTS.jsonl', 'REMOTE_HISTORY.jsonl', 'HTTP_REQUESTS.jsonl'))}
        record['evidence'] = {}
        for name, meta in selected.items():
            response = get(base + prefix + '/' + meta['object'])
            body = response.content
            assert len(body) == meta['bytes'] and hashlib.sha256(body).hexdigest() == meta['sha256']
            path = out / kind / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            if name.endswith('.jsonl'):
                rows = [json.loads(line) for line in body.splitlines() if line.strip()]
                record['evidence'][name] = dict(rows=len(rows), tail=safe(rows[-12:]))
            elif name == 'CONTRACT.json':
                contract = json.loads(body)
                record['evidence'][name] = {key:contract[key] for key in ('environment', 'sources') if key in contract}
            else:
                record['evidence'][name] = safe(json.loads(body))
    result['destinations'][kind] = record
result['public_read_requests'] = calls
(out / 'PUBLIC_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
# Avoid printing long manifests and full source inventories.
summary = {**result, 'destinations': {kind:{key:value for key,value in record.items() if key not in ('file_names',)}
                                      for kind,record in result['destinations'].items()}}
print(json.dumps(summary, indent=2))
