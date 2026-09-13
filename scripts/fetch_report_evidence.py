"""Fetch only the small verified reporting package; never datasets or weights."""
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'docs/report/evidence'
REV='22d5a6bc9f953ba3bf2a75919edc7db3193b317b'
PREFIX='s10/reporting-r1/2a333e2a6469905ad8cb822821ea46a357364e6d17bdd53c153c8b7e51fd217e'
REPO='Shanmuk4622/tyre-wear-study'

def main():
    token=os.environ.get('HF_TOKEN','')
    env=ROOT/'.env'
    if not token and env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            key,sep,value=line.partition('=')
            if sep and key.strip()=='HF_TOKEN':token=value.strip().strip('"\'');break
    session=requests.Session()
    if token:session.headers['Authorization']='Bearer '+token
    total=0
    def get(name,expected=None):
        nonlocal total
        path=(DEST/name).resolve();assert path.is_relative_to(DEST.resolve())
        if expected and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return path.read_bytes()
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{PREFIX}/{name}'
        with session.get(url,stream=True,timeout=90) as r:
            r.raise_for_status();parts=[];size=0
            for chunk in r.iter_content(65536):
                size+=len(chunk);total+=len(chunk)
                if size>8*2**20 or total>20*2**20:raise RuntimeError('Small-download safety budget exceeded')
                parts.append(chunk)
        raw=b''.join(parts)
        if expected:assert hashlib.sha256(raw).hexdigest()==expected,name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return raw
    es=json.loads(get('EVIDENCE_STATUS.json'));rs=json.loads(get('REPORT_STATUS.json'))
    manifest=json.loads(get('evidence_manifest.json',es['manifest_sha256']))
    for item in manifest:get(item['local'],item['sha256'])
    for name,sha in rs['artifact_sha256'].items():get(name,sha)
    record=dict(repo=REPO,revision=REV,prefix=PREFIX,downloaded_bytes=total,
                files=[str(p.relative_to(DEST)) for p in DEST.rglob('*') if p.is_file()])
    (DEST/'LOCAL_FETCH.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Verified small reporting package; downloaded {total/2**20:.2f} MiB. No model or dataset files.')

if __name__=='__main__':main()
