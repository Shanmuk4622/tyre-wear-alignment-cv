"""NB24 public reader. No credentials, remote writes or notebook execution."""
import hashlib,json,sys,time
from pathlib import Path
import nb17_public as public
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch'
sys.path.insert(0,str(SRC))
from common import read,atomic,file_meta,sha
OUT=Path(read(ROOT/'tread_two/nb24_completion/LATEST.json')['path'])
META=read(OUT/'METADATA.json');REPO='Shanmuk4622/tyre-wear-study'
MANIFESTS={s:read(OUT/(s.upper()+'_MANIFEST.json')) for s in META['stages']}
def url(stage,name):
    s=META['stages'][stage];return f'https://huggingface.co/datasets/{REPO}/resolve/{s["revision"]}/{s["prefix"]}/{name}'
def fetch(stage,name):
    m=MANIFESTS[stage]['files'][name];expected={k:m[k] for k in ('bytes','sha256')}
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    path=OUT/'remote'/stage/name
    if path.exists() and file_meta(path)==expected:return path
    path.parent.mkdir(parents=True,exist_ok=True)
    # Identical files across prep/engineering/fit are reused by checksum.
    for other in META['stages']:
        candidate=OUT/'remote'/other/name
        if candidate!=path and candidate.exists() and file_meta(candidate)==expected:
            import shutil;shutil.copyfile(candidate,path);return path
    for attempt in range(5):
        try:
            body=public.request(url(stage,m['object'])).content;break
        except Exception as exc:
            code=getattr(getattr(exc,'response',None),'status_code',None)
            if code not in (408,499) or attempt==4:raise
            print('Retry transient public resolver response',code,stage,name,flush=True);time.sleep(min(20,2**attempt))
    assert len(body)==m['bytes'] and hashlib.sha256(body).hexdigest()==m['sha256'],(stage,name)
    part=path.with_name(path.name+'.part');part.write_bytes(body);part.replace(path)
    return path
