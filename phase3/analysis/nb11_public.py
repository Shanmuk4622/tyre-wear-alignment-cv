"""Pinned, cached public HF reads. No local credentials, writes or SDK cache."""
import hashlib,json,threading,time,uuid
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb11_completion/LATEST.json').read_text())['path'])
REVISION='3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8'
PROTOCOL='ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28'
PREFIX='phase3/tread-two/atlas-lc-r2/fit/'+PROTOCOL
RANGE_BYTES=1024**2
BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{REVISION}/{PREFIX}/'
TLS=threading.local();LOCK=threading.Lock();CALLS=0
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def request(url,stream=False,headers=None):
    global CALLS
    if not hasattr(TLS,'session'):
        TLS.session=requests.Session();TLS.session.trust_env=False
    for attempt in range(8):
        with LOCK:CALLS+=1
        try:
            r=TLS.session.get(url,timeout=(20,150),stream=stream,headers=headers)
            if r.status_code==429 or r.status_code>=500:
                raw=r.headers.get('Retry-After','0')
                try:delay=float(raw)
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    delay=max(0,parsedate_to_datetime(raw).timestamp()-time.time())
                if r.status_code==429:
                    import re
                    reset=re.search(r'\bt=(\d+)',r.headers.get('RateLimit',''))
                    if reset:delay=max(delay,float(reset[1]))
                r.close()
                if attempt==7:raise RuntimeError('Public HF retry budget exhausted; cached verified files retained')
                time.sleep(max(delay,min(60,2**attempt)));continue
            r.raise_for_status();return r
        except (requests.ConnectionError,requests.Timeout):
            if attempt==7:raise
            time.sleep(min(30,2**attempt))
def manifest():
    value=request(BASE+'MANIFEST.json').json();owner=request(BASE+'OWNER.json').json()
    assert value['contract']==PROTOCOL and owner==dict(owner=None,expires=0)
    assert not any(n.endswith(('.zip','.npz','.sqlite','.db')) for n in value['files'])
    (OUT/'REMOTE_MANIFEST.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return value
def fetch(item):
    name,meta=item
    assert not Path(name).is_absolute() and '..' not in Path(name).parts and '..' not in Path(meta['object']).parts
    path=OUT/'remote'/name
    if path.exists() and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256']:return path
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.'+uuid.uuid4().hex+'.part')
    if meta['bytes']>RANGE_BYTES:
        # HF supports HTTP byte ranges. Bounded transfers avoid slow full-object
        # streams; the complete assembled bytes still require the pinned SHA.
        def range_body(start):
                end=min(meta['bytes']-1,start+RANGE_BYTES-1)
                for attempt in range(6):
                    try:
                        with request(BASE+meta['object']+f'?download=true&range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}'}) as r:
                            assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{meta["bytes"]}',name
                            body=r.content;assert len(body)==end-start+1
                        return body
                    except (requests.ConnectionError,requests.Timeout):
                        if attempt==5:raise
                        time.sleep(min(30,2**attempt))
        with part.open('wb') as f,ThreadPoolExecutor(max_workers=4) as ranges:
            for body in ranges.map(range_body,range(0,meta['bytes'],RANGE_BYTES)):f.write(body)
    else:
        with request(BASE+meta['object'],stream=True) as response,part.open('wb') as f:
            for block in response.iter_content(256*1024):f.write(block)
    assert part.stat().st_size==meta['bytes'] and sha(part)==meta['sha256'],name
    for attempt in range(12):
        try:part.replace(path);break
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(5,attempt+1))
    return path
