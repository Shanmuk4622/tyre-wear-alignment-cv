"""Pinned credential-free public NB17 reader; cached byte verification and safe retries."""
import hashlib,json,threading,time,re
from collections import deque
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb17_completion/LATEST.json').read_text())['path'])
REVISION='1113bf8a9dd69687b23570fe985ed3a979a5f0e1'
PROTOCOL='7d1bc8554813ae16ba3d73d5d4af2d8c05c977e9282d1bf324ff9dbef6e89bdb'
PREFIX='phase3/tread-two/geo-r1/fit/'+PROTOCOL
BASE=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{REVISION}/{PREFIX}/'
TLS=threading.local();LOCK=threading.Lock();CALLS=0
TIMES={'api':deque(),'resolver':deque()};BLOCKED={'api':0.,'resolver':0.};LIMITS={'api':400,'resolver':2400}
def reserve(bucket):
    while True:
        now=time.time()
        with LOCK:
            stamps=TIMES[bucket]
            while stamps and stamps[0]<=now-300:stamps.popleft()
            delay=max(0.,BLOCKED[bucket]-now,stamps[0]+300-now if len(stamps)>=LIMITS[bucket] else 0.)
            if delay<=0:stamps.append(now);return
        time.sleep(min(30,delay))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def request(url):
    global CALLS
    if not hasattr(TLS,'session'):
        TLS.session=requests.Session();TLS.session.trust_env=False
    for attempt in range(8):
        bucket='api' if '/api/' in url else 'resolver';reserve(bucket)
        with LOCK:CALLS+=1
        try:
            r=TLS.session.get(url,timeout=(20,150))
            for response in [*r.history,r]:
                policy=re.search(r'\bq=(\d+)\s*;\s*w=(\d+)',response.headers.get('RateLimit-Policy',''))
                remaining=re.search(r'\br=(-?\d+)\s*;\s*t=(\d+)',response.headers.get('RateLimit',''))
                with LOCK:
                    if policy:LIMITS[bucket]=min(400 if bucket=='api' else 2400,max(1,int(.8*int(policy[1]))))
                    if remaining and int(remaining[1])<=0:BLOCKED[bucket]=max(BLOCKED[bucket],time.time()+int(remaining[2]))
            if r.status_code==429 or r.status_code>=500:
                raw=r.headers.get('Retry-After','0') or '0'
                try:delay=float(raw)
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    delay=max(0.,parsedate_to_datetime(raw).timestamp()-time.time())
                reset=re.search(r'\bt=(\d+)',r.headers.get('RateLimit',''))
                delay=max(delay,float(reset[1]) if reset else 0.,min(60,2**attempt));r.close()
                with LOCK:BLOCKED[bucket]=max(BLOCKED[bucket],time.time()+delay)
                print('Public HF audit backoff',delay,'seconds; verified cache retained.',flush=True)
                for _ in range(int(delay//30)):time.sleep(30)
                time.sleep(delay%30);continue
            r.raise_for_status();return r
        except (requests.ConnectionError,requests.Timeout):
            if attempt==7:raise
            time.sleep(min(30,2**attempt))
    raise RuntimeError('Public read retries exhausted')
def json_get(name):return request(BASE+name).json()
def manifest():
    path=OUT/'REMOTE_MANIFEST.json'
    value=json_get('MANIFEST.json');assert value['contract']==PROTOCOL
    assert json_get('OWNER.json')==dict(owner=None,expires=0)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return value
def fetch(item):
    name,meta=item
    assert not Path(name).is_absolute() and '..' not in Path(name).parts and '..' not in Path(meta['object']).parts
    target=OUT/'remote'/name
    if target.exists() and target.stat().st_size==meta['bytes'] and sha(target)==meta['sha256']:return target
    target.parent.mkdir(parents=True,exist_ok=True)
    body=request(BASE+meta['object']).content
    assert len(body)==meta['bytes'] and hashlib.sha256(body).hexdigest()==meta['sha256'],name
    part=target.with_name(target.name+'.part');part.write_bytes(body);part.replace(target)
    return target
