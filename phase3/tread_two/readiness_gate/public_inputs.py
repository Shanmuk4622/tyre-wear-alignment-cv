"""Pinned public-only evidence reads; verified caches and bounded transfers."""
import hashlib,json,os,re,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):result.update(block)
    return result.hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
class Inputs:
    def __init__(self,spec,cache):
        self.spec=json.loads(Path(spec).read_text());self.cache=Path(cache);self.tls=threading.local();self.calls=0;self.lock=threading.Lock();self.used={}
    def request(self,url,byte_range=None):
        if not hasattr(self.tls,'session'):
            self.tls.session=requests.Session();self.tls.session.trust_env=False
        for attempt in range(8):
            with self.lock:self.calls+=1
            try:
                headers={'Range':f'bytes={byte_range[0]}-{byte_range[1]}'} if byte_range else None
                with self.tls.session.get(url,headers=headers,timeout=(20,150),stream=True) as response:
                    if response.status_code==429 or response.status_code>=500:
                        raw=response.headers.get('Retry-After','0')
                        try:delay=float(raw)
                        except ValueError:
                            from email.utils import parsedate_to_datetime
                            delay=max(0,parsedate_to_datetime(raw).timestamp()-time.time())
                        reset=re.search(r'\bt=(\d+)',response.headers.get('RateLimit',''))
                        delay=max(delay,float(reset[1]) if reset and response.status_code==429 else 0,min(60,2**attempt))
                        if attempt==7:response.raise_for_status()
                    else:
                        response.raise_for_status()
                        if byte_range:
                            start,end,total=byte_range
                            assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}','Invalid byte range'
                        return response.content
                deadline=time.monotonic()+delay
                print('Public HF read delayed; preserving verified cache.',flush=True)
                while time.monotonic()<deadline:time.sleep(min(10,max(0,deadline-time.monotonic())))
            except (requests.Timeout,requests.ConnectionError):
                if attempt==7:raise
                time.sleep(min(30,2**attempt))
        raise RuntimeError('Read retries exhausted; preserve cache and rerun')
    def file(self,key):
        meta=self.spec['inputs'][key];name=meta['name'];assert '..' not in Path(name).parts and not Path(name).is_absolute()
        target=self.cache/key;assert target.resolve().is_relative_to(self.cache.resolve())
        if not target.exists() or target.stat().st_size!=meta['bytes'] or sha(target)!=meta['sha256']:
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
            base=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/{meta["revision"]}/{meta["prefix"]}/{meta["object"]}'
            if meta['bytes']>1024**2:
                def block(start):
                    end=min(start+1024**2-1,meta['bytes']-1)
                    body=self.request(base+f'?download=true&range_start={start}&range_end={end}',(start,end,meta['bytes']));assert len(body)==end-start+1;return body
                with temp.open('wb') as f,ThreadPoolExecutor(max_workers=4) as pool:
                    for body in pool.map(block,range(0,meta['bytes'],1024**2)):f.write(body)
            else:temp.write_bytes(self.request(base))
            assert temp.stat().st_size==meta['bytes'] and sha(temp)==meta['sha256'],key
            os.replace(temp,target)
        self.used[key]=dict(bytes=meta['bytes'],sha256=meta['sha256'],revision=meta['revision'],name=name)
        return target
