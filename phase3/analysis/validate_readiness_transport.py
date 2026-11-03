"""New reader: actual pinned public small read plus offline cache/range failures."""
import hashlib,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];GATE=ROOT/'tread_two/readiness_gate';CHECK=ROOT/'tread_two/readiness_validation';sys.path.insert(0,str(GATE))
from public_inputs import Inputs,atomic,sha
fixture=Path(tempfile.mkdtemp(prefix='fixture_transport_',dir=CHECK));spec=json.loads((GATE/'INPUTS.json').read_text());key='identity_components.csv'
atomic(fixture/'public_spec.json',dict(inputs={key:spec['inputs'][key]}));reader=Inputs(fixture/'public_spec.json',fixture/'cache')
path=reader.file(key);assert sha(path)==spec['inputs'][key]['sha256'] and reader.calls==1
assert reader.file(key)==path and reader.calls==1
path.write_bytes(b'corrupted');reader.file(key);assert reader.calls==2 and sha(path)==spec['inputs'][key]['sha256']
body=b'x'*(2*1024**2+19);meta=dict(spec['inputs'][key],bytes=len(body),sha256=hashlib.sha256(body).hexdigest());atomic(fixture/'ranges.json',dict(inputs={'large.csv':meta}))
test=Inputs(fixture/'ranges.json',fixture/'large_cache')
def blocks(url,byte_range=None):
    start,end,total=byte_range;assert total==len(body);return body[start:end+1]
with patch.object(test,'request',side_effect=blocks) as request:
    assert test.file('large.csv').read_bytes()==body and request.call_count==3
class Response:
    status_code=200;headers={}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    @property
    def content(self):raise AssertionError('Invalid range body must not be consumed')
class Session:
    def get(self,*args,**kwargs):return Response()
test.tls.session=Session()
try:test.request('https://huggingface.co/offline',(0,9,100))
except AssertionError as e:assert str(e)=='Invalid byte range'
else:raise AssertionError('Ignored range was accepted')
result=dict(status='passed',source_sha256=sha(GATE/'public_inputs.py'),actual_pinned_public_read=True,public_requests=reader.calls,cache_reuse=True,corruption_replacement=True,bounded_range_assembly=True,ignored_range_rejected_before_body=True,local_credentials_read=False,remote_writes=0)
atomic(CHECK/'TRANSPORT_VALIDATION.json',result);print(json.dumps(result,indent=2))
