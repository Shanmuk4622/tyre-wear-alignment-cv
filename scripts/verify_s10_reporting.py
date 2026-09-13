"""Full CPU dry-run against public pinned evidence, with every HF write disabled."""
import ast
import base64
import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s10_reporting as s

for name in ('NB19_S10_Evidence.ipynb','NB20_S10_Report.ipynb'):
    nb=json.loads((ROOT/'notebooks'/name).read_text())
    assert not nb['metadata']['kaggle']['isGpuEnabled']
    code=[]
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));code.append(''.join(c['source']));assert not c['outputs']
    assert base64.b64encode(Path(s.__file__).read_bytes()).decode() in '\n'.join(code)
if '--public' in sys.argv:
    root=Path(tempfile.mkdtemp(prefix='tyre_s10_qa_'))
    def download(item):
        local,remote=item
        r=requests.get(f'https://huggingface.co/datasets/{s.REPO}/resolve/{s.SOURCE}/{remote}',timeout=90);r.raise_for_status()
        return remote,r.content
    inputs=dict(concurrent.futures.ThreadPoolExecutor(8).map(download,s.sources().items()))
    class OfflineContext(s.Context):
        def __init__(self,sess):
            self.root=root;self.code=s.sha(Path(s.__file__).read_bytes());self.prefix='s10/reporting-r1/'+self.code
            self.token=False;self.sess=sess;self.last=__import__('time').monotonic()
        def read(self,path,rev=s.SOURCE):
            if path.startswith(self.prefix+'/'):return (root/path[len(self.prefix)+1:]).read_bytes()
            return inputs[path]
        def enqueue(self,paths):
            for p in paths:assert (root/p).is_file()
        def flush(self,reason):self.last=__import__('time').monotonic()
    with patch.object(s,'Context',OfflineContext),patch.object(s,'HfApi',return_value=NS(repo_info=lambda *a,**k:NS(sha='offline-evidence'))):
        s.evidence(NS());s.report(NS())
    status=json.loads((root/'REPORT_STATUS.json').read_text())
    assert status['figure_count']==14 and status['full_project_complete']==False
    for p,sha in status['artifact_sha256'].items():assert s.sha((root/p).read_bytes())==sha
    # A corrupt bundle must never be rendered as valid.
    path=root/'tables/s9_fusion_by_run.csv';saved=path.read_bytes();path.write_bytes(saved+b'corruption')
    try:s.validate(root)
    except AssertionError:pass
    else:raise AssertionError('Corrupt source passed validation')
    path.write_bytes(saved)
    print('QA_DIRECTORY',root)
    print('PASS complete NB19/NB20 dry-run, pinned public inputs,14 figures, report hashes, corrupt evidence rejection; ZERO HF WRITES')
print('PASS generated CPU notebook syntax and embedded source checks')
