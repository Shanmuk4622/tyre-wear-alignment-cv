"""Execute the real NB14 setup twice with only Kaggle paths/SDK metadata stubbed."""
import hashlib,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];CHECK=ROOT/'tread_two/context_review_validation'
delivery=json.loads((CHECK/'NOTEBOOK.json').read_text());nb=nbformat.read(delivery['notebook'],as_version=4)
cell=next(c.source for c in nb.cells if c.cell_type=='code')
fixture=Path(tempfile.mkdtemp(prefix='fixture_setup_',dir=CHECK));work=fixture/'tread_two_nb14_context_review'
cell=cell.replace("Path('/kaggle/working')",f'Path({str(fixture)!r})').replace("Path('/kaggle/working/tread_two_nb14_context_review')",f'Path({str(work)!r})')
assert '/kaggle/' not in cell
context={}
with patch('importlib.metadata.version',return_value='0.36.2'):
    exec(compile(cell,'NB14_setup_cell','exec'),context)
    assert context['WORK']==work
    for n,s in delivery['sources'].items():assert hashlib.sha256((work/n).read_bytes()).hexdigest()==s
    exec(compile(cell,'NB14_setup_cell_rerun','exec'),context)
    target=work/'review/review_context.py';original=target.read_bytes();target.write_bytes(original+b'\n# fixture corruption\n')
    try:exec(compile(cell,'NB14_setup_cell_mismatch','exec'),context)
    except AssertionError:pass
    else:raise AssertionError('Changed source must be rejected')
result=dict(status='passed',notebook_sha256=delivery['sha256'],actual_setup_cell_executed=True,empty_directory_extract=True,identical_rerun=True,changed_source_rejected=True,
    path_and_SDK_metadata_stubs_only=True,Kaggle_runtime_test=False,remote_writes=0,model_forwards=0,training_updates=0)
(CHECK/'SETUP_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
