"""Run the delivered saved-data review on the byte-verified local HF cache."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
if '--audit-in-progress' not in sys.argv:
    assert json.loads((done/'VERIFICATION.json').read_text())['status']=='passed'
else:
    early=json.loads((done/'CHECKPOINT_EARLY.json').read_text());assert early['phase']=='completed' and early['owner_released']
sys.path.insert(0,str(ROOT/'tread_two/context_review'))
from review_context import Inputs,audit
from render_context import render
inputs=Inputs(done/'remote',done/'REMOTE_MANIFEST.json')
out=ROOT/'tread_two/context_analysis_report';result=audit(inputs,out);render(out,inputs)
if '--audit-in-progress' not in sys.argv:assert inputs.reads==0,'Verified full cache must satisfy all science inputs without network'
print(json.dumps({k:v for k,v in result.items() if k!='csv_tables'},indent=2))
