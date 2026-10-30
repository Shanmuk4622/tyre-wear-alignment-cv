"""Bind the independently verified NB09 result to the next separate gate."""
import hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=Path(json.loads((ROOT/'tread_two/nb09_completion/LATEST.json').read_text())['path'])
proof=json.loads((path/'VERIFICATION.json').read_text())
assert proof['status']=='passed' and proof['csv_rows']==246157 and proof['owner_released']
result=dict(NB09_revision=proof['revision'],NB09_destination=proof['destination'],NB09_review_contract=proof['review_contract'],
    executed_NB09_sha256=proof['executed_notebook_sha256'],independent_verification_sha256=hashlib.sha256((path/'VERIFICATION.json').read_bytes()).hexdigest(),
    published_files=51,published_csv_tables=30,published_derived_rows=246157,source_fit_scientific_files_preserved=1177,
    meaning='Completed analysis motivates a training-only gate, not a new measured fit or checkpoint replacement')
(ROOT/'tread_two/appearance_gate/UPSTREAM.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
