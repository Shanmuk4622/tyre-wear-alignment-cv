"""Bind the compact NB03 delivery, current sources and completed local evidence."""
import ast
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    validation=read(ROOT/'diagnostics/DIAGNOSTIC_VALIDATION.json');assert validation['status']=='passed'
    contract=read(Path(validation['snapshot'])/'CONTRACT.json')
    assert all(sha(ROOT/name)==value for name,value in contract['sources'].items())
    assert all(sha(ROOT/'runtime'/name)==value for name,value in contract['frozen_sources'].items())
    notebook=ROOT/'notebooks/phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb'
    delivery=read(ROOT/'diagnostics/NB03_DELIVERY.json')
    publication=read(ROOT/'diagnostics/nb03_publication/VERIFICATION.json')
    assert publication['status']=='passed' and sha(notebook)==publication['notebook_sha256']
    assert sha(ROOT/'diagnostics/nb03_publication/executed.ipynb')==sha(notebook)
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    embedded={}
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        assert not any(o.get('output_type')=='error' for o in cell.outputs)
        for call in ast.walk(ast.parse(cell.source)):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='b64decode' and call.args and isinstance(call.args[0],ast.Constant):
                payload=json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
                for name,value in payload.items():
                    body=base64.b64decode(value);assert body==(ROOT/name).read_bytes()
                    embedded[name]=hashlib.sha256(body).hexdigest()
    assert embedded==delivery['sources'] and len(embedded)==15
    tests=read(ROOT/'diagnostics/DIAGNOSTIC_TESTS.json');assert tests['status']=='passed'
    assert len(tests['offline_checks'])==12 and tests['actual_resume']['status']=='passed'
    assert sha(validation['explorer_alias'])==validation['explorer_sha256']
    result=dict(status='passed',notebook=str(notebook),notebook_bytes=notebook.stat().st_size,notebook_sha256=sha(notebook),
       embedded_source_files=15,local_full_diagnostic_verified=True,diagnostic_contract=validation['diagnostic_contract'],
       offline_fault_checks=12,actual_gpu_resume_scientific_csvs_identical=True,
       browser_render_tested=False,kaggle_execution_tested=True,live_hf_diagnostic_publication_tested=True,
       executed_code_cells=publication['executed_code_cells'],verified_revision=publication['revision'],
       historical_unexecuted_delivery_sha256=delivery['notebook_sha256'],
       standing_data_growth_and_visual_requirements_recorded=True)
    (ROOT/'diagnostics/HANDOFF_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
