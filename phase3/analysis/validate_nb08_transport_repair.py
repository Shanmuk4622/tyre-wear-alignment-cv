"""Bind executed failure, unchanged experiment, repaired notebook and checks."""
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path
import nbformat

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'tread_two/pilot'
OUT = ROOT / 'tread_two/pilot_validation/nb08_repair'
FAILED = OUT / '5bb076e30cd8'
sys.path.insert(0,str(PILOT))
from common import atomic, digest, file_meta, read, sha


def payload(nb):
    result = None
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
                value = json.loads(base64.b64decode(node.args[0].value))
                assert result is None
                result = {name:base64.b64decode(body) for name,body in value.items()}
    return result


def main():
    delivery = read(ROOT / 'tread_two/pilot_validation/NOTEBOOK_DELIVERY.json')
    tests = read(OUT / 'TRANSPORT_TESTS.json')
    extracted = read(ROOT / 'tread_two/pilot_validation/NOTEBOOK_VALIDATION.json')
    failed = FAILED / 'executed.ipynb'
    assert sha(failed) == '5bb076e30cd8651b352281fffdfd38d38f91f1ec04c8bd9f1e58d8cf0d7fc862'
    nb = nbformat.read(delivery['notebook'],as_version=4)
    nbformat.validate(nb)
    old = nbformat.read(failed,as_version=4)
    before, after = payload(old), payload(nb)
    assert before == after and len(after) == 25
    assert all(body == (PILOT / name).read_bytes() for name,body in after.items())
    assert delivery['sources'] == read(FAILED / 'ORIGINAL_NOTEBOOK_DELIVERY.json')['sources']
    assert all(not cell.outputs and cell.execution_count is None for cell in nb.cells if cell.cell_type=='code')
    assignments = {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                    for target in node.targets:
                        if isinstance(target,ast.Name):
                            assignments[target.id] = node.value.value
    overlay = base64.b64decode(assignments['REPAIR_PAYLOAD'])
    assert overlay == (ROOT / 'analysis/nb08_transport_launcher.py').read_bytes()
    assert hashlib.sha256(overlay).hexdigest() == delivery['transport_launcher_sha256'] == tests['transport_launcher_sha256']
    assert assignments['SOURCE_INVENTORY_SHA'] == delivery['frozen_inventory_sha256'] == sha(PILOT / 'FILES.json') == tests['source_inventory_sha256']
    compile(overlay.decode('utf-8'),'nb08_transport_launcher.py','exec')
    assert tests['status'] == extracted['status'] == 'passed' and tests['count'] == 13
    assert extracted['notebook_sha256'] == delivery['notebook_sha256'] == sha(Path(delivery['notebook']))
    assert extracted['actual_batch'] == 8 and extracted['extracted_worker_completed_updates'] == 1
    assert extracted['extracted_model_optimizer_RNG_and_scientific_CSV_match']
    assert tests['network_requests'] == extracted['network_requests'] == 0
    # Reconstruct the actual Kaggle fit identity from the original recipe,
    # hashes and captured environment: the repair must keep fe7dc763... .
    remote = read(FAILED / 'remote/CONTRACT.json')
    audit = read(FAILED / 'REMOTE_FAILURE_AUDIT.json')
    recipe = read(PILOT / 'RECIPE.json')
    fit = dict(recipe, **{name:remote[name] for name in ('sources','source_checksum','overlay_sha256','environment','train_device','eval_device')})
    assert digest(remote) == audit['engineering_destination'].split('/')[-1]
    assert digest(fit) == audit['fit_destination'].split('/')[-1] == 'fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd'
    current = read(PILOT / 'FILES.json')
    assert remote['sources'] == {name:item['sha256'] for name,item in current.items()}
    assert all(file_meta(PILOT/name) == item for name,item in current.items())
    proof_bytes = (FAILED / 'remote/PREFLIGHT_PROOF.json').read_bytes()
    assert hashlib.sha256(proof_bytes).hexdigest() == audit['engineering_proof_sha256']
    assert read(FAILED / 'remote/PREFLIGHT_PROOF.json')['actual_live_HF_restore']
    assert not audit['measured_manifest_present'] and audit['remote_writes'] == 0 and not audit['local_credentials_read']
    # Original preservation receipts bind all prior source/model paths.
    station = read(ROOT / 'workstation_validation/DELIVERY.json')
    assert all(sha(ROOT/name) == value for name,value in station['source_hashes'].items())
    original = read(ROOT / 'workstation_validation/ORIGINAL_BASELINE.json')
    assert all(sha(ROOT.parent/name) == value for name,value in original.items())
    result = dict(status='passed',failed_notebook_sha256=sha(failed),repaired_notebook_sha256=delivery['notebook_sha256'],
                  notebook_bytes=Path(delivery['notebook']).stat().st_size,code_cells=6,
                  original_embedded_files_byte_identical=25,frozen_source_assets_unchanged=24,
                  transport_checks=tests['count'],packaged_real_batch=8,
                  packaged_model_optimizer_RNG_and_CSV_equivalent=True,
                  actual_engineering_protocol=digest(remote),actual_fit_protocol_preserved=digest(fit),
                  published_engineering_proof_hash_verified=True, inspected_remote_revision=audit['revision'],
                  fit_manifest_absent_at_inspection=True, original_workstation_files_preserved=len(original),
                  separate_workstation_source_files_preserved=len(station['source_hashes']),
                  local_test_network_requests=0, public_audit_read_requests=audit['public_read_requests'],
                  local_credentials_read=False,remote_writes=0,repaired_Kaggle_execution_tested=False,
                  transport_launcher_sha256=tests['transport_launcher_sha256'],
                  check_sources={name:sha(ROOT/name) for name in ('analysis/nb08_transport_launcher.py','analysis/test_nb08_transport_repair.py','analysis/build_tread_two_notebook.py','analysis/validate_nb08_transport_repair.py')})
    atomic(OUT / 'REGRESSION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
