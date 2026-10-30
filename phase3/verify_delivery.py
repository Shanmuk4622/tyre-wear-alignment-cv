"""Validate source data, delivered notebook contents, documentation and prior-file integrity."""
from PIL import Image
import ast
import base64
import gzip
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
from common import atomic, read, sha
from data import plan
from phase2_dataset_verify import verify


def run():
    source = ROOT.parent/'phase2/data/phase2_dataset_v1'
    result = verify(source)
    p = plan(source)
    atomic(ROOT/'validation/DATA_PREFLIGHT.json', dict(source=result, split=p['counts'], groups=p['groups'],
           geometry=p['geometry_counts'], scope='Local full byte/structural source verification; no HF publication.'))
    original = read(ROOT/'audit/markdown-baseline.json')
    assert all(sha(ROOT.parent/r['path']) == r['sha256'] for r in original)
    for name in ('phase2_training_data.py', 'phase2_dataset_verify.py'):
        assert sha(ROOT/'runtime'/name) == sha(ROOT.parent/'phase2'/name)
    for path in list((ROOT/'runtime').glob('*.py'))+list(ROOT.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    import nbformat
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'b64decode' and node.args and isinstance(node.args[0], ast.Constant):
                        payload = json.loads(gzip.decompress(base64.b64decode(node.args[0].value)))
                        for name, value in payload.items():
                            if name.endswith('.py'):
                                assert base64.b64decode(value) == (ROOT/'runtime'/name).read_bytes()
    delivery = read(ROOT/'releases/DELIVERY.json')
    for name, expected in delivery['files'].items():
        assert sha(ROOT/name) == expected
    with zipfile.ZipFile(ROOT/'releases/TREAD_ONE_Kaggle.zip') as z:
        assert z.testzip() is None
        for name in delivery['files']:
            assert z.read(name) == (ROOT/name).read_bytes()
    broken = []
    links = 0
    for path in ROOT.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|mailto:|#)', target):
                continue
            links += 1
            if not (path.parent/target.split('#')[0]).exists():
                broken.append([path.name, target])
    assert not broken, broken
    tests = {name: read(ROOT/'validation'/name)['status'] for name in ('LOCAL_TESTS.json', 'PERSISTENCE_TESTS.json', 'LIFECYCLE_TESTS.json')}
    assert set(tests.values()) == {'passed'}
    smoke_sources = read(ROOT/'validation/LOCAL_TESTS.json')['fresh_process']['runtime_sources']
    assert all(sha(ROOT/'runtime'/name) == expected for name, expected in smoke_sources.items()), 'GPU seam predates a runtime edit'
    atomic(ROOT/'validation/DELIVERY_VALIDATION.json', dict(status='passed', notebooks=2,
           source_files_verified=result['checksummed_files'], image_mask_sets=result['image_and_mask_sets'],
           protected_original_markdown=len(original), exact_vendored_sources=2,
           notebook_payloads_match_sources=True, notebook_schema_and_python_syntax=True,
           archive_roundtrip=True, local_links=links, broken_links=broken, tests=tests,
           kaggle_dual_t4_tested=False, live_hf_tested=False, long_training_completed=False,
           delivery_zip_sha256=sha(ROOT/'releases/TREAD_ONE_Kaggle.zip')))
    print('DELIVERY VALIDATION PASSED; KAGGLE/LIVE-HF/LONG TRAINING STILL PENDING')


if __name__ == '__main__':
    run()
