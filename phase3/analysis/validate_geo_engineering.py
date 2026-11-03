"""Numerical/local preparation proof; separate from actual Kaggle/live HF proof."""
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'tread_two/geometry_first/runtime'
sys.path.insert(0, str(SOURCE))
from common import atomic, file_meta, digest

# Refresh this new, unmeasured package only. Completed packages are never touched.
inventory = {p.name:file_meta(p) for p in SOURCE.iterdir() if p.is_file() and p.name!='FILES.json'}
atomic(SOURCE/'FILES.json', inventory)
from engineering_geo import run_fixture
proof = run_fixture(ROOT/('tread_two/geometry_first/validation/fixture_local_'+digest(inventory)[:12]),
                    ROOT/'workstation/models/weights.pt', device='cpu', anchor_device='cpu', live=False)
atomic(ROOT/'tread_two/geometry_first/validation/LOCAL_ENGINEERING.json', proof)
print(json.dumps(proof, indent=2))
