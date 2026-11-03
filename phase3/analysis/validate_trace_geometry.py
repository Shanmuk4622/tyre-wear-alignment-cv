"""Check isolated frozen geometry against the original implementation, CPU only."""
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tread_two/geo_pilot'))
from common import read, sha, atomic
from frozen_anchor import TreadOne
from anchor_geo import Anchor
from model_geo import Geo
from data_geo import locate, dataset, load_batch
sys.path.insert(0, str(ROOT / 'tread_two/trace_design/runtime'))
from frozen_geometry import FrozenGeometry, EXPORT_SHA256

torch.set_num_threads(4)
folder = Path(read(ROOT / 'tread_two/nb17_completion/LATEST.json')['path'])
export = folder / 'remote/exports/final.pt'
source = ROOT / 'tread_two/geo_pilot/frozen_anchor.py'
assert sha(export) == EXPORT_SHA256
geometry = FrozenGeometry(export, source, TreadOne)
assert not geometry.training and not any(p.requires_grad for p in geometry.parameters())
geometry.train(True)
assert not any(m.training for m in geometry.modules())
before = {k: v.clone() for k, v in geometry.state_dict().items()}
data = locate(str(ROOT.parent / 'phase2/data/phase2_dataset_v1')); _, train = dataset(data, verify=False)
batch = load_batch(data, train[:2], 'cpu')
anchor = Anchor(folder / 'remote/anchor.pt', 'cpu')
reference = Geo().eval()
reference.load_state_dict(torch.load(export, map_location='cpu', weights_only=True)['trainable'])
with torch.no_grad():
    base = anchor(batch['rgb'], batch['guides'], 'cpu')
    expected = reference.refine(base)
    actual = geometry(batch['rgb'], batch['guides'])
assert all(torch.equal(actual[k], expected[k]) for k in ('seg_logits', 'masks', 'points'))
assert all(torch.equal(v, before[k]) for k, v in geometry.state_dict().items())
assert sum(p.numel() for p in geometry.parameters()) == 1372159 + 27068
try:
    FrozenGeometry(source, source, TreadOne)
except ValueError:
    bad_rejected = True
else:
    raise AssertionError('Unpinned export accepted')
result = {'status': 'passed', 'scope': 'CPU_two_existing_train_images_implementation_equivalence',
          'export_sha256': EXPORT_SHA256, 'source_sha256': sha(source),
          'wrapper_sha256': sha(ROOT / 'tread_two/trace_design/runtime/frozen_geometry.py'),
          'frozen_parameters': 1399227, 'exact_logits_masks_points_equal_original_GEO': True,
          'all_state_tensors_unchanged': True, 'parent_train_cannot_unfreeze': True,
          'texture_quality_parameters_loaded': False, 'wrong_export_rejected': bad_rejected,
          'image_model_forwards': 4, 'unique_train_images': 2, 'heldout_forwards': 0,
          'optimizer_updates': 0, 'HF_writes': 0, 'new_accuracy_claim': False}
atomic(ROOT / 'tread_two/trace_design/validation/GEOMETRY.json', result)
print(json.dumps(result, indent=2), flush=True)
