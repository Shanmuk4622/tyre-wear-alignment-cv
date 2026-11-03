"""Compare retained descriptor spaces; no new images, fit or held-out access."""
import json
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/trace_design'
records = []
for revision, folder in [('v0', 'actual_train_v0'), ('v2', 'actual_train')]:
    root = BASE / 'validation' / folder
    x = np.load(root / 'full_targets.npy').astype('float64')
    w = np.load(root / 'full_target_weights.npy').astype('float64')
    norm = pd.read_csv(root / 'normalization.csv')
    z = (x - norm['mean'].to_numpy()[None,None,:,None,None]) / norm['std'].to_numpy()[None,None,:,None,None]
    registry = pd.read_csv(root / 'target_registry.csv')
    groups = registry.component.to_numpy()
    # Only grid cells observed in ALL train images for the spatial diagnostic;
    # a missingness pattern is not made into descriptor variation.
    common = (w > 0).all(0).repeat(6, axis=1)
    spatial = z[:, common]
    pooled = (z * w).sum((1,3,4)) / np.maximum(1, w.sum((1,3,4)))
    spatial96 = (z * w).sum(1) / np.maximum(1, w.sum(1))
    for name, a in [('channel_mean6', pooled), ('spatial_cell_mean96', spatial96.reshape(len(x), -1)), ('common_full_spatial', spatial)]:
        for component_center in (False, True):
            value = a.copy()
            if component_center:
                for group in np.unique(groups):
                    use = groups == group; value[use] -= value[use].mean(0)
            centered = value - value.mean(0)
            gram = centered @ centered.T / (len(value) - 1)
            ev = np.maximum(0, np.linalg.eigvalsh(gram))[::-1]
            # Remove roundoff-only null eigenvalues, not measured small signals.
            ev[ev < ev[0] * 1e-12] = 0
            p = ev[ev > 0] / ev.sum()
            records.append(dict(target_revision=revision, role='train', images=len(x), space=name, dimensions=a.shape[1],
                                within_component_centered=component_center, effective_rank=float(np.exp(-(p*np.log(p)).sum())),
                                leading_variance_fraction=float(ev[0]/ev.sum()), common_observed_spatial_dimensions=int(common.sum()),
                                interpretation='Fixed image-derived targets; not a model embedding or held-out success'))
result = pd.DataFrame(records)
result.to_csv(BASE / 'validation/TARGET_SPACES.csv', index=False)
result.to_csv(ROOT / 'tread_two/trace_report/target_spaces.csv', index=False)
print(result[['target_revision','space','within_component_centered','effective_rank','leading_variance_fraction']].to_string(index=False), flush=True)
