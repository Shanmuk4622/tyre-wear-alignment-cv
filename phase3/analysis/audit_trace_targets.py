"""Build every train-role TRACE target from existing photos and saved final grids.

No held-out photo reads, image-model forwards, fitting updates, remote writes or
credential lookup. This is a design audit, not the future Kaggle data stage.
"""
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/trace_design'
OUT = BASE / 'validation/actual_train'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'tread_two/geo_postrun'))
from common import atomic, read, sha, file_meta
from inputs import Saved, PIN
folder = Path(read(ROOT / 'tread_two/nb17_completion/LATEST.json')['path'])
assert read(folder / 'VERIFICATION.json')['status'] == 'passed'
manifest = read(folder / 'REMOTE_MANIFEST.json')
saved = Saved(folder / 'remote', manifest)
sys.path.insert(0, str(BASE / 'runtime'))
from targets import fixed_targets, fit_statistics, standardized, tiles, choose_hidden, CHANNELS, TARGET_REVISION

torch.set_num_threads(4)
dataset = ROOT.parent / 'phase2/data/phase2_dataset_v1'
assert sha(dataset / 'SHA256SUMS.txt') == PIN['source_checksum']
expected = {name: digest for digest, name in (line.split('  ', 1) for line in (dataset / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines())}
original = {row['image_id']: row for row in read(dataset / 'manifests/images.json')}
verified = {}


def verify(path):
    relative = path.relative_to(saved.root).as_posix()
    assert file_meta(path) == {k: manifest['files'][relative][k] for k in ('bytes', 'sha256')}, relative
    verified[relative] = file_meta(path)


for family in ('sample_observations', 'source_images', 'native_chunks_index'):
    for path in saved.paths(family):
        verify(path)
source = saved.table('source_images').set_index('image_id')
obs = saved.table('sample_observations')
endpoint = obs[obs.stage == 'endpoint'].merge(source[['role', 'leakage_component']], left_on='image_id', right_index=True)
train = endpoint[endpoint.role == 'train'].sort_values('image_id').reset_index(drop=True)
assert len(train) == 386 and train.source_sha256.nunique() == 386 and train.leakage_component.nunique() == 6
index = saved.table('native_chunks_index')
ids = train['sample'].to_numpy(dtype=np.int64)
bodies = {}
for kind in ('row_parameters', 'row_valid', 'geometry_logits'):
    body = None
    seen = np.zeros(len(ids), dtype=bool)
    for row in index[index.kind == kind].itertuples():
        choose = (ids >= row.first_sample) & (ids <= row.last_sample)
        if not choose.any(): continue
        path = saved.root / row.file
        verify(path)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if body is None: body = np.empty((len(ids), *value.shape[1:]), dtype=value.dtype)
        assert not seen[choose].any()
        body[choose] = value[ids[choose] - int(row.first_sample)]
        seen[choose] = True
    assert seen.all()
    bodies[kind] = body

targets, weights, rows, examples, hidden_plans = [], [], [], [], []
start = time.monotonic()
for slot, item in enumerate(train.itertuples()):
    original_row = original[item.image_id]
    image_path = (dataset / original_row['image_path']).resolve()
    ignore_path = (dataset / original_row['ignore_mask']).resolve()
    for path in (image_path, ignore_path):
        assert path.is_relative_to(dataset) and sha(path) == expected[path.relative_to(dataset).as_posix()]
    assert sha(image_path) == item.source_sha256
    with Image.open(image_path) as image:
        rgb = torch.from_numpy(np.asarray(image.convert('RGB').resize((384, 512), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1)[None].float() / 255
    with Image.open(ignore_path) as image:
        ignored = torch.from_numpy((np.asarray(image.resize((384, 512), Image.Resampling.NEAREST)) > 0).copy())[None, None].float()
    raw = torch.from_numpy(bodies['geometry_logits'][slot:slot+1])
    binary = (raw >= 0).repeat_interleave(4, -2).repeat_interleave(4, -1)
    support = (binary[:, :1] & binary[:, 1:2]).float()
    params = torch.from_numpy(bodies['row_parameters'][slot:slot+1])
    flags = torch.from_numpy(bodies['row_valid'][slot:slot+1])
    yy, left, right = params.unbind(-1)
    xx = left[..., None] + torch.linspace(0, 1, 256) * (right - left)[..., None]
    grid = torch.stack((2 * xx / 383 - 1, (2 * yy / 511 - 1)[..., None].expand(-1, -1, 256)), -1).masked_fill(~flags[..., None, None], 0)
    valid = F.grid_sample(support, grid, mode='nearest', align_corners=True) * flags[:, None, :, None]
    atlas = F.grid_sample(rgb, grid, mode='bilinear', align_corners=True) * valid
    ignore_atlas = F.grid_sample(ignored, grid, mode='nearest', align_corners=True)
    result = fixed_targets(tiles(atlas), tiles(valid), tiles(1 - ignore_atlas))
    targets.append(result['target']); weights.append(result['weight'])
    hidden = choose_hidden(tiles(valid), [1000003 + slot])
    hidden_plans.append(hidden)
    rows.append({'slot': slot, 'image_id': item.image_id, 'source_sha256': item.source_sha256, 'role': 'train',
                 'component': item.leakage_component, 'original_sample': int(item.sample),
                 'supported_cells': int(result['available'].sum()), 'target_pixels': int(result['weight'].sum()),
                 'supported_tiles': int((tiles(valid).mean((-3, -2, -1)) >= .25).sum()),
                 'hidden_tiles': int(hidden.sum()), 'mask_seed': 1000003 + slot})
    if slot in (0, 100, 200, 300):
        examples.append({'image_id': item.image_id, 'atlas': atlas.numpy(), 'target': result['target'].numpy(), 'weight': result['weight'].numpy()})
    if (slot + 1) % 64 == 0:
        print('Train-only fixed-target audit:', slot + 1, '/386', flush=True)
target, weight = torch.cat(targets), torch.cat(weights)
stats = fit_statistics(target, weight, ['train'] * 386)
z = standardized(target, stats)
np.save(OUT / 'full_targets.npy', target.numpy(), allow_pickle=False)
np.save(OUT / 'full_target_weights.npy', weight.numpy(), allow_pickle=False)
np.save(OUT / 'hidden_plans.npy', torch.cat(hidden_plans).numpy(), allow_pickle=False)
np.save(OUT / 'train_position_mean.npy', stats['position_mean'].numpy(), allow_pickle=False)
np.save(OUT / 'train_position_count.npy', stats['position_count'].numpy(), allow_pickle=False)
pd.DataFrame(rows).to_csv(OUT / 'target_registry.csv', index=False)
pd.DataFrame([dict(channel=name, mean=float(stats['mean'][k]), std=float(stats['std'][k]), pixels=float(stats['count'][0])) for k, name in enumerate(CHANNELS)]).to_csv(OUT / 'normalization.csv', index=False)

# Full target construction is directly inspectable as six values per grid cell.
with (OUT / 'target_cells.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['slot', 'image_id', 'tile', 'cell_y', 'cell_x', 'valid_pixels', *CHANNELS])
    a, w = target.numpy(), weight.numpy()
    for i in range(386):
        for t in range(32):
            for y in range(4):
                for x in range(4):
                    writer.writerow([i, rows[i]['image_id'], t, y, x, int(w[i, t, 0, y, x]), *[float(v) for v in a[i, t, :, y, x]]])

# Descriptive training-target spectra, including within-component centering.
mean_cells = ((z * weight).sum((1, 3, 4)) / weight.sum((1, 3, 4)).clamp_min(1)).double().numpy()
components = train.leakage_component.to_numpy()
def spectrum(values):
    cov = np.cov(values, rowvar=False)
    eigen = np.maximum(0, np.linalg.eigvalsh(cov))[::-1]
    p = eigen[eigen > 0] / eigen.sum() if eigen.sum() else np.array([])
    return {'rank': float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.,
            'leading_fraction': float(eigen[0] / eigen.sum()) if eigen.sum() else None,
            'eigenvalues': eigen.tolist()}
centered = mean_cells - mean_cells.mean(0)
within = np.zeros_like(centered)
for component in np.unique(components):
    choose = components == component
    within[choose] = mean_cells[choose] - mean_cells[choose].mean(0)
between = centered - within
assert np.isclose((centered ** 2).sum(), (within ** 2).sum() + (between ** 2).sum(), atol=1e-10)

# Three control families, scored on deterministic masked TRAIN cells only.
from objective import reduction
controls = []
for i in range(386):
    hidden = hidden_plans[i]
    value = z[i:i+1]
    ww = weight[i:i+1]
    position = standardized(stats['position_mean'][None], stats)
    neighbor_raw = torch.zeros_like(target[i:i+1]); neighbor_ok = torch.zeros_like(ww)
    for band in range(8):
        sl = slice(4*band, 4*band+4)
        vw = ww[:, sl] * (~hidden[:, sl])[..., None, None, None]
        count = vw.sum(1, keepdim=True)
        predicted = (target[i:i+1, sl] * vw).sum(1, keepdim=True) / count.clamp_min(1)
        neighbor_raw[:, sl] = predicted
        neighbor_ok[:, sl] = (count > 0).expand(-1, 4, -1, -1, -1)
    common_weight = ww * neighbor_ok * (stats['position_count'][None] > 0)
    full_count = float((ww * hidden[..., None, None, None]).sum() * 6)
    for name, prediction in [('channel_mean', torch.zeros_like(value)),
                              ('position_mean', position),
                              ('visible_band_mean', standardized(neighbor_raw, stats))]:
        use_weight = common_weight
        loss, detail = reduction(prediction, value, use_weight, hidden)
        controls.append({'slot': i, 'image_id': rows[i]['image_id'], 'role': 'train', 'control': name,
                         'huber_sum': float(detail['sum'][0]), 'weighted_channel_count': float(detail['count'][0]),
                         'available': bool(detail['available'][0]), 'loss': float(loss) if bool(detail['available'][0]) else None,
                         'original_masked_channel_count': full_count,
                         'common_control_coverage': float(detail['count'][0]) / full_count if full_count else None,
                         'scope': 'training_descriptive_not_checkpoint_eligibility'})
pd.DataFrame(controls).to_csv(OUT / 'control_losses.csv', index=False)
pd.DataFrame([dict(file=name, **meta) for name, meta in verified.items()]).to_csv(OUT / 'saved_inputs_verified.csv', index=False)
atomic(OUT / 'TARGET_SPECTRUM.json', {'scope': 'fixed_training_image_descriptors_not_a_trained_representation',
       'channel_mean6': spectrum(mean_cells), 'within_component_channel_mean6': spectrum(within),
       'between_fraction': float((between ** 2).sum() / (centered ** 2).sum())})

# Durable actual-data figure and compact image/target payload for the design explorer.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
report = ROOT / 'tread_two/trace_report'
(report / 'figures').mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 4, figsize=(13, 8), constrained_layout=True)
payload = []
for k, example in enumerate(examples):
    atlas = np.transpose(example['atlas'][0], (1, 2, 0))
    axes[0, k].imshow(atlas); axes[0, k].set_title('Existing train image ' + str((0,100,200,300)[k]))
    channels = []
    for channel in range(6):
        a = example['target'][0, :, channel].reshape(8, 4, 4, 4).transpose(0, 2, 1, 3).reshape(32, 16)
        mask = example['weight'][0, :, 0].reshape(8, 4, 4, 4).transpose(0, 2, 1, 3).reshape(32, 16) > 0
        channels.append({'name': CHANNELS[channel], 'values': a.tolist(), 'available': mask.tolist()})
        if channel in (1, 2):
            axes[channel, k].imshow(np.ma.masked_where(~mask, a), cmap='magma', interpolation='nearest')
            axes[channel, k].set_title(CHANNELS[channel] + ' / fixed observed target')
    for row in range(3): axes[row, k].axis('off')
    buffer = io.BytesIO(); Image.fromarray((np.clip(atlas,0,1)*255).astype('uint8')).resize((192,384)).save(buffer, format='PNG')
    payload.append({'image_id': example['image_id'], 'role': 'train', 'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), 'channels': channels})
fig.suptitle('TRACE design audit / fixed image descriptors, not model predictions', fontsize=15)
fig.savefig(report / 'figures/TRACE_Fixed_Targets.png', dpi=140); plt.close(fig)
atomic(report / 'EXAMPLES.json', payload)
import runpy
runpy.run_path(str(ROOT / 'analysis/render_trace_figure.py'), run_name='__main__')
for name in ('target_registry.csv', 'normalization.csv', 'control_losses.csv', 'saved_inputs_verified.csv', 'TARGET_SPECTRUM.json'):
    (report / name).write_bytes((OUT / name).read_bytes())
artifacts = {p.name: file_meta(p) for p in OUT.iterdir() if p.is_file() and p.name != 'AUDIT.json'}
result = {'status': 'passed', 'target_revision': TARGET_REVISION, 'scope': 'local_design_target_audit_only', 'images': 386, 'roles': ['train'],
          'target_source_sha256': sha(BASE / 'runtime/targets.py'),
          'conservative_components': 6, 'new_photos': 0, 'geometry_model_forwards': 0, 'optimizer_updates': 0,
          'target_cells': 386 * 32 * 4 * 4, 'target_channel_values': target.numel(),
          'supported_cells': int((weight > 0).sum()), 'full_target_maps_retained': True,
          'input_revision': PIN['revision'], 'input_contract': PIN['protocol'],
          'geometry_input': 'GEO_r1_epoch60_unselected_research_final_saved_grids',
          'saved_objects_size_SHA_verified': len(verified), 'original_photo_SHA_verified': 386,
          'original_ignore_mask_SHA_verified': 386, 'heldout_photo_reads': 0, 'credentials_read': False,
          'remote_writes': 0, 'new_HF_publication': False, 'seconds': time.monotonic() - start,
          'artifacts': artifacts, 'target_descriptor_spectrum': read(OUT / 'TARGET_SPECTRUM.json')}
atomic(OUT / 'AUDIT.json', result)
atomic(BASE / 'validation/ACTUAL_TARGETS.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2), flush=True)
