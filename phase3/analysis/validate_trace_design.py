"""Disposable TRACE numerical proofs, not a measured candidate or fit runner."""
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tread_two/trace_design/runtime'
sys.path.insert(0, str(SOURCE))
from targets import fixed_targets, fit_statistics, standardized, choose_hidden, input_view
from model import Trace
from objective import objective, reduction, diversity

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.manual_seed(20261102)
np.random.seed(20261102)
random.seed(20261102)
OUT = ROOT / 'tread_two/trace_design/validation'
OUT.mkdir(parents=True, exist_ok=True)
checks = {}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def state_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((str(y.dtype), tuple(y.shape))).encode()); h.update(y.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (tuple, list)):
            for child in x: visit(child)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def restore(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])


rgb = torch.rand(8, 32, 3, 64, 64)
valid = torch.ones(8, 32, 1, 64, 64)
valid[0, 0, :, :8] = 0
ignore = valid.clone(); ignore[0, 1, :, 20:25, 20:25] = 0
fixed = fixed_targets(rgb, valid, ignore)
stats = fit_statistics(fixed['target'], fixed['weight'], ['train'] * 8)
target = standardized(fixed['target'], stats)
hidden = choose_hidden(valid, range(8))
assert hidden.sum(1).tolist() == [8] * 8
assert all(int(hidden[:, 4 * i:4 * i + 4].sum(1).max()) <= 1 for i in range(8))
checks['deterministic_per_band_mask'] = torch.equal(hidden, choose_hidden(valid, range(8)))
assert checks['deterministic_per_band_mask']
assert not fixed['target'].requires_grad and not stats['mean'].requires_grad
checks['fixed_target_detached'] = True
for role in ('validation', 'test', 'mixed'):
    try:
        fit_statistics(fixed['target'], fixed['weight'], ['train'] * 7 + [role])
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out normalization accepted')
checks['heldout_statistics_rejected'] = True

# Independent NumPy reduction, not a second call to the target pooling function.
body = fixed['input'].numpy()
z, dx, dy = body[:, :, :1], body[:, :, 1:2], body[:, :, 2:3]
e2 = dx ** 2 + dy ** 2
padded = np.pad(z, ((0,0), (0,0), (0,0), (1,1), (1,1)))
lap = (padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1] + padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2] - 4 * z) / 4
curvature = np.minimum(1., np.abs(lap) / (np.abs(dx) + np.abs(dy) + .05))
fields = np.concatenate((z, np.sqrt(e2), (dx ** 2 - dy ** 2) / (e2 + .0025),
                         2 * dx * dy / (e2 + .0025), curvature, (np.abs(z) > .15).astype('float32')), 2)
mask = fixed['pixel_weight'].numpy()
maximum = 0.
for r, c in ((0, 0), (1, 2), (3, 3)):
    a = fields[..., r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
    w = mask[..., r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
    count = w.sum((-2, -1), dtype=np.float64)
    reference = (a * w).sum((-2, -1), dtype=np.float64) / np.maximum(1, count)
    reference *= count >= 64
    delta = np.abs(reference - fixed['target'][..., r, c].numpy()).max()
    # Bounded FP32 pool reduction of up to 256 values vs NumPy FP64 accumulation.
    assert delta < 1e-6, delta
    maximum = max(maximum, float(delta))
checks['independent_target_cell_reduction_max_absolute_difference'] = maximum

# Entire hidden RGB altered; visible inputs and masked predictions stay exact.
changed = torch.where(hidden[..., None, None, None], 1 - rgb, rgb)
factors = torch.tensor([[.9, 1.1, .95]]).repeat(8, 1)
assert torch.equal(input_view(rgb, valid, factors, hidden), input_view(changed, valid, factors, hidden))
net = Trace()
out = net(rgb, valid, hidden, factors)
other = net(changed, valid, hidden, factors)
assert torch.equal(out['masked_prediction'], other['masked_prediction'])
checks['hidden_raw_pixel_perturbation_masked_prediction_bit_exact'] = True
not_hidden = ~hidden
assert torch.equal(fixed_targets(changed, valid, ignore)['target'][not_hidden], fixed['target'][not_hidden])
checks['hidden_target_changes_cannot_change_visible_targets'] = True
altered_ignore = ignore.clone(); altered_ignore[:, :, :, 30:35, 30:35] = 0
assert torch.equal(fixed_targets(rgb, valid, altered_ignore)['input'], fixed['input'])
checks['annotation_ignore_does_not_enter_student_features'] = True
rgb_invalid = torch.where(valid.bool(), rgb, 1 - rgb)
assert torch.equal(fixed_targets(rgb_invalid, valid, ignore)['target'], fixed['target'])
checks['invalid_raw_pixels_cannot_change_supported_targets'] = True

# Shape, task-boundary and exact objective reconstruction from per-image sums.
assert out['projected'].shape == (8, 6) and out['pooled'].shape == (8, 240)
probe = torch.zeros(1, 32, 48); probe[0, 0, 0] = 1
mass = torch.ones(1, 32) / 32
assert not torch.equal(net.spatial_pool(probe, mass), net.spatial_pool(probe.reshape(1,8,4,48).flip(1).reshape_as(probe), mass))
assert not torch.equal(net.spatial_pool(probe, mass), net.spatial_pool(probe.reshape(1,8,4,48).flip(2).reshape_as(probe), mass))
checks['explicit_spatial_moments_respond_to_row_and_column_reversal'] = True
second = net(rgb, valid, choose_hidden(valid, range(8, 16)), torch.tensor([[1.1, .9, 1.05]]).repeat(8, 1))
loss, terms, audit = objective([out, second], target, fixed['weight'])
local = np.mean([float((a['local']['sum'] / a['local']['count'].clamp_min(1))[a['local']['available']].mean().detach()) for a in audit])
masked = np.mean([float((a['masked']['sum'] / a['masked']['count'].clamp_min(1))[a['eligible']].mean().detach()) for a in audit])
independent = local + masked + .1 * float(terms['pair'].detach()) + .1 * float(terms['variance'].detach()) + .01 * float(terms['covariance'].detach())
assert abs(independent - float(loss.detach())) < 2e-6
checks['per_image_loss_reconstruction_max_absolute_difference'] = abs(independent - float(loss.detach()))
checks['trainable_parameters'] = sum(p.numel() for p in net.parameters())
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
checks['all_parameter_gradients_finite_and_connected'] = True
assert not rgb.requires_grad and not valid.requires_grad
checks['geometry_inputs_detached'] = True
small = torch.randn(8, 6, requires_grad=True)
_, _, present = diversity(small, torch.tensor([True] * 6 + [False] * 2))
assert not present
_, _, present = diversity(small, torch.tensor([True] * 7 + [False]))
assert present
checks['covariance_guard_requires_at_least_seven_eligible_rows'] = True
empty = fixed_targets(torch.zeros_like(rgb[:1]), torch.zeros_like(valid[:1]))
assert not empty['available'].any() and empty['weight'].sum() == 0 and torch.isfinite(empty['target']).all()
zero, record = reduction(torch.zeros_like(target[:1], requires_grad=True), target[:1], torch.zeros_like(fixed['weight'][:1]), hidden[:1])
assert float(zero.detach()) == 0 and not record['available'].any()
checks['empty_support_explicitly_unavailable'] = True
try:
    bad = rgb.clone(); bad[0, 0, 0, 0, 0] = float('nan'); fixed_targets(bad, valid)
except ValueError:
    checks['nonfinite_input_rejected'] = True
else:
    raise AssertionError('Nonfinite input accepted')

# Feasibility witness: 8 rows, 6 unit-variance, zero-covariance dimensions.
q, _ = torch.linalg.qr(torch.eye(8, dtype=torch.float64)[:, :6] - torch.ones(8, 6, dtype=torch.float64) / 8)
q *= 7 ** .5
cov = torch.cov(q.T)
assert torch.allclose(cov, torch.eye(6, dtype=torch.float64), atol=2e-15, rtol=0)
checks['batch8_dim6_unit_covariance_feasible_max_difference'] = float((cov - torch.eye(6)).abs().max())

# Disposable 6-update complete-state proof. Native file round trip at update 3.
# No study image, original optimizer, epoch checkpoint or selected model touched.
initial_rng = rng(); initial = copy.deepcopy(net.state_dict())
def run(stop=None, resume=None):
    m = Trace(); m.load_state_dict(initial)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001)
    if resume:
        m.load_state_dict(resume['model']); opt.load_state_dict(resume['optimizer']); restore(resume['rng']); start = resume['next_update']
    else:
        restore(initial_rng); start = 0
    history = list(resume['history']) if resume else []
    for update in range(start, 6 if stop is None else stop):
        opt.zero_grad(set_to_none=True)
        # Consume all three retained RNG streams to exercise exact restoration.
        ids = torch.randperm(8)[:2]
        seed = random.randrange(100000) + int(np.random.randint(100000))
        images = rgb[ids]
        masks = valid[ids]
        views = [m(images, masks, choose_hidden(masks, [seed + 2 * j + v for j in range(2)])) for v in (0, 1)]
        value, parts, detail = objective(views, target[ids], fixed['weight'][ids])
        value.backward(); norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True); opt.step()
        history.append({'update': update + 1, 'loss': float(value.detach()), 'pre_clip_norm': float(norm),
                        'source_indices': ids.tolist(), **{k: float(v.detach()) for k, v in parts.items()},
                        'eligible_images': int(detail[0]['eligible'].sum()), 'diversity_available': detail[0]['diversity_available']})
    return {'model': copy.deepcopy(m.state_dict()), 'optimizer': copy.deepcopy(opt.state_dict()), 'rng': rng(),
            'next_update': len(history), 'history': history, 'precision': 'FP32', 'scheduler': None, 'scaler': None}

continuous = run()
interrupted = run(stop=3)
fixture = OUT / 'fixture_state'
fixture.mkdir(exist_ok=True)
torch.save(interrupted, fixture / 'resume.pt')
resumed = run(resume=torch.load(fixture / 'resume.pt', map_location='cpu', weights_only=False))
assert state_digest(continuous) == state_digest(resumed)
checks['CPU_completed_update_native_restore_exact_model_optimizer_RNG_cursor_history'] = True
with (OUT / 'FIXTURE_UPDATES.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=['execution_branch', *continuous['history'][0].keys()], lineterminator='\n')
    writer.writeheader()
    for branch, history in [('continuous', continuous['history']), ('interrupted', interrupted['history']), ('restored', resumed['history'][3:])]:
        for row in history:
            writer.writerow({'execution_branch': branch, **row, 'source_indices': json.dumps(row['source_indices'])})
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}
write('NUMERICAL.json', {'status': 'passed', 'scope': 'untrained_design_and_disposable_synthetic_engineering_only', 'checks': checks,
                         'source_sha256': source_hashes,
                         'synthetic_training_updates_executed': 12, 'measured_candidate_updates': 0,
                         'new_photos': 0, 'HF_writes': 0, 'credentials_read': False,
                         'dual_T4_recovery_verified': False, 'live_HF_recovery_verified': False,
                         'limitations': ['CPU fixture is not the future full transactional trainer.', 'No measured accuracy or generalization finding.']})
print(json.dumps({'status': 'passed', 'checks': checks}, indent=2), flush=True)
