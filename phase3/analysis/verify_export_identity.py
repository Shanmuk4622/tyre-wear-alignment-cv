"""Verify compact export identity and current protected source against pinned evidence."""
from PIL import Image
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
import torch
from fetch_completed_results import fetch, sha, OUT, ROOT


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


def main():
    tree = json.loads((OUT/'remote_tree.json').read_text())
    entries = {r['path']: r for r in tree['files']}
    contract = json.loads((OUT/'hf/CONTRACT.json').read_text())
    export = json.loads((OUT/'hf/report/EXPORT.json').read_text())
    sources = []
    for name, expected in contract['sources'].items():
        payload = (ROOT/'runtime'/name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest() == entries['sources/'+name]['blob_id']
        sources.append(name)
    path = fetch('report/weights.pt')
    assert sha(path) == export['weights_sha256'] == entries['report/weights.pt']['lfs_sha256']
    weights = torch.load(path, map_location='cpu', weights_only=True)
    state = torch.load(OUT/'extracted/seed3/state.pt', map_location='cpu', weights_only=False)
    assert weights.keys() == state['best'].keys()
    assert all(torch.equal(weights[k], state['best'][k]) for k in weights)
    assert state['best_epoch'] == export['epoch'] == 20 and export['seed'] == 3
    optimizer_finite = []
    for seed in (1, 2, 3):
        s = state if seed == 3 else torch.load(OUT/f'extracted/seed{seed}/state.pt', map_location='cpu', weights_only=False)
        assert finite(s['optimizer']) and finite(s['scheduler']) and finite(s['scaler'])
        optimizer_finite.append(seed)
    image_checks = []
    for c in range(3):
        name = f'report/gallery/example_class{c}.png'
        p = fetch(name)
        assert sha(p) == entries[name]['lfs_sha256']
        image_checks.append(name)
    (OUT/'EXPORT_VERIFICATION.json').write_text(json.dumps(dict(status='passed', source_files=sources,
        export_matches_validation_winner_bitwise=True, seed=3, epoch=20, weights_sha256=sha(path),
        all_optimizer_scheduler_scaler_tensors_finite=optimizer_finite, gallery_image_hashes_checked=image_checks,
        inference_or_training_performed=False), indent=2)+'\n')
    print('Compact export matches seed 3/epoch20 best weights bitwise; all 9 source blobs and state finiteness verified.')


if __name__ == '__main__':
    main()
