"""Portable frozen intent; paths excluded, actual environment included for strict resume."""
from pathlib import Path
from common import HERE, digest, environment, file_meta, read, sha
from data_atlas import DATA_CHECKSUM, OVERLAY_SHA


def sources():
    files = read(HERE / 'FILES.json')
    assert all(file_meta(HERE / name) == meta for name, meta in files.items()), 'Pilot source/asset drift'
    return {name: meta['sha256'] for name, meta in files.items()}


def configuration(root, weights, device='cuda:0', eval_device='cuda:1', engineering=False, base_weights=None):
    recipe = read(HERE / 'RECIPE.json')
    base_weights=Path(base_weights) if base_weights else Path(weights).parent/'atlas_lc_selected_epoch7.pt'
    assert sha(base_weights)==read(HERE/'BASE_INPUT.json')['sha256'],'Wrong frozen LC base'
    if engineering:
        recipe = dict(recipe, mode='engineering_fixture', epochs=2, batches_per_epoch=2,
                      evaluation_images_per_class=1, test_role='validation fixtures only')
    contract = dict(recipe, sources=sources(), source_checksum=DATA_CHECKSUM, overlay_sha256=OVERLAY_SHA,
                    environment=environment(), train_device=device, eval_device=eval_device)
    protocol = digest(contract)
    return dict(contract=contract, protocol=protocol, run=protocol[:12], data_root=str(Path(root).resolve()),
                weights=str(Path(weights).resolve()),base_weights=str(base_weights.resolve()))


def evaluation_rows(plan, cfg, stage):
    role = 'validation' if stage == 'validate' or cfg['contract']['mode'] == 'engineering_fixture' else 'test'
    rows = sorted([row for row in plan['rows'] if row['role'] == role], key=lambda row: row['image_id'])
    if cfg['contract']['mode'] == 'engineering_fixture':
        rows = [next(row for row in rows if row['class_index'] == cls and row['point_kind'] == 'human') for cls in (0, 2)]
    return rows
