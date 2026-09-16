"""Pinned seed-1/final-epoch inference exports. No training or HF writes."""
import bootstrap
import json
import os
import torch
from bootstrap import ROOT
from prepare_models import download, digest
from registry import REPO

SPECS = {
    'hrnet': dict(revision='96166fb19f9bf63d1acb2bd2ee8b55b0ffdacadd',
        protocol='351c6638783996f46cf9efd99ec6689de726e045288054f240193cea61385712',
        namespace='hrnet-geometry-2026-09-15-r1', audit='hrnet_final_audit', contract='report_CONTRACT.json'),
    'matched': dict(revision='aeee706c3aa60dad2706fff56939d7ed86ab01ca',
        protocol='8013bf1e6418e6f884a353efa7ce892c31fbaac450f9d7caede636d9e61e2a18',
        namespace='segformer-matched-2026-09-15-r1', audit='segformer_completion_audit', contract='comparison_CONTRACT.json'),
}


def path(name):
    return ROOT/'checkpoints'/f'geometry-{name}.pt'


def prepare(name):
    spec = SPECS[name]
    target = path(name)
    meta_path = target.with_suffix('.json')
    if target.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['revision'] == spec['revision'] and digest(target) == meta['inference_sha256']:
            print(name+': verified offline export', flush=True); return
    audit = ROOT.parent/'outputs'/spec['audit']
    contract = json.loads((audit/spec['contract']).read_text())
    from hrnet_protocol import canonical, sha
    assert sha(canonical(contract)) == spec['protocol'], 'Contract hash mismatch'
    status = json.loads((audit/'runs_seed1_STATUS.json').read_text())
    remote = f"s9/{spec['namespace']}/{spec['protocol']}/runs/seed1/state.pt"
    raw = download(REPO, remote, spec['revision'], ROOT/'.cache'/f'geometry-{name}-source.pt')
    assert digest(raw) == status['checkpoint_sha256'], 'Audited checkpoint hash mismatch'
    state = torch.load(raw, map_location='cpu', weights_only=False)
    assert state['protocol'] == spec['protocol'] and state['seed'] == 1 and state['epoch'] == 60 and state['cursor'] == 0
    compact = dict(model=state['model'], contract=contract)
    if name == 'matched':
        cfg = download('nvidia/mit-b0', 'config.json', '80983a413c30d36a39c20203974ae7807835e2b4', ROOT/'.cache'/'segformer_config.json', repo_type='model')
        compact['config'] = json.loads(cfg.read_text())
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.tmp'); torch.save(compact, temporary); os.replace(temporary, target)
    meta = dict(spec, repo=REPO, path=remote, seed=1, epoch=60, source_sha256=digest(raw),
                inference_sha256=digest(target), bytes=target.stat().st_size)
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'{name}: verified compact export {target.stat().st_size/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    for name in SPECS:
        prepare(name)
