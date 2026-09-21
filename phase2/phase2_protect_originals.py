"""Capture/check hashes of existing source, docs, notebooks, registry and inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BASELINE = ROOT/'manifests'/'phase2_originals_baseline.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        if BASELINE.exists():
            raise SystemExit('Baseline already exists; refusing to replace it')
        tracked = subprocess.check_output(['git','ls-files','-z'], cwd=REPO).decode().split('\0')
        paths = {p for p in tracked if p and not p.startswith('phase2/') and (REPO/p).is_file()}
        paths.update(p.relative_to(REPO).as_posix() for p in REPO.rglob('*.md')
                     if ROOT not in p.parents and '.git' not in p.parts)
        paths.update(p.relative_to(REPO).as_posix() for p in (REPO/'Videos').glob('*.mp4'))
        # Include the existing understanding note even when not tracked.
        if (REPO/'undertand3.md').exists(): paths.add('undertand3.md')
        manifest = {p:dict(bytes=(REPO/p).stat().st_size, sha256=digest(REPO/p)) for p in sorted(paths)}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Protected {len(manifest)} pre-existing files; baseline saved inside phase2')
    else:
        manifest = json.loads(BASELINE.read_text(encoding='utf-8'))
        changed = [p for p, r in manifest.items() if not (REPO/p).is_file() or digest(REPO/p) != r['sha256']]
        result = dict(checked=len(manifest), unchanged=not changed, changed=changed)
        (ROOT/'manifests'/'phase2_originals_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
        if changed: raise SystemExit(1)


if __name__ == '__main__': main()
