"""Read-only Git add-preview audit. Never stages, deletes, or reads secret files.

Run from the project environment: phase3/run_cv.cmd -B scripts/check_git_content.py
Use --inventory to summarize visible files before changing ignore rules.
"""
import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = {'.pt', '.pth', '.ckpt', '.bin', '.onnx', '.safetensors', '.engine', '.tflite', '.pkl', '.pickle', '.joblib'}
SECRET_PARTS = {'.aws', '.ssh', '.kaggle', '.huggingface', '.secrets'}
TOKEN_PATTERNS = [re.compile(p) for p in (
    r'\bhf_[A-Za-z0-9]{25,}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{40,}\b', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    r'\bsk-proj-[A-Za-z0-9_-]{40,}\b', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
)]
TEXT_SUFFIXES = {'.py', '.md', '.ipynb', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.cmd', '.ps1', '.sh', '.js', '.cjs', '.mjs', '.html', '.css', '.txt', '.csv', '.ts', '.tsx', '.xml', '.tex', '.cff'}

def git(*args, input_bytes=None):
    return subprocess.run(['git', *args], cwd=ROOT, input=input_bytes, capture_output=True, check=False)

def paths(*args):
    result = git('ls-files', '-z', *args)
    assert result.returncode == 0, 'Cannot list Git candidates'
    return {p.decode('utf-8') for p in result.stdout.split(b'\0') if p}

def secret_name(name):
    p = Path(name); lower = p.name.lower()
    return ((lower == '.env' or lower.startswith('.env.')) and lower not in {'.env.example', '.env.template'}
            or lower == 'kaggle.json' or lower.endswith(('.pem', '.key', '.token'))
            or lower.startswith('hf_token') or bool(set(x.lower() for x in p.parts) & SECRET_PARTS)
            or ('credentials' in lower and lower.endswith('.json')))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--history', action='store_true', help='Also inspect committed blob sizes and secret filenames; never rewrites history')
    args = parser.parse_args()
    tracked = paths('--cached'); untracked = paths('--others', '--exclude-standard')
    candidates = sorted(n for n in tracked | untracked if (ROOT / n).is_file())
    sizes = {n: (ROOT / n).stat().st_size for n in candidates}
    groups = collections.defaultdict(lambda: [0, 0])
    for n, size in sizes.items():
        parts = n.split('/'); key = '/'.join(parts[:3]) if parts[0] == 'phase3' else parts[0]
        groups[key][0] += 1; groups[key][1] += size
    record = dict(candidate_files=len(candidates), candidate_bytes=sum(sizes.values()),
        tracked_files=len(tracked), untracked_candidates=len(untracked),
        largest_files=[dict(path=n, bytes=sizes[n]) for n in sorted(sizes, key=sizes.get, reverse=True)[:15]],
        largest_groups=[dict(path=n, files=groups[n][0], bytes=groups[n][1]) for n in sorted(groups, key=lambda n: groups[n][1], reverse=True)[:18]])
    if not args.inventory:
        bad = []
        for n in candidates:
            p = ROOT / n
            if secret_name(n): bad.append(dict(path=n, reason='secret filename is Git-visible')); continue
            if p.suffix.lower() in MODEL_SUFFIXES: bad.append(dict(path=n, reason='model/checkpoint is Git-visible')); continue
            if sizes[n] >= 100 * 1024**2: bad.append(dict(path=n, reason='file exceeds the project 100 MiB guard'))
            if p.suffix.lower() in TEXT_SUFFIXES or p.name == '.gitignore':
                # Only Git-visible text: .env and credential files were excluded above.
                try: content = p.read_text(encoding='utf-8')
                except UnicodeDecodeError: continue
                if any(pattern.search(content) for pattern in TOKEN_PATTERNS):
                    bad.append(dict(path=n, reason='credential-like content; values are never printed'))
        # Frozen source packages must be rebuildable from a checkout, including
        # the ten small deliberate NPY reference arrays; no blanket binary exception.
        missing = []
        for folder in ('phase3/tread_two/pilot', 'phase3/tread_two/lc_pilot', 'phase3/tread_two/context_pilot'):
            package = ROOT / folder
            if not (package / 'FILES.json').exists(): continue
            for name in json.loads((package / 'FILES.json').read_text(encoding='utf-8')):
                relative = folder + '/' + name
                check = git('check-ignore', '--no-index', '-q', '--', relative)
                if not (ROOT / relative).is_file() or check.returncode == 0: missing.append(relative)
        record.update(status='passed' if not bad and not missing else 'failed', violations=bad,
            missing_or_ignored_frozen_source_files=missing, credential_values_printed=False,
            secret_files_read=False, index_changed=False, file_contents_changed=False)
    else: record['status'] = 'inventory_only'
    if args.history:
        objects = git('rev-list', '--objects', '--all')
        assert objects.returncode == 0, 'Cannot inspect history'
        blobs = git('cat-file', '--batch-check=%(objecttype) %(objectsize) %(rest)', input_bytes=objects.stdout)
        assert blobs.returncode == 0, 'Cannot inspect object sizes'
        large, secret_paths = [], set()
        for line in blobs.stdout.decode('utf-8', errors='replace').splitlines():
            parts = line.split(' ', 2)
            if len(parts) != 3 or parts[0] != 'blob': continue
            size, name = int(parts[1]), parts[2]
            if size >= 100 * 1024**2: large.append(dict(path=name, bytes=size))
            if secret_name(name): secret_paths.add(name)
        record['history'] = dict(blobs_at_least_100_MiB=large, secret_filenames=sorted(secret_paths), history_rewritten=False)
    if args.report:
        report = args.report.resolve()
        assert report.is_relative_to(ROOT), 'Report must stay inside the workspace'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    return 1 if record['status'] == 'failed' else 0

if __name__ == '__main__': raise SystemExit(main())
