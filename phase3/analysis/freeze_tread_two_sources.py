"""Refresh the isolated pilot's exact source/asset inventory before packaging/checks."""
import hashlib
import json
import sys
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
HERE = Path(__file__).resolve().parents[1] / 'tread_two/pilot'
files = {}
for path in sorted(HERE.rglob('*')):
    if not path.is_file() or path.name == 'FILES.json' or '__pycache__' in path.parts:
        continue
    if path.suffix not in ('.py', '.json', '.csv', '.npy'):
        continue
    files[path.relative_to(HERE).as_posix()] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
(HERE / 'FILES.json').write_text(json.dumps(files, sort_keys=True, indent=2) + '\n', encoding='utf-8')
print('Frozen', len(files), 'isolated source/asset files')
