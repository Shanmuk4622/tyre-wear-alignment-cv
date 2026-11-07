"""Host the original bench in Phase 3 without importing its global bootstrap."""
import builtins
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from types import ModuleType

from paths import CACHE, DATA, PHASE3, ROOT


ORIGINAL = ROOT.parents[1] / 'prototype'
DEFAULT_BENCH = (DATA / 'model_bench').resolve()
BENCH = Path(os.environ.get('TREAD_PHASE3_BENCH_ROOT', DEFAULT_BENCH)).resolve()
if BENCH != DEFAULT_BENCH and not (
    BENCH.name == 'model_bench'
    and BENCH.parent.name.startswith('unified_')
    and BENCH.parent.parent == (PHASE3 / 'workstation_validation').resolve()
):
    raise ValueError('Model bench records must stay in the designated Phase 3 folders.')
_MODULES = {
    'alignment', 'alignment_ui', 'app', 'classification', 'edge_diagram',
    'edge_geometry', 'engine', 'evidence', 'geometry_comparison', 'geometry_diagram',
    'learned_diagram', 'learned_geometry', 'media_export', 'phase2_adapter',
    'phase2_video_labels', 'prepare_learned', 'prepare_models', 'registry',
    'shape_geometry', 'theme', 'video',
}
_PRIVATE_ROOT = {'app', 'alignment_ui', 'evidence', 'geometry_comparison'}
_PREFIX = '_phase3_original_bench.'


def prepare_dependencies():
    # Phase 3 owns Qt. Only the other dependencies fall through to the original vendor.
    for position, path in ((1, ORIGINAL / '.vendor'), (2, ORIGINAL.parent / 'tyrelib')):
        if str(path) not in sys.path:
            sys.path.insert(position, str(path))
    for key, name in (('HF_HOME', 'huggingface'), ('YOLO_CONFIG_DIR', 'ultralytics'),
                      ('MPLCONFIGDIR', 'matplotlib')):
        folder = CACHE / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['HF_HUB_OFFLINE'] = '1'


def load(name):
    """Resolve the original's flat imports inside this one dedicated namespace."""
    key = _PREFIX + name
    if key in sys.modules:
        return sys.modules[key]
    if name not in _MODULES:
        raise ImportError(name)
    spec = importlib.util.spec_from_file_location(key, ORIGINAL / f'{name}.py')
    module = importlib.util.module_from_spec(spec)

    def scoped_import(target, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            if target == 'bootstrap':
                # Writable workflows and checkpoint lookup have different roots.
                config = ModuleType(_PREFIX + 'bootstrap')
                config.ROOT = BENCH if name in _PRIVATE_ROOT else ORIGINAL
                return config
            if name == 'alignment_ui' and target == 'engine' and fromlist == ('read_image',):
                # Alignment needs oriented RGB, without loading the legacy Torch engine.
                from imaging import read_image
                image_reader = ModuleType(_PREFIX + 'alignment_image_reader')
                image_reader.read_image = lambda path: read_image(path)[0]
                return image_reader
            if target in _MODULES:
                return load(target)
        return builtins.__import__(target, globals, locals, fromlist, level)

    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=scoped_import)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module


def open_bench(device):
    prepare_dependencies()
    (BENCH / '.cache').mkdir(parents=True, exist_ok=True)
    (BENCH / 'data').mkdir(exist_ok=True)
    # The unchanged bench launches ROOT/media_export.py in an owned QProcess.
    # This shim gives the child the same private imports and immutable weights.
    script = BENCH / 'media_export.py'
    source = (
        'import sys\n'
        f'sys.path.insert(0, {str(ROOT)!r})\n'
        'import bootstrap\n'
        'from legacy_bench import prepare_dependencies, load\n'
        'prepare_dependencies()\n'
        "raise SystemExit(load('media_export').main())\n"
    )
    if not script.exists() or script.read_text(encoding='utf-8') != source:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=BENCH, delete=False) as pending:
            pending.write(source)
        Path(pending.name).replace(script)
    return load('app').Station(device)
