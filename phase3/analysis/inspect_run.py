"""Read executed notebooks as data; retain exact copies and bounded text evidence."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'completion_20261028'
OUT.mkdir(exist_ok=True)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def text(value):
    return ''.join(value) if isinstance(value, list) else value or ''


def main():
    records = []
    all_lines = []
    with zipfile.ZipFile(ROOT/'releases/TREAD_ONE_Kaggle.zip') as released:
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            raw = path.read_bytes()
            nb = json.loads(raw)
            original = json.loads(released.read('notebooks/'+path.name))
            archive = OUT/'executed_notebooks'/path.name
            archive.parent.mkdir(exist_ok=True)
            if archive.exists():
                assert archive.read_bytes() == raw, 'Existing archive differs; use a new dated snapshot'
            else:
                archive.write_bytes(raw)
            cells = []
            for i, cell in enumerate(nb['cells']):
                output_lines = []
                errors = []
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        errors.append(dict(name=output.get('ename'), value=output.get('evalue'), traceback=output.get('traceback')))
                    if output.get('output_type') == 'stream':
                        output_lines += text(output.get('text')).splitlines()
                    else:
                        output_lines += text(output.get('data', {}).get('text/plain')).splitlines()
                original_cell = original['cells'][i] if i < len(original['cells']) else None
                source_match = bool(original_cell and text(original_cell['source']) == text(cell['source']))
                cells.append(dict(index=i, type=cell['cell_type'], execution_count=cell.get('execution_count'),
                                  source_matches_delivery=source_match, errors=errors, output_lines=len(output_lines)))
                all_lines += [f'NOTEBOOK {path.name} CELL {i}']+output_lines
                selected = [line for line in output_lines if any(word in line.lower() for word in
                            ('run_prefix', 'hf snapshot', 'epoch 60', 'error', 'failed', 'status:', 'export:', 'complete', 'paused', 'summary', 'balanced_accuracy', 'mean_two_region', 'human_point', 'selected', 'validation_utility'))]
                print(json.dumps(dict(notebook=path.name, cell=i, execution_count=cell.get('execution_count'),
                      source_match=source_match, errors=errors, selected_lines=selected[-22:]), ensure_ascii=False))
            records.append(dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=sha(raw), cells=cells,
                                metadata=nb.get('metadata', {})))
    (OUT/'notebook_text_outputs.txt').write_text('\n'.join(all_lines)+'\n', encoding='utf-8')
    (OUT/'notebook_inspection.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
