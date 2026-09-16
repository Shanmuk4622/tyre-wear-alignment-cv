"""Fail-closed local checks for the report and the frozen small evidence cache."""
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/report'
EVIDENCE = REPORT / 'evidence'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_evidence():
    manifest_path = EVIDENCE / 'evidence_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    evidence_status = json.loads((EVIDENCE / 'EVIDENCE_STATUS.json').read_text())
    assert sha(manifest_path) == evidence_status['manifest_sha256']
    assert len(manifest) == 38
    for item in manifest:
        path = (EVIDENCE / item['local']).resolve()
        assert path.is_relative_to(EVIDENCE.resolve())
        assert path.stat().st_size == item['bytes'], item['local']
        assert sha(path) == item['sha256'], item['local']
    status = json.loads((EVIDENCE / 'REPORT_STATUS.json').read_text())
    assert len(status['artifact_sha256']) == 7
    for name, expected in status['artifact_sha256'].items():
        path = (EVIDENCE / name).resolve()
        assert path.is_relative_to(EVIDENCE.resolve())
        assert sha(path) == expected, name
    for filename, expected in {'classification_master_architectures': 17,
             'classification_stage_a_quarantined': 9, 'classification_baselines_by_fold': 15,
             'stage_b_effects': 108, 's4b_architecture_effects': 6,
             's5_inventory': 81, 's5_localisation_by_run': 81,
             's5_roi_by_run': 405, 's9_fusion_by_run': 405, 'conformal': 3}.items():
        with (EVIDENCE / 'tables' / (filename + '.csv')).open(encoding='utf-8-sig', newline='') as f:
            items = list(csv.DictReader(f))
        assert len(items) == expected, filename
        if filename == 's5_inventory':
            assert len({r['run_id'] for r in items}) == 81
            assert all(r['epoch'] == '60' and r['status'] == 'completed_verified' for r in items)

class Document(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.images=[]; self.ids=[]; self.tables=0
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        if tag == 'a': self.links.append(attrs.get('href',''))
        if tag == 'img':
            assert attrs.get('alt'), 'Missing image alternative text'
            self.images.append(attrs['src'])
        if tag == 'table': self.tables += 1

def check_link(base, target):
    parsed=urlsplit(target)
    if parsed.scheme or parsed.netloc: return
    if not parsed.path: return
    path = (base / unquote(parsed.path)).resolve()
    assert path.is_relative_to(ROOT.resolve()), f'Link escapes project: {target}'
    if path == (REPORT / 'VALIDATION.json').resolve():
        return  # This verification run creates the linked output only after all checks pass.
    assert path.exists(), f'Missing local link: {target}'

def main():
    verify_evidence()
    text=(REPORT/'REPORT.md').read_text(encoding='utf-8')
    assert not re.search(r'\{\{[A-Z_]+\}\}', text)
    doc=Document(); doc.feed((REPORT/'REPORT.html').read_text(encoding='utf-8'))
    assert len(doc.ids)==len(set(doc.ids))
    assert len(doc.images)==22, len(doc.images)
    assert doc.tables >= 11, doc.tables
    for link in doc.links + doc.images:
        check_link(REPORT, link)
        if link.startswith('#'): assert link[1:] in doc.ids, link
    for target in doc.images:
        path=REPORT/target
        if path.suffix=='.svg': ElementTree.parse(path)
        else:
            with Image.open(path) as img: img.verify()
    docs=list(REPORT.glob('*.md')) + [ROOT/'docs/DOCUMENTATION_INDEX.md', ROOT/'docs/REPOSITORY_GUIDE.md']
    for path in docs:
        if path.name=='manuscript.source.md': continue
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            check_link(path.parent, match.group(1))
    used={int(n) for n in re.findall(r'\[(\d+)\](?!\()',text)}
    assert used.issubset(set(range(1,15))), used
    build=json.loads((REPORT/'BUILD_PROVENANCE.json').read_text())
    extension=REPORT/'evidence/geometry_manifest.json'
    assert sha(extension)==build['geometry_manifest_sha256']
    for item in json.loads(extension.read_text())['files']:
        path=REPORT/item['local'] if 'local' in item else ROOT/item['source']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item
    for name, expected in build['artifacts_sha256'].items(): assert sha(ROOT/name)==expected, name
    # Compare only in memory; never display token or .env content.
    env=ROOT/'.env'; token=''
    if env.exists():
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            key, sep, value=line.partition('=')
            if sep and key.strip()=='HF_TOKEN': token=value.strip().strip('\"\'')
    for path in REPORT.rglob('*'):
        if path.is_file() and path.suffix in {'.md','.html','.json','.csv','.svg'}:
            raw=path.read_text(encoding='utf-8-sig')
            assert not token or token not in raw, 'Secret detected in report output'
            assert not re.search(r'hf_[A-Za-z0-9]{25,}',raw), 'Token-like credential detected'
    result={'status':'passed','evidence_files':38,'report_artifact_hashes':7,
            'visuals':len(doc.images),'tables':doc.tables,'word_count':build['word_count'],'geometry_and_local_evidence':'hash_verified',
            'limits':['No training rerun','No large checkpoint re-download','No venue pagination certification',
                      'Inherited saliency source has disclosed title overlap'],
            'full_original_project_complete':False}
    (REPORT/'VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
