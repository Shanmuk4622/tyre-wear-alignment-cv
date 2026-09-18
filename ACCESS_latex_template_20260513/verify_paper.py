"""Check paper dependencies, citations, source hashes, and compiled PDF integrity."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import unicodedata
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed', action='store_true', help='Record a completed human/agent page-render review of this exact PDF')
    args = parser.parse_args()
    sources = [HERE/'main.tex'] + sorted((HERE/'sections').glob('*.tex')) + sorted((HERE/'generated').glob('*.tex'))
    text = '\n'.join(p.read_text(encoding='utf-8') for p in sources)
    citations = set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}', text) for k in group.split(','))
    bib = (HERE/'references.bib').read_text(encoding='utf-8')
    entries = re.findall(r'@\w+\s*\{\s*([^,]+),', bib)
    assert len(entries) == len(set(entries)), 'Duplicate bibliography key'
    assert citations == set(entries), f'Citation mismatch: {citations ^ set(entries)}'
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    assert len(labels) == len(set(labels)), 'Duplicate figure/table/equation label'
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)
    assert set(refs) <= set(labels), f'Unknown references: {set(refs)-set(labels)}'
    assert not re.search(r'\b(?:TODO|FIXME|TBD|lorem ipsum)\b', text, re.I)
    for name in re.findall(r'\\input\{([^}]+)\}', text):
        path=HERE/(name if name.endswith('.tex') else name+'.tex')
        assert path.is_file(), f'Missing source {name}'
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
        assert (HERE/'figures'/name).is_file(), f'Missing figure {name}'
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.S)[1]
    words = len(abstract.split())
    assert 150 <= words <= 250, f'Abstract length: {words}'
    log = (HERE/'main.log').read_text(encoding='utf-8',errors='replace')
    for pattern in [r'^!',r'Overfull \\[hv]box',r'Missing character:',r'Citation .+ undefined',
                    r'Reference .+ undefined',r'There were undefined references',r'Font shape .+ undefined']:
        assert not re.search(pattern,log,re.M), f'Build problem: {pattern}'
    assert 'Done.' in (HERE/'main.blg').read_text(), 'Bibliography did not finish'
    # Require the current compiled PDF, not a stale output left by a failed run.
    pdf = HERE/'main.pdf'
    assert pdf.stat().st_mtime >= max(p.stat().st_mtime for p in sources+[HERE/'references.bib',HERE/'engine_compat.tex'])
    reader = PdfReader(pdf)
    pages = [unicodedata.normalize('NFKC',p.extract_text()) for p in reader.pages]
    full = '\n'.join(pages)
    assert all(len(t) > 150 for t in pages), 'Potentially blank output page'
    assert '??' not in full, 'Unresolved reference in PDF'
    for expected in ['23.75','1.312','1.721','0.91310','0.99746','0.01324','REFERENCES','Confidence Diagnostics','0.7393','0.1094']:
        assert expected.lower() in full.lower(), f'Missing expected content: {expected}'
    assert 'AUTHOR-REVIEW DRAFT' in full
    assert 'VOLUME 11, 2023' not in full, 'Inherited false publication metadata'
    assert 'FIGURE' not in full[full.rfind('REFERENCES'):], 'Figure stranded after bibliography'
    bbl=(HERE/'main.bbl').read_text(encoding='utf-8')
    assert len(re.findall(r'\\bibitem\{', bbl)) == len(entries)
    numerical=json.loads((HERE/'NUMERICAL_AUDIT.json').read_text())
    revision=json.loads((HERE/'REVISION_ANALYSIS.json').read_text())
    source_checks=0
    if (REPO/'docs/report/evidence').is_dir():
        for rel, digest in numerical['sources_sha256'].items():
            assert sha(REPO/rel)==digest, f'Changed evidence: {rel}'
            source_checks+=1
        for rel, digest in revision['sources_sha256'].items():
            assert sha(REPO/rel)==digest, f'Changed revision evidence: {rel}'
            source_checks+=1
        upstream=json.loads((REPO/'docs/report/evidence/evidence_manifest.json').read_text())
        for entry in upstream:
            assert sha(REPO/'docs/report/evidence'/entry['local'])==entry['sha256'], entry['local']
        extension=json.loads((REPO/'docs/report/evidence/geometry_manifest.json').read_text())
        for entry in extension['files']:
            path = REPO/'docs/report'/entry['local'] if 'local' in entry else REPO/entry['source']
            assert sha(path)==entry['sha256'], str(path)
    files = sources+[HERE/'references.bib',HERE/'engine_compat.tex']+sorted((HERE/'figures').glob('*'))
    result = dict(status='passed', pages=len(pages), abstract_words=words, references=len(entries),
        figures=len([x for x in labels if x.startswith('fig:')]), tables=len([x for x in labels if x.startswith('tab:')]),
        checked_source_hashes=source_checks, pdf_sha256=sha(pdf),
        compile_errors=0, unresolved_citations_or_references=0, overfull_boxes=0,
        visual_review_of_exact_pdf=args.visual_reviewed,
        remaining_author_actions=['Confirm proposed corresponding author','Confirm competing interests and proposed contributions',
                                  'Confirm collection and figure-use permissions','Review scientific interpretation and approve submission'],
        files_sha256={p.relative_to(HERE).as_posix():sha(p) for p in files})
    (HERE/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':
    main()
