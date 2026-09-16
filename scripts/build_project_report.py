"""Build an offline author-review report from immutable small HF evidence."""
import csv
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/report'
TABLES = REPORT / 'evidence/tables'

def rows(name):
    with (TABLES / (name + '.csv')).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(value, digits=5):
    try:
        x = float(value)
        return f'{x:.{digits}f}' if math.isfinite(x) else '—'
    except (ValueError, TypeError):
        return '—'

def mean(items, field):
    values = [float(r[field]) for r in items if r.get(field) and math.isfinite(float(r[field]))]
    return statistics.mean(values) if values else float('nan')

def table(headers, data):
    def line(values):
        return '| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |'
    return '\n'.join([line(headers), line(['---'] * len(headers))] + [line(r) for r in data])

def report_tables():
    baselines = rows('classification_baselines_by_fold')
    base = []
    for name in sorted({r['baseline'] for r in baselines}):
        group = sorted([r for r in baselines if r['baseline'] == name], key=lambda r: int(r['fold']))
        assert [r['fold'] for r in group] == ['0', '1', '2']
        base.append([name] + [number(r['f1_macro']) for r in group] + [number(mean(group, 'f1_macro'))])
    arch = rows('classification_master_architectures')
    s4b = rows('s4b_architecture_effects')
    loc = [r for r in rows('s5_localisation_by_run') if r['fold'] == '1']
    dense = []
    for name in sorted({r['model'] for r in loc}):
        group = [r for r in loc if r['model'] == name]
        assert len(group) == 3
        dense.append([name] + [number(mean(group, f)) for f in
                    ['bbox_map_50_95', 'segm_map_50_95', 'tread_iou', 'tread_boundary_f1_2px']])
    fusion = rows('s9_fusion_by_run')
    arms = ['full', 'tyre_only', 'tread_only', 'tyre_tread_equal', 'full_tyre_tread_equal']
    fus = []
    for arm in arms:
        group = [r for r in fusion if r['arm'] == arm]
        assert len(group) == 81
        fus.append([arm, len(group), number(mean(group, 'delta_vs_full'), 6)])
    cal = rows('calibration')
    conf = rows('conformal')
    return {
        'BASELINES': table(['Baseline', 'Fold 0', 'Fold 1', 'Fold 2', 'Mean'], base),
        'ARCHITECTURES': table(['Architecture ID', 'Runs', 'Selected macro-F1', 'Final macro-F1'],
                              [[r['arch'], r['n'], number(r['f1_mean']), number(r['final_f1'])] for r in arch]),
        'S4B': table(['Architecture', 'Factor', 'Selected Δ', 'Final Δ', 'Discovery direction repeated?'],
                     [[r['arch'], r['factor'], number(r['mean_selected_delta']), number(r['mean_final_delta']),
                       'Yes' if r['same_direction'] == 'True' else 'No'] for r in s4b]),
        'LOCALISATION': table(['Model', 'Box AP50:95', 'Mask AP50:95', 'Tread IoU', 'Tread boundary F1'], dense),
        'FUSION': table(['Fixed arm', 'Rows', 'Mean Δ macro-F1'], fus),
        'CALIBRATION': table(['Fold', 'Confidence', 'Test n', 'Macro-F1', 'ECE', 'NLL', 'Brier'],
                              [[r['fold'], r['kind'], r['n']] + [number(r[f]) for f in
                               ['f1_macro', 'ece', 'nll', 'brier']] for r in cal]),
        'CONFORMAL': table(['Fold', 'Calibration n', 'Test n', 'Coverage', 'Mean set size', 'Recorded abstention'],
                           [[r['fold'], r['n_cal'], r['n_test']] + [number(r[f]) for f in
                            ['coverage_90', 'mean_set_size', 'abstain_rate']] for r in conf]),
    }

def plot_endpoints():
    data = sorted(rows('classification_master_architectures'), key=lambda r: float(r['final_f1']))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(10.5, 8), facecolor='white')
    for i, r in enumerate(data):
        ax.plot([float(r['final_f1']), float(r['f1_mean'])], [i, i], color='#c0cbd3', lw=3, zorder=1)
    ax.scatter([float(r['f1_mean']) for r in data], range(len(data)), color='#d18c26', s=60,
               label='Validation-selected mean', zorder=3)
    ax.scatter([float(r['final_f1']) for r in data], range(len(data)), color='#126d78', s=60,
               label='Fixed-final mean', zorder=3)
    ax.set_yticks(range(len(data)), [r['arch'] for r in data])
    ax.set_xlim(.40, 1.02)
    ax.set_xlabel('Macro-F1 · mean of 3 folds × 3 seeds')
    ax.grid(axis='x', color='#e5eaee')
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.035), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Checkpoint selection changes the reported result', x=.04, ha='left', fontsize=18, fontweight='bold')
    fig.text(.04, .925, '17 retained architectures · existing folds, including flagged folds 0 and 2', fontsize=11, color='#566570')
    fig.text(.04, .025, 'Source: frozen classification_master_architectures.csv | Descriptive, not independent-test uncertainty', fontsize=9, color='#566570')
    fig.tight_layout(rect=(0, .05, 1, .91))
    fig.savefig(REPORT / 'assets/endpoint_comparison.png', dpi=170)
    plt.close(fig)

CSS = '''
:root{--ink:#182e3b;--muted:#526773;--teal:#126d78;--line:#dce5ea;--paper:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef3f5;color:var(--ink);font:17px/1.75 Georgia,'Times New Roman',serif}
.masthead{background:#122e3c;color:#fff;padding:20px 5vw;font:12px/1.5 Arial,sans-serif;letter-spacing:2px;text-transform:uppercase}
.layout{display:grid;grid-template-columns:250px minmax(0,1020px);gap:34px;max-width:1360px;margin:35px auto;padding:0 22px}
nav{position:sticky;top:22px;align-self:start;max-height:92vh;overflow:auto;font:13px/1.55 Arial,sans-serif;padding:12px 0}
nav strong{display:block;letter-spacing:2px;font-size:11px;text-transform:uppercase;margin-bottom:15px}nav a{display:block;padding:7px 10px;border-left:2px solid var(--line);text-decoration:none;color:var(--muted)}nav a:hover{border-color:var(--teal);color:var(--teal);background:white}
main{background:var(--paper);padding:55px 65px 65px;box-shadow:0 8px 36px #1432410b;min-width:0;border-top:7px solid var(--teal)}
h1,h2,h3{font-family:Arial,sans-serif;line-height:1.25;color:var(--ink);scroll-margin-top:20px}h1{font-size:39px;letter-spacing:-1.3px;margin:0 0 24px}h2{font-size:27px;border-top:1px solid var(--line);padding-top:30px;margin:52px 0 22px}h3{font-size:20px;margin:30px 0 12px}
main>h2:first-of-type{font-size:22px;color:var(--teal);border:0;padding:0;margin-top:0;font-weight:normal}p{margin:16px 0}a{color:#086d81;text-underline-offset:3px;overflow-wrap:anywhere}blockquote{background:#eef6f6;border-left:4px solid var(--teal);margin:28px 0;padding:10px 24px;font-family:Arial,sans-serif;font-size:15px}
.table-scroll{overflow-x:auto;margin:26px 0}table{width:100%;border-collapse:collapse;font:12px/1.6 Arial,sans-serif}th{text-align:left;background:#183c4b;color:white;padding:11px 10px}td{border-bottom:1px solid var(--line);padding:10px;vertical-align:top;overflow-wrap:anywhere}tbody tr:nth-child(even){background:#f4f8fa}
img{display:block;max-width:100%;height:auto;margin:28px auto 12px}p:has(img)+p{font:13px/1.6 Arial,sans-serif;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:34px}code{font:12px/1.7 Consolas,monospace;background:#eff3f5;padding:2px 4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;padding:18px;background:#eff3f5}ul,ol{padding-left:26px}li{margin:8px 0}
.footer{font:12px/1.6 Arial,sans-serif;color:var(--muted);border-top:1px solid var(--line);margin-top:40px;padding-top:20px}
@media(max-width:1000px){.layout{display:block;max-width:900px}nav{position:static;max-height:260px;margin-bottom:24px}main{padding:35px 25px}h1{font-size:31px}}
@media print{@page{size:A4;margin:18mm 17mm}body{background:white;font-size:10.5pt;line-height:1.5}.masthead,nav{display:none}.layout{display:block;margin:0;padding:0;max-width:none}main{padding:0;box-shadow:none;border:0}h1{font-size:27pt}h2{font-size:18pt;break-after:avoid;margin-top:24pt;padding-top:14pt}h3{font-size:13pt;break-after:avoid}p{orphans:3;widows:3}img{max-height:215mm;object-fit:contain;break-inside:avoid}table{font-size:8pt}thead{display:table-header-group}tr{break-inside:avoid}.table-scroll{overflow:visible}a{color:inherit;text-decoration:none}blockquote{break-inside:avoid}p:has(img)+p{font-size:9pt}h2[id^="appendix"]{break-before:page}}
'''

def render(text):
    md = MarkdownIt('commonmark', {'html': True}).enable('table')
    tokens = md.parse(text)
    contents = []
    used = set()
    for i, tok in enumerate(tokens):
        if tok.type == 'heading_open':
            title = tokens[i+1].content
            slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            stem = slug
            n = 2
            while slug in used:
                slug = f'{stem}-{n}'; n += 1
            used.add(slug); tok.attrSet('id', slug)
            if tok.tag == 'h2':
                contents.append(f'<a href="#{slug}">{html.escape(title)}</a>')
    body = md.renderer.render(tokens, md.options, {})
    body = body.replace('<table>', '<div class="table-scroll"><table>').replace('</table>', '</table></div>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="description" content="Evidence-grounded tyre image pilot study: full author-review report.">'
            '<title>Tyre Study — Full Project Report</title><style>' + CSS + '</style></head><body>'
            '<header class="masthead">VIT-AP · Research report · Updated 15 September 2026 · Versioned evidence</header>'
            '<div class="layout"><nav aria-label="Contents"><strong>Report contents</strong>' + ''.join(contents) +
            '<a href="REPRODUCIBILITY.md">Reproducibility appendix</a><a href="SUBMISSION_CHECKLIST.md">Submission checklist</a>'
            '</nav><main>' + body + '<footer class="footer">Author-review edition · Local documentation build · '
            'No claim of physical wear certification or full original-plan completion.</footer></main></div></body></html>')

def main():
    from verify_project_documentation import verify_evidence
    verify_evidence()
    (REPORT / 'assets').mkdir(exist_ok=True)
    source = (REPORT / 'manuscript.source.md').read_text(encoding='utf-8')
    from build_report_geometry import build_geometry
    for key, value in {**report_tables(),**build_geometry(table)}.items():
        source = source.replace('{{' + key + '}}', value)
    refs = (REPORT / 'REFERENCES.md').read_text(encoding='utf-8').split('<!-- bibliography:start -->')[1].split('<!-- bibliography:end -->')[0].strip()
    source = source.replace('{{REFERENCES}}', refs)
    assert not re.search(r'\{\{[A-Z_]+\}\}', source), 'Unresolved authoring marker'
    plot_endpoints()
    (REPORT / 'REPORT.md').write_text(source, encoding='utf-8')
    (REPORT / 'REPORT.html').write_text(render(source), encoding='utf-8')
    files = [p for p in REPORT.rglob('*') if p.is_file() and 'evidence' not in p.relative_to(REPORT).parts
             and p.name not in {'BUILD_PROVENANCE.json', 'VALIDATION.json'}]
    scripts = [ROOT / 'scripts' / n for n in ['build_project_report.py', 'build_report_geometry.py', 'verify_project_documentation.py', 'fetch_report_evidence.py']]
    record = {'edition_date': '2026-09-15', 'status': 'full_report_refreshed_for_author_review',
              'full_original_project_complete': False, 'python': sys.version.split()[0],
              'packages': {p: importlib.metadata.version(p) for p in ['markdown-it-py', 'matplotlib', 'numpy', 'Pillow']},
              'evidence_revision': '22d5a6bc9f953ba3bf2a75919edc7db3193b317b',
              'geometry_revision':'bbe586c6f00cf12ae4cac8b2e9cb4f875b272abb',
              'geometry_manifest_sha256':hashlib.sha256((REPORT/'evidence/geometry_manifest.json').read_bytes()).hexdigest(),
              'word_count': len(re.findall(r"\b[\w’-]+\b", source)),
              'artifacts_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(files + scripts)}}
    (REPORT / 'BUILD_PROVENANCE.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Built full report: {record["word_count"]:,} words, 22 visuals. No network or training used.')

if __name__ == '__main__':
    main()
