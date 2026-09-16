"""Mechanical current-status cross-links across maintained Markdown.

Preserve historical prose, frozen downloaded evidence, generated outputs and
annotation packages. Substantive completion changes are edited separately.
"""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/n for n in subprocess.check_output(['rg','--files','-g','*.md'],cwd=ROOT,text=True).splitlines()]
inventory=[];patches=[]
for path in paths:
    relative=path.relative_to(ROOT); parts=relative.parts
    # Only maintained project documentation, never datasets/caches/execution evidence.
    maintained=(len(parts)==1 or (parts[0]=='docs' and 'evidence' not in parts)
        or (parts[0]=='prototype' and len(parts)==2) or relative.as_posix()=='notebooks/README.md')
    if not maintained:continue
    special=relative.as_posix() in {'docs/CURRENT_STATUS.md','docs/report/REPORT.md','docs/report/manuscript.source.md'}
    original=path.read_text(encoding='utf-8-sig')
    if not special and '<!-- current-status:start -->' not in original:
        link=os.path.relpath(ROOT/'docs/CURRENT_STATUS.md',path.parent).replace(os.sep,'/')
        banner='\n<!-- current-status:start -->\n> **Current status (15 September 2026):** [Completed work and remaining validation]('+link+'). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.\n<!-- current-status:end -->\n'
        first=original.splitlines()[0]
        patches.append('*** Update File: '+str(path)+'\n@@\n '+first+'\n'+''.join('+'+line+'\n' for line in banner.splitlines()))
    inventory.append(dict(path=relative.as_posix(),status='special/generated or canonical' if special else 'current_status_link_synchronised',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
if '--check' in sys.argv:
    assert not patches, 'Some maintained Markdown files lack current-status navigation'
    out=ROOT/'outputs/documentation_refresh';out.mkdir(parents=True,exist_ok=True)
    (out/'MARKDOWN_COVERAGE.json').write_text(json.dumps(dict(files=inventory,
        preserved='Frozen evidence, generated report source/output handled by builder, ignored caches and annotation packages are not mechanically rewritten'),indent=2),encoding='utf-8')
    print(f'Checked {len(inventory)} maintained Markdown files; canonical/current-status navigation present.')
else:
    print('*** Begin Patch\n'+''.join(patches)+'*** End Patch')
