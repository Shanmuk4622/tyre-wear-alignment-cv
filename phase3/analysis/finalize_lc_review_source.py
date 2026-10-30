"""Remove superseded v1 report prose from the adapted r2 renderer."""
import sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
folder=Path(__file__).resolve().parents[1]/'tread_two/lc_review'
p=folder/'render_lc_review.py';s=p.read_text(encoding='utf-8')
a=s.index("    (out/'README.md').write_text(");b=s.index('\n\n\ndef write_r2_notes',a)
s=s[:a]+'    write_r2_notes(out)\n'+s[b:]
s=s.replace('TREAD-TWO / ATLAS ·','TREAD-TWO / ATLAS-LC r2 ·').replace('Grayscale changes both query and bank. Brightness changes query only.','LC uses channel-mean luminance, so grayscale stability is built into this representation. Brightness and gamma change the query only.')
p.write_text(s,encoding='utf-8')
p=folder/'review_lc.py';s=p.read_text(encoding='utf-8').replace('from render_lc_review import render,write_r2_notes','from render_lc_review import render').replace('    write_r2_notes(Path(args.out))\n','');p.write_text(s,encoding='utf-8')
print('R2 report now owns its prose directly; no intermediate v1 README')
