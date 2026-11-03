"""Create isolated package and viewer checks; leave the completed LC checks intact."""
import sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'analysis/check_nb11_package.py').read_text()
s=s.replace('tread_two/lc_pilot','tread_two/context_pilot').replace('tread_two/lc_validation','tread_two/context_validation')
s=s.replace('phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot','phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot')
s=s.replace('==31','==33').replace('embedded_files=31','embedded_files=33').replace('scientific_CSV_families_match=24','scientific_CSV_families_match=27').replace('nb11_extract_','nb13_extract_')
(ROOT/'analysis/check_nb13_package.py').write_text(s,encoding='utf-8')
s=(ROOT/'analysis/test_lc_viewer.cjs').read_text()
s=s.replace("const raw=", "assert.match(html,/Frozen case-branch/);assert.match(html,/context_delta/);assert.match(html,/base_logits/);\nconst raw=")
(ROOT/'analysis/test_context_viewer.cjs').write_text(s,encoding='utf-8')
print('Prepared context-only package and interactive logic checks.')
