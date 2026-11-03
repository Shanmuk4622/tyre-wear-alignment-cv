"""Build the current NB13A delivery; the historical r1 generator is archived."""
import runpy,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
runpy.run_path(str(Path(__file__).with_name('build_nb13_recovery_r2.py')),run_name='__main__')
