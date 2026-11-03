"""Remove only unintended line-ending churn in previously untouched history docs."""
import subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[2]
names=['TREAD_ONE_DESIGN.md','TREAD_ONE_EXPERIMENT.md','TREAD_ONE_RUN_GUIDE.md','TREAD_ONE_VALIDATION.md',
       'phase3ClassifierDiagnostic.md','phase3ClassifierDiagnosticResults.md','phase3CompletionAudit.md','phase3DocumentationConflicts.md','phase3Environment.md',
       'phase3NB02Completion.md','phase3NB02Repair.md','phase3NB03Completion.md','phase3NB04Completion.md','phase3NB05Repair.md',
       'phase3NB08CompletionStatus.md','phase3NB08Repair.md','phase3NB09Completion.md','phase3TreadOneWorkstation.md']
restored=[]
for name in names:
    path=ROOT/'phase3'/name;relative=path.relative_to(ROOT).as_posix()
    result=subprocess.run(['git','show','HEAD:'+relative],cwd=ROOT,capture_output=True)
    if result.returncode:continue
    current=path.read_bytes();original=result.stdout
    if current!=original and current.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'):
        path.write_bytes(original);restored.append(name)
print(dict(line_endings_only_restored=restored,semantic_content_changed=False,index_changed=False))
