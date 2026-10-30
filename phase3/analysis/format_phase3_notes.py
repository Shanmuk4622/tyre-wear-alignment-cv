"""Polish maintained prose spacing while preserving code, hashes and link targets."""
import re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
FILES=['README.md','phase3NB04Completion.md','phase3R2PairedPilot.md','phase3R2EvidenceSchema.md','phase3Progress.md','phase3Understanding.md',
 'phase3TrainingEvidenceGate.md','phase3NextSteps.md','phase3OpenItems.md','phase3EvidenceRegister.md','phase3SessionLog.md','phase3Decisions.md',
 'phase3RepositoryMap.md','phase3DocumentationInventory.md','phase3DataRetentionPolicy.md','phase3DataGrowth.md']
PROTECTED=re.compile(r'(`[^`]*`|\]\([^)]*\))')

def prose(text):
 text=re.sub(r'\b(all|with|both|two|one|each|over|of|about|at|below|above|every|seed|no|is)(?=\d)',r'\1 ',text,flags=re.I)
 text=re.sub(r'(?<=\d)(?=(?:bytes|files|epochs|minutes|seconds|hours|GiB|MiB|GB|TB|updates|conditions|sources?|CSV|fields|scientific|core|native|published|direct|frozen|columns?|records?|samples?|steps?|original|paired|inputs?|matching|tyres?|models?|images?|surrogate|local)\b)',' ',text)
 text=re.sub(r'October(?=20\d\d)','October ',text)
 text=re.sub(r'\b(PyTorch|NumPy|Pillow)(?=\d)',r'\1 ',text)
 text=re.sub(r'(?<!\d),(?=\d)',', ',text)
 return text

def main():
 changed=[]
 for name in FILES:
  path=ROOT/name;original=path.read_text(encoding='utf-8');result=[];fenced=False
  for line in original.splitlines(keepends=True):
   if line.lstrip().startswith('```'):fenced=not fenced;result.append(line);continue
   if fenced:result.append(line);continue
   parts=PROTECTED.split(line);result.append(''.join(part if i%2 else prose(part) for i,part in enumerate(parts)))
  output=''.join(result)
  if output!=original:
   tmp=path.with_name(path.name+'.format.tmp');tmp.write_text(output,encoding='utf-8');tmp.replace(path);changed.append(name)
 print('Prose spacing polished in',len(changed),'maintained notes; code and link destinations preserved.')

if __name__=='__main__':main()
