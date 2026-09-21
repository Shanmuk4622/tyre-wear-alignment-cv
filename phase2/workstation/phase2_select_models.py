"""Switch the workstation between verified Phase 2 and preserved legacy models."""
import argparse,pathlib,json,hashlib
root=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('mode',choices=['phase2','legacy']);a=p.parse_args()
flag=root/'ACTIVE'
if a.mode=='phase2':
 registry=json.loads((root/'registry.json').read_text())
 for item in registry.values():
  weights=root/'models'/item['job']['model']/'weights.pt'
  assert hashlib.sha256(weights.read_bytes()).hexdigest()==item['sha256']
 flag.write_text('Phase 2 verified exports enabled\n')
else:
 if flag.exists():flag.rename(root/'INACTIVE') if not (root/'INACTIVE').exists() else flag.unlink()
print('Selected '+a.mode+'. Close and reopen Tread Station to apply.')
