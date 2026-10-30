"""Document the supplied repeat without replacing the first completed audit."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,atomic
def write(name,text):(ROOT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def append(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if text.strip() not in s:write(name,s.rstrip()+'\n\n'+text)

def main():
 latest=read(ROOT/'data_gate_validation/nb07_reruns/LATEST.json');folder=Path(latest['path']);comparison=read(folder/'COMPARISON.json');r=read(folder/'VERIFICATION.json');first=read(ROOT/'data_gate_validation/nb07_publication/VERIFICATION.json')
 assert comparison['status']==r['status']=='passed' and comparison['same_code_cells'] and comparison['first_archive_preserved']
 assert comparison['same_publication_contract'] and comparison['same_data_run_id'] and comparison['same_summary'] and comparison['same_environment'] and comparison['unchanged_payload_files']==45 and comparison['changed_payload_files']==[]
 assert comparison['reported_uploaded_objects']==0
 rel=folder.relative_to(ROOT).as_posix();url=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/{r["revision"]}/{r["destination"]}'
 write('phase3NB07RepeatCheck.md',f'''# NB07 repeat checked — no further run needed now

**29 October 2026.** The latest supplied notebook completed successfully.
All five code cells passed and the independently verified HF publication has
the same results as the first execution. **Do not rerun NB07 again with the
current inputs.** Earlier setup/run instructions are historical instructions
for executing the audit, not a request to repeat a completed one.

The user reran after interpreting earlier guidance as requiring another run.
This record makes the next action explicit: review identities and obtain actual
independent intake; rerun the audit only when those inputs or its policy change.

## First execution versus supplied repeat

| Item | First verified execution | Latest verified repeat |
|---|---|---|
| Saved notebook bytes | {first['executed_notebook_bytes']:,} | {r['executed_notebook_bytes']:,} |
| Notebook SHA-256 | `{first['executed_notebook_sha256']}` | `{r['executed_notebook_sha256']}` |
| Code cells | Five successful cells | Five successful cells; code exactly identical |
| HF revision | `{first['revision']}` | `{r['revision']}` |
| Publication contract | `{first['publication_contract']}` | Same contract |
| Data run | `{first['run_id']}` | Same run identity |
| Scientific payload | 45 files + selecting manifest | All 45 payload hashes unchanged; 46 logical files verified |
| Newly uploaded payload objects | 45 | **0 — verified reuse** |
| Structured CSV | 22 tables / 7,614 rows | Identical to the first publication |
| Total published CSV files | 26, including raw controls | 26 |
| New data / model fitting | None | None |

The new revision records another publication/ownership cycle; it is not
additional training, new metrics or another set of unique payload files.
Notebook bytes differ because execution outputs/metadata differ. Code,
environment, input hashes, source contract, summary and payload are unchanged.

```mermaid
flowchart LR
    A["First NB07 complete<br/>45 payloads + manifest"] --> B["User reruns unchanged audit"]
    B --> C["Same code / inputs / environment<br/>same run and publication contract"]
    C --> D["HF reuses all payload objects<br/>0 new uploads; new receipt revision"]
    D --> E["Independent recheck passes<br/>stop repeating unchanged NB07"]
    E --> F["Next: identity review + real new intake"]
```

## Verification and preservation

| Evidence | Result |
|---|---|
| [Repeat comparison]({rel}/COMPARISON.json) | All 45 payload SHA-256 values unchanged; new receipt and notebook archived separately |
| [Independent remote check]({rel}/VERIFICATION.json) | Manifest, every Git/LFS identity, all file SHA-256 values and released writer verified at the new revision |
| [Latest published snapshot]({url}) | Pinned repeat receipt; original publication remains available |
| First archive | Preserved byte-for-byte in the original completed-audit folder |
| Current notebook | Preserved as supplied; no cells executed or regenerated locally |

Public reads reused verified cached files, then checked them against the new
pinned remote tree. The verifier made zero remote writes and read no alternate
credentials. Earlier and repeated execution receipts are kept separately.

## Readiness=false is not a request to rerun

The audit passes source integrity and historical grouping. It still reports
**prospective_training_ready=false** because no new independent tyre intake,
actual subject minima or frozen prospective policy were supplied. There are
12 explicit readiness gaps, just as before. Repeating the same inputs cannot
fill these gaps.

| Situation | Next action |
|---|---|
| Current notebook and inputs | **No more NB07 execution needed** |
| Known clip identity with supporting records | Use the [review/export tool](phase3CollectionReview.md); keep uncertain parents unresolved |
| Actual additional independent tyres/images | Register native files, labels, sessions and annotations using the [intake guide](data_intake/README.md) |
| Reviewed mappings / intake / policy actually change | Preserve this completed audit; run NB07 once against the changed inputs and retain its new contract |
| Considering another model fit | First freeze actual cohort, role/class minima, selection and task/failure criteria |

The local comparison/capture pack remains usable. No new notebook or model run
is required to inspect it. [Original completion and scientific insights](phase3NB07Completion.md).
''')
 atomic(ROOT/'data_gate_outputs/PUBLISHED_LATEST.json',dict(status='independently_verified_repeat',revision=r['revision'],destination=r['destination'],publication_contract=r['publication_contract'],run_id=r['run_id'],executed_notebook_sha256=r['executed_notebook_sha256'],published_files=46,newly_uploaded_payload_objects=0,all_45_payloads_unchanged=True,first_publication_preserved=True,verification=rel+'/VERIFICATION.json',comparison=rel+'/COMPARISON.json',rerun_needed_now=False))
 notice='''**Repeat checked, 29 October 2026:** the latest supplied NB07 also passes.
All 45 payload files are unchanged and HF uploaded zero new payload objects.
**No further NB07 run is needed now.** A false readiness flag requires actual
new intake/protocol work, not another identical execution. [Repeat check](phase3NB07RepeatCheck.md).'''
 for name in ('README.md','phase3Progress.md','phase3Understanding.md','phase3OpenItems.md','phase3DataGrowth.md','phase3DataRetentionPolicy.md'):
  p=ROOT/name;s=p.read_text(encoding='utf-8')
  if notice not in s:write(name,s.replace('<!-- END NB07 CURRENT -->',notice+'\n<!-- END NB07 CURRENT -->',1))
 for name in ('phase3NB07Completion.md','phase3NB07DataGate.md','phase3ProspectiveDataGate.md','phase3NextSteps.md'):
  p=ROOT/name;s=p.read_text(encoding='utf-8');heading,rest=s.split('\n',1)
  if notice not in s:write(name,heading+'\n\n'+notice+'\n\n'+rest.lstrip())
 # Exact identity table refers to the first archive; the current file is the repeat.
 p=ROOT/'phase3NB07Completion.md';s=p.read_text(encoding='utf-8');s=s.replace('| Executed notebook SHA-256 |','| First executed notebook SHA-256 |').replace('| Notebook | Cells 1–5, zero saved errors; 232,871 bytes |','| First notebook | Cells 1–5, zero saved errors; 232,871 bytes |');write(p.name,s)
 append('phase3EvidenceRegister.md',f'''## NB07 supplied repeat — 29 October 2026

| Item | Verified result |
|---|---|
| Repeat notebook | SHA `{r['executed_notebook_sha256']}`; {r['executed_notebook_bytes']:,} bytes; five successful cells |
| New receipt revision | `{r['revision']}` |
| Code / run / environment / contract | Unchanged from first publication |
| Payload comparison | 45/45 file SHA-256 values unchanged; 0 newly uploaded objects; 46 logical files verified |
| [Comparison receipt]({rel}/COMPARISON.json) | Both archives preserved, source summary/readiness unchanged |
| [Remote verification]({rel}/VERIFICATION.json) | New pinned manifest/Git/LFS/SHA-256 and released writer checked with public reads only |

[Repeat check and explicit next action](phase3NB07RepeatCheck.md). No additional
unchanged-input execution is needed. New publication metadata do not add
scientific observations or subjects.''')
 append('phase3SessionLog.md',f'''## 29 October 2026 — recheck user-supplied NB07 repeat

The user reran after interpreting earlier wording as requiring it, then asked
for another check and to continue. Preserved the first archive and created a
separate hash-named repeat archive. Five cells pass; code is identical. The
new pinned receipt is `{r['revision']}` with **0 uploaded objects**. Independent
remote verification confirms the same contract/run/environment, summary and
all 45 payload hashes. There are still 46 logical published files, 22 structured
CSV tables / 7,614 rows, 12 original tyres, no new intake and false prospective
readiness. This was a repeated CPU audit, not model training.

Clarified active documentation: **do not rerun NB07 with current inputs**.
Actual reviewed mappings, independent acquisition and protocol changes are
needed before another data-gate run is useful. The local identity/capture tool
remains the next action. Updated audit inspection to accept ordered positive
execution counters and avoid rewriting identical receipts; cached file copies
are reverified against the new remote tree. No training/runtime source or
executed notebook was modified. [Repeat report](phase3NB07RepeatCheck.md).''')
 append('phase3DocumentationInventory.md','''## NB07 repeat record — 29 October 2026

[phase3NB07RepeatCheck.md](phase3NB07RepeatCheck.md) distinguishes the original
and repeat notebook/revision, confirms identical payload hashes and zero new
uploads, and states that unchanged NB07 must not be rerun. Active status and
next-step notes now link that clarification. Both executed archives are retained.''')
 p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8')
 if 'P3-D052' not in s:write(p.name,s.replace('\n\nAppend a new decision','\n| P3-D052 | 2026-10-29 | Preserve and verify the repeated NB07 separately; explicitly stop unchanged-input reruns | Same contract/run and 45 payload hashes, zero new uploads; false readiness requires actual intake/protocol changes, not another identical audit; [repeat report](phase3NB07RepeatCheck.md) |\n\nAppend a new decision'))
 print('NB07 repeat documented; original archive and publication records retained; no further run needed now.')

if __name__=='__main__':main()
