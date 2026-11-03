"""Update current navigation only; preserve all source evidence and old history."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prepend(name,tag,body):
    body=re.sub(r'\b(covers|is|checks|from|all|every|and|gain|against|rank|by|through|epoch|view|fit|across|snapshot|selected|step|scope|over)(?=\d)',r'\1 ',body)
    body=re.sub(r'(?<=\d)(families|executions|epochs|updates|images|components|fit|files|local|source)\b',r' \1',body)
    body=body.replace('Phase2','Phase 2').replace('Phase3','Phase 3')
    path=ROOT/name;old=path.read_text(encoding='utf-8');block='<!-- '+tag+' -->\n'+body.rstrip()+'\n<!-- END_'+tag+' -->'
    if '<!-- '+tag+' -->' in old:new=re.sub(re.escape('<!-- '+tag+' -->')+r'.*?'+re.escape('<!-- END_'+tag+' -->'),lambda _:block,old,flags=re.S)
    else:
        title,rest=old.split('\n',1);new=title+'\n\n'+block+'\n'+rest
    if new!=old:path.write_text(new,encoding='utf-8',newline='\n')
current='''## Latest analysis — whole-project retrospective through NB24

**2 November 2026:** [Full project analysis](phase3ProjectRetrospective.md)
connects original research, Phase2, TREAD-ONE, ATLAS/LC/context, GEO, TRACE,
both workstations and manuscript scope. The inventory covers236 maintained
Markdown documents. Phase3 measured bookkeeping is7 fit families /10 fit
executions /600 epochs /29,400 updates, excluding engineering and avoiding
double-counting NB13A or notebook masters.

A new saved-only [TRACE component review](phase3TRACEComponentReview.md)
checks570 image records /10 conservative components from444 verified CSV
inputs. Every image beats all3 masked controls for final clean view0 and its
recorded plan; validation gain21.63%, historical-test gain17.85%. Per-position
local tokens are more varied than pooled summaries. This does not alter
selected=None, establish a pooling cause or validate physical wear.

[Interactive evidence explorer](retrospective_20261102/Project_Evidence_Explorer.html)
and10 direct CSV inventories/derived tables are local; no new HF publication,
fit, forward, photo or workstation change. Next investigate local evidence and
aggregation under existing-data controls before justifying another model.
The completed NB24 status below remains authoritative for that run.'''
for name in ('AGENTS.md','README.md','phase3Progress.md','phase3Understanding.md','phase3NextSteps.md','phase3OpenItems.md'):
    prepend(name,'PROJECT_RETROSPECTIVE_20261102',current)
records={
'phase3Decisions.md':'''## Whole-project retrospective — 2 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D181 | Reconcile by experiment scope and later controlling evidence | Original study /Phase2 /Phase3 /manuscript are separate;236-document pre-analysis inventory |
| P3-D182 | Count fits rather than notebook numbers |7 measured families /10 executions; NB13A continues NB13, masters wrap stages |
| P3-D183 | Preserve every original validation selection and research rejection | No test-selected MobileNet seed swap; GEO/TRACE selected=None |
| P3-D184 | Add saved-only TRACE component/image diagnostics |570 images /10 components;444 full-hash inputs; no new forward or fit |
| P3-D185 | Distinguish local reconstruction utility from pooled diversity | All images beat3 controls for one recorded plan; per-tile statistics do not prove a causal pooling fix |
| P3-D186 | Prioritize supported local evidence and aggregation diagnosis | Existing photos only; no automatic new fit, workstation or manuscript change |
''',
'phase3SessionLog.md':'''## 2 November 2026 — whole-project analysis requested

1. Inventoried236 maintained Markdown documents and reconciled original-study, Phase2, Phase3, application and manuscript scopes with40 controlling source records.
2. Built measured-fit and selected/final endpoint ledgers; distinguished10 Phase3 fit executions from notebook counts, reviews, engineering and continuation.
3. Read/checksummed444 actual saved CSV bodies and calculated570 final-view image comparisons,10 component summaries,3 role summaries and6 retained tile-statistic summaries.
4. Reconciled role means to original NB24 endpoints. Both validation/test components improve reconstruction; original global diversity rejection remains unchanged.
5. Added whole-project and component Markdown reports, two scientific figures,10 CSV tables and a standalone interactive explorer. New derivatives are local, not HF-published.
6. Preserved source evidence, completed notebooks, both workstations and all original selections. No model forward/training, new photo, network/HF write or paper rewrite.
''',
'phase3EvidenceRegister.md':'''## Whole-project retrospective evidence

| Record | Scope |
|---|---|
| [Markdown inventory](retrospective_20261102/MARKDOWN_INVENTORY.csv) |236 pre-analysis maintained documents, bytes/hashes and scope |
| [Source evidence](retrospective_20261102/SOURCE_EVIDENCE.csv) |40 controlling report/data/state inputs |
| [Actual consumed science](retrospective_20261102/CONSUMED_SCIENCE_INPUTS.csv) |444 full-hash saved CSV bodies for the new TRACE analysis |
| [Analysis receipt](retrospective_20261102/ANALYSIS.json) |570 images /10 components /zero fitting, forwards, network calls or remote writes |
| [Fit ledger](retrospective_20261102/PHASE3_FIT_LEDGER.csv) |7 families /10 measured fit executions;600 epochs /29,400 updates |
| [Explorer and dictionary](retrospective_20261102/README.md) |Local new derivatives; original raw evidence already on HF |
''',
'phase3DocumentationInventory.md':'''## Whole-project retrospective additions

| Document | Purpose |
|---|---|
| [Project retrospective](phase3ProjectRetrospective.md) |All-stage results, causal limits, dataset/task support, applications/report and next research |
| [New TRACE component review](phase3TRACEComponentReview.md) |Per-image/component utility and retained local-versus-pooled statistics |
| [Evidence package dictionary](retrospective_20261102/README.md) |10 direct CSV tables, provenance, units, missingness and local/publication boundary |
| [Interactive evidence explorer](retrospective_20261102/Project_Evidence_Explorer.html) |Family navigation, original role/component bars and searchable570-image diagnostics |
''',
'phase3RepositoryMap.md':'''## Whole-project retrospective locations

| Location | Role |
|---|---|
| `phase3ProjectRetrospective.md` |Current synthesis across scopes; no completed experiment redefined |
| `phase3TRACEComponentReview.md` |New bounded saved-only analysis |
| `retrospective_20261102/` |Compact direct CSVs, input hashes, figures, offline explorer and analysis receipt |
| `analysis/*project_retrospective.py`, `analysis/build_retrospective_explorer.py` |Read-only numerical/report generation; no training, inference or network |
''',
'phase3DocumentationConflicts.md':'''## Retrospective reconciliation — 2 November 2026

| Apparent conflict | Controlling interpretation |
|---|---|
| Root September “current” versus November Phase3 | Different scope/date; use the relevant exact artifact and completion receipt |
|24 notebook numbers imply24 trained models | Phase3 measured ledger records10 fit executions across7 families; engineering/review/recovery/master stages are separate |
| NB13A adds a model /NB24 adds another independent NB22 fit | Continuation /automatic wrapper of the same frozen fit |
| Strong Dice means wear classifier is solved | Dense localization and proxy classification have distinct targets and observed outcomes |
| TRACE failed diversity therefore learned no useful texture | New final-view audit finds every image beats all3 controls; global selection still rejects0/60 eligible epochs |
| Rich telemetry adds data support | Derived rows/repeated views are not new physical tyres or physical labels |
| New retrospective CSVs already saved on HF | They are local derivatives; only original raw evidence has earlier verified HF publication |
| Local versus pooled rank proves a pooling defect | Different statistics/support; motivates a hypothesis, not a causal remedy or gate relaxation |

[Full synthesis](phase3ProjectRetrospective.md) and [bounded new analysis](phase3TRACEComponentReview.md)
preserve all original snapshots and selections. Existing-photo scope supersedes
historical new-intake instructions; it does not remove generalization limits.
'''}
for name,body in records.items():prepend(name,'RETROSPECTIVE_RECORD',body)
path=ROOT/'.gitignore';old=path.read_text(encoding='utf-8');line='!/retrospective_20261102/figures/*.png'
if line not in old:path.write_text(old.rstrip()+'\n\n# Curated whole-project analysis figures only.\n'+line+'\n',encoding='utf-8',newline='\n')
print('Current records linked to the retrospective; historical evidence preserved.')
