"""Generate the isolated one-seed context pilot using the established launcher."""
import ast,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'analysis/build_nb11_lc.py').read_text(encoding='utf-8')
s=s.replace("SOURCE=ROOT/'tread_two/lc_pilot'","SOURCE=ROOT/'tread_two/context_pilot'")
s=s.replace("ROOT/'tread_two/lc_validation/NOTEBOOK.json'","ROOT/'tread_two/context_validation/NOTEBOOK.json'")
s=s.replace('phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb','phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb')
s=s.replace('NB11','NB13').replace('ATLAS-LC r2','ATLAS-CONTEXT r3').replace('atlas-lc-nb11','atlas-context-nb13').replace('tread_two_atlas_lc_r2','tread_two_atlas_context_r3')
s=s.replace('NB08–NB10 are completed evidence, preserved unchanged. This is a separately\nversioned hypothesis: four local-contrast texture channels, 18 distinct\ntraining-photo reference patches, source-SHA self exclusion, gentler group\nresampling and two-view appearance consistency. Its accuracy is **unmeasured**.',
'''NB08–NB12 are completed evidence, preserved unchanged. Both previous classifiers
discard band order at pooling. A train-only eight-tyre structural check confirmed
this invariance; it does not prove layout caused the classification failures.
This separate hypothesis keeps validation-selected LC epoch7 fully frozen and
learns a bounded spatial correction using a warm-start patch encoder, explicit
2D positions and two four-head attention blocks. The correction head starts at
zero, so initial predictions exactly reproduce LC. Accuracy is **unmeasured**.''')
s=s.replace('There are 81,274 trainable / 1,453,433 total parameters. No workstation changes.',
 'There are **231,555 trainable / 1,684,988 total parameters**. The old1,453,433-parameter LC model stays frozen. No workstation changes.')
s=s.replace('This verifies every prepared-dataset checksum, role overlay, full bank-source\nSHA and frozen anchor SHA.',
 'This verifies every prepared-dataset checksum, role overlay, full bank-source\nSHA, frozen anchor SHA and exact selected LC checkpoint SHA. Both upstream\nweights are downloaded from pinned HF objects; dataset access stays on Kaggle.')
s=s.replace('all24 CSV families','all27 CSV families')
s=s.replace('The frozen dense anchor runs once; the shared encoder sees both LC atlases and\nthe same literal reference bank. Objective:',
'''The frozen dense anchor runs once; frozen LC supplies its prior and exact
transformed LC tiles. A trainable copy of its patch encoder supplies32tokens.
Coverage, fixed2D positions and two96D/4-head context blocks predict bounded
per-patch logit corrections. The reference bank remains part of the frozen prior.
Objective:''')
s=s.replace('0.5*(CE(A)+CE(B)) +0.1*JS(PA,PB) +0.05*sum(alpha*(1-cos(qA,qB)))',
 '0.5*(CE(A)+CE(B)) +0.1*JS(PA,PB) +0.05*sum(alpha*(1-cos(contextA,contextB))) +0.02*mean(total_correction_squared)')
s=s.replace('53,342 planned raw observations:47,040 online training views,4,860 validation,\n206 selected/final,1,236 fixed diagnostic observations.',
 '53,754 planned raw observations:47,040 online training views,4,860 validation,\n206 selected/final,1,648 fixed diagnostic observations.')
s=s.replace('gradient summaries and resource readings. Native state keeps exact recovery;',
'''gradient summaries and resource readings. Three additional CSV families keep
every weighted context correction, per-head attention summaries on every view,
and full selected/final identity attention matrices. Native captures retain
base/context embeddings and the separate reference/correction accounting.
Full per-update attention matrices are not saved; their summaries are retained
and this limit is explicit. Native state keeps exact recovery;''')
s=s.replace('phase3/tread-two/atlas-lc-r2/fit/','phase3/tread-two/atlas-context-r3/fit/')
s=s.replace('Six predeclared diagnostics at both endpoints: grayscale-both, brightness0.8/1.2,\ngamma0.8/1.2, top-signed-patch occlusion.',
'''Eight predeclared diagnostics at both endpoints: grayscale-both,
brightness0.8/1.2,gamma0.8/1.2,top-signed-patch occlusion,reverse band order and
reverse columns within bands. The last two rearrange the saved local token
correspondence; they are structural probes,not realistic new tyre photographs.
The source-patch mapping is retained for every observation.''')
s=s.replace('Native tensor weights/maps','Native tensor weights/maps')
s=s.replace('points; signed reference evidence exactly accounts for classification logits.', 'points. Frozen reference evidence plus the separately recorded spatial correction\nexactly account for classification logits. Attention is an internal diagnostic,\nnot causal physical-wear truth.')
s=s.replace('single head','spatial correction').replace('HTTP five-minute budgets and separate hourly commit guard\nare included natively in this r2','HTTP five-minute budgets and separate hourly commit guard\nare included natively in this r3')
s=s.replace('Completed one measured ATLAS-LC candidate.','Completed one measured ATLAS-CONTEXT candidate.')
s=s.replace('53,754 planned raw observations:47,040','53,754 planned raw observations: 47,040').replace('views,4,860','views, 4,860').replace('206 selected/final,1,648','206 selected/final, 1,648')
s=s.replace('old1,453,433','old 1,453,433').replace('supplies32tokens','supplies 32 tokens').replace('fixed2D','fixed 2D').replace('two96D/4-head','two 96D/4-head').replace('all27 CSV','all 27 CSV')
s=s.replace('brightness0.8/1.2,gamma0.8/1.2,top-signed-patch occlusion,reverse band order','brightness 0.8/1.2, gamma 0.8/1.2, top-signed-patch occlusion, reverse band order').replace('probes,not','probes, not')
# Final report explicitly separates the frozen case explanation from learned context.
s=s.replace("The model remains research-only:","The model remains research-only:")
ast.parse(s)
exec(compile(s,str(ROOT/'analysis/build_nb13_context.py'),'exec'),{'__file__':str(ROOT/'analysis/build_nb13_context.py'),'__name__':'__main__'})
