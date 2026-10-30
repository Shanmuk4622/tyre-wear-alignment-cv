"""Adapt the preserved v1 post-hoc audit into a separately versioned r2 review."""
import ast,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'tread_two/review';NEW=ROOT/'tread_two/lc_review'
NEW.mkdir(exist_ok=True)
assert not (NEW/'review_lc.py').exists(),'Initial scaffold only; preserve delivered/reviewed sources and edit a new reviewed revision.'
def change(s,a,b):assert a in s,a[:90];return s.replace(a,b)
s=(OLD/'review_atlas.py').read_text(encoding='utf-8')
for a,b in [('82e7e1c5774f54c786571cdef7b3b60c1e96630f','3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8'),('fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd','ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28'),('phase3/tread-two/atlas-v1/fit/','phase3/tread-two/atlas-lc-r2/fit/'),('29410','53342'),('train=23520','train=47040'),('diagnostics=824','diagnostics=1236'),('1030','1442'),('1030','1442'),('1030','1442'),('1030','1442'),('1030','1442'),('824 diagnostic','1236 diagnostic'),('All378','All386'),('23520 draws','23520 paired draws'),('len(ex)==23520','len(ex)==47040'),('==378','==386'),('for epoch in [18,60]','for epoch in [7,60]'),("('selected',18,882)","('selected',7,343)"),('==18','==7'),('epoch18','epoch7'),('selected_epoch=18','selected_epoch=7'),('from render_atlas_review import render','from render_lc_review import render')]:s=s.replace(a,b)
s=change(s,"assert not p.loc[p.stage=='train','is_bank'].astype(int).any()","assert source.loc[source.role=='train','is_query'].astype(int).eq(1).all()")
s=change(s,"save('class_metrics',class_rows);save('confusion',confusion_rows)","save('class_metrics',class_rows);save('confusion',confusion_rows)")
# The two online view subcohorts remain separate and also receive a combined epoch denominator.
s=change(s,"summary=save('cohort_metrics',rows)","rows += [dict(stage='train',endpoint='paired_online_combined',variant='paired_views_combined',epoch=e,**scores(g)) for e,g in p[p.stage=='train'].groupby('epoch')]\n    summary=save('cohort_metrics',rows)")
s=change(s,"if row.stage=='train': endpoint='online_before_update'","if row.stage=='train': endpoint,variant='paired_online_combined','paired_views_combined'")
s=change(s,"assert set(validation.prediction.unique())=={1,2} and not validation[validation.label==0].correct.any()\n    assert all(summary.loc[summary.stage=='validate','balanced_accuracy_observed']==.5)","assert math.isclose(selection.validation_ba_observed.max(),.5625)\n    assert validation[validation.label==0].correct.any()")
s=change(s,"assert exposures.groupby('update').size().eq(8).all()", "assert exposures.groupby('update').size().eq(16).all()")
s=change(s,"exposures[['brightness','contrast','saturation']].ge(.9-1e-6).all().all() and exposures[['brightness','contrast','saturation']].le(1.1+1e-6).all().all()", "exposures[['brightness','contrast','gamma']].ge(.8-1e-6).all().all() and exposures[['brightness','contrast','gamma']].le(1.2+1e-6).all().all()")
s=change(s,"checks.append('All386", "checks.append('All386")
# Review every actual extra scientific seam before computing the derived inventory.
s=change(s,"save('input_inventory',[", "from paired_review import extend\n    extend(inputs,out,p,source,exposures,updates,coef,save,checks)\n    save('input_inventory',[")
s=s.replace('Inputs are immutable NB08','Inputs are immutable NB11').replace('never alter NB08','never alter NB11')
(NEW/'review_lc.py').write_text(s,encoding='utf-8')
s=(OLD/'render_atlas_review.py').read_text(encoding='utf-8')
s=s.replace("ax.axvline(18,", "ax.axvline(7,")
s=s.replace("variants=['identity','brightness_080','brightness_120','grayscale_both','occlude_top_patch']","variants=['identity','brightness_080','brightness_120','gamma_080','gamma_120','grayscale_both','occlude_top_patch']")
s=s.replace('np.arange(5)','np.arange(7)').replace('range(5)','range(7)')
s=change(s,"['Original','Dim ×0.8','Bright ×1.2','Gray both','Top patch']","['Original','Dim ×0.8','Bright ×1.2','Gamma0.8','Gamma1.2','Gray both','Top patch']")
s=s.replace('epoch 18','epoch 7').replace('epoch18','epoch7').replace('NB08','NB11').replace('NB09','NB12').replace('1,030','1,442').replace('ATLAS completed pilot','ATLAS-LC completed pilot')
s=change(s,'<option value="occlude_top_patch">','<option value="gamma_080">Query gamma0.8</option><option value="gamma_120">Query gamma1.2</option><option value="occlude_top_patch">')
# Old descriptive prose is replaced by actual r2 result prose after rendering.
s += '''\n\ndef write_r2_notes(out):
    out=Path(out)
    (out/'README.md').write_text("""# ATLAS-LC r2 — completed evidence review

**Saved evidence only: zero fitting, zero model forwards, no new checkpoint selection.**
The one seed completed60epochs/2,940updates. Validation chose epoch7 at56.25% BA.
Selected and final historical endpoints both have50% BA,39.806% accuracy:
62low→mid and41high→high. No held-out mid target exists.
Selected NLL0.96127 improves over preserved v1's1.17830; final r2 NLL1.36987.
Lower probability loss does not mean low-tyre recognition or validated calibration.

All386training images,23,520draws/two views each,full source-SHA permissions,
per-draw paired losses,gradient summaries and reference/projection evidence are
audited. Grayscale stability is built into LC,not learned causal wear evidence.
Brightness/gamma/occlusion probes and confidence changes remain inspectable.
The selected/final image atlas thumbnails are base-image thumbnails; intervened
numbers are from saved probes. No new probe images are synthesized.

| Open | Purpose |
|---|---|
| [Interactive review](ATLAS_Failure_Explorer.html) | All1,442 endpoint/probe cases,paired scores,signed patches and18literal references |
| [CSV inventory](CSV_INDEX.md) | Derived table counts and complete field dictionary |
| [Methods](METHODS.md) | Denominators,formulas,selection and scientific limitations |
| [Training/confusion](figures/fit_and_confusion.png) | Online views and validation history with observed failures |
| [Interventions](figures/probes_and_affinities.png) | All six saved diagnostics and pooled reference affinity |
| [Exposure/calibration](figures/exposure_embedding_calibration.png) | Correlated repeated observations,reference embeddings and raw confidence |
| [Paired learning](figures/paired_learning.png) | CE,JS,embedding consistency and view disagreement |
| [v1 versus r2](v1_r2_comparison.csv) | Descriptive preserved endpoint comparison; changed protocols,not a paired single-factor experiment |

The candidate is not promoted: low endpoint recognition remains0/62 despite
better validation/probability scores and appearance stability. Complete this
evidence review/publication before defining another model experiment. Both
workstations remain unchanged. Existing photos only; no new physical tyres,
depth/safety truth or independent confirmation is claimed.
""",encoding='utf-8')
    with (out/'METHODS.md').open('a',encoding='utf-8') as f:
        f.write("""\n## R2 paired training and source permissions

Training observations are two views per draw. The combined online epoch metric
has784view observations per epoch; view-specific metrics have392. All retain
pre-update state ownership. CE/JS/weighted embedding losses are independently
reconstructed from both saved predictions and patch distances; joint update loss
is the mean of eight draw totals. Seeds regenerate brightness/contrast/gamma
factors; no fresh tensor or model inference is used.

All386train photos/frames are eligible queries. Full source SHA suppresses a
matching literal reference and scales its reference class by6/n_allowed,using
no query target. The held-out endpoint has no identical bank source and all18
references allowed. Thus endpoint reference sums use base signed coefficients;
training SHA permissions/scales are audited separately for every observation.
Greyscale invariance follows LC's RGB mean; it is not learned improvement.

The v1 comparison uses pinned NB09 cohort metrics on the same historical
endpoint; r2 additionally has two gamma probes unavailable in v1. Absent v1 rows
remain absent. Changed bank/query/sampler/objective and representation prevent
single-factor causal attribution. Confidence risk curves are descriptive; no
threshold is selected from this endpoint for deployment.
""")
'''
s=change(s,"atomic(Path(args.cache)/'REVIEW_READ_STATUS", "atomic(Path(args.cache)/'REVIEW_READ_STATUS") if False else s
(NEW/'render_lc_review.py').write_text(s,encoding='utf-8')
s=(NEW/'review_lc.py').read_text(encoding='utf-8');s=change(s,'from render_lc_review import render','from render_lc_review import render,write_r2_notes');s=change(s,'render(Path(args.out),inputs)','render(Path(args.out),inputs)\n    write_r2_notes(Path(args.out))')
(NEW/'review_lc.py').write_text(s,encoding='utf-8')
for p in NEW.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print('Separate LC review scaffold written; original v1 review preserved')
