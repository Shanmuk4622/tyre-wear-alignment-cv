"""Exact source data dictionary, accepted identities and an illustrative gradient figure."""
from PIL import Image
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from metrics_r2 import SCHEMAS
from transactions import read,sha,atomic
from write_training_schema import describe

EXTRA={
 'condition':('category','baseline / shared_projection; stored in every core CSV row.'),
 'schedule_epoch0':('integer','Zero-based original r1 sampler/augmentation epoch; CSV epoch is one higher.'),
 'raw_maps_retained':('0/1','Whether this update retained full native augmented input/target/output maps under the frozen schedule.'),
 'unit':('identifier','Condition/endpoint/epoch/image-index inference-unit identity.'),
 'job':('identifier','Condition identity in the inherited diagnostic schema.'),
 'seed':('integer','Predeclared seed1; fixture seed and measured seed share initialization but never checkpoint state.'),
 'endpoint':('category','epoch_validation / selected / final.'),
 'split':('category','validation / test; engineering endpoints use validation_fixture and never read test images.'),
 'group':('identifier','Cohort all, physical tyre, or inherited clip/tyre grouping as named by the table.'),
 'tyre':('identifier','Physical tyre source ID; alias of tyre_id in training tables.'),
 'clip':('identifier','Video source where applicable; alias of source_video.'),
 'variant':('category','normal only in NB05. Self-comparison delta fields are zero by definition, not intervention measurements.'),
 'class_margin':('fraction','Largest class probability minus the second largest.'),
 'true_class_probability':('fraction','Probability assigned to the provided mileage-proxy label.'),
 'normal_prediction':('class index','Normal-forward prediction; equals prediction for the normal-only NB05 inference.'),
 'probability_l1_delta':('fraction','Reserved self-comparison field, zero for normal-only inference.'),
 'true_probability_delta':('fraction','Reserved self-comparison field, zero for normal-only inference.'),
 'normal_predicted_class_probability_delta':('fraction','Reserved self-comparison field, zero for normal-only inference.'),
 'prediction_changed':('0/1','Reserved normal self-comparison field, zero; no input intervention was performed.'),
 'human_point_available':('0/1','Whether this image has human point targets; weak/unknown targets are not human truth.'),
 'gate_entropy_nats':('nats','Spatial entropy of the normalized detached predicted tread gate.'),
 'gate_mass_inside_true_tread':('fraction','Gate mass weighted by valid true-tread4×4 area fractions.'),
 'gate_support_fraction':('fraction','Gate grid cells with mass above1e-6, divided by all grid cells.'),
 'evidence_residual':('absolute logit','Maximum absolute signed-map-sum plus class bias minus class logit.'),
 'target_kind':('category','Inherited point provenance: human / polygon_weak / none.'),
 'frozen_r1_selection_point_mae':('width fraction','Torch arithmetic used by original r1 utility; distinct from float64 diagnostic coordinate reconstruction.'),
 'class_objective':('weighted loss','A=CE+0.2 ordinal+0.2 background KL.'),
 'dense_objective':('weighted loss','B=segmentation+2 point+0.1 containment+0.1 alignment.'),
 'shared_parameters':('scalar parameter count','Number of declared shared scalar parameters, not the number of tensors.'),
 'class_norm':('gradient L2 norm','Unscaled shared-gradient norm for A.'),
 'dense_norm':('gradient L2 norm','Unscaled shared-gradient norm for B.'),
 'dot_before':('gradient dot product','Combined shared A/B dot product before routing.'),
 'cosine_before':('[-1,1]','Combined shared A/B cosine before routing; denominator clamped for zero gradients.'),
 'dot_after':('gradient dot product','Dot product of the two routed components; unchanged in baseline.'),
 'cosine_after':('[-1,1]','Cosine of routed components; zero-denominator convention is explicit in code.'),
 'projection_applied':('0/1','One only in the projection condition when the finite shared dot product is negative.'),
 'class_projected_norm':('gradient L2 norm','Norm of routed class component; original norm when routing is not applied.'),
 'dense_projected_norm':('gradient L2 norm','Norm of routed dense component; original norm when routing is not applied.'),
 'shared_gradient_change_norm':('gradient L2 norm','Actual shared update change relative to the original total backward gradient.'),
 'head_gradient_max_change':('gradient magnitude','Measured maximum task-head gradient change during routing; must be zero.'),
 'projection_seconds':('seconds','Measured routing/statistics segment; an observed software duration, not pure GPU kernel time.'),
 'level':('category','cohort / tyre.'),
 'accuracy':('fraction','Correct classifications divided by all image observations in the group.'),
 'balanced_accuracy_observed':('fraction','Mean recall over classes actually supported by that cohort; no held-out mid recall is invented.'),
 'human_point_images':('image count','Human-labelled images contributing point MAE.'),
 'joint_validation_utility':('selection utility','0.5 observed BA+0.35 mean Dice+0.15 max(0,1-frozen point MAE/0.05).'),
 'support':('image count','Ground-truth support for the stated class.'),
 'precision':('fraction','TP/(TP+FP); blank when the denominator is zero.'),
 'recall':('fraction','TP/(TP+FN); blank for an absent true class.'),
 'f1':('fraction','2TP/(2TP+FP+FN); blank for a zero denominator.'),
 'bin_index':('0..9','Ten equal confidence bins; exact1.0 belongs to the last bin.'),
 'lower_inclusive':('fraction','Confidence-bin lower edge, inclusive.'),
 'upper_inclusive_last':('fraction','Upper edge; exclusive except for the final bin.'),
 'confidence_sum':('probability sum','Sum of predicted-class confidences in this bin; retain with images/correct for later ECE calculation.'),
}

def definition(field,table):
 if field=='epoch':return ('one-based epoch','Training/validation epoch; selected/final inference rows name the actual checkpoint epoch.')
 if field=='engineering_only':return ('0/1','One for short engineering fixtures; zero for the declared measured pilot.')
 if field=='class_index' and table=='eval_class_metrics':return ('0..2','Class receiving the one-versus-rest confusion/support metrics.')
 if field in EXTRA:return EXTRA[field]
 if field in ('tp','fp','fn','tn') and table=='eval_class_metrics':return ('image count','One-versus-rest class confusion count in this group.')
 if field=='correct' and table=='calibration_bins':return ('image count','Number correct in this bin; not an accuracy fraction.')
 if field in ('load_seconds','total_seconds'):return ('seconds','Measured inference input-loading or total observation duration as named.')
 if field=='lr_after_scheduler':return ('learning rate','Compatibility field: rate at update end; actual post-validation scheduler rate is learning_rate_next_epoch in epoch_summaries.')
 if field=='validation_seconds':return ('seconds','Compatibility field blank in NB05; validation costs are in eval_timings and commit/lifecycle CSVs.')
 if field=='missing_reason':return ('text','Explicit unavailable/nonfinite/scheduled resource or task-gradient reason.')
 if field=='prediction' and table=='eval_points':return ('width fraction','Predicted horizontal point coordinate at the fixed evaluation checkpoint.')
 return describe(field,'geometry_points' if table=='eval_points' else table)

def main():
 previous_path=ROOT/'revision2_validation/ACCEPTED_IDENTITIES.json'
 previous=read(previous_path) if previous_path.exists() else None
 validation=read(ROOT/'revision2_validation/VALIDATION.json');delivery=read(ROOT/'revision2_validation/DELIVERY.json')
 assert validation['status']=='passed' and validation['source_hashes']==delivery['sources']
 assert all(sha(ROOT/n)==h for n,h in delivery['sources'].items())
 lines=['# TREAD-ONE NB05 — exact evidence dictionary','',
  '**28 October2026.** Source-derived headers and definitions for the paired pilot. [Protocol/run guide](phase3R2PairedPilot.md). Blank means unavailable/not applicable; observed zero remains zero. Repeated identity columns are not independent metrics.','',
  f'The source defines **{len(SCHEMAS)} core CSV families / {sum(map(len,SCHEMAS.values()))} column positions**. The compatibility `validation_predictions` family is intentionally empty: full validation now lives in the richer `eval_predictions` and companion tables. Normal-only self-comparison fields do not imply an intervention experiment.','',
  '## Planned measured record coverage','','| Family / scope | Per condition | Both conditions |','|---|---:|---:|',
  '| Completed training updates |2,940|5,880|','| Actual augmented training occurrences |23,520|47,040|','| Validation image observations |4,860|9,720|','| Selected/final development-test image observations |206|412|',
  '| Training point records |141,120|282,240|','| Training latent records |282,240|564,480|','',
  'These are planned observation counts for complete runs, not new subjects or already achieved results. A local engineering fixture is much smaller.','']
 for table,fields in SCHEMAS.items():
  lines += [f'## `{table}.csv`','',f'{len(fields)} ordered columns.','', '| Position | Column | Units / type | Meaning |','|---:|---|---|---|']
  for i,name in enumerate(fields):
   units,meaning=definition(name,table);lines.append(f'|{i+1}|`{name}`|{units}|{meaning}|')
  lines.append('')
 lines += ['## Native arrays and auxiliary observations','',
  '| File | Meaning |','|---|---|',
  '| `inputs.npy` | Float32 normalized input; training batch `[8,3,512,384]`, fixed validation example `[3,512,384]` |',
  '| `targets.npy` | Uint8 tyre/tread/ignore masks on the corresponding512×384 working grid |',
  '| `outputs.npy` / endpoint chunk | Float32 mask2, latent12, signed evidence3, gate1 channels on128×96; full endpoint chunks contain up to8 images |',
  '| `INITIALIZATION.json` | Actual initial model tensor-content SHA-256, parameter count, architecture and seed |',
  '| `COMMIT_TIMINGS.csv` | Unit, updates, phase_after_commit, seconds, UTC and post-commit observation scope |',
  '| `WORKER_LIFECYCLE.csv` | Start/resume and exit UTC, main-process elapsed seconds, updates, phase and termination status; includes input/evaluation/checkpoint work, excludes stopped-worker HF waiting |',
  '| `NUMERICAL_ATTEMPTS.csv` | Unique failed-attempt ID, UTC, candidate update/epoch/cursor, AMP scales, reason and explicit uncommitted-attempt status |',
  '| `STATE_VERSIONS.csv` | Every local unit/current optimizer cursor, state hash, unit-row hash, observation-chain hash and native-presence flag |',
  '| `CAPACITY.json` | Actual saved-output logical bytes and free filesystem bytes at publication |',
  '| Native `.pt` | Model, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG, training/evaluation cursors, partial statistics and row/hash-chain anchors |','',
  'Auxiliary post-commit timings or exit rows can be incomplete after a forced kill. They are not the authoritative scientific cursor. The native/journal unit and published manifest establish that cursor. Lifecycle durations are process observations, not a claim of continuous energy or complete end-to-end notebook wall time.','']
 (ROOT/'phase3R2EvidenceSchema.md').write_text('\n'.join(lines),encoding='utf-8')
 atomic(ROOT/'revision2_validation/SCHEMA.json',dict(families=SCHEMAS,column_positions=sum(map(len,SCHEMAS.values())),family_count=len(SCHEMAS),planned_runs=2,seed=1))
 cfg=spec.configuration(spec.locate(),'cuda:0','baseline');ident=dict(study_intent=cfg['study_id'],notebook_bytes=delivery['notebook_bytes'],notebook_sha256=delivery['notebook_sha256'],source_files=delivery['source_files'],
   original_math=next(r['original_r1_update_math'] for r in validation['actual_cases'] if r['condition']=='baseline'),
   actual_cases=[{k:r[k] for k in ('condition','initial_model_tensor_sha256','real_conflicts','projections')} for r in validation['actual_cases']],
   offline_checks=len(validation['offline_checks']),measured_training_started=False,live_HF_r2_tested=False)
 atomic(ROOT/'revision2_validation/ACCEPTED_IDENTITIES.json',ident)
 if previous:
  replacements=[(previous['notebook_sha256'],ident['notebook_sha256']),(previous['study_intent'],ident['study_intent']),
    (f"{previous['notebook_bytes']:,}",f"{ident['notebook_bytes']:,}"),('15 offline','16 offline'),('15 new journal','16 new journal'),('|15 checks:','|16 checks:')]
  for name in ('phase3R2PairedPilot.md','phase3SessionLog.md','phase3EvidenceRegister.md'):
   p=ROOT/name;text=p.read_text(encoding='utf-8')
   for old,new in replacements:text=text.replace(old,new)
   p.write_text(text,encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 out=ROOT/'revision2_validation/summary_exports';out.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
 for ax,title,vectors in [(axes[0],'Original shared components · dot = −1',[(1,0,'Class A','#c75b54'),(-1,1,'Dense B','#387ca6'),(0,1,'Sum','#7752a6')]),(axes[1],'Projected components · dot = +0.5',[(.5,.5,"Class A′",'#c75b54'),(0,1,"Dense B′",'#387ca6'),(.5,1.5,'Sum','#7752a6')])]:
  ax.axhline(0,color='#aaa',lw=.8);ax.axvline(0,color='#aaa',lw=.8)
  for x,y,label,color in vectors:ax.quiver(0,0,x,y,angles='xy',scale_units='xy',scale=1,color=color,width=.012);ax.text(x+.04,y+.02,label,color=color,fontsize=10)
  ax.set(xlim=(-1.35,1.4),ylim=(-.25,1.9),title=title,xlabel='Illustrative shared-gradient coordinate1',ylabel='Coordinate2');ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Two-dimensional mathematical example · not a measured tyre-model result',fontsize=11);fig.savefig(out/'shared_gradient_projection_example.png',dpi=170);plt.close(fig)
 print(json.dumps(ident,indent=2))

if __name__=='__main__':main()
