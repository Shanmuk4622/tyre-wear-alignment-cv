"""Compact actual findings from retained CSVs and executed recovery output."""
import json,re,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path']);out=ROOT/'tread_two/context_analysis_report'
analysis=json.loads((out/'ANALYSIS.json').read_text());assert analysis['status']=='passed'
def table(name):return pd.read_csv(out/(name+'.csv'))
inspection=json.loads((done/'INSPECTION.json').read_text());streams='\n'.join(o['text'] for c in inspection['cells'] for o in c['outputs'] if o['type']=='stream')
restored=[int(x) for x in re.findall(r'HF recovery verified at update (\d+)',streams)]
identity=table('cohort_metrics').query("stage in ['selected_eval','final_eval']")
context=table('context_by_variant').query("stage in ['selected_eval','final_eval']")
prior=table('frozen_LC_prior_metrics');validation_prior=prior[prior.stage=='validate'];exposure=table('query_draw_exposure')
result=dict(status='passed',actual_HF_restore_updates=restored,derived_tables=len(analysis['csv_tables']),derived_rows=sum(v['rows'] for v in analysis['csv_tables'].values()),
    review_inputs=analysis['input_files'],review_input_bytes=analysis['input_bytes'],identity_metrics=identity.to_dict('records'),
    frozen_LC_validation_BA=sorted(validation_prior.balanced_accuracy_observed.unique().tolist()),
    frozen_LC_validation_NLL=float(validation_prior.nll.iloc[0]),context_by_class=context.to_dict('records'),
    all_query_images_exposed=int(exposure.actual_draws.gt(0).sum()),actual_draws=int(exposure.actual_draws.sum()),maximum_image_draws=int(exposure.actual_draws.max()),
    blocked_view_reference_rows=int(table('reference_permission_summary').blocked_rows.sum()),
    maximum_logit_reconstruction_residual=float(table('logit_reconstruction').residual.max()),
    maximum_patch_residual=float(table('endpoint_patch_accounting').maximum_patch_residual.max()),
    maximum_pair_loss_residual=float(table('paired_loss_audit').loss_residual.max()),
    maximum_attention_row_sum_residual=float(table('attention_head_audit').row_sum_max_residual.max()),
    maximum_attention_entropy_residual=float(table('attention_head_audit').entropy_residual.max()),
    probe_flips=int(table('probe_summary').flips.sum()),probes=table('probe_summary').to_dict('records'),
    dense_regions=json.loads(table('dense_region_by_role').to_json(orient='records')),dense_points=json.loads(table('dense_points_by_role').to_json(orient='records')),
    model_forwards=0,training_updates=0,new_independent_tyres=0)
(done/'SCIENTIFIC_FINDINGS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('dense_points','dense_regions','probes','context_by_class')},indent=2))
