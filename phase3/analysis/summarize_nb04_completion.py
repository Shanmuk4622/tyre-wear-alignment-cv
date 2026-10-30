"""Measured NB04 stage figure and friendly aliases from audited Kaggle files."""
from PIL import Image
import json,re,shutil,subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 audit=json.loads((ROOT/'training_gates/nb04_publication/VERIFICATION.json').read_text());assert audit['status']=='passed'
 out=ROOT/'training_gates/summary_exports';out.mkdir(exist_ok=True);measurements=[]
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 stages=['data_seconds','forward_seconds','backward_seconds','optimizer_seconds','telemetry_seconds'];labels=['Input loading / checks','Forward','Backward','Optimizer','Observation / copies']
 fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained');colors=['#355c7d','#58a4b0','#66c2a5','#f5b567','#bd80b6']
 for i,d in enumerate(audit['devices']):
  means=[float(np.mean([float(r[s]) for r in d['step_measurements']])) for s in stages]
  later=[float(np.mean([float(r[s]) for r in d['step_measurements'][1:]])) for s in stages]
  for ax,values in zip(axes,(means,later)):
   left=0
   for label,color,value in zip(labels,colors,values):ax.barh(i,value,left=left,color=color,label=label if i==0 else None);left+=value
  measurements.append(dict(device=d['device'],stage_mean_seconds=dict(zip(stages,means)),mean_update_compute_seconds=float(np.mean([float(r['update_compute_seconds']) for r in d['step_measurements']])),
    first_update_compute_seconds=float(d['step_measurements'][0]['update_compute_seconds']),mean_later_update_compute_seconds=float(np.mean([float(r['update_compute_seconds']) for r in d['step_measurements'][1:]])),
    later_stage_mean_seconds=dict(zip(stages,later)),max_loss_reconstruction_error=d['max_loss_reconstruction_error'],sampled_cuda_peak_bytes=max(int(r['cuda_peak_bytes']) for r in d['resource_samples']),published_bytes=d['published_bytes']))
  alias=out/f'TREAD_ONE_NB04_Kaggle_GPU{i}.html';shutil.copy2(Path(d['remote_directory'])/'cases/live_resume/EVIDENCE_GATE.html',alias)
  script=re.search(r'<script>(.*?)</script>',alias.read_text(encoding='utf-8'),re.S).group(1);target=out/f'nb04_gpu{i}_syntax.js';target.write_text(script,encoding='utf-8')
  result=subprocess.run([shutil.which('node'),'--check',str(target)],capture_output=True,text=True);assert result.returncode==0,result.stderr
 for ax,title in zip(axes,('All three updates · includes cold start','Updates 2–3 only · tiny warm sample')):
  ax.set_yticks([0,1],['T4 · cuda:0','T4 · cuda:1']);ax.set_xlabel('Mean seconds per update');ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(loc='upper center',bbox_to_anchor=(1.05,-.22),ncol=3,frameon=False);fig.suptitle('NB04 measured stages · engineering observations, not a training-runtime benchmark',fontsize=11)
 fig.savefig(out/'nb04_kaggle_stage_times.png',dpi=170);plt.close(fig)
 result=dict(status='passed',measurements=measurements,JS_syntax_checked=True,browser_render_checked=False,published_files=572,core_case_rows=7524,new_independent_tyres=0)
 (ROOT/'training_gates/nb04_publication/SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
