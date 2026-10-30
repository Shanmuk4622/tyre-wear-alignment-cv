"""Matched-class summaries and static research figures from completed NB03 CSV.

This adds local derived presentation artifacts; the diagnostic snapshot itself
remains immutable and all underlying table data are already directly exported.
"""
from PIL import Image
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'diagnostic'))
from storage import read,atomic,write_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    validated=read(ROOT/'diagnostics/DIAGNOSTIC_VALIDATION.json');assert validated['status']=='passed'
    snapshot=Path(validated['snapshot']);out=ROOT/'diagnostics/summary_exports';out.mkdir(exist_ok=True)
    def table(name):
        with open(snapshot/name,encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))
    classes=table('summary/class_metrics.csv');summary=table('summary/cohort_group_metrics.csv')
    jobs=[j['key'] for j in read(snapshot/'CONTRACT.json')['jobs']]
    recalls={(r['job'],r['split'],r['variant'],int(r['class_index'])):r for r in classes if r['level']=='cohort'}
    matrix=np.array([[float(recalls[(job,split,'normal',c)]['recall']) for split,c in [('train',0),('train',1),('train',2),('validation',0),('validation',2)]] for job in jobs])
    plt.rcParams.update({'font.size':11,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(11,5.2),constrained_layout=True)
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(5),['Train low\nn=67','Train mid\nn=249','Train high\nn=70','Validation low\nn=40','Validation high\nn=41'])
    ax.set_yticks(range(len(jobs)),jobs);ax.set_xlabel('Cohort / class (nominal post-hoc inference)');ax.set_ylabel('Saved checkpoint')
    for i in range(len(jobs)):
        for j in range(5): ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Recall (fraction)');ax.set_title('TREAD-ONE: training-identity fit and validation transfer\nNo new training or test evaluation')
    fig.savefig(out/'class_recall_by_checkpoint.png',dpi=180);plt.close(fig)
    variants=['normal','uniform_gate','oracle_tread_gate','background_neutralized','top10_image_occluded']
    selected=[r for r in summary if r['level']=='cohort' and r['split']=='validation']
    lookup={(r['job'],r['variant']):r for r in selected}
    matrix=np.array([[float(lookup[(job,v)]['balanced_accuracy_observed']) for v in variants] for job in jobs])
    fig,ax=plt.subplots(figsize=(11,5.2),constrained_layout=True)
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(5),['Normal','Uniform gate','Oracle tread\ngate','Neutral\nbackground','Top 10% input\nocclusion']);ax.set_yticks(range(len(jobs)),jobs)
    ax.set_xlabel('Fixed-weight diagnostic intervention');ax.set_ylabel('Saved checkpoint')
    for i in range(len(jobs)):
        for j in range(5): ax.text(j,i,f'{100*matrix[i,j]:.2f}%',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Balanced accuracy (fraction)');ax.set_title('Validation intervention results, observed low/high classes\nOracle variants are diagnostic only; no checkpoint reselection')
    fig.savefig(out/'validation_interventions.png',dpi=180);plt.close(fig)
    comparisons=[]
    normal={(r['job'],r['split']):r for r in summary if r['level']=='cohort' and r['variant']=='normal'}
    for job in jobs:
        rec=dict(job=job)
        for split in ('train','validation'):
            r=normal[(job,split)]
            rec[split+'_images']=int(r['images']);rec[split+'_ba_all_observed']=float(r['balanced_accuracy_observed'])
            for c in (0,1,2):
                rr=recalls[(job,split,'normal',c)]
                rec[split+f'_recall_{c}']=float(rr['recall']) if rr['recall'] else None
                rec[split+f'_support_{c}']=int(rr['support'])
            rec[split+'_low_high_ba']=(rec[split+'_recall_0']+rec[split+'_recall_2'])/2
        comparisons.append(rec)
    write_csv(out/'matched_low_high_summary.csv',comparisons,list(comparisons[0]))
    write_csv(out/'validation_intervention_summary.csv',selected,list(selected[0]))
    gradients=table('csv/gradients.csv');pairs=[r for r in gradients if r['row_type']=='cosine']
    group=[r for r in pairs if r['task']=='classification_plus_ordinal' and r['other_task']=='segmentation']
    atomic(ROOT/'diagnostics/DIAGNOSTIC_INSIGHTS.json',dict(status='derived',diagnostic_contract=validated['diagnostic_contract'],
        matched_summaries=comparisons,validation_interventions=selected,
        classification_segmentation_gradient_cosines=group,
        figures=['summary_exports/class_recall_by_checkpoint.png','summary_exports/validation_interventions.png'],
        scope='Local presentation derivatives of saved NB03 tables; no new inference, training or test evaluation.'))
    print(json.dumps(dict(matched_summaries=comparisons,classification_segmentation_gradient_cosines=group),indent=2))


if __name__=='__main__': main()
