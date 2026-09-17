"""Recompute paper tables/figures from frozen local evidence; no model runs."""
from pathlib import Path
import csv
import json
import hashlib
import shutil
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = REPO/'docs/report/evidence'
FIG = HERE/'figures'
GEN = HERE/'generated'
FIG.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)
used = {}


def record(path):
    path = Path(path)
    used[path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def table(name):
    with record(EVIDENCE/'tables'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def data(name):
    return json.loads(record(EVIDENCE/'geometry'/name).read_text(encoding='utf-8'))


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def tex_table(name, caption, columns, header, rows, wide=False):
    env = 'table*' if wide else 'table'
    text = '\\begin{'+env+'}[t]\n\\caption{'+caption+'}\n\\label{tab:'+name+'}\n'
    text += '\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' & '.join(header)+r' \\'+'\n\\midrule\n'
    text += '\n'.join(' & '.join(row)+r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{'+env+'}\n'
    (GEN/(name+'.tex')).write_text(text, encoding='utf-8')


names = {'mobilenetv4':'MobileNetV4','resnet50':'ResNet-50','swin_t':'Swin-T','swin_s':'Swin-S',
    'effnetv2s':'EfficientNetV2-S','clip_b16':'CLIP ViT-B/16','regnety016':'RegNetY-016',
    'vit_s':'ViT-S','resnext50':'ResNeXt-50','maxvit_t':'MaxViT-T','dinov2_b':'DINOv2-B',
    'convnextv2_t':'ConvNeXtV2-T','densenet121':'DenseNet-121','vgg16bn':'VGG16-BN',
    'dinov2_s':'DINOv2-S','coatnet0':'CoAtNet-0','deit3_s':'DeiT-III-S',
    'deeplabv3plus_r34':'DeepLabV3+/R34','unet_r34':'U-Net/R34','segformer_b0':'SegFormer-B0',
    'segformer_b2':'SegFormer-B2','rtdetrv2_r18':'RT-DETRv2-R18','yolo26n_det':'YOLO26-n det.',
    'yolo26s_det':'YOLO26-s det.','yolo26n_seg':'YOLO26-n seg.','yolo26s_seg':'YOLO26-s seg.'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
teal, orange, navy = '#087f83', '#b75a18', '#24394c'

arch = table('classification_master_architectures.csv')
assert len(arch) == 17 and sum(int(r['n']) for r in arch) == 153
tex_table('architectures', 'Classification endpoints on the historical folds. Each row averages nine runs (three folds and three seeds); folds 0 and 2 retain overlap concerns. These are internal validation results, not an independent-test ranking.',
          'lrr', ['Architecture','Selected','Final'],
          [[names[r['arch']],f"{float(r['f1_mean']):.5f}",f"{float(r['final_f1']):.5f}"] for r in arch])
fig, ax = plt.subplots(figsize=(6.8,4.7))
for i, row in enumerate(arch):
    a, b = float(row['f1_mean']), float(row['final_f1'])
    ax.plot([b,a],[i,i],color='#aab5be',linewidth=1.5)
ax.scatter([float(r['f1_mean']) for r in arch], range(17), color=teal, label='Validation-selected', s=27,zorder=3)
ax.scatter([float(r['final_f1']) for r in arch], range(17), color=orange, marker='s',label='Fixed final',s=23,zorder=3)
ax.set_yticks(range(17),[names[r['arch']] for r in arch]); ax.invert_yaxis()
ax.set(xlim=(.4,1.025),xlabel='Mean macro-F1 (nine runs; dependent internal folds)')
ax.grid(axis='x',alpha=.18); ax.legend(loc='lower left',frameon=False)
fig.tight_layout(); fig.savefig(FIG/'endpoints.pdf'); plt.close(fig)

base = table('classification_baselines_by_fold.csv')
# Preserve the frozen table in the package for direct audit, without treating
# the legacy random-init baseline as the matched 60-epoch ResNet-50 control.
effects = table('s4b_architecture_effects.csv')
factors = {'sampler_classweighted':'Class-weighted','sampler_uniform':'Uniform','transfer_random':'Random init.'}
tex_table('confirmation','Same-fold confirmation: mean macro-F1 change relative to each matched baseline over three seeds. Selection and final endpoints can give different sampling conclusions.',
    'llrr',['Architecture','Factor',r'$\Delta$ selected',r'$\Delta$ final'],
    [[names[r['arch']],factors[r['factor']],f"{float(r['mean_selected_delta']):+.5f}",f"{float(r['mean_final_delta']):+.5f}"] for r in effects])

loc = table('s5_localisation_by_run.csv')
assert len(loc) == 81 and len({r['run_id'] for r in loc}) == 81
models = sorted({r['model'] for r in loc})
def mean_field(rows, field):
    vals=[float(r[field]) for r in rows if r[field] not in ('','nan')]
    return st.mean(vals) if vals else None
def fmt(x): return '--' if x is None else f'{x:.5f}'
local_rows=[]
for model in models:
    rows=[r for r in loc if r['model']==model and r['fold']=='1']
    assert len(rows)==3
    local_rows.append([names[model]]+[fmt(mean_field(rows,k)) for k in ['bbox_map_50_95','segm_map_50_95','tread_iou','tread_boundary_f1_2px']])
tex_table('localisation','Fold-1 localisation means over three seeds. AP denotes COCO-style AP at IoU thresholds 0.50:0.95. Boundary F1 uses a two-native-pixel tolerance. Dashes denote undefined mask metrics for box-only models.',
    'lrrrr',['Model','Box AP','Mask AP','Tread IoU','Boundary F1'],local_rows,wide=True)
mask_models=[m for m in models if any(r['model']==m and r['tread_iou'] not in ('','nan') for r in loc)]
fig, axes=plt.subplots(1,2,figsize=(6.8,2.7),sharey=True)
for ax,field,title in zip(axes,['tread_iou','tread_boundary_f1_2px'],['(a) Area overlap','(b) Boundary agreement']):
    for i,m in enumerate(mask_models):
        values=[float(r[field]) for r in loc if r['model']==m and r['fold']=='1']
        ax.errorbar(st.mean(values),i,xerr=st.stdev(values),fmt='o',color=teal,capsize=3)
    ax.set_yticks(range(len(mask_models)),[names[m] for m in mask_models]); ax.set_title(title,loc='left',fontsize=10)
    ax.set_xlim((.8,1) if field=='tread_iou' else (0,.5)); ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Tread IoU' if field=='tread_iou' else 'Boundary F1 (2 px)')
axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(FIG/'localisation.pdf');plt.close(fig)

fusion=table('s9_fusion_by_run.csv'); roi=table('s5_roi_by_run.csv')
assert len(fusion)==405 and len(roi)==405
arm_names={'full':'Full frame','tyre_only':'Tyre only','tread_only':'Tread only','tyre_tread_equal':'Tyre + tread','full_tyre_tread_equal':'Full + tyre + tread'}
fusion_rows=[]; deltas=[]
for arm,name in arm_names.items():
    rows=[r for r in fusion if r['arm']==arm]; assert len(rows)==81
    delta=mean_field(rows,'delta_vs_full');deltas.append(delta)
    fusion_rows.append([name,'81',f'{delta:+.5f}'])
tex_table('fusion','Fixed probability fusion: mean paired macro-F1 changes over 81 source runs. Repeated images and shared classifiers make these rows dependent.',
          'lrr',['Arm','Rows',r'Mean $\Delta$'],fusion_rows)
fig,ax=plt.subplots(figsize=(3.35,2.35))
ax.barh(list(arm_names.values())[1:],[x*100 for x in deltas[1:]],color=orange,height=.55)
ax.axvline(0,color=navy,linewidth=.8);ax.set_xlabel('Mean macro-F1 change\n(percentage points)',fontsize=8)
ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(FIG/'fusion.pdf');plt.close(fig)

cal=table('calibration.csv');conf=table('conformal.csv')
tex_table('calibration','Confidence diagnostics on held-out calibration-test subsets. ECE is the recorded expected calibration error; values near zero on flagged folds do not imply external reliability.',
    'llrrrr',['Fold','Scores',r'$n$','F1','ECE','NLL'],
    [[r['fold'],'Temp.' if r['kind']=='temperature' else 'Raw',r['n']]+[f"{float(r[k]):.4f}" for k in ['f1_macro','ece','nll']] for r in cal])
tex_table('conformal','Prediction-set diagnostics at nominal 90\% coverage. The recorded abstention field does not count every empty-set event.',
    'rrrrrr',['Fold',r'$n_{cal}$',r'$n_{test}$','Coverage','Set size','Abstain'],
    [[r['fold'],r['n_cal'],r['n_test']]+[f"{float(r[k]):.4f}" for k in ['coverage_90','mean_set_size','abstain_rate']] for r in conf])

comparison=data('comparison_REPORT.json'); paired=data('comparison_PAIRED_POINTS.json'); contract=data('comparison_CONTRACT.json')
assert len(paired)==432 and len({r['tyre'] for r in paired})==2 and len({r['pilot_id'] for r in paired})==24
assert not any(r['fallback_used'] for r in paired)
for row in comparison['results']:
    records=[r for r in paired if r['seed']==row['seed']]
    assert len(records)==144
    assert abs(st.mean(r['hrnet_error'] for r in records)-row['hrnet_mean']) < 1e-12
    assert abs(st.mean(r['segformer_error'] for r in records)-row['segformer_fallback_mean']) < 1e-12
hr=st.mean(r['hrnet_error'] for r in paired);seg=st.mean(r['segformer_error'] for r in paired)
metrics={'hrnet_percent_width':hr*100,'segformer_percent_width':seg*100,
         'relative_reduction_percent':100*(seg-hr)/seg,'hrnet_px':hr*1151,'segformer_px':seg*1151,
         'fusion_delta':deltas[-1],'retained_classification_runs':153,'s5_runs':81,'paired_points':432}
geometry_rows=[[str(r['seed']),f"{r['hrnet_mean']*100:.3f}",f"{r['segformer_fallback_mean']*100:.3f}",'100'] for r in comparison['results']]
geometry_rows.append(['Mean',f'{hr*100:.3f}',f'{seg*100:.3f}','100'])
tex_table('geometry','Matched fixed-epoch-60 horizontal point error (\% of native width minus one). Each seed uses the same 24 images from two test tyres; coverage is the SegFormer raw point coverage.',
    'lrrr',['Seed','HRNet','SegFormer',r'Coverage (\%)'],geometry_rows)
fig,axes=plt.subplots(1,2,figsize=(6.8,2.6))
for ax in axes: ax.grid(axis='y',alpha=.2); ax.set_ylabel('Horizontal error (% width)')
seeds=[r['seed'] for r in comparison['results']]
axes[0].plot(seeds,[r['hrnet_mean']*100 for r in comparison['results']],'-o',color=teal,label='HRNet-W18')
axes[0].plot(seeds,[r['segformer_fallback_mean']*100 for r in comparison['results']],'-s',color=orange,label='SegFormer-B0')
axes[0].set_xticks(seeds);axes[0].set_xlabel('Training seed');axes[0].set_ylim(0,3);axes[0].legend(frameon=False,fontsize=8)
axes[0].set_title('(a) Same test images, each seed',loc='left',fontsize=10)
tyres=list(comparison['results'][0]['per_tyre'])
for i,tyre in enumerate(tyres):
    for key,offset,color,marker in [('hrnet',-.13,teal,'o'),('segformer_with_fallback',.13,orange,'s')]:
        values=[r['per_tyre'][tyre][key]*100 for r in comparison['results']]
        axes[1].scatter([i+offset]*3,values,color=color,marker=marker,s=27)
        axes[1].plot([i+offset-.08,i+offset+.08],[st.mean(values)]*2,color=color)
axes[1].set_xticks([0,1],['High-mileage tyre','New tyre']);axes[1].set_ylim(0,3)
axes[1].set_title('(b) Only two test tyres',loc='left',fontsize=10)
fig.tight_layout();fig.savefig(FIG/'geometry.pdf');plt.close(fig)

# Diagram is a schematic, not a measured result. All labels are embedded.
fig,ax=plt.subplots(figsize=(7,3.1));ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,title,body,color=teal):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor='#f1f5f8',edgecolor=color,linewidth=1.3))
    ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=8,fontweight='bold',color=color)
    ax.text(x+w/2,y+h-.65,body,ha='center',va='top',fontsize=7.5,linespacing=1.45,color=navy)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':navy,'lw':1.2})
box(.1,3.1,2.65,1.65,'SOURCE COLLECTION','418 clean photographs\n12 confirmed tyre identities\nMileage labels + masks')
box(3.35,3.1,2.9,1.65,'RECOGNITION / REGIONS','153 retained classifier runs\n81 localisation runs\nFixed crops and fusion')
box(6.9,3.1,2.9,1.65,'MATCHED POINT STUDY','120 images; 72 / 24 / 24 split\nHRNet vs. SegFormer\n3 seeds; 2 test tyres')
arrow((2.8,3.9),(3.25,3.9));arrow((6.3,3.9),(6.8,3.9))
box(.1,.45,5.95,1.8,'INSPECTION WORKSTATION','Exact-frame overlays + raw evidence + failure flags\nImage-space lines / rim candidates / temporal display\nDownloaded PNG frames and background 10 fps video')
box(6.9,.45,2.9,1.8,'PHYSICAL VALIDATION','Dual-target software exists\nReference error unmeasured\nNo target-free angle claim',orange)
arrow((4.8,3),(4.8,2.35));arrow((8.3,3),(5.6,2.35));arrow((6.15,1.3),(6.8,1.3))
fig.savefig(FIG/'workflow.pdf');plt.close(fig)

for filename in ['learned-image-check.png','video1-learned-diagram.png','alignment-bench-check.png']:
    source=record(REPO/'docs/report/assets'/filename)
    shutil.copy2(source,FIG/filename)
# Use the original saved workstation overview; keep screenshot provenance clear.
source=record(REPO/'docs/report/assets/video1-learned-workstation.png')
shutil.copy2(source,FIG/source.name)
for name in ['prototype_learned-integration-check.json','prototype_learned-ui-check.json']:
    data(name)
for name in ['classification_hypothesis_outcomes.csv','classification_stage_a_quarantined.csv','stress_tests.csv']:
    table(name)
for file in ['docs/12_DATASET_FINAL_V1.md','docs/36_MATCHED_GEOMETRY_RESULTS_AND_INTEGRATION.md',
             'prototype/VALIDATION.md','prototype/alignment.py','prototype/edge_geometry.py',
             'prototype/LEARNED_GEOMETRY_LOG.md','CITATION.cff', 'tyrelib/tyrelib.py',
             'tyrelib/hrnet_runtime.py', 'tyrelib/hrnet_protocol.py', 'tyrelib/build_later_notebooks.py']:
    record(REPO/file)
(HERE/'NUMERICAL_AUDIT.json').write_text(json.dumps({'status':'passed','metrics':metrics,'sources_sha256':used},indent=2)+'\n',encoding='utf-8')
print(json.dumps(metrics,indent=2))
