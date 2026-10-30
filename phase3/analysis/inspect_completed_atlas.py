"""Exploratory read of byte-verified NB08 tables; no model loading/fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/pilot_validation/nb08_followup/82e7e1c5774f/fit'

def table(name):
    return pd.concat([pd.read_csv(p) for p in sorted((BASE/'csv').glob('epoch_*/'+name+'.csv'))],ignore_index=True)

p=table('predictions')
print('PREDICTIONS',len(p))
print(p.groupby(['stage','endpoint','variant','label']).agg(n=('correct','size'),correct=('correct','sum'),confidence=('probability_2','mean'),low_p=('probability_0','mean'),empty=('empty_support','sum'),support=('supported_patches','mean')).to_string())
print('TRAIN LAST',p[(p.stage=='train') & (p.epoch==60)].groupby('label').agg(n=('correct','size'),accuracy=('correct','mean'),nll=('nll','mean')).to_string())
print('VAL PRED COUNTS',p[p.stage=='validate'].groupby(['prediction','label']).size().to_dict())
m=table('metrics')
print('DENSE',m[(m.level=='cohort') & m.stage.isin(['selected_eval','final_eval']) & m.metric.str.contains('dice|iou|point')][['stage','metric','value','n','unavailable_reason']].to_string(index=False))
print('SOURCE PATHS',list((BASE/'csv').glob('epoch_*/source_images.csv')))
print('VARIANTS',p[p.stage=='diagnostics'].groupby(['endpoint','variant','prediction']).size().to_dict())
for ep in ['selected','final']:
    b=p[(p.endpoint==ep)&p.stage.isin(['selected_eval','final_eval'])].set_index('image_id')
    for variant,q in p[(p.endpoint==ep)&(p.stage=='diagnostics')].groupby('variant'):
        q=q.set_index('image_id').loc[b.index]
        print('PROBE',ep,variant,'flips',int((q.prediction!=b.prediction).sum()),'maxdp',float(np.abs(q[[f'probability_{c}' for c in range(3)]].values-b[[f'probability_{c}' for c in range(3)]].values).max()),'dtrue',float(np.mean([q.iloc[i][f'probability_{int(r.label)}']-r[f'probability_{int(r.label)}'] for i,(_,r) in enumerate(b.iterrows())])))
u=table('train_updates')
print('UPDATES',u[['loss','grad_norm_before_clip','grad_norm_after_clip','compute_seconds','record_seconds']].describe().to_string())
print('SOURCE REGISTRY', pd.read_csv(list((BASE/'csv').glob('epoch_*/source_images.csv'))[0]).groupby(['role','class_index','domain']).size().to_dict())
for epoch in [1,18,60]:
    c=table('coefficients')
    c=c[c.parameter_update==epoch*49]
    print('COEFF',epoch,c[c.kind!='signed_weight'].to_dict('records'))
print('BANK JSON',pd.read_csv(ROOT/'tread_two/pilot/assets/prototype_bank.csv').groupby('proxy_class').source_tyre_id.nunique().to_dict())
for ep in ['selected','final']:
    rows=p[(p.endpoint==ep)&p.stage.isin(['selected_eval','final_eval'])]
    probability=rows[[f'anchor_probability_{c}' for c in range(3)]].to_numpy()
    print('FROZEN_CONTROL',ep,'confusion',rows.groupby(['label','anchor_prediction']).size().to_dict(),
          'nll',float(-np.log(probability[np.arange(len(rows)),rows.label.to_numpy(int)].clip(1e-12)).mean()),
          'confidence',float(probability.max(1).mean()))
