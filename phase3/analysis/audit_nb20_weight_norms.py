"""Quantify reported FP32 norm diagnostics against original tensors in FP64."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from common import read,file_meta,atomic
out=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);original=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path']);manifest=read(original/'REMOTE_MANIFEST.json')
rows=[];audit=pd.read_csv(out/'remote/review/epoch_parameter_audit.csv')
for epoch,group in audit.groupby('epoch'):
    name=f'epochs/epoch_{int(epoch):03d}.pt';path=original/'remote'/name
    assert file_meta(path)=={k:manifest['files'][name][k] for k in ('bytes','sha256')}
    weights=torch.load(path,map_location='cpu',weights_only=True)
    for row in group.itertuples():
        values=weights[row.parameter].numpy().astype('float64');norm=float(np.sqrt(np.square(values).sum(dtype='float64')))
        reported=float(row.l2);difference=abs(reported-norm);relative=difference/max(norm,1e-30)
        u=float(np.finfo('float32').eps/2);operations=values.size+2;bound=operations*u/(1-operations*u)
        assert relative<=bound+u if norm else reported==0
        ulps=abs(int(np.array(reported,dtype='float32').view('int32'))-int(np.array(norm,dtype='float32').view('int32')))
        rows.append(dict(epoch=int(epoch),parameter=row.parameter,numel=values.size,HF_reported_FP32_norm=reported,independent_FP64_norm=norm,absolute_difference=difference,relative_difference=relative,FP32_ULP_distance_from_rounded_FP64=ulps,conservative_FP32_reduction_relative_bound=bound+u,passed=True))
frame=pd.DataFrame(rows);frame.to_csv(out/'FP64_WEIGHT_NORM_AUDIT.csv',index=False)
summary=dict(status='passed',rows=len(frame),max_absolute_difference=float(frame.absolute_difference.max()),max_relative_difference=float(frame.relative_difference.max()),max_ULP_distance_from_rounded_FP64=int(frame.FP32_ULP_distance_from_rounded_FP64.max()),all_original_epoch_bytes_verified=True,
             criterion='Each norm lies within a conservative FP32 sum-of-squares reduction roundoff bound; exact model tensors remain unchanged')
atomic(out/'FP64_WEIGHT_NORM_SUMMARY.json',summary);print(summary);print(frame.nlargest(3,'relative_difference')[['epoch','parameter','numel','relative_difference','FP32_ULP_distance_from_rounded_FP64']].to_string(index=False))
