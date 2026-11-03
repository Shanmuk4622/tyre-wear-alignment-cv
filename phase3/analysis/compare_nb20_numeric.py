"""Measure cross-platform differences before choosing field-specific checks."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from common import read
out=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);rows=[]
for stage,reference in [('review',ROOT/'tread_two/nb17_report'),('passport',ROOT/'tread_two/geo_passport_report')]:
    for name,spec in read(out/'remote'/stage/'TABLES.json').items():
        if name=='provenance':continue
        a=pd.read_csv(out/'remote'/stage/spec['file']);b=pd.read_csv(reference/spec['file'])
        for column in a.select_dtypes(include='number'):
            if column not in b or not pd.api.types.is_numeric_dtype(b[column]):continue
            av=a[column].to_numpy(dtype='float64');bv=b[column].to_numpy(dtype='float64');delta=np.abs(av-bv);valid=np.isfinite(delta)
            if valid.any() and np.max(delta[valid])>0:
                rows.append(dict(stage=stage,table=name,column=column,max_abs=float(np.max(delta[valid])),max_rel=float(np.max(delta[valid]/np.maximum(np.abs(bv[valid]),1e-12))),fails_1e8=not np.allclose(av,bv,atol=1e-8,rtol=1e-8,equal_nan=True)))
frame=pd.DataFrame(rows);frame.to_csv(out/'NUMERIC_DIFFERENCES.csv',index=False)
print(frame[frame.fails_1e8].to_string(index=False));print('Numerically different columns:',len(frame))
