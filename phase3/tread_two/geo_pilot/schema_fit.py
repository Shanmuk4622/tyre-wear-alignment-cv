"""Executable typed CSV contract: all approved families, explicit pre-fit refinements."""
import csv,io,math
import numpy as np
from common import HERE,read

SCHEMA=read(HERE/'SCHEMAS.json')['families']
TYPE_FIXES={'source_images':{'role':'utf8','domain':'utf8'},'epoch_metrics':{'task':'utf8'},
            'selection':{'rejection_reasons':'utf8'},'atlas_rows':{'row_valid':'bool','grid_parameter_native_key':'utf8'},
            'dense_pixels':{'predicted':'bool'},'endpoint_neighbours':{'rank':'int64'},
            'interventions':{'base_sample':'int64'},'numeric_events':{'replay_from_update':'int64'},
            'quality_bins':{'count':'int64'},'HF_requests':{'event_id':'utf8'},'HF_publications':{'attempt':'utf8'}}
for family,fixes in TYPE_FIXES.items():
    for col in SCHEMA[family]['columns']:
        if col['name'] in fixes:col['dtype']=fixes[col['name']]
# Disaggregate the anchor penalty: mask-valid and coordinate denominators differ.
EXTRA={'geometry_losses': [('anchor_mask_sum','float64'),('anchor_mask_count','int64'),
                            ('anchor_point_sum','float64'),('anchor_point_count','int64')],
       'batch_objectives':[('invariance','float64'),('masked_huber_sum','float64'),('masked_huber_count','float64'),
                           ('mean_predictor_huber_sum','float64'),('mean_predictor_huber_count','float64')],
       'view_plans':[('masking_policy','utf8')],
       'unit_ledger':[('state_semantic_sha256','utf8')],
       'HF_requests':[('server_quota','int64'),('server_window_seconds','int64'),('time_utc','utf8')],
       'pipeline_timings':[('data_loading_seconds','float64'),('backward_optimizer_seconds','float64'),('previous_commit_seconds','float64')]}
EXTRA['geometry_drift']=[('anchor_tyre_box_available','bool'),('anchor_tread_box_available','bool')]
EXTRA['atlas_bands']=[('point_mask_gap_count','int64'),('point_mask_gap_exceeds_flag','bool')]
EXTRA['interventions']=[('probe_available','bool'),('source_patch','int64'),('source_patch_mapping','utf8'),('unavailable_reason','utf8')]
for family,fields in EXTRA.items():
    for name,dtype in fields:
        SCHEMA[family]['columns'].append(dict(name=name,dtype=dtype,definition='Executable pre-fit refinement; see runtime documentation',missing='Blank only when unavailable',units='field-specific'))

def fields(family):return [c['name'] for c in SCHEMA[family]['columns']]
def normalize(family,record,run):
    assert not set(record)-set(fields(family)),(family,set(record)-set(fields(family)))
    row=[]
    for column in SCHEMA[family]['columns']:
        name,dtype=column['name'],column['dtype']
        value=record.get(name,run if name=='run' else None)
        if value is None:row.append('');continue
        if dtype=='utf8':row.append(str(value))
        elif dtype=='bool':
            assert value in (True,False,0,1),(family,name,value)
            row.append(str(int(value)))
        elif dtype in ('int64','uint64'):
            number=int(value);assert number==value and (number>=0 if dtype=='uint64' else True)
            row.append(str(number))
        else:
            number=float(value);assert math.isfinite(number),(family,name)
            row.append(format(number,'.9g' if dtype=='float32' else '.17g'))
    return row

def encoded(family,records,run,header=False):
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    if header:writer.writerow(fields(family))
    count=0
    for record in records:
        writer.writerow(normalize(family,record,run));count+=1
    return stream.getvalue().encode(),count

def specification():
    return dict(status='executable_pre_fit_contract',families=SCHEMA,CSV_families=len(SCHEMA),
                columns=sum(len(v['columns']) for v in SCHEMA.values()),design_columns=443,
                refinements=EXTRA,missing='empty field with count/availability/reason context',
                corrected_design_types=TYPE_FIXES,
                float32_precision=9,float64_precision=17)
