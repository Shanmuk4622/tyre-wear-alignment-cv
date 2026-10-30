"""CSV/array meanings accompany every portable inference capture."""
def units(column,table):
 if column in ('accuracy','dice','point_error'):return 'unavailable without reference truth; intentionally blank'
 if column.endswith('_sha256'):return 'SHA-256 hex'
 if column.endswith('_ms'):return 'milliseconds, measured in the worker'
 if column=='timestamp_seconds':return 'seconds; see source timestamp_source'
 if column in ('utc',):return 'UTC ISO-8601'
 if column in ('mean_activation','gated_mass'):return 'dimensionless latent-feature activation / gate-weighted mass'
 if column=='threshold':return 'sigmoid probability threshold'
 if column=='seed':return 'recorded integer RNG seed for the completed training run'
 if column=='epoch':return 'one-based completed training epoch of the selected checkpoint'
 if column=='guide':return 'one-based fixed guide-row index 1–3'
 if column.startswith('probability') or column=='native_area_fraction':return 'fraction; model probability is not calibrated accuracy'
 if 'fraction' in column:return 'fraction of width-1 or height-1, as named'
 if 'logit' in column or 'evidence' in column or column in ('bias','reconstruction_error'):return 'class-logit units'
 if column in ('native_width','native_height','positive_native_pixels','x0','y0','x1_exclusive','y1_exclusive') or column.endswith('_native_pixels'):return 'native oriented pixels; boxes have exclusive upper bounds'
 if column in ('point_index','guide','class_index','frame_index','seed','epoch','channel','row','column'):return 'zero-based index except guide/epoch; class 0/1/2 = low/mid/high proxy'
 if column=='value':return {'mask_probabilities.csv':'sigmoid probability','mask_logits.csv':'segmentation-logit units','signed_evidence.csv':'signed class-logit contribution','latent_textures.csv':'nonnegative latent activation','tread_gate.csv':'spatial weight, sums to one'}.get(table,'numeric value')
 return 'recorded text/identifier/status; blank means unknown/not applicable'
GRAINS={'prediction.csv':'one analysed frame','class_scores.csv':'one frame × class','points.csv':'one frame × ordered boundary point','regions.csv':'one frame × overlapping region','latent_mass.csv':'one frame × latent channel',
 'mask_probabilities.csv':'one region × quarter-grid cell','mask_logits.csv':'one region × quarter-grid cell','signed_evidence.csv':'one class × quarter-grid cell','latent_textures.csv':'one latent channel × quarter-grid cell','tread_gate.csv':'one quarter-grid cell'}
def describe(tables,arrays):
 return dict(schema='tread-one-station-capture-v1',CSV_tables={name:dict(grain=GRAINS[name],rows=len(rows) if isinstance(rows,list) else rows['rows'],columns=[dict(name=k,units=units(k,name)) for k in (list(rows[0]) if isinstance(rows,list) else rows['fields'])]) for name,rows in tables.items()},
  raw_arrays={name:dict(shape=list(v.shape),dtype=str(v.dtype),format='native NPY; allow_pickle=False') for name,v in arrays.items()},
  mask_channels=['tyre','tread'],class_channels=['low mileage proxy','mid mileage proxy','high mileage proxy'],latent_channels='indices 0–11 are learned features, not human-labelled damage types',
  grid=dict(height=128,width=96,input_height=512,input_width=384),notes='Dense CSV values retain every quarter-grid output; native image, model-input PNG and normalization metadata reconstruct input preprocessing.')
