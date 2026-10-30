"""Real exported-model fidelity and direct-evidence checks; no fitting."""
import csv,importlib.util,json,shutil,sys,uuid
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'workstation';OUT=ROOT/'workstation_validation';sys.path.insert(0,str(APP))
import numpy as np,torch
from engine import Engine
from imaging import read_image,working_image,fingerprint,render,masks_at
from records import save_capture,load_capture,Session
from paths import read,sha,write_json

def main():
 checks=[];baseline=read(OUT/'ORIGINAL_BASELINE.json');assert all(sha(ROOT.parent/n)==h for n,h in baseline.items());checks.append('original_workstation_and_registry_unchanged')
 with (ROOT/'data_gate_outputs/csv/source_images.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 row=next(r for r in rows if r['historical_study_split']=='train' and r['proxy_class']=='low' and r['domain']=='original_photo');path=ROOT.parent/'phase2/data/phase2_dataset_v1'/row['image_path'];rgb,_=read_image(path)
 engine=Engine('cpu');result=engine.predict(rgb);a=result['arrays'];checks.append('pinned_validation_selected_export_real_CPU_inference')
 spec=importlib.util.spec_from_file_location('tread_one_original_reference',ROOT/'runtime/model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);reference=module.TreadOne().eval();reference.load_state_dict(torch.load(APP/'models/weights.pt',weights_only=True,map_location='cpu'))
 from PIL import Image
 with Image.open(path) as image:array=np.asarray(image.convert('RGB').resize((384,512),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1).copy()/255.
 tensor=torch.tensor(array)[None];tensor=(tensor-tensor.new_tensor([.485,.456,.406])[None,:,None,None])/tensor.new_tensor([.229,.224,.225])[None,:,None,None]
 with torch.inference_mode():out=reference(tensor)
 differences={}
 for name,value in a.items():
  expected=out[name].detach().numpy() if name=='bias' else out[name][0].detach().numpy();differences[name]=float(np.max(np.abs(expected-value)));np.testing.assert_allclose(value,expected,atol=1e-6,rtol=1e-6)
 checks.append('all_ten_output_arrays_match_frozen_original_reference')
 assert abs(float(a['probabilities'].sum())-1)<1e-6 and a['points'].shape==(6,) and all(a['points'][i]<a['points'][i+1] for i in (0,2,4));assert result['summary']['evidence_residual']<1e-5;checks.append('probabilities_ordered_points_and_exact_additive_evidence')
 for view in ('Combined','Masks','Boundaries','Signed evidence','Tread attention','Texture channel','Original'):
  image=render(rgb,result,view,evidence_class=2,concept=11);assert image.ndim==3 and image.dtype==np.uint8
 checks.append('all_seven_views_render_without_another_forward_pass')
 fixture=OUT/('engine_'+uuid.uuid4().hex[:8]);fixture.mkdir();source=dict(path=str(path),media_kind='photo',reference_proxy_class='low',reference_scope='Training fixture; not an accuracy benchmark')
 capture=save_capture(rgb,result,source,'CSV, "quotes" and native evidence',fixture/'captures');restored,second,meta,note=load_capture(capture);np.testing.assert_array_equal(rgb,restored)
 for name in a:np.testing.assert_array_equal(a[name],second['arrays'][name])
 assert note=='CSV, "quotes" and native evidence';checks.append('full_direct_CSV_native_capture_roundtrip_exact')
 csv_rows={}
 for p in capture.glob('*.csv'):
  with p.open(newline='',encoding='utf-8') as f:csv_rows[p.name]=sum(1 for _ in csv.DictReader(f))
 assert len(csv_rows)==10 and sum(csv_rows.values())==245784
 saved_schema=read(capture/'CSV_SCHEMA.json');assert len(saved_schema['CSV_tables'])==10 and sum(t['rows'] for t in saved_schema['CSV_tables'].values())==245784
 for name,filename in {'masks':'mask_probabilities','seg_logits':'mask_logits','contributions':'signed_evidence','concepts':'latent_textures','gate':'tread_gate'}.items():
  restored_grid=np.empty_like(a[name])
  with (capture/(filename+'.csv')).open(newline='',encoding='utf-8') as f:
   for r in csv.DictReader(f):restored_grid[int(r['channel']),int(r['row']),int(r['column'])]=float(r['value'])
  np.testing.assert_array_equal(restored_grid,a[name])
 checks.append('all_245760_dense_CSV_values_roundtrip_exact_with_units_schema')
 with (capture/'prediction.csv').open(newline='',encoding='utf-8') as f:saved=next(csv.DictReader(f));assert saved['accuracy']==saved['dice']==saved['point_error']==''
 checks.append('ten_CSV_tables_245784_rows_and_unknown_metrics_not_invented')
 session=Session(engine.identity,fixture/'sessions');session.prediction(rgb,result,source);assert len(list(session.root.glob('*.csv')))==5;checks.append('automatic_per_prediction_class_point_region_and_latent_ledger')
 damaged=fixture/'damaged';shutil.copytree(capture,damaged);p=damaged/'points.npy';body=bytearray(p.read_bytes());body[-1]^=1;p.write_bytes(body)
 try:load_capture(damaged);raise AssertionError('Corrupt capture accepted')
 except AssertionError as exc:assert 'checksum' in str(exc).lower()
 checks.append('corrupted_capture_rejected_before_loading_arrays')
 gpu_difference=None
 if torch.cuda.is_available():
  gpu=Engine('cuda:0');other=gpu.predict(rgb);gpu_difference={name:float(np.max(np.abs(a[name]-other['arrays'][name]))) for name in ('probabilities','points','masks')}
  for name in gpu_difference:np.testing.assert_allclose(a[name],other['arrays'][name],atol=5e-4,rtol=1e-3)
  checks.append('real_GPU_FP32_inference_agrees_with_CPU_with_recorded_tolerance')
 assert all(sha(ROOT.parent/n)==h for n,h in baseline.items());checks.append('original_files_still_byte_unchanged_after_inference_and_export')
 report=dict(status='passed',checks=checks,check_count=len(checks),source_image=str(path),source_frame_sha256=fingerprint(rgb),frozen_reference_max_differences=differences,GPU_max_differences=gpu_difference,
  CSV_tables=10,CSV_rows=sum(csv_rows.values()),CSV_rows_by_table=csv_rows,fixture_capture=str(capture),fixture_root=str(fixture),model_seed=3,model_epoch=20,new_training=False,new_independent_tyres=0,new_inference=True,original_files_preserved=len(baseline))
 write_json(OUT/'ENGINE_VALIDATION.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
