"""Real GPU checks of Phase 2 exports, app interfaces and three development videos."""
import sys,pathlib,json,threading,traceback,time,gc
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'prototype'));import bootstrap
import numpy as np
import torch
import phase2_adapter as a
from engine import Engine,read_image
from evidence import save_capture
from video import configure_capture,portrait_frame
import cv2

@torch.inference_mode()
def main():
 a.enabled=lambda:True
 torch.set_num_threads(4)
 reg=json.loads(a.REGISTRY.read_text());data=ROOT/'phase2/data/phase2_dataset_v1';rows=json.loads((data/'manifests/images.json').read_text());byid={r['image_id']:r for r in rows}
 engine=Engine('cuda:0');result={'device':torch.cuda.get_device_name(),'models':{},'videos':[]}
 for name in ['mobilenetv4','resnet50','segformer_b0','yolo26n_seg']:
  spec=a.spec(name);final=json.loads((ROOT/'phase2/completion/runs'/spec['job']['id']/'FINAL.json').read_text());r=final['test']['records'][0];rgb=read_image(data/byid[r['image_id']]['image_path'])
  record,masks=engine.predict(rgb,name);again,mm=engine.predict(rgb,name)
  if masks is None:
   assert record['prediction']==r['prediction'];np.testing.assert_allclose(record['scores'],again['scores'],atol=1e-6)
  else:assert masks.shape==(2,*rgb.shape[:2]) and np.array_equal(masks,mm)
  (model,_,_),_=engine.load(name);import phase2_models as models
  expected=models.predict(model,{'img':a.tensor(rgb,name,engine.device)},a.ALIASES[name])
  if masks is not None:
   native=torch.nn.functional.interpolate(expected.float(),rgb.shape[:2],mode='nearest')[0].bool().cpu().numpy();assert np.array_equal(native,masks)
  result['models'][name]=dict(seed=spec['job']['seed'],inference_ms=again['inference_ms'],load_ms=record['load_ms'],repeatable=True,training_adapter_parity=True)
  print(name,result['models'][name],flush=True)
 hr=a.spec('hrnet');f=json.loads((ROOT/'phase2/completion/runs'/hr['job']['id']/'FINAL.json').read_text());r=f['test']['records'][0];rgb=read_image(data/byid[r['image_id']]['image_path'])
 from learned_geometry import LearnedEngine
 learned=LearnedEngine('cuda:0');lr=learned.inspect(rgb,'paired');xs=np.array([p[0]/(rgb.shape[1]-1) for p in lr['models']['hrnet']['points']]);np.testing.assert_allclose(xs,r['prediction'],atol=2e-5)
 result['models']['hrnet']=dict(seed=hr['job']['seed'],inference_ms=lr['models']['hrnet']['inference_ms'],export_prediction_matches=True)
 del learned;gc.collect();torch.cuda.empty_cache()
 for video in sorted((ROOT/'Videos').glob('*.mp4')):
  cap=cv2.VideoCapture(str(video));angle=configure_capture(cap);records=[]
  for fraction in [.1,.5,.9]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)*fraction));ok,bgr=cap.read();assert ok
   frame,_=portrait_frame(bgr,angle);rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   inspection,masks=engine.inspect(rgb,classifier='mobilenetv4',region='yolo26n_seg',learned='paired')
   assert all(m.shape==(2,*rgb.shape[:2]) for m in masks.values())
   records.append(dict(fraction=fraction,total_ms=inspection['total_ms'],peak_vram_mb=inspection['peak_vram_mb'],models=inspection['models'],learned_geometry=inspection['learned_geometry']))
  cap.release();result['videos'].append(dict(video=video.name,frames=records));print(video.name,'3 frames passed',flush=True)
 folder=save_capture(rgb,inspection,masks,'phase2 validation','Phase 2 GPU integration check; training-cohort video, not unseen evaluation.')
 assert json.loads((folder/'inspection.json').read_text())['learned_geometry']==inspection['learned_geometry']
 result['evidence']=str(folder);result['status']='passed';(ROOT/'phase2/workstation/validation.json').write_text(json.dumps(result,indent=2));print('PASS',flush=True)

if __name__=='__main__':
 threading.stack_size(32*1024*1024);errors=[]
 def run():
  try:main()
  except BaseException as e:errors.append(e);traceback.print_exc()
 t=threading.Thread(target=run);t.start();t.join()
 if errors:raise SystemExit(1)



