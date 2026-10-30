"""Pinned TREAD-ONE inference. No original workstation imports or training calls."""
import importlib.util,time
import numpy as np
import torch
from paths import MODELS,ROOT,sha,model_record
from imaging import working_image,mask_boxes,fingerprint

class Engine:
 def __init__(self,device='auto'):
  self.identity=model_record();r=self.identity
  if sha(MODELS/'weights.pt')!=r['weights_sha256'] or sha(MODELS/'tread_one_r1.py')!=r['model_source_sha256']:raise RuntimeError('Station model checksum differs from its declared export. Restore the pinned local package.')
  self.device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
  if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable. Select CPU.')
  torch.set_num_threads(min(4,torch.get_num_threads()));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  spec=importlib.util.spec_from_file_location('tread_one_station_pinned',MODELS/'tread_one_r1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  self.model=module.TreadOne().to(self.device).eval();self.model.load_state_dict(torch.load(MODELS/'weights.pt',map_location='cpu',weights_only=True),strict=True)
  assert sum(p.numel() for p in self.model.parameters())==r['parameters']
  self.environment=dict(device=str(self.device),device_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',torch=torch.__version__,numpy=np.__version__,precision='float32; no autocast; TF32 disabled')
  self.source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')}
 def predict(self,rgb):
  if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected an RGB uint8 image')
  start=time.perf_counter();small=working_image(rgb);image=torch.from_numpy(small.astype(np.float32).transpose(2,0,1).copy()/255.)[None].to(self.device)
  image=(image-image.new_tensor(self.identity['mean'])[None,:,None,None])/image.new_tensor(self.identity['std'])[None,:,None,None]
  if self.device.type=='cuda':torch.cuda.synchronize(self.device)
  ready=time.perf_counter()
  with torch.inference_mode():out=self.model(image)
  if self.device.type=='cuda':torch.cuda.synchronize(self.device)
  elapsed=(time.perf_counter()-ready)*1000
  arrays={k:out[k].detach().float().cpu().numpy().copy() if k=='bias' else out[k][0].detach().float().cpu().numpy().copy() for k in ('probabilities','logits','bias','masks','seg_logits','points','contributions','concepts','gate','guides')}
  if not all(np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Non-finite model output; capture was not accepted')
  residual=float(np.max(np.abs(arrays['contributions'].sum((-2,-1))+arrays['bias']-arrays['logits'])))
  if residual>1e-5:raise RuntimeError('Signed evidence does not reconstruct the logits within tolerance')
  flags=[]
  for i,y in enumerate(arrays['guides']):
   line=arrays['masks'][1,round(float(y)*127)]>=.5;xx=np.where(line)[0]
   if not len(xx):flags.append('Guide '+str(i+1)+': empty predicted tread row')
   elif xx[0]==0 or xx[-1]==95:flags.append('Guide '+str(i+1)+': tread touches image edge')
   elif float(np.max(np.abs(arrays['points'][i*2:i*2+2]-xx[[0,-1]]/95)))>.05:flags.append('Guide '+str(i+1)+': point/mask disagreement over 5% width')
  working=np.repeat(np.repeat(arrays['masks']>=.5,4,axis=-2),4,axis=-1)
  return dict(arrays=arrays,summary=dict(prediction=int(arrays['probabilities'].argmax()),probabilities=arrays['probabilities'].tolist(),logits=arrays['logits'].tolist(),bias=arrays['bias'].tolist(),
   boxes_model_grid=mask_boxes(working),region_fraction=[float(m.mean()) for m in working],points_width_fraction=arrays['points'].tolist(),guides_height_fraction=arrays['guides'].tolist(),
   evidence_residual=residual,model_input_sha256=fingerprint(small),model_ms=elapsed,preprocess_ms=(ready-start)*1000,device=str(self.device),precision=self.environment['precision'],review_flags=flags,flags_scope='Uncalibrated image-space review heuristics'),
   identity=self.identity,environment=self.environment,app_source_hashes=self.source_hashes)
