"""CPU engineering checks and training-source atlas illustrations. No fitting."""
import csv,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import numpy as np,torch
from PIL import Image,ImageDraw
from atlas import canonicalize,patches,band_weights
from model import TreadTwoAtlas,load_anchor,WEIGHTS_SHA
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;OUT=HERE/'prototype_report';DATA=ROOT.parent/'phase2/data/phase2_dataset_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(name,rows,fields=None):
 rows=list(rows);fields=fields or list(rows[0]);p=OUT/(name+'.csv')
 with p.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 return dict(rows=len(rows),columns=fields,sha256=sha(p))
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rgb(row):
 p=(DATA/row['image_path']).resolve();assert p.is_relative_to(DATA.resolve()) and sha(p)==row['image_sha256']
 with Image.open(p) as image:return torch.from_numpy(np.array(image.convert('RGB').resize((384,512),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1).copy()/255.)

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261029);station=read(ROOT/'workstation_validation/DELIVERY.json');protected={n:sha(ROOT/n) for n in station['source_hashes']}
 assert protected==station['source_hashes'];write(OUT/'WORKSTATION_SNAPSHOT.json',protected)
 with (ROOT/'data_gate_outputs/csv/source_images.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 train=sorted([r for r in rows if r['historical_study_split']=='train'],key=lambda r:r['image_id']);original=[r for r in train if r['domain']=='original_photo'];anchor=load_anchor();mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 bank=[];labels=[];provenance=[];bank_ids=set();bank_images=[]
 for cls in range(3):
  groups=sorted({r['confirmed_original_tyre_id'] for r in original if int(r['class_index'])==cls});allocation=[6//len(groups)+(i<6%len(groups)) for i in range(len(groups))]
  for tid,count in zip(groups,allocation):
   options=[r for r in original if r['confirmed_original_tyre_id']==tid];chosen=None
   for row in options:
    image=rgb(row)[None]
    with torch.no_grad():dense=anchor((image-mean)/std);atlas=canonicalize(image,dense['masks'])
    alpha,coverage=band_weights(atlas['valid']);valid=[i for i in range(32) if float(coverage[0,i])>=.75]
    if len(valid)>=count:chosen=(row,atlas,valid);break
   assert chosen is not None,'No eligible fixed training-source bank photo for '+tid
   row,atlas,valid=chosen;bank_ids.add(row['image_id']);bank_images.append(dict(image_id=row['image_id'],tyre_id=tid,proxy_class=row['proxy_class'],image_sha256=row['image_sha256'],historical_split='train'))
   # Spread fixed patch choices through the atlas; no candidate class output,
   # validation image or test image participates in source selection.
   selected=[valid[int(i)] for i in np.linspace(0,len(valid)-1,count).round().astype(int)];tiles=patches(atlas['rgb'])[0]
   for index in selected:
    pid=len(bank);bank.append(tiles[index]);labels.append(cls);p=OUT/f'bank_{pid:02d}.png';Image.fromarray((tiles[index].permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)).save(p)
    provenance.append(dict(prototype_index=pid,class_index=cls,proxy_class=row['proxy_class'],source_image_id=row['image_id'],source_image_sha256=row['image_sha256'],source_tyre_id=tid,source_split='train',atlas_patch_index=index,atlas_row=index//4,atlas_column=index%4,valid_fraction=float(coverage[0,index]),clipped_source_rows_fraction=atlas['diagnostics'][0]['clipped_row_fraction'],anchor_weights_sha256=WEIGHTS_SHA,new_independent_tyre=False,model_status='UNTRAINED_ARCHITECTURE_PROTOTYPE'))
 assert len(bank)==18 and len(bank_ids)==8
 bank_tensor=torch.stack(bank);bank_labels=torch.tensor(labels);np.save(OUT/'bank_rgb.npy',bank_tensor.numpy(),allow_pickle=False);np.save(OUT/'bank_labels.npy',bank_labels.numpy(),allow_pickle=False);model=TreadTwoAtlas(bank_tensor,bank_labels,anchor);query_rows=[next(r for r in original if int(r['class_index'])==cls and r['image_id'] not in bank_ids) for cls in range(3)];images=torch.stack([rgb(r) for r in query_rows]);model.train()
 output=model(images)
 with torch.no_grad():reference=anchor((images-mean)/std)
 mask_difference=float((output['masks']-reference['masks']).abs().max());point_difference=float((output['points']-reference['points']).abs().max());assert mask_difference==point_difference==0
 residual=float((model.bias+output['patch_evidence'].sum(1)-output['logits']).abs().max().detach());assert residual<1e-6 and torch.isfinite(output['probabilities']).all();assert torch.allclose(output['probabilities'].sum(1),torch.ones(3),atol=1e-6)
 torch.nn.functional.cross_entropy(output['logits'],torch.arange(3)).backward();assert all(p.grad is None and not p.requires_grad for p in anchor.parameters()) and not anchor.training
 gradient_rows=[dict(parameter=n,gradient_norm=float(p.grad.norm()) if p.grad is not None else None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,scope='engineering backward only; no optimizer step') for n,p in model.named_parameters() if p.requires_grad];assert any(r['gradient_norm'] and r['gradient_norm']>0 for r in gradient_rows) and all(r['finite'] for r in gradient_rows if r['finite'] is not None)
 checks=['bank_uses_only_existing_training_photos','one_bank_photo_per_eight_original_training_tyres','all_bank_images_excluded_from_query_pool','six_real_patch_cases_per_class','dense_masks_bit_equal_to_frozen_anchor','dense_points_bit_equal_to_frozen_anchor','anchor_stays_eval_and_has_no_gradients','new_branch_has_finite_nonzero_gradients','class_probabilities_finite_and_normalized','patch_evidence_exactly_reconstructs_logits']
 with torch.no_grad():single=anchor((images[:1]-mean)/std)
 with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):mixed=model(images[:1])
 autocast_difference=max(float((mixed[k]-single[k]).abs().max()) for k in ('masks','points'));assert autocast_difference==0;checks.append('outer_mixed_precision_context_cannot_change_FP32_dense_anchor')
 toy=torch.rand(1,3,512,384);mask=torch.ones(1,2,128,96);mask[:,:,40:60]=0;hole=canonicalize(toy,mask);assert (hole['valid'][0,0,180:220]==0).all();empty=canonicalize(toy,torch.zeros_like(mask));weights,cover=band_weights(empty['valid']);assert empty['rgb'].sum()==weights.sum()==cover.sum()==0;checks.extend(['unsupported_rows_are_not_filled_with_center_pixels','empty_support_yields_zero_atlas_and_zero_evidence_weights'])
 weights,_=band_weights(output['atlas']['valid']);assert torch.allclose(weights.sum(1),torch.ones(3),atol=1e-6);checks.append('valid_row_band_weights_conserve_total_mass')
 catalog={};catalog['prototype_bank']=emit('prototype_bank',provenance);catalog['bank_images']=emit('bank_images',bank_images);catalog['head_gradient_check']=emit('head_gradient_check',gradient_rows)
 catalog['bank_pixels']=emit('bank_pixels',(dict(prototype_index=p,channel=c,row=y,column=x,value=float(bank_tensor[p,c,y,x])) for p in range(18) for c in range(3) for y in range(64) for x in range(64)),['prototype_index','channel','row','column','value'])
 atlas_rows=[];affinity=[];evidence=[];queries=[]
 for i,row in enumerate(query_rows):
  source=(images[i].permute(1,2,0).numpy()*255).round().astype(np.uint8);atlas_rgb=(output['atlas']['rgb'][i].permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8);Image.fromarray(source).save(OUT/f'query_{i}_input.png');Image.fromarray(atlas_rgb).save(OUT/f'query_{i}_atlas.png')
  queries.append(dict(image_id=row['image_id'],class_index=i,source_split='train',source_image_sha256=row['image_sha256'],is_bank_image=False,status='geometry illustration; classifier branch untrained'))
  grid=output['atlas']['grid'][i].cpu().numpy();valid=output['atlas']['valid'][i].cpu().numpy()[0]
  for y in range(512):
   ok=bool(valid[y].any());atlas_rows.append(dict(image_id=row['image_id'],atlas_row=y,source_y_input_pixels=float((grid[y,0,1]+1)*511/2) if ok else None,left_input_pixels=float((grid[y,0,0]+1)*383/2) if ok else None,right_input_pixels=float((grid[y,-1,0]+1)*383/2) if ok else None,valid_fraction=float(valid[y].mean())))
  for tile in range(32):
   for p in range(18):affinity.append(dict(image_id=row['image_id'],query_patch=tile,prototype_index=p,cosine=float(output['cosine'][i,tile,p].detach()),affinity=float(output['affinity'][i,tile,p].detach()),model_status='untrained; no accuracy claim'))
   for c in range(3):evidence.append(dict(image_id=row['image_id'],query_patch=tile,class_index=c,weight=float(output['patch_weights'][i,tile]),contribution=float(output['patch_evidence'][i,tile,c].detach()),model_status='untrained; engineering accounting only'))
 catalog['query_manifest']=emit('query_manifest',queries);catalog['atlas_rows']=emit('atlas_rows',atlas_rows);catalog['untrained_affinities']=emit('untrained_affinities',affinity);catalog['untrained_patch_evidence']=emit('untrained_patch_evidence',evidence)
 canvas=Image.new('RGB',(1000,615),'#10222e');draw=ImageDraw.Draw(canvas);draw.text((18,10),'TREAD-TWO / ATLAS: existing training photos -> predicted visible-tread coordinates',fill='#d8f4eb')
 for i in range(3):
  x=15+i*330;source=Image.open(OUT/f'query_{i}_input.png').resize((174,232));mapped=Image.open(OUT/f'query_{i}_atlas.png').resize((116,232));canvas.paste(source,(x,45));canvas.paste(mapped,(x+180,45));draw.text((x,287),['low','mid','high'][i]+' training example; NOT a new result',fill='#d7e5ef')
 draw.text((18,323),'Actual training-source patch bank: 6 cases per class. Independent source tyres: 1 / 3 / 4.',fill='#d8f4eb')
 for index,p in enumerate(provenance):
  x=18+(index%6)*155;y=353+(index//6)*79;im=Image.open(OUT/f'bank_{index:02d}.png').resize((64,64));canvas.paste(im,(x,y));draw.text((x+69,y+9),p['proxy_class']+'\ncase '+str(index),fill='#d7e5ef')
 canvas.save(OUT/'TREAD_TWO_Atlas_Concept.png')
 assert all(sha(ROOT/n)==h for n,h in protected.items());checks.append('both_workstation_source_trees_unchanged')
 query_pool=[r for r in train if r['image_id'] not in bank_ids];assert len(query_pool)==378
 report=dict(status='architecture_checks_passed',model_status='UNTRAINED_ARCHITECTURE_PROTOTYPE',checks=checks,check_count=len(checks),total_parameters=sum(p.numel() for p in model.parameters()),frozen_parameters=sum(p.numel() for p in anchor.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
  masks_max_difference=mask_difference,points_max_difference=point_difference,outer_autocast_dense_max_difference=autocast_difference,logit_accounting_residual=residual,bank_cases=18,bank_original_images=8,bank_physical_tyres_by_class=[1,3,4],classifier_query_pool=378,classifier_query_pool_by_class={str(c):sum(int(r['class_index'])==c for r in query_pool) for c in range(3)},atlas_shape=[512,256],patch_grid=[8,4],patch_size=64,embedding_dimension=96,
  engineering_query_images=3,optimizer_updates=0,measured_training=False,trained_classifier_accuracy=None,additional_photo_collection=False,workstation_changes=False,new_independent_tyres=0,grid_sample_CUDA_backward_doc_warns_nondeterminism='nondeterministic' in torch.nn.functional.grid_sample.__doc__,full_training_recipe_frozen=False,Kaggle_training_notebook_delivered=False,
  environment=dict(device='CPU',torch=torch.__version__,numpy=np.__version__,Pillow=__import__('PIL').__version__,python=sys.version),source_hashes={p.name:sha(p) for p in HERE.glob('*.py')},exact_bank_float_values=221184,bank_RGB_SHA256=sha(OUT/'bank_rgb.npy'))
 write(OUT/'CSV_SCHEMA.json',catalog);write(OUT/'PROOF.json',report);write(OUT/'PROTOTYPE_STATUS.json',dict(status='untrained',classifier_weights_published=False,HF_uploads=0,scope='Local architecture/provenance checks, not model performance'))
 write(OUT/'MANIFEST.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
