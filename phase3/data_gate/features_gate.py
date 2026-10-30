"""Measured image/annotation descriptors; no neural network or label fitting."""
from PIL import Image
import numpy as np
from common_gate import inside
HW=(512,384)

def dhash(image):
 gray=np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR),dtype=np.uint8)
 value=0
 for bit in (gray[:,1:]>gray[:,:-1]).ravel():value=(value<<1)|int(bit)
 return f'{value:016x}'

def region_geometry(mask,valid,width,height):
 a=mask&valid;ys,xs=np.where(a);pixels=int(a.sum());result=dict(valid_pixels=int(valid.sum()),positive_pixels=pixels,area_fraction=pixels/max(int(valid.sum()),1))
 result.update({k:None for k in ('x0','y0','x1_exclusive','y1_exclusive','centroid_x_fraction','centroid_y_fraction','width_fraction','height_fraction')})
 if pixels:
  result.update(x0=int(xs.min()),y0=int(ys.min()),x1_exclusive=int(xs.max())+1,y1_exclusive=int(ys.max())+1,
   centroid_x_fraction=float(xs.mean()/max(width-1,1)),centroid_y_fraction=float(ys.mean()/max(height-1,1)),
   width_fraction=float((xs.max()-xs.min()+1)/width),height_fraction=float((ys.max()-ys.min()+1)/height))
 return result

def describe(root,row,geometry):
 with Image.open(inside(root,row['image_path'])) as image:
  assert image.size==(row['width'],row['height']);orientation=image.getexif().get(274,1)
  decoded=image.convert('RGB');fingerprint=dhash(decoded)
  rgb=np.asarray(decoded.resize((HW[1],HW[0]),Image.Resampling.BILINEAR),dtype=np.float32)
 masks=[]
 for name in ('tyre_mask','tread_mask','ignore_mask'):
  with Image.open(inside(root,row[name])) as image:
   assert image.size==(row['width'],row['height']);a=np.asarray(image)
   assert a.ndim==2 and set(np.unique(a))<=set((0,1,255)),(row['image_id'],name,'binary mask expected')
   masks.append(a>0)
 tyre,tread,ignore=masks;valid=~ignore;records=[]
 for name,mask in (('tyre',tyre),('tread',tread)):records.append(dict(image_id=row['image_id'],region=name,native_width=row['width'],native_height=row['height'],**region_geometry(mask,valid,row['width'],row['height'])))
 work=[]
 for mask in masks:work.append(np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((HW[1],HW[0]),Image.Resampling.NEAREST))>0)
 t,w,ig=work;ok=~ig;gray=rgb@np.array([.299,.587,.114],dtype=np.float32)
 lap=gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1]
 hist=np.bincount(np.clip(gray.astype(int),0,255).ravel(),minlength=256)/gray.size;positive=hist>0
 fg=(t|w)&ok;bg=~(t|w)&ok
 def mean(mask):return float(gray[mask].mean()) if mask.any() else None
 saturation=np.divide(rgb.max(2)-rgb.min(2),rgb.max(2),out=np.zeros(gray.shape,dtype=np.float32),where=rgb.max(2)>0)
 feature=dict(image_id=row['image_id'],domain=row['domain'],class_index=row['class_index'],historical_split=row['role'],dhash64=fingerprint,
  quality_height=HW[0],quality_width=HW[1],native_width=row['width'],native_height=row['height'],native_aspect_ratio=row['width']/row['height'],
  exif_orientation=int(orientation),exif_transform_applied=False,brightness_0_255=float(gray.mean()),contrast_std_0_255=float(gray.std()),
  gray_entropy_bits=float(-(hist[positive]*np.log2(hist[positive])).sum()),focus_laplacian_variance=float(lap.var()),mean_saturation_0_1=float(saturation.mean()),
  red_mean_0_255=float(rgb[:,:,0].mean()),green_mean_0_255=float(rgb[:,:,1].mean()),blue_mean_0_255=float(rgb[:,:,2].mean()),
  foreground_brightness_0_255=mean(fg),background_brightness_0_255=mean(bg),foreground_pixels_on_quality_grid=int(fg.sum()),background_pixels_on_quality_grid=int(bg.sum()),
  tread_outside_tyre_native_pixels=int((tread&~tyre).sum()),tread_outside_tyre_valid_pixels=int((tread&~tyre&valid).sum()),
  source_reported_annotation_conflict_pixels=row.get('annotation_conflict_pixels'),native_ignored_pixels=int(ignore.sum()),
  quality_scope='deterministic 512x384 stored-orientation descriptors; no physical wear or causal shortcut measurement')
 if feature['foreground_brightness_0_255'] is not None and feature['background_brightness_0_255'] is not None:
  feature['foreground_minus_background_brightness']=feature['foreground_brightness_0_255']-feature['background_brightness_0_255']
 else:feature['foreground_minus_background_brightness']=None
 points=[];record=geometry.get(row['image_id']);kind='human' if row['point_kind']=='human' else 'weak_polygon_proposal' if record else 'unavailable'
 if record:
  for index,point in enumerate(record['points']):
   x,y=float(point['x']),float(point['y']);assert -.000001<=x<=row['width']-1+.000001 and 0<=y<=row['height']-1
   points.append(dict(image_id=row['image_id'],point_index=index,point_name=point['name'],point_kind=kind,human_eligible=kind=='human',
    source_x_pixels=x,source_y_pixels=y,x_width_fraction=x/(row['width']-1),y_height_fraction=y/(row['height']-1),
    native_width=row['width'],native_height=row['height'],source_label=record['label_source']))
 coverage=dict(image_id=row['image_id'],point_kind=kind,point_records=len(points),human_point_records=len(points) if kind=='human' else 0,
  optimization_weight=row['point_weight'],historical_split=row['role'],independent_human_truth=kind=='human')
 return dict(quality=feature,regions=records,points=points,point_coverage=coverage)
