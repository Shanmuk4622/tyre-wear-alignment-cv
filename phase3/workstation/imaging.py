"""Deterministic model input, reversible display coordinates and signed overlays."""
import base64,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageDraw
COLORS=((245,174,70),(46,216,179))
VIEWS=['Combined','Masks','Boundaries','Signed evidence','Tread attention','Texture channel','Original']
def read_image(path):
 with Image.open(path) as im:
  orientation=im.getexif().get(274,1);rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
 return rgb,dict(exif_orientation=int(orientation),exif_transform_applied=orientation!=1)
def working_image(rgb):return np.array(Image.fromarray(rgb).resize((384,512),Image.Resampling.BILINEAR))
def fingerprint(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
def pack(array):
 a=np.ascontiguousarray(array);return dict(shape=list(a.shape),dtype=a.dtype.str,data=base64.b64encode(a.tobytes()).decode('ascii'))
def unpack(value):
 dtype=np.dtype(value['dtype']);shape=tuple(value['shape']);assert dtype.kind in 'fbiu' and 0<len(shape)<=4 and all(type(n) is int and 0<n<=4096 for n in shape)
 length=int(np.prod(shape))*dtype.itemsize;assert length<=16*1024**2
 raw=base64.b64decode(value['data'],validate=True);assert len(raw)==length
 return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
def masks_at(arrays,size):
 # Repeat-by-four matches frozen reporting; native nearest resize is display.
 masks=np.repeat(np.repeat(arrays['masks']>=.5,4,axis=-2),4,axis=-1)
 return np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize(size,Image.Resampling.NEAREST))>0 for m in masks])
def mask_boxes(masks):
 result=[]
 for mask in masks:
  yy,xx=np.where(mask);result.append(None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1])
 return result
def heat_rgb(values):
 a=np.asarray(values,dtype=np.float32);scale=max(float(np.max(np.abs(a))),1e-12);v=np.clip(a/scale,-1,1)
 neutral=np.full((*a.shape,3),28.,dtype=np.float32);positive=np.array([255,116,89]);negative=np.array([67,171,255]);color=np.where((v>=0)[...,None],positive,negative)
 return (neutral*(1-np.abs(v)[...,None])+color*np.abs(v)[...,None]).astype(np.uint8)
def render(rgb,result,view='Combined',opacity=.35,show_tyre=True,show_tread=True,show_boxes=True,show_points=True,show_guides=True,evidence_class=None,concept=0):
 if result is None or view=='Original':return rgb.copy()
 # Display is bounded; preserve original pixels separately in evidence.
 h,w=rgb.shape[:2];scale=min(1.,1100/max(h,w));size=(max(1,round(w*scale)),max(1,round(h*scale)))
 base=np.array(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR));a=result['arrays'];masks=masks_at(a,size);cls=int(np.argmax(a['probabilities'])) if evidence_class is None else int(evidence_class)
 if view in ('Signed evidence','Tread attention','Texture channel'):
  values=a['contributions'][cls] if view=='Signed evidence' else a['gate'][0] if view=='Tread attention' else a['concepts'][int(concept)]
  color=heat_rgb(values if view=='Signed evidence' else values/max(float(np.max(values)),1e-12))
  layer=np.array(Image.fromarray(color).resize(size,Image.Resampling.BILINEAR));base=(base*.35+layer*.65).astype(np.uint8)
 if view in ('Combined','Masks'):
  for mask,color,show in zip(masks,COLORS,(show_tyre,show_tread)):
   if show:base[mask]=(base[mask]*(1-opacity)+np.array(color)*opacity).astype(np.uint8)
 image=Image.fromarray(base);draw=ImageDraw.Draw(image);ww,hh=size;line=max(1,round(ww/300))
 if view in ('Combined','Masks') and show_boxes:
  for box,color,show in zip(mask_boxes(masks),COLORS,(show_tyre,show_tread)):
   if box and show:draw.rectangle([box[0],box[1],box[2]-1,box[3]-1],outline=color,width=line)
 if view in ('Combined','Boundaries'):
  for i,y in enumerate(a['guides']):
   yy=float(y)*(hh-1);left,right=[float(x)*(ww-1) for x in a['points'][2*i:2*i+2]]
   if show_guides:draw.line((0,yy,ww-1,yy),fill=(121,151,168),width=1)
   if show_points:
    draw.line((left,yy,right,yy),fill=(252,231,139),width=line+1)
    radius=max(3,ww//140)
    for x in (left,right):draw.ellipse((x-radius,yy-radius,x+radius,yy+radius),fill=(255,231,122),outline=(15,29,42),width=line)
 return np.array(image)
