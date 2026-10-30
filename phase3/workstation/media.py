"""Owned local video/camera reader; never opens a camera automatically."""
import math
import cv2
from PySide6.QtCore import QObject,QTimer,Signal
class Media(QObject):
 frame=Signal(object,object);error=Signal(str);ended=Signal()
 def __init__(self,parent=None):super().__init__(parent);self.timer=QTimer(self);self.timer.timeout.connect(self.step);self.cap=None;self.rgb=None;self.meta={};self.index=-1;self.fps=25.;self.count=0;self.angle=0
 @property
 def playing(self):return self.timer.isActive()
 def open(self,path,camera=False):
  self.close();self.cap=cv2.VideoCapture(0 if camera else str(path))
  if not self.cap.isOpened():self.close();raise RuntimeError('Could not open the camera.' if camera else 'Could not decode this video.')
  self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0);angle=self.cap.get(cv2.CAP_PROP_ORIENTATION_META);self.angle=int(round(angle))%360 if math.isfinite(angle) else 0
  if self.angle not in (0,90,180,270):self.angle=0
  fps=self.cap.get(cv2.CAP_PROP_FPS);self.fps=fps if math.isfinite(fps) and 1<=fps<=240 else 25.;self.count=max(0,int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) if not camera else 0
  self.meta=dict(path='Camera 0' if camera else str(path),media_kind='camera' if camera else 'video',rotation_clockwise=self.angle,fps=self.fps,source_media_sha256=None,source_hash_reason='Decoded-frame RGB SHA-256 recorded; full video bytes not hashed during live playback')
  self.index=-1;self.step()
 def step(self):
  if self.cap is None:return
  ok,frame=self.cap.read()
  if not ok:self.pause();self.ended.emit();return
  self.index+=1
  if self.angle:frame=cv2.rotate(frame,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[self.angle])
  self.rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);self.frame.emit(self.rgb,dict(self.meta,frame_index=self.index,timestamp_seconds=self.index/self.fps,timestamp_source='decoded frame index / reported fps'))
 def play(self):
  if self.cap is not None:self.timer.start(max(8,round(1000/self.fps)))
 def pause(self):self.timer.stop()
 def seek(self,index):
  if self.cap is not None and self.count:
   self.pause();index=max(0,min(self.count-1,int(index)));self.cap.set(cv2.CAP_PROP_POS_FRAMES,index);self.index=index-1;self.step()
 def close(self):
  self.timer.stop()
  if self.cap is not None:self.cap.release()
  self.cap=None;self.rgb=None;self.meta={};self.count=0
