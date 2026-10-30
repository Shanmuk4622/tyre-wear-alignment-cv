import numpy as np
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QPainter,QPen,QColor,QFont
from PySide6.QtWidgets import QWidget
def qimage(rgb):
 a=np.ascontiguousarray(rgb);return QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format_RGB888).copy()
class Canvas(QWidget):
 def __init__(self):
  super().__init__();self.original=None;self.processed=None;self.wipe=False;self.fraction=.5;self.zoom=1.;self.pan=QPointF();self.drag=None;self.setMinimumSize(360,330);self.setMouseTracking(True)
 def set_frame(self,rgb,processed=None):self.original=qimage(rgb);self.processed=qimage(processed) if processed is not None else None;self.update()
 def reset_view(self):self.zoom=1.;self.pan=QPointF();self.update()
 def area(self):
  area=self.rect().adjusted(18,36,-18,-30);size=self.original.size().scaled(area.size(),Qt.KeepAspectRatio);w,h=size.width()*self.zoom,size.height()*self.zoom;return QRectF(area.center().x()-w/2+self.pan.x(),area.center().y()-h/2+self.pan.y(),w,h)
 def mousePressEvent(self,event):self.drag=event.position()
 def mouseMoveEvent(self,event):
  if not self.original or not event.buttons()&Qt.LeftButton:return
  if self.wipe and self.processed:self.fraction=max(0,min(1,(event.position().x()-self.area().x())/self.area().width()))
  elif self.drag is not None:self.pan+=event.position()-self.drag;self.drag=event.position()
  self.update()
 def wheelEvent(self,event):self.zoom=max(1.,min(5.,self.zoom*(1.15 if event.angleDelta().y()>0 else 1/1.15)));self.update()
 def paintEvent(self,event):
  p=QPainter(self);p.fillRect(self.rect(),QColor('#08151e'));p.setPen(QPen(QColor('#132733'),1))
  for x in range(0,self.width(),28):p.drawLine(x,0,x,self.height())
  for y in range(0,self.height(),28):p.drawLine(0,y,self.width(),y)
  if self.original is None:
   p.setPen(QColor('#66d9bd'));p.setFont(QFont('Segoe UI',25,QFont.Bold));p.drawText(self.rect().adjusted(0,-40,0,0),Qt.AlignCenter,'One model. Four views.');p.setFont(QFont('Segoe UI',11));p.setPen(QColor('#89a8b9'));p.drawText(self.rect().adjusted(25,55,-25,0),Qt.AlignCenter,'Open an existing photo or choose a study example.\nMasks · boundaries · mileage proxy · signed evidence');return
  p.setRenderHint(QPainter.SmoothPixmapTransform);r=self.area();p.drawImage(r,self.processed if self.processed is not None and not self.wipe else self.original)
  if self.wipe and self.processed is not None:
   split=r.x()+r.width()*self.fraction;p.save();p.setClipRect(QRectF(split,r.top(),r.right()-split,r.height()));p.drawImage(r,self.processed);p.restore();p.setPen(QPen(QColor('#65eac6'),2));p.drawLine(QPointF(split,r.top()),QPointF(split,r.bottom()))
  p.setFont(QFont('Segoe UI',9));p.setPen(QColor('#89a7b7'));p.drawText(14,22,'ORIGINAL  /  TREAD-ONE' if self.wipe else 'TREAD-ONE INSPECTION');p.drawText(14,self.height()-10,'Drag to compare' if self.wipe else 'Scroll to zoom · drag to pan')
class ClassBars(QWidget):
 def __init__(self):super().__init__();self.values=None;self.setMinimumHeight(155)
 def set_values(self,values):self.values=values;self.update()
 def paintEvent(self,event):
  p=QPainter(self);colors=['#66ddbd','#65b9e8','#f1b25d'];labels=['Low mileage proxy','Mid mileage proxy','High mileage proxy']
  for i,label in enumerate(labels):
   y=12+i*49;p.setPen(QColor('#cadde6'));p.drawText(0,y+10,label);value=0 if self.values is None else float(self.values[i]);p.drawText(self.width()-65,y+10,'—' if self.values is None else f'{100*value:.1f}%');p.setPen(Qt.NoPen);p.setBrush(QColor('#233d4b'));p.drawRoundedRect(QRectF(0,y+21,self.width(),6),3,3);p.setBrush(QColor(colors[i]));p.drawRoundedRect(QRectF(0,y+21,self.width()*value,6),3,3)
