import numpy as np
from PySide6.QtCore import Qt,QRectF,QPointF
from PySide6.QtGui import QImage,QPainter,QPen,QColor,QFont
from PySide6.QtWidgets import QWidget
def qimage(rgb):
 a=np.ascontiguousarray(rgb);return QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format_RGB888).copy()
class Canvas(QWidget):
 def __init__(self):
  super().__init__();self.original=None;self.processed=None;self.wipe=False;self.fraction=.5;self.zoom=1.;self.pan=QPointF();self.drag=None;self.setMinimumSize(310,260);self.setMouseTracking(True)
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
  p=QPainter(self);p.fillRect(self.rect(),QColor('#252729'))
  if self.original is None:
   p.setPen(QColor('#c7c9cb'));p.setFont(QFont('Segoe UI',11));p.drawText(self.rect(),Qt.AlignCenter,'No frame loaded');return
  p.setRenderHint(QPainter.SmoothPixmapTransform);r=self.area();p.drawImage(r,self.processed if self.processed is not None and not self.wipe else self.original)
  if self.wipe and self.processed is not None:
   split=r.x()+r.width()*self.fraction;p.save();p.setClipRect(QRectF(split,r.top(),r.right()-split,r.height()));p.drawImage(r,self.processed);p.restore();p.setPen(QPen(QColor('#f0f0ec'),2));p.drawLine(QPointF(split,r.top()),QPointF(split,r.bottom()))
  p.setFont(QFont('Segoe UI',9));p.setPen(QColor('#c2c5c7'));p.drawText(14,22,'Original / overlay' if self.wipe else 'Analysis frame');p.drawText(14,self.height()-10,'Drag to compare' if self.wipe else 'Scroll to zoom · drag to pan')
class ClassBars(QWidget):
 def __init__(self):super().__init__();self.values=None;self.setMinimumHeight(111)
 def set_values(self,values):self.values=values;self.update()
 def paintEvent(self,event):
  p=QPainter(self);labels=['Low mileage proxy','Mid mileage proxy','High mileage proxy']
  for i,label in enumerate(labels):
   y=5+i*34;p.setPen(QColor('#252729'));p.drawText(0,y+10,label);value=0 if self.values is None else float(self.values[i]);p.drawText(self.width()-51,y+10,'—' if self.values is None else f'{100*value:.1f}%');p.fillRect(QRectF(0,y+17,self.width(),5),QColor('#e0dfda'));p.fillRect(QRectF(0,y+17,self.width()*value,5),QColor('#526d80'))
