"""Phase 3 native inspection workstation."""
import bootstrap
from legacy_bench import prepare_dependencies
# Use the bench's existing OpenBLAS NumPy wheel before either tab loads NumPy.
# This keeps Conda's OpenMP runtime out of the UI process when geometry runs first.
prepare_dependencies()

import argparse,csv,json,sys,time,traceback
from pathlib import Path
from PySide6.QtCore import Qt,QTimer,QUrl
from PySide6.QtGui import QPixmap,QShortcut,QKeySequence,QDesktopServices
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QLabel,QPushButton,QVBoxLayout,QHBoxLayout,QFrame,QComboBox,QLineEdit,QFileDialog,QListWidget,QListWidgetItem,QSplitter,QScrollArea,QSpinBox,QSlider,QCheckBox,QTabWidget,QTableWidget,QTableWidgetItem,QHeaderView,QDialog,QTextBrowser,QLayout)
import numpy as np
from paths import ROOT,PHASE3,DATA,CACHE,read,write_json,sha,model_record
from imaging import read_image,render,working_image,fingerprint,VIEWS,masks_at,mask_boxes
from widgets import Canvas,ClassBars,qimage
from bridge import InferenceBridge,CaptureBridge
from media import Media
from records import Session,load_capture
from theme import apply

def label(text='',name=None):
 w=QLabel(text);w.setWordWrap(True)
 if name:w.setObjectName(name)
 return w
def button(text,callback,primary=False):
 w=QPushButton(text);w.setMinimumHeight(29);w.clicked.connect(callback)
 if primary:w.setObjectName('primary')
 return w
def section(layout,title):layout.addWidget(label(title,'section'))
def scroll(widget):
 area=QScrollArea();area.setWidgetResizable(True);area.setWidget(widget);area.setFrameShape(QFrame.NoFrame);return area
def clock(seconds):
 if seconds is None:return ''
 seconds=float(seconds);return f'{int(seconds)//60:02d}:{seconds%60:06.3f}'

class Station(QMainWindow):
 def __init__(self,device='auto',auto_example=False,capture_root=None,session_root=None):
  super().__init__();self.setWindowTitle('TREAD-ONE');self.setMinimumSize(960,570)
  area=QApplication.primaryScreen().availableGeometry();self.resize(min(1510,area.width()-40),min(980,area.height()-18));self.rgb=None;self.result=None;self.source={};self.input_rgb=None;self.input_source={};self.generation=0;self.request_id=0;self.pending={};self.inflight=False;self.model_ready=False;self.device=device;self.worker=None;self.exporter=None;self.closing=False;self.auto_example=auto_example;self.last_export=None;self.errors=[]
  self.capture_root=Path(capture_root or DATA/'captures').resolve();self.capture_root.mkdir(parents=True,exist_ok=True)
  self.identity=model_record();self.session=Session(self.identity,session_root);self.session.event('opened','TREAD-ONE Studio; separate workstation; existing data only')
  self.alignment_dialog=None
  self.media=Media(self);self.media.frame.connect(self.media_frame);self.media.ended.connect(self.media_ended);self.media.error.connect(self.show_error)
  self.live_timer=QTimer(self);self.live_timer.setInterval(700);self.live_timer.timeout.connect(self.live_tick)
  self.build();self.setAcceptDrops(True);self.reload_captures();self.start_model(device)
  QShortcut(QKeySequence('Ctrl+O'),self,activated=self.choose_photo);QShortcut(QKeySequence('Ctrl+S'),self,activated=self.save_capture);QShortcut(QKeySequence('Space'),self,activated=self.toggle_play)
  if auto_example:self.load_example()
 def build(self):
  host=QWidget();root=QVBoxLayout(host);root.setContentsMargins(10,8,10,7);root.setSpacing(7);self.setCentralWidget(host)
  body=QHBoxLayout();body.setSpacing(8);root.addLayout(body,1)
  rail=QFrame();rail.setObjectName('rail');rail.setFixedWidth(222);rl=QVBoxLayout(rail);rl.setContentsMargins(13,16,13,14);rl.setSpacing(9);rl.setSizeConstraint(QLayout.SetMinimumSize)
  section(rl,'Source');rl.addWidget(button('Open photo…',self.choose_photo));rl.addWidget(button('Open video…',self.choose_video));rl.addWidget(button('Open camera',self.open_camera))
  rl.addWidget(label('Existing study examples','muted'));self.examples=QComboBox();self.examples.addItem('Choose a study example',None);self.examples.setMaximumWidth(194);self.sample_rows=self.existing_examples()
  for r in self.sample_rows:self.examples.addItem(r['proxy_class'].title()+' · '+r['historical_study_split'],r)
  self.examples.currentIndexChanged.connect(self.example_changed);rl.addWidget(self.examples)
  section(rl,'Playback');self.preview=label('No live preview','muted');self.preview.setObjectName('feed');self.preview.setAlignment(Qt.AlignCenter);self.preview.setMinimumHeight(96);self.preview.setMaximumHeight(142);self.preview.hide();rl.addWidget(self.preview)
  controls=QHBoxLayout();self.play_btn=button('Play',self.toggle_play);self.play_btn.setEnabled(False);controls.addWidget(self.play_btn);self.freeze_btn=button('Freeze',self.freeze);self.freeze_btn.setEnabled(False);controls.addWidget(self.freeze_btn);rl.addLayout(controls)
  self.seek=QSlider(Qt.Horizontal);self.seek.setRange(0,0);self.seek.sliderReleased.connect(self.seek_frame);rl.addWidget(self.seek);self.feed_time=label('No video open','muted');rl.addWidget(self.feed_time)
  self.live=QCheckBox('Inspect while playing');self.live.setToolTip('At most one inference in flight; completed snapshots stay tied to their exact decoded frame.');self.live.toggled.connect(self.live_changed);rl.addWidget(self.live)
  section(rl,'Processing');self.device_choice=QComboBox();self.device_choice.addItem('Auto · local GPU / CPU','auto');self.device_choice.addItem('CPU','cpu');self.device_choice.addItem('GPU 0','cuda:0');self.device_choice.setCurrentIndex(max(0,self.device_choice.findData(self.device)));self.device_choice.currentIndexChanged.connect(self.change_device);rl.addWidget(self.device_choice)
  self.ready_label=label('Loading the pinned model…','muted');rl.addWidget(self.ready_label)
  section(rl,'Saved captures');self.captures=QListWidget();self.captures.setMinimumHeight(65);self.captures.setMaximumHeight(112);self.captures.itemDoubleClicked.connect(self.restore_item);rl.addWidget(self.captures,1);rl.addWidget(button('Restore from folder…',self.choose_capture));rl.addWidget(button('Open evidence folder',self.open_evidence));rl.addStretch();self.rail_scroll=scroll(rail);self.rail_scroll.setFixedWidth(234);self.rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);body.addWidget(self.rail_scroll)
  splitter=QSplitter(Qt.Horizontal);splitter.setHandleWidth(9);body.addWidget(splitter,1)
  middle=QWidget();ml=QVBoxLayout(middle);ml.setContentsMargins(0,0,0,0);ml.setSpacing(8)
  self.source_title=label('No frame selected','hero');ml.addWidget(self.source_title);self.source_detail=label('Open a photo or video, or select a study example.','muted');ml.addWidget(self.source_detail)
  toolbar=QHBoxLayout();self.view=QComboBox();self.view.addItems(VIEWS);self.view.currentTextChanged.connect(self.render_frame);toolbar.addWidget(self.view,1)
  self.evidence_class=QComboBox();self.evidence_class.addItems(['Predicted class','Low evidence','Mid evidence','High evidence']);self.evidence_class.currentIndexChanged.connect(self.render_frame);toolbar.addWidget(self.evidence_class,1)
  self.concept=QSpinBox();self.concept.setRange(1,12);self.concept.setPrefix('Texture ');self.concept.valueChanged.connect(self.render_frame);self.concept.setVisible(False);toolbar.addWidget(self.concept);toolbar.addWidget(button('Fit',self.fit));ml.addLayout(toolbar)
  self.canvas=Canvas();ml.addWidget(self.canvas,1)
  layers=QHBoxLayout();self.layer_checks={}
  for key,title in [('tyre','Tyre'),('tread','Tread'),('boxes','Boxes'),('points','Points'),('guides','Guides')]:
   check=QCheckBox(title);check.setChecked(True);check.toggled.connect(self.render_frame);self.layer_checks[key]=check;layers.addWidget(check)
  self.wipe=QCheckBox('Compare');self.wipe.toggled.connect(self.render_frame);layers.addWidget(self.wipe);layers.addStretch();ml.addLayout(layers)
  blend=QHBoxLayout();blend.addWidget(label('Mask blend','muted'));self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(0,70);self.opacity.setValue(35);self.opacity.valueChanged.connect(self.render_frame);blend.addWidget(self.opacity);self.inspect_btn=button('Inspect this frame',self.inspect,True);self.inspect_btn.setEnabled(False);blend.addWidget(self.inspect_btn);ml.addLayout(blend)
  alignment=QHBoxLayout();alignment.addWidget(button('Wheel alignment…',self.open_alignment));alignment.addWidget(button('Compare boundaries and rim…',self.open_geometry_comparison));alignment.addStretch();ml.addLayout(alignment)
  splitter.addWidget(middle)
  readout=QFrame();readout.setObjectName('readout');readout.setMinimumWidth(270);readout.setMaximumWidth(350);rr=QVBoxLayout(readout);rr.setContentsMargins(10,10,10,8);rr.setSpacing(6);head=QHBoxLayout();head.addWidget(label('TREAD-ONE results','section'),1);head.addWidget(button('Model record',self.about));rr.addLayout(head);self.prediction=label('Waiting for inspection','hero');rr.addWidget(self.prediction);self.run_detail=label('Classification · masks · boundaries · explanation','muted');rr.addWidget(self.run_detail)
  tabs=QTabWidget();rr.addWidget(tabs,1);read=QWidget();readl=QVBoxLayout(read);readl.setContentsMargins(0,14,0,4);readl.setSpacing(10)
  self.bars=ClassBars();readl.addWidget(self.bars);readl.addWidget(label('Model probabilities, not measured wear or calibrated accuracy.','muted'));section(readl,'REGION COVERAGE');self.region_label=label('Tyre  —\nTread  —');readl.addWidget(self.region_label);section(readl,'IMAGE-SPACE BOUNDARIES')
  self.geometry=QTableWidget(3,3);self.geometry.setHorizontalHeaderLabels(['Guide','Left px','Right px']);self.geometry.verticalHeader().hide();self.geometry.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.geometry.setEditTriggers(QTableWidget.NoEditTriggers);self.geometry.setSelectionMode(QTableWidget.NoSelection);self.geometry.setMaximumHeight(132);readl.addWidget(self.geometry);self.geometry_note=label('Six proposals at three guide rows.\nPixels are relative to the oriented source image.','muted');readl.addWidget(self.geometry_note);section(readl,'REVIEW NOTES');self.flags=label('Inspection will record any empty rows, edge contacts or point/mask disagreements.','muted');readl.addWidget(self.flags);readl.addStretch();tabs.addTab(scroll(read),'Readout')
  explain=QWidget();el=QVBoxLayout(explain);el.setContentsMargins(0,14,0,4);el.setSpacing(12);section(el,'SIGNED LOGIT EVIDENCE');el.addWidget(label('Coral supports the selected class logit. Blue opposes it. The global bias is separate.','muted'));self.evidence_detail=label('Inspect a frame to view the accounting.');el.addWidget(self.evidence_detail);section(el,'LATENT TEXTURE BANK');self.texture_detail=label('12 learned texture channels. They are not human-labelled damage concepts.','muted');el.addWidget(self.texture_detail);el.addWidget(label('Changing evidence class or texture channel changes the display only. Input pixels can influence several locations through the encoder.','muted'));el.addStretch();tabs.addTab(scroll(explain),'Explain');splitter.addWidget(readout);splitter.setStretchFactor(0,1);splitter.setStretchFactor(1,0)
  self.error_banner=label('','error');self.error_banner.hide();root.addWidget(self.error_banner)
  bottom=QHBoxLayout();self.note=QLineEdit();self.note.setPlaceholderText('Optional note for this saved capture');bottom.addWidget(self.note,1);self.capture_btn=button('Save capture + CSV',self.save_capture,True);self.capture_btn.setEnabled(False);bottom.addWidget(self.capture_btn);root.addLayout(bottom)
  self.statusBar().showMessage('TREAD-ONE · local inference')
 def existing_examples(self):
  path=PHASE3/'data_gate_outputs/csv/source_images.csv';data_root=PHASE3.parent/'phase2/data/phase2_dataset_v1'
  if not path.exists() or not data_root.exists():return []
  with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
  result=[]
  for split in ('train','validation','test'):
   for cls in ('low','mid','high'):
    group=sorted((r for r in rows if r['domain']=='original_photo' and r['historical_study_split']==split and r['proxy_class']==cls),key=lambda r:r['image_id'])
    if group:
     row=dict(group[0]);p=(data_root/row['image_path']).resolve()
     if p.is_relative_to(data_root.resolve()) and p.exists():row['absolute_path']=str(p);result.append(row)
  return result
 def example_changed(self,index):
  item=self.examples.itemData(index)
  if item:self.load_source(item['absolute_path'],sample=item)
 def load_example(self):
  if self.sample_rows:self.examples.setCurrentIndex(1)
 def choose_photo(self):
  path,_=QFileDialog.getOpenFileName(self,'Open an existing photo','','Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)')
  if path:self.load_source(path)
 def choose_video(self):
  path,_=QFileDialog.getOpenFileName(self,'Open a video','','Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.webm)')
  if path:self.load_source(path)
 def load_source(self,path,sample=None):
  try:
   self.stop_media();self.generation+=1;self.result=None;self.note.clear();self.clear_readout();self.error_banner.hide()
   if Path(path).suffix.lower() in ('.mp4','.avi','.mov','.mkv','.m4v','.webm'):
    self.media.open(path);self.play_btn.setEnabled(True);self.freeze_btn.setEnabled(True);self.seek.setRange(0,max(0,self.media.count-1));self.rgb=self.input_rgb.copy();self.source=dict(self.input_source)
   else:
    rgb,orientation=read_image(path)
    if rgb.shape[0]*rgb.shape[1]>30_000_000:raise ValueError('Choose an image with at most 30 million pixels for this workstation.')
    self.rgb=self.input_rgb=rgb;self.source=self.input_source=dict(path=str(Path(path).resolve()),media_kind='photo',source_media_sha256=sha(path),**orientation)
    if sample:
     if self.source['source_media_sha256']!=sample['image_sha256']:raise ValueError('Existing example checksum changed.')
     self.source.update(reference_proxy_class=sample['proxy_class'],reference_scope='Existing '+sample['historical_study_split']+' study example; not a new blind test',source_image_id=sample['image_id'])
   self.canvas.reset_view();self.update_source_title();self.render_frame();self.inspect_btn.setEnabled(self.model_ready and not self.inflight);self.capture_btn.setEnabled(False);self.session.event('source_opened',str(path))
  except Exception as exc:self.show_error(str(exc))
 def receive_bench_frame(self,rgb,source):
  self.stop_media();self.generation+=1;self.result=None;self.note.clear();self.clear_readout();self.error_banner.hide()
  self.rgb=self.input_rgb=rgb.copy();self.source=self.input_source=dict(source)
  self.canvas.reset_view();self.update_source_title();self.render_frame()
  self.inspect_btn.setEnabled(self.model_ready and not self.inflight)
  self.session.event('frame_from_model_bench',fingerprint(self.rgb))
  self.statusBar().showMessage('Model bench frame received. Inspect to run TREAD-ONE.')
 def open_camera(self):
  try:
   self.stop_media();self.generation+=1;self.result=None;self.clear_readout();self.media.open('Camera 0',camera=True);self.play_btn.setEnabled(True);self.freeze_btn.setEnabled(True);self.rgb=self.input_rgb.copy();self.source=dict(self.input_source);self.update_source_title();self.render_frame();self.media.play();self.play_btn.setText('Pause');self.inspect_btn.setEnabled(self.model_ready and not self.inflight)
  except Exception as exc:self.show_error(str(exc))
 def media_frame(self,rgb,meta):
  self.input_rgb=rgb;self.input_source=meta;self.preview.show();self.preview.setPixmap(QPixmap.fromImage(qimage(rgb)).scaled(194,136,Qt.KeepAspectRatio,Qt.SmoothTransformation));self.feed_time.setText(clock(meta['timestamp_seconds'])+'  ·  frame '+str(meta['frame_index']))
  if not self.seek.isSliderDown():self.seek.setValue(meta['frame_index'])
 def media_ended(self):self.play_btn.setText('Play');self.statusBar().showMessage('End of video · last inspected snapshot remains available')
 def toggle_play(self):
  if self.media.cap is None:return
  if self.media.playing:self.media.pause();self.play_btn.setText('Play')
  else:
   if self.media.count and self.media.index>=self.media.count-1:self.media.seek(0)
   self.media.play();self.play_btn.setText('Pause')
 def freeze(self):
  if self.media.cap is None:return
  self.media.pause();self.play_btn.setText('Play');self.live.setChecked(False);self.generation+=1;self.rgb=self.input_rgb.copy();self.source=dict(self.input_source);self.result=None;self.clear_readout();self.update_source_title();self.render_frame();self.inspect_btn.setEnabled(self.model_ready and not self.inflight)
 def seek_frame(self):
  if self.media.cap is not None:self.media.seek(self.seek.value());self.freeze()
 def stop_media(self):
  self.live_timer.stop();self.live.setChecked(False);self.media.close();self.play_btn.setEnabled(False);self.freeze_btn.setEnabled(False);self.play_btn.setText('Play');self.seek.setRange(0,0);self.preview.hide();self.feed_time.setText('No video open')
 def live_changed(self,on):
  if on and self.media.cap is not None:self.live_timer.start()
  else:self.live_timer.stop()
 def live_tick(self):
  if self.media.cap is not None and self.media.playing and self.model_ready and not self.inflight:self.inspect(live=True)
 def start_model(self,device):
  self.model_ready=False;self.inspect_btn.setEnabled(False);self.ready_label.setText('Loading the pinned model…');self.worker=InferenceBridge(device);self.worker.ready.connect(self.on_ready);self.worker.result.connect(self.receive);self.worker.error.connect(self.worker_error);self.worker.start()
 def change_device(self,index):
  device=self.device_choice.itemData(index)
  if device==self.device:return
  self.live.setChecked(False)
  if self.worker:
   self.worker.request_stop()
   if not self.worker.wait(1000):self.worker.kill_owned_process();self.worker.wait(4000)
  self.pending.clear();self.inflight=False;self.device=device;self.start_model(device)
 def on_ready(self,packet):
  self.model_ready=True;self.error_banner.hide();env=packet['environment'];self.ready_label.setText(env['device_name']+'\nPinned seed 3 · epoch 20');self.inspect_btn.setEnabled(self.rgb is not None and not self.inflight);self.session.event('model_ready',json.dumps(env));self.statusBar().showMessage('Model ready · '+env['precision'])
  if self.auto_example and self.rgb is not None:self.auto_example=False;self.inspect()
 def inspect(self,checked=False,live=False):
  if not self.model_ready or self.inflight:return
  if self.media.cap is not None:
   if not live:self.media.pause();self.play_btn.setText('Play')
   rgb=self.input_rgb.copy();source=dict(self.input_source)
  else:
   if self.rgb is None:return
   rgb=self.rgb.copy();source=dict(self.source)
  self.request_id+=1;key=self.request_id;self.pending[key]=dict(rgb=rgb,source=source,generation=self.generation);self.inflight=True;self.inspect_btn.setEnabled(False);self.statusBar().showMessage('Inspecting this frame…')
  try:self.worker.submit(key,rgb)
  except Exception as exc:self.pending.pop(key,None);self.inflight=False;self.show_error(str(exc))
 def receive(self,key,result):
  job=self.pending.pop(key,None);self.inflight=False;self.inspect_btn.setEnabled(self.model_ready and self.rgb is not None)
  if job is None:return
  if fingerprint(working_image(job['rgb']))!=result['summary']['model_input_sha256']:self.show_error('Frame identity check failed; result was not displayed.');return
  try:self.session.prediction(job['rgb'],result,job['source'])
  except Exception as exc:self.show_error('Prediction completed, but session CSV could not be saved: '+str(exc))
  if job['generation']!=self.generation:return
  self.rgb=job['rgb'];self.source=job['source'];self.result=result;self.update_source_title();self.update_readout();self.render_frame();self.capture_btn.setEnabled(self.exporter is None or not self.exporter.isRunning());self.statusBar().showMessage(f"Inspection saved to the session CSV · model {result['summary']['model_ms']:.0f} ms · {result['summary']['device']}")
 def worker_error(self,message):self.model_ready=False;self.inflight=False;self.pending.clear();self.inspect_btn.setEnabled(False);self.ready_label.setText('Model unavailable · choose CPU or restart this window');self.show_error(message)
 def clear_readout(self):
  self.prediction.setText('Waiting for inspection');self.bars.set_values(None);self.region_label.setText('Tyre  —\nTread  —');self.geometry.clearContents();self.flags.setText('Inspect a frame to populate model outputs.');self.evidence_detail.setText('Inspect a frame to view the accounting.');self.capture_btn.setEnabled(False)
 def update_source_title(self):
  self.source_title.setText(Path(self.source.get('path','Current frame')).name);parts=[self.source.get('media_kind','photo').title()]
  if self.source.get('timestamp_seconds') is not None:parts.append('Analysed snapshot '+clock(self.source['timestamp_seconds']))
  if self.rgb is not None:parts.append(f'{self.rgb.shape[1]} × {self.rgb.shape[0]}')
  if self.source.get('reference_scope'):parts.append('Known proxy: '+self.source['reference_proxy_class']+' · '+self.source['reference_scope'])
  self.source_detail.setText('  ·  '.join(parts))
 def update_readout(self):
  a=self.result['arrays'];s=self.result['summary'];names=['Low mileage proxy','Mid mileage proxy','High mileage proxy'];self.prediction.setText(names[s['prediction']]);self.bars.set_values(a['probabilities']);self.run_detail.setText(f"One forward pass · {s['model_ms']:.0f} ms · {s['device']}")
  masks=masks_at(a,(self.rgb.shape[1],self.rgb.shape[0]));self.region_label.setText('Tyre  '+f'{100*masks[0].mean():.1f}%'+' of image\nTread  '+f'{100*masks[1].mean():.1f}%'+' of image')
  for i in range(3):
   for j,value in enumerate((['Upper','Middle','Lower'][i],f"{a['points'][2*i]*(self.rgb.shape[1]-1):.1f}",f"{a['points'][2*i+1]*(self.rgb.shape[1]-1):.1f}")):self.geometry.setItem(i,j,QTableWidgetItem(value))
  self.flags.setText('\n'.join(s['review_flags']) or 'No configured image-space review flags. This is not an accuracy estimate.');self.update_explanation()
 def update_explanation(self):
  if self.result is None:return
  a=self.result['arrays'];cls=int(a['probabilities'].argmax()) if self.evidence_class.currentIndex()==0 else self.evidence_class.currentIndex()-1;e=a['contributions'][cls];positive=float(e[e>0].sum());negative=float(e[e<0].sum());bias=float(a['bias'][cls]);logit=float(a['logits'][cls]);names=['Low','Mid','High'];self.evidence_detail.setText(f'{names[cls]} class\n\nPositive spatial evidence  {positive:+.5f}\nNegative spatial evidence  {negative:+.5f}\nGlobal bias                         {bias:+.5f}\n\nReconstructed logit          {positive+negative+bias:+.5f}\nModel logit                         {logit:+.5f}\nAccounting residual          {self.result["summary"]["evidence_residual"]:.2g}\n\nThese are logit contributions, not probabilities or physical measurements.')
  mass=(a['concepts']*a['gate']).sum((-2,-1));self.texture_detail.setText('Learned channels, not named damage labels.\n\n'+'\n'.join(f'Texture {i+1:02d}    gated mass {float(v):.4f}' for i,v in enumerate(mass)))
 def render_frame(self,*_):
  self.concept.setVisible(self.view.currentText()=='Texture channel');self.evidence_class.setVisible(self.view.currentText()=='Signed evidence');self.canvas.wipe=self.wipe.isChecked()
  if self.rgb is None:return
  cls=None if self.evidence_class.currentIndex()==0 else self.evidence_class.currentIndex()-1;checks={k:v.isChecked() for k,v in self.layer_checks.items()}
  processed=render(self.rgb,self.result,self.view.currentText(),self.opacity.value()/100,checks['tyre'],checks['tread'],checks['boxes'],checks['points'],checks['guides'],cls,self.concept.value()-1) if self.result is not None else None
  self.canvas.set_frame(self.rgb,processed);self.update_explanation()
 def fit(self):self.canvas.reset_view()
 @property
 def captured_rgb(self):
  return self.rgb if self.result is not None else None
 @property
 def masks(self):
  if self.captured_rgb is None:return {}
  h,w=self.rgb.shape[:2]
  return {'TREAD-ONE':masks_at(self.result['arrays'],(w,h))}
 learned_frame=None  # The legacy dialog's optional HRNet/SegFormer record.

 def open_alignment(self):
  if self.alignment_dialog is not None:
   self.alignment_dialog.show();self.alignment_dialog.raise_();self.alignment_dialog.activateWindow()
   return
  from legacy_bench import BENCH,load
  (BENCH/'data').mkdir(parents=True,exist_ok=True)

  class AlignmentDialog(load('alignment_ui').AlignmentDialog):
   def __init__(dialog,station):
    super().__init__(station)
    dialog.setWindowTitle('Wheel alignment | TREAD-ONE')
    dialog.live.setText('Follow analysed TREAD-ONE frames')
    dialog.live.toggled.connect(dialog.follow_changed)

   def follow_changed(dialog,enabled):
    if enabled:
     dialog.last_frame=object()
     dialog.tick()

   def tick(dialog):
    if not dialog.live.isChecked():return
    rgb=dialog.station.captured_rgb
    if rgb is dialog.last_frame:return
    dialog.last_frame=rgb
    if rgb is None:
     dialog.invalidate()
     dialog.preview.setText('Waiting for an analysed TREAD-ONE frame.')
    else:dialog.capture()

  dialog=AlignmentDialog(self)
  dialog.setAttribute(Qt.WA_DeleteOnClose)
  dialog.destroyed.connect(lambda: setattr(self,'alignment_dialog',None))
  self.alignment_dialog=dialog
  dialog.show()

 def open_geometry_comparison(self):
  if self.captured_rgb is None:
   self.statusBar().showMessage('Inspect a TREAD-ONE frame before comparing boundaries and rim.')
   return
  from legacy_bench import load
  dialog=load('geometry_comparison').GeometryComparison(self.captured_rgb.copy(),self.masks,self)
  dialog.setAttribute(Qt.WA_DeleteOnClose)
  self.comparison_dialog=dialog
  dialog.show()

 def alignment_busy(self):
  return self.alignment_dialog is not None and self.alignment_dialog.worker is not None and self.alignment_dialog.worker.isRunning()

 def save_capture(self):
  if self.result is None or self.rgb is None or self.exporter is not None and self.exporter.isRunning():return
  self.capture_btn.setEnabled(False);self.capture_btn.setText('Saving full evidence…');self.exporter=CaptureBridge(self.rgb.copy(),self.result,dict(self.source),self.note.text(),self.capture_root);self.exporter.saved.connect(self.capture_saved);self.exporter.error.connect(self.show_error);self.exporter.finished.connect(self.export_finished);self.exporter.start()
 def capture_saved(self,path):self.last_export=Path(path);self.session.event('capture_saved',path);self.statusBar().showMessage('Capture saved · direct CSV, original image and native arrays');self.reload_captures()
 def export_finished(self):
  self.capture_btn.setText('Save capture + CSV');self.capture_btn.setEnabled(self.result is not None)
  if self.closing:QTimer.singleShot(0,self.close)
 def reload_captures(self):
  self.captures.clear()
  for path in sorted(self.capture_root.glob('*/CAPTURE.json'),reverse=True)[:15]:
   try:
    meta=read(path);item=QListWidgetItem(path.parent.name[:15]+' · '+meta['record']['predicted_label'].split()[0]);item.setData(Qt.UserRole,str(path.parent));item.setToolTip(meta.get('note','')+'\n'+str(path.parent));self.captures.addItem(item)
   except Exception:continue
 def restore_item(self,item):self.restore_capture(item.data(Qt.UserRole))
 def choose_capture(self):
  path=QFileDialog.getExistingDirectory(self,'Restore a TREAD-ONE capture',str(self.capture_root))
  if path:self.restore_capture(path)
 def restore_capture(self,path):
  try:
   rgb,result,source,note=load_capture(path)
   if result['identity']['weights_sha256']!=self.identity['weights_sha256']:raise ValueError('This capture belongs to a different checkpoint.')
   self.stop_media();self.generation+=1;self.rgb=self.input_rgb=rgb;self.source=self.input_source=source;self.result=result;self.note.setText(note);self.update_source_title();self.update_readout();self.render_frame();self.capture_btn.setEnabled(True);self.inspect_btn.setEnabled(self.model_ready and not self.inflight);self.session.event('capture_restored',str(path));self.statusBar().showMessage('Saved capture restored with verified file hashes · no new inference')
  except Exception as exc:self.show_error(str(exc))
 def open_evidence(self):QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.capture_root)))
 def show_error(self,message):
  self.errors.append(str(message));self.error_banner.setText(str(message));self.error_banner.show();self.statusBar().showMessage('Review the message above.')
  try:self.session.event('error',message)
  except Exception:pass
 def about(self):
  dialog=QDialog(self);dialog.setWindowTitle('TREAD-ONE model record');dialog.resize(680,590);layout=QVBoxLayout(dialog);browser=QTextBrowser();browser.setPlainText('TREAD-ONE Studio — separate research workstation\n\nOne custom 1,372,159-parameter model; no external backbone.\nDeclared r1 validation selection: seed 3, epoch 20.\n\nOutputs\n• Three mileage-proxy probabilities\n• Overlapping tyre and tread masks; mask-derived boxes\n• Six ordered image-space boundary proposals\n• Exact additive signed class-logit evidence and 12 latent texture channels\n\nCurrent study\n570 existing images, 12 original physical tyres.\nSelected export: 50% balanced accuracy, 98.419% mean Dice and 1.196% image-width human-point error on the limited low/high holdout. All selected r1 models missed the low test tyre. No held-out mid class or independent new-tyre validation.\n\nDesktop inference uses FP32; the Kaggle evaluation used its recorded precision. Display probabilities are not calibrated wear, depth or safety measurements. Boundary rows do not measure physical alignment.\n\nThis window owns its own checkpoint, settings/cache, session CSV and evidence captures. It does not replace the original workstation.\n\nCheckpoint SHA-256\n'+self.identity['weights_sha256']+'\n\nEvidence folder\n'+str(DATA/'captures'));layout.addWidget(browser);layout.addWidget(button('Close',dialog.accept));dialog.exec()
 def dragEnterEvent(self,event):
  if event.mimeData().hasUrls():event.acceptProposedAction()
 def dropEvent(self,event):
  urls=event.mimeData().urls()
  if urls and urls[0].isLocalFile():self.load_source(urls[0].toLocalFile())
 def closeEvent(self,event):
  if self.alignment_busy():self.statusBar().showMessage('Wait for wheel calibration to finish before closing.');event.ignore();return
  if self.exporter is not None and self.exporter.isRunning():self.closing=True;self.statusBar().showMessage('Finishing the evidence capture before closing…');event.ignore();return
  if self.alignment_dialog is not None:self.alignment_dialog.close()
  self.stop_media()
  if self.worker:
   self.worker.request_stop()
   if not self.worker.wait(1800):self.worker.kill_owned_process();self.worker.wait(4000)
   if self.worker.isRunning():event.ignore();self.statusBar().showMessage('Waiting for the owned inference process to stop…');QTimer.singleShot(1000,self.close);return
  self.session.event('closed','Clean shutdown');write_json(DATA/'settings.json',dict(window_width=self.width(),window_height=self.height(),device=self.device));event.accept()

def main():
 from workspace import Workspace
 parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['auto','cpu','cuda:0'],default='auto');parser.add_argument('--example',action='store_true');parser.add_argument('--photo',default='');parser.add_argument('--bench',action='store_true',help='Open the model bench tab first');parser.add_argument('--preview',default='',help='Diagnostic screenshot after the first real inference; path must stay in phase3');args=parser.parse_args()
 if sys.platform=='win32':
  import ctypes
  ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Tyre.Phase3.TreadStation')
 app=QApplication(sys.argv[:1]);app.setApplicationName('Tread Station | Phase 3');app.setOrganizationName('Tyre Study');apply(app)
 def failure(kind,value,tb):
  with (CACHE/'ui-errors.log').open('a',encoding='utf-8') as f:traceback.print_exception(kind,value,tb,file=f)
  custom.show_error(str(value))
 custom=Station(args.device,args.example);window=Workspace(custom,args.device);sys.excepthook=failure
 if args.photo:custom.load_source(args.photo)
 window.show()
 if args.bench:window.tabs.setCurrentIndex(1)
 if args.preview:
  target=Path(args.preview).resolve();assert target.is_relative_to(PHASE3.resolve());target.parent.mkdir(parents=True,exist_ok=True);timer=QTimer(window)
  def preview_ready():
   if custom.result is None:return
   window.grab().save(str(target));import os
   write_json(target.with_suffix('.json'),dict(status='native_window_ready',pid=os.getpid(),worker_pid=custom.worker.process.pid,Qt_platform=app.platformName(),title=window.windowTitle(),checkpoint=custom.identity['weights_sha256'],device=custom.result['summary']['device'],model_ms=custom.result['summary']['model_ms'],source=custom.source,session=str(custom.session.root),screenshot=str(target),new_training=False));timer.stop()
  timer.timeout.connect(preview_ready);timer.start(350)
 sys.exit(app.exec())
if __name__=='__main__':main()
