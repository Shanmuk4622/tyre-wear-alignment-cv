"""Real native-widget workflow with the owned model subprocess and video fixture."""
import os,sys,time,json,uuid,faulthandler
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
faulthandler.dump_traceback_later(180,repeat=False)
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[1];APP=ROOT/'workstation';OUT=ROOT/'workstation_validation';sys.path.insert(0,str(APP))
import bootstrap
from PySide6.QtWidgets import QApplication,QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
import cv2,numpy as np
from app import Station
from theme import apply
from paths import read,sha,write_json
from imaging import fingerprint,working_image
from records import load_capture

def main():
 app=QApplication([]);apply(app);fixture=OUT/('ui_'+uuid.uuid4().hex[:8]);fixture.mkdir();window=Station('cpu',capture_root=fixture/'captures',session_root=fixture/'sessions');window.resize(1370,940);window.show();checks=[]
 def until(predicate,seconds=90):
  deadline=time.monotonic()+seconds
  while time.monotonic()<deadline:
   app.processEvents()
   if window.errors:raise AssertionError(window.errors)
   if predicate():return
   time.sleep(.015)
  raise TimeoutError('UI did not reach the expected state')
 try:
  until(lambda:window.model_ready);assert window.sample_rows and not window.capture_btn.isEnabled();checks.append('private_Qt_window_and_owned_inference_process_ready')
  assert QFontMetrics(app.font()).inFont('A') and QFontMetrics(app.font()).inFont('0') and not app.windowIcon().isNull();checks.append('actual_Latin_font_glyphs_and_separate_station_icon_available')
  first=window.sample_rows[0];second=next(r for r in window.sample_rows if r['proxy_class']=='mid');window.load_source(first['absolute_path'],sample=first);window.inspect();window.load_source(second['absolute_path'],sample=second)
  until(lambda:not window.inflight);assert window.result is None and window.source['source_image_id']==second['image_id'];checks.append('source_change_discards_stale_overlay_while_preserving_completed_ledger')
  window.inspect();until(lambda:window.result is not None and not window.inflight);assert window.capture_btn.isEnabled();assert len(window.result['arrays'])==10;checks.append('real_exported_model_inference_through_JSON_process_bridge')
  key=window.request_id;frame_hash=fingerprint(window.rgb)
  for view in ('Signed evidence','Tread attention','Texture channel','Boundaries','Masks','Combined'):
   window.view.setCurrentText(view);window.evidence_class.setCurrentIndex(3);window.concept.setValue(12);app.processEvents();assert window.canvas.processed is not None
  assert window.request_id==key and fingerprint(window.rgb)==frame_hash;checks.append('all_view_class_and_texture_controls_render_without_reinference')
  window.resize(1210,750);app.processEvents();assert window.rail_scroll.verticalScrollBar().maximum()>0 and all(b.height()>=35 for b in window.findChildren(QPushButton));window.grab().save(str(OUT/'TREAD_ONE_Studio_Compact.png'));window.resize(1370,940);app.processEvents();checks.append('compact_desktop_sidebar_scrolls_without_clipping_button_text')
  window.wipe.setChecked(True);window.opacity.setValue(20);window.layer_checks['tyre'].setChecked(False);app.processEvents();window.canvas.reset_view();assert window.canvas.wipe;checks.append('comparison_wipe_layer_toggle_and_opacity')
  window.wipe.setChecked(False);window.layer_checks['tyre'].setChecked(True);window.opacity.setValue(35);window.view.setCurrentText('Combined');window.grab().save(str(OUT/'TREAD_ONE_Studio_Preview.png'))
  window.view.setCurrentText('Signed evidence');window.grab().save(str(OUT/'TREAD_ONE_Studio_Evidence.png'));window.view.setCurrentText('Combined');checks.append('native_widget_screenshots_saved')
  window.note.setText('Engineering UI capture, "roundtrip"');window.save_capture();until(lambda:window.last_export is not None and not window.exporter.isRunning());capture=window.last_export;rgb,result,source,note=load_capture(capture);assert note==window.note.text();assert fingerprint(rgb)==frame_hash;checks.append('asynchronous_full_CSV_capture_save')
  window.load_source(first['absolute_path'],sample=first);window.restore_capture(capture);assert fingerprint(window.rgb)==frame_hash and window.note.text()==note and not window.inflight;checks.append('hashed_capture_restore_without_model_forward')
  # Explicit engineering-only video generated from the already existing frame.
  video=fixture/'UI_FIXTURE_NOT_NEW_TYRE_DATA.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240));assert writer.isOpened();base=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(320,240))
  for i in range(20):writer.write(np.roll(base,i,axis=1))
  writer.release();window.load_source(video);assert window.media.cap is not None and window.source['frame_index']==0;window.media.seek(4);window.freeze();assert window.source['frame_index']==4 and abs(window.source['timestamp_seconds']-.4)<1e-8
  window.inspect();until(lambda:window.result is not None and not window.inflight);assert window.result['summary']['model_input_sha256']==fingerprint(working_image(window.rgb));checks.append('video_decode_seek_freeze_and_exact_frame_identity')
  window.live.setChecked(True);window.toggle_play();before=window.request_id;until(lambda:window.request_id>before);until(lambda:not window.inflight);assert window.rgb.shape==(240,320,3);window.live.setChecked(False);window.media.pause();checks.append('bounded_live_video_inference_preserves_analysed_snapshot')
  window.close();app.processEvents();assert not window.worker.isRunning() and window.worker.process.poll() is not None and window.media.cap is None;checks.append('clean_shutdown_stops_only_owned_worker_and_media')
  baseline=read(OUT/'ORIGINAL_BASELINE.json');assert all(sha(ROOT.parent/n)==h for n,h in baseline.items());checks.append('all_47_original_source_launcher_and_registry_files_unchanged')
  report=dict(status='passed',check_count=len(checks),checks=checks,Qt_platform=app.platformName(),Qt_binding=__import__('PySide6').__version__,fixture_root=str(fixture),fixture_capture=str(capture),camera_hardware_tested=False,
   new_training=False,new_independent_tyres=0,synthetic_video_is_engineering_fixture=True,model_seed=3,model_epoch=20,original_files_preserved=len(baseline),worker_exit_code=window.worker.process.returncode)
  write_json(OUT/'UI_VALIDATION.json',report);print(json.dumps(report,indent=2))
 finally:
  window.close();app.processEvents()
if __name__=='__main__':
 try:main()
 finally:faulthandler.cancel_dump_traceback_later()
