"""Custom-rate encoding and angle/display checks without retraining."""
import os,sys,pathlib,tempfile,math,json
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'prototype'));import bootstrap
from PySide6.QtWidgets import QApplication
import cv2,numpy as np
from media_export import export_video
from phase2_video_labels import tread_angle,draw_footer,footer_lines
from check_media_export import ProbeEngine

def main():
 app=QApplication([]);out=[]
 with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'.cache') as temp:
  folder=pathlib.Path(temp)
  for source_fps in (5,30):
   source=folder/f'{source_fps}.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),source_fps,(120,160));assert writer.isOpened()
   for i in range(source_fps):writer.write(np.full((160,120,3),30+i*5,np.uint8))
   writer.release()
   for fps in (1,7.5,24,60):
    dest=folder/f'{source_fps}-{fps}.mp4';job=dict(source=str(source),destination=str(dest),classifier='test',region='test',overlay=None,threshold=.25,assist=False,learned='off',opacity=.3,layers=[True,True],geometry=False,edges=None,show_learned=False,fps=fps,information_panel=True);e=ProbeEngine()
    count=export_video(job,e);assert count==math.ceil(fps);np.testing.assert_allclose(e.levels,[30+int(i*source_fps/fps+1e-7)*5 for i in range(count)],atol=3)
    cap=cv2.VideoCapture(str(dest));assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<.01;n=0
    while True:
     ok,frame=cap.read()
     if not ok:break
     assert frame.shape[0]>160 and frame.shape[1]==120;n+=1
    cap.release();assert n==count and abs(count/fps-1)<1/fps+1e-6;out.append(dict(source_fps=source_fps,export_fps=fps,frames=count))
  for bad in (0,61,float('nan'),float('inf')):
   try:export_video(dict(job,fps=bad));raise AssertionError('Invalid FPS accepted')
   except ValueError:pass
 for degree in (-15,0,15):
  points=[]
  for y in (100,200,300):
   center=200-math.tan(math.radians(degree))*y;points.extend([[center-30,y],[center+30,y]])
  rec=dict(models=dict(hrnet=dict(ordered=True,points=points)))
  assert abs(tread_angle(rec)-degree)<1e-8
  rec['models']['hrnet']['ordered']=False;assert tread_angle(rec) is None
 rgb=np.full((200,300,3),80,np.uint8)
 assert draw_footer(rgb,['a']).shape==draw_footer(rgb,['very long text '*20]*7).shape
 np.testing.assert_array_equal(draw_footer(rgb,['a'])[:200],rgb)
 (pathlib.Path('phase2/workstation')/'phase2_video_controls_tests.json').write_text(json.dumps(dict(status='passed',rates=out,angles='signed/vertical/invalid passed',panel='fixed canvas; source pixels preserved'),indent=2));print('PASS custom FPS, fractional duration, repetition, invalid rates, signed angles, fixed panel canvas')
if __name__=='__main__':main()
