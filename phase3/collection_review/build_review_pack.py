"""Prepare existing-identity review and actual-capture priorities from verified NB07."""
from PIL import Image
import base64,csv,io,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,digest,atomic,inside,table,file_meta
OUT=ROOT/'collection_review_outputs';PUB=ROOT/'data_gate_validation/nb07_publication'

def write_csv(name,rows,fields):
 with (OUT/name).open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)

def main():
 verified=read(PUB/'VERIFICATION.json');assert verified['status']=='passed'
 source=PUB/'remote';manifest=read(source/'REMOTE_MANIFEST.json')
 for name,info in manifest['files'].items():assert sha(inside(source,name))==info['sha256'],name
 OUT.mkdir(exist_ok=True);data_root=ROOT.parent/'phase2/data/phase2_dataset_v1'
 rows=table(source/'csv/source_images.csv');tyres=table(source/'csv/source_tyres.csv');clips=table(source/'csv/source_clips.csv');candidates=table(source/'csv/video_parent_review.csv');support=table(source/'csv/historical_class_support.csv')
 mid=[r['tyre_id'] for r in tyres if r['proxy_class']=='mid'];assert len(mid)==3
 gallery={};images=[];groups=[]
 def select(records):
  records=sorted(records,key=lambda r:(float(r['timestamp_seconds']) if r['timestamp_seconds'] else 0,r['image_id']));return [records[i] for i in sorted({0,len(records)//2,len(records)-1})]
 def group(gid,label,kind,records,extra):
  selected=select(records);assert 1<=len(selected)<=3
  for row in selected:
   path=inside(data_root,row['image_path']);assert sha(path)==row['image_sha256']
   with Image.open(path) as image:
    buffer=io.BytesIO();image.convert('RGB').resize((384,512),Image.Resampling.BILINEAR).save(buffer,format='JPEG',quality=82)
   gallery[row['image_id']]=dict(preview='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode(),timestamp_seconds=row['timestamp_seconds'])
   images.append(dict(review_group=gid,source_image_id=row['image_id'],source_sha256=row['image_sha256'],source_path=row['image_path'],domain=row['domain'],proxy_class=row['proxy_class'],historical_split=row['historical_study_split'],confirmed_original_tyre_id=row['confirmed_original_tyre_id'],source_video=row['source_video'],timestamp_seconds=row['timestamp_seconds'],selection_rule='first/middle/last in source ID or source-video time order',new_independent_tyre=False))
  groups.append(dict(group_id=gid,label=label,kind=kind,image_ids=[r['image_id'] for r in selected],**extra))
 for clip in clips:group(clip['video_id'],clip['video_id'],'clip',[r for r in rows if r['source_video']==clip['video_id']],dict(video_id=clip['video_id']))
 for tyre in tyres:group(tyre['tyre_id'],tyre['tyre_id'],'original',[r for r in rows if r['domain']=='original_photo' and r['confirmed_original_tyre_id']==tyre['tyre_id']],dict(proxy_class=tyre['proxy_class'],historical_split=tyre['historical_study_split']))
 reasons={'low':'Only one historical training tyre; both NB05 selected models miss all 62 low test photos. Prioritize additional independent low tyres across capture conditions.',
 'mid':'All three historical mid tyres are training-only; validation/test contain no mid. Obtain additional independent reserved mid tyres.',
 'high':'Historical training has four tyres; reserved sets each one. Broaden independent high-subject and capture coverage while preserving low/mid priorities.'}
 priorities=[]
 for label in ('low','mid','high'):
  item=dict(proxy_class=label)
  for role in ('train','validation','test'):item['historical_'+role+'_tyres']=int(next(r for r in support if r['historical_split']==role and r['proxy_class']==label)['confirmed_original_tyres'])
  train_tyres=[r for r in tyres if r['proxy_class']==label and r['historical_study_split']=='train'];counts=[int(r['original_photo_count']) for r in train_tyres]
  item.update(historical_train_original_photos=sum(counts),fewest_original_photos_per_train_tyre=min(counts),most_original_photos_per_train_tyre=max(counts),historical_train_video_frames=sum(r['domain']=='new_video_frame' and r['proxy_class']==label for r in rows),collection_reason=reasons[label],new_subject_count_required=None,new_subject_count_decided=False,minimum_to_be_declared_against_actual_study_budget=True);priorities.append(item)
 write_csv('review_images.csv',images,list(images[0]));write_csv('identity_candidates.csv',candidates,list(candidates[0]));write_csv('capture_priorities.csv',priorities,list(priorities[0]))
 resolutions=table(source/'input_csv/video_identity_resolutions.csv');fields=['video_id','confirmed_original_tyre_id','review_status','evidence_source','notes'];write_csv('video_identity_resolutions_UNREVIEWED.csv',resolutions,fields)
 data=dict(clips=clips,candidate_ids=mid,groups=groups,images=gallery,candidates=candidates,priorities=priorities,resolution_fields=fields)
 script=(ROOT/'collection_review/review_ui.js').read_text(encoding='utf-8');payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
 html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'unsafe-inline';style-src 'unsafe-inline';img-src data:;connect-src 'none'"><title>TREAD-ONE identity review and capture planner</title><style>body{font:16px/1.5 system-ui;background:#112431;color:#eff6fa;max-width:1300px;margin:24px auto;padding:0 20px}h1,h2{line-height:1.2}.card{background:#203b4b;border-radius:12px;padding:18px;margin:16px 0}h3{overflow-wrap:anywhere}.strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;max-height:440px;object-fit:contain;background:#09151e}figcaption{font-size:12px;overflow-wrap:anywhere}input[type=text],textarea,select{display:block;box-sizing:border-box;width:100%;font:inherit;padding:9px;margin:10px 0;border-radius:6px}input[type=checkbox]{width:20px;height:20px;vertical-align:middle}button{font:inherit;padding:12px 20px;cursor:pointer}button:disabled{cursor:default;opacity:.5}#errors{padding:12px;background:#642c24}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #476574;text-align:left;padding:8px}.notice{background:#1e4b59;padding:16px;border-radius:10px}details{margin:16px 0}@media(max-width:600px){.strip{grid-template-columns:1fr}}</style></head><body><h1>TREAD-ONE identity review and capture planner</h1><p class="notice">NB07 is verified complete. These 45 pictures show the existing 12 tyres and three existing clips. They add zero independent subjects. Review with supporting capture records; visual similarity alone cannot establish physical identity.</p><h2>1 · Compare the three existing mid clips</h2><p>All clips show distinct existing mid tyres. Their original parents remain unresolved. Each clip has first/middle/last frames; compare with the original tyre strips below. No parent is preselected.</p><div id="clips"></div><h2>2 · Original tyre reference strips</h2><p>Three existing views per physical tyre, selected by source order. These are review aids, not a representative sample or new ground truth.</p><div id="tyres"></div><details><summary>Similarity candidates from NB07</summary><p>Lower hash distance does not confirm identity. Ties and conflicting capture conditions remain possible.</p><table><thead><tr><th>Clip</th><th>Possible parent</th><th>Minimum dHash distance</th></tr></thead><tbody id="distances"></tbody></table></details><h2>3 · Export reviewed mappings</h2><p>Confirm only mappings supported by actual records. Leaving all three unresolved is valid. This page never edits project inputs or uploads anything. Download the CSV to preserve your review; refreshing the page clears unsaved form entries.</p><div id="errors" hidden></div><p id="status" role="status"></p><button id="download">Download video_identity_resolutions.csv</button><h2>4 · Actual independent capture priorities</h2><p>Subject counts and annotation budget must be decided against an actual cohort and full prospective protocol. This planner sets no invented collection target or split ratio. Keep all views/derivatives of one tyre together.</p><table><thead><tr><th>Class</th><th>Train tyres</th><th>Validation tyres</th><th>Test tyres</th><th>Collection reason</th></tr></thead><tbody id="priorities"></tbody></table><p>Record stable tyre identity, mileage-proxy source, session/date/camera/lighting/background, original/derivative parent, native image hashes and annotation review. Reserved new subjects need reviewed masks and human geometry under the declared policy. Mileage proxies do not measure physical wear, depth or safety.</p><script>const DATA=__DATA__;
__SCRIPT__</script></body></html>'''.replace('__DATA__',payload).replace('__SCRIPT__',script)
 html=html.replace('These 45 pictures',f'These {len(images)} pictures').replace('Three existing views per physical tyre, selected by source order.','Up to three existing views per physical tyre, selected by source order. One high training tyre has only two source photos, so both are shown without duplicating a view.')
 (OUT/'TREAD_ONE_Identity_And_Capture_Review.html').write_text(html,encoding='utf-8')
 selected_groups=[g for g in groups if g['kind']=='clip' or g['group_id'] in mid];fig,axes=plt.subplots(6,3,figsize=(9,19),layout='constrained')
 for i,g in enumerate(selected_groups):
  for j,key in enumerate(g['image_ids']):
   with Image.open(io.BytesIO(base64.b64decode(gallery[key]['preview'].split(',',1)[1]))) as image:axes[i,j].imshow(image)
   axes[i,j].axis('off');axes[i,j].set_title(g['label'].replace('__session_','\nsession ')+'\n'+key,fontsize=7)
 fig.suptitle('Existing mid clips and original candidates · identities remain unconfirmed',fontsize=12);fig.savefig(OUT/'NB07_Identity_Review_Contact_Sheet.png',dpi=150);plt.close(fig)
 contract=dict(source_revision=verified['revision'],source_publication_contract=verified['publication_contract'],source_notebook_sha256=verified['executed_notebook_sha256'],source_csv_hashes={n:sha(source/'csv'/(n+'.csv')) for n in ('source_images','source_tyres','source_clips','video_parent_review','historical_class_support')},review_source_hashes={p.name:sha(p) for p in (ROOT/'collection_review').glob('*') if p.suffix in ('.py','.js')},existing_original_tyres=12,existing_video_clips=3,selected_source_images=len(images),new_independent_tyres=0,confirmed_mappings_created_by_agent=0,new_training=False,new_inference=False,publication_status='local_preparation_not_published')
 atomic(OUT/'CONTRACT.json',contract)
 schema={p.name:dict(rows=len(table(p)),columns=list(table(p)[0]) if table(p) else []) for p in OUT.glob('*.csv')};atomic(OUT/'CSV_SCHEMA.json',schema)
 atomic(OUT/'MANIFEST.json',dict(contract=digest(contract),files={p.name:file_meta(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
 print(f'Prepared local identity/capture review: {len(images)} existing images, 3 planning CSVs / {len(images)+12} rows, unresolved control CSV / 3 rows. No mappings assigned or uploads performed.')

if __name__=='__main__':main()
