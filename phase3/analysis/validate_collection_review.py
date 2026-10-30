"""Check offline review evidence, CSV export rules and existing identity validator."""
from PIL import Image
import base64,io,json,re,shutil,subprocess,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,sha,atomic,table
from run_gate import resolutions
OUT=ROOT/'data_gate_validation/collection_review';PACK=ROOT/'collection_review_outputs'

def main():
 OUT.mkdir(exist_ok=True);contract=read(PACK/'CONTRACT.json');manifest=read(PACK/'MANIFEST.json')
 assert contract['new_independent_tyres']==contract['confirmed_mappings_created_by_agent']==0
 assert all(sha(PACK/n)==v['sha256'] for n,v in manifest['files'].items())
 assert len(table(PACK/'review_images.csv'))==44 and len(table(PACK/'identity_candidates.csv'))==9 and len(table(PACK/'capture_priorities.csv'))==3
 assert all(not r['confirmed_original_tyre_id'] and r['review_status']=='unresolved' for r in table(PACK/'video_identity_resolutions_UNREVIEWED.csv'))
 html=(PACK/'TREAD_ONE_Identity_And_Capture_Review.html').read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',html,re.S).group(1);syntax=OUT/'script.js';syntax.write_text(script,encoding='utf-8')
 node=shutil.which('node');checked=subprocess.run([node,'--check',str(syntax)],capture_output=True,text=True);assert checked.returncode==0,checked.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0]
 assert len(data['images'])==44 and len(data['groups'])==15 and len(data['candidate_ids'])==3
 for r in data['images'].values():
  with Image.open(io.BytesIO(base64.b64decode(r['preview'].split(',',1)[1]))) as image:assert image.size==(384,512);image.verify()
 # Exercise the exact shipped logic in Node without claiming a browser render.
 fixture=OUT/'logic_test.cjs';fixture.write_text('''const fs=require('fs'),vm=require('vm'),assert=require('assert');
const sandbox={module:{exports:{}}};vm.runInNewContext(fs.readFileSync(process.argv[2],'utf8'),sandbox);
const {reviewRows,csvText}=sandbox.module.exports;const data=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));let checks=[];
const initial=reviewRows(data,{});assert.equal(initial.errors.length,0);assert(initial.rows.every(r=>r.review_status==='unresolved'&&!r.confirmed_original_tyre_id));checks.push('default_export_never_invents_mapping');
const state={};data.clips.forEach((c,i)=>state[c.video_id]={parent:data.candidate_ids[i],evidence:'supporting_record_'+i,confirmed:true,notes:'Reviewed, with "source"'});
const good=reviewRows(data,state);assert.equal(good.errors.length,0);assert(good.rows.every(r=>r.review_status==='confirmed'));checks.push('three_distinct_evidenced_mappings_export');
const copy=()=>JSON.parse(JSON.stringify(state));let bad=copy();bad[data.clips[0].video_id].evidence='';assert(reviewRows(data,bad).errors.length);checks.push('missing_evidence_blocks_export');
bad=copy();bad[data.clips[1].video_id].parent=data.candidate_ids[0];assert(reviewRows(data,bad).errors.length);checks.push('known_distinct_clips_cannot_share_parent');
bad=copy();bad[data.clips[0].video_id].parent='invented';assert(reviewRows(data,bad).errors.length);checks.push('unknown_parent_rejected');
bad=copy();bad[data.clips[0].video_id].confirmed=false;assert(reviewRows(data,bad).errors.length);checks.push('selection_without_confirmation_cannot_silently_export');
bad=copy();bad[data.clips[0].video_id].evidence='=1+1';assert(reviewRows(data,bad).errors.length);checks.push('formula_like_evidence_rejected');
const csv=csvText(good.rows,data.resolution_fields);assert(csv.includes('"Reviewed, with ""source"""'));checks.push('CSV_quotes_and_commas_preserved');
fs.writeFileSync(process.argv[4],csv);fs.writeFileSync(process.argv[5],JSON.stringify(checks));''',encoding='utf-8')
 datafile=OUT/'data.json';atomic(datafile,{k:data[k] for k in ('clips','candidate_ids','resolution_fields')});exported=OUT/'reviewed_fixture.csv';checksfile=OUT/'checks.json'
 result=subprocess.run([node,str(fixture),str(ROOT/'collection_review/review_ui.js'),str(datafile),str(exported),str(checksfile)],capture_output=True,text=True);assert result.returncode==0,result.stderr
 checks=read(checksfile)
 with tempfile.TemporaryDirectory(prefix='identity_fixture_',dir=OUT) as temporary:
  target=Path(temporary);shutil.copy2(exported,target/'video_identity_resolutions.csv');identity=read(ROOT.parent/'phase2/data/phase2_dataset_v1/provenance/phase2_video_identity.json');mapped,issues=resolutions(target,identity);assert len(mapped)==3 and not issues
 checks.append('exported_CSV_passes_unchanged_NB07_identity_validator')
 assert all(not r['confirmed_original_tyre_id'] for r in table(ROOT/'data_intake/video_identity_resolutions.csv'))
 checks.append('actual_project_resolution_inputs_still_unmodified')
 report=dict(status='passed',logic_checks=checks,logic_check_count=len(checks),source_images_verified=44,preview_dimensions=[384,512],original_tyres=12,existing_clips=3,planning_CSV_tables=3,planning_rows=56,unreviewed_control_CSV_rows=3,
  source_NB07_revision=contract['source_revision'],confirmed_mappings_created_by_agent=0,new_independent_tyres=0,JS_syntax_valid=True,browser_render_verified=False,new_training=False,new_inference=False,HF_publication_status='local_preparation_not_published')
 atomic(OUT/'VALIDATION.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
