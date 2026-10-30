/* Offline identity review. Choices require human evidence and never write inputs. */
function csvText(rows,fields){
  const quote=v=>{const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
  return fields.map(quote).join(',')+'\r\n'+rows.map(r=>fields.map(k=>quote(r[k])).join(',')).join('\r\n')+'\r\n';
}
function reviewRows(data,states){
  const errors=[],seen=new Set(),rows=[];
  for(const clip of data.clips){
    const s=states[clip.video_id]||{},parent=String(s.parent||''),evidence=String(s.evidence||'').trim(),notes=String(s.notes||'').trim();
    if(s.confirmed){
      if(!data.candidate_ids.includes(parent))errors.push(clip.video_id+': choose an allowed original mid tyre.');
      if(!evidence)errors.push(clip.video_id+': a supporting capture/identity record is required.');
      if(/^[=+@\-\t\r\n]/.test(evidence)||/^[=+@\-\t\r\n]/.test(notes))errors.push(clip.video_id+': use plain-text evidence references and notes.');
      if(parent&&seen.has(parent))errors.push('Distinct clips cannot be confirmed as the same physical tyre.');
      if(parent)seen.add(parent);
    }else if(parent||evidence){errors.push(clip.video_id+': confirm the evidence-based selection or clear it to keep unresolved.');}
    rows.push({video_id:clip.video_id,confirmed_original_tyre_id:s.confirmed?parent:'',review_status:s.confirmed?'confirmed':'unresolved',evidence_source:s.confirmed?evidence:'phase2_video_identity.json',notes:notes||'Known existing mid tyre; exact original session not confirmed'});
  }
  return {errors,rows};
}
if(typeof module!=='undefined')module.exports={csvText,reviewRows};
if(typeof document!=='undefined'){
  const $=id=>document.getElementById(id),states={};
  function text(tag,value,className){const node=document.createElement(tag);node.textContent=value;if(className)node.className=className;return node;}
  function strip(group){const holder=document.createElement('div');holder.className='strip';for(const id of group.image_ids){const r=DATA.images[id],item=document.createElement('figure'),img=document.createElement('img');img.src=r.preview;img.alt=id;img.loading='lazy';item.append(img,text('figcaption',id+(r.timestamp_seconds!==''?' · '+r.timestamp_seconds+' s':'')));holder.append(item);}return holder;}
  function refresh(){const result=reviewRows(DATA,states);$('errors').replaceChildren(...result.errors.map(v=>text('p',v)));$('errors').hidden=!result.errors.length;$('download').disabled=Boolean(result.errors.length);$('status').textContent=result.errors.length?'Resolve the listed input issues before exporting.':result.rows.filter(r=>r.review_status==='confirmed').length+' confirmed mappings; all others remain unresolved.';}
  function update(id,key,value){states[id][key]=value;refresh();}
  for(const group of DATA.groups){
    const card=document.createElement('section');card.className='card';card.append(text('h3',group.label),text('p',group.kind==='clip'?'Existing mid tyre · original parent unresolved':'Existing '+group.proxy_class+' tyre · historical '+group.historical_split),strip(group));
    if(group.kind==='clip'){
      const id=group.video_id;states[id]={parent:'',evidence:'',notes:'',confirmed:false};const label=text('label','Original parent '),select=document.createElement('select');select.append(new Option('Unresolved',''));for(const tid of DATA.candidate_ids)select.append(new Option(tid,tid));select.addEventListener('change',()=>update(id,'parent',select.value));label.append(select);
      const evidence=document.createElement('input');evidence.type='text';evidence.placeholder='Supporting capture / identity record reference';evidence.setAttribute('aria-label',id+' supporting evidence');evidence.addEventListener('input',()=>update(id,'evidence',evidence.value));
      const notes=document.createElement('textarea');notes.placeholder='Reviewer notes; leave uncertain identities unresolved';notes.setAttribute('aria-label',id+' review notes');notes.addEventListener('input',()=>update(id,'notes',notes.value));
      const check=document.createElement('input');check.type='checkbox';check.addEventListener('change',()=>update(id,'confirmed',check.checked));const confirm=text('label',' Confirm this mapping from supporting records');confirm.prepend(check);card.append(label,evidence,notes,confirm);
    }
    (group.kind==='clip'?$('clips'):$('tyres')).append(card);
  }
  for(const row of DATA.candidates){const tr=document.createElement('tr');for(const value of [row.video_id,row.possible_original_tyre_id,row.min_dhash_hamming_to_original_photos])tr.append(text('td',value));$('distances').append(tr);}
  for(const row of DATA.priorities){const tr=document.createElement('tr');for(const key of ['proxy_class','historical_train_tyres','historical_validation_tyres','historical_test_tyres','collection_reason'])tr.append(text('td',row[key]));$('priorities').append(tr);}
  $('download').addEventListener('click',()=>{const result=reviewRows(DATA,states);if(result.errors.length){refresh();return;}const blob=new Blob([csvText(result.rows,DATA.resolution_fields)],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='video_identity_resolutions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  refresh();
}
