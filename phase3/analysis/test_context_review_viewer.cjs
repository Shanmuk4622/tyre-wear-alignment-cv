// Actual report JavaScript, synthetic DOM only; no native rendering claim.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1],d=JSON.parse(raw),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class E{constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.textContent=''}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}addEventListener(k,f){this.handlers[k]=f}}
const e={};for(const id of ['data','summary','endpoint','variant','target','filter','class','search','count','cases','previous','next','component','layer','head'])e['#'+id]=new E();
for(const [id,v] of Object.entries({data:raw,endpoint:'selected',variant:'identity',target:'all',filter:'all',class:'pred',component:'combined',layer:'0',head:'0'}))e['#'+id][id==='data'?'textContent':'value']=v;
vm.runInNewContext(script,{document:{querySelector:s=>e[s],createElement:()=>new E()},console});
function set(id,v){e[id].value=v;e[id].handlers.input()}
assert.equal(d.cases.length,1854);assert.equal(d.bank.length,18);assert.equal(d.cases.filter(x=>x.attention).length,206);
assert.match(e['#count'].textContent,/^103 records/);assert.equal(e['#cases'].children.length,8);
e['#next'].onclick();assert.match(e['#count'].textContent,/page 2/);e['#previous'].onclick();
set('#component','context');set('#class','0');assert.equal(e['#cases'].children[0].children[0].children[2].children.length,32);
set('#layer','1');set('#head','3');
const photos=e['#cases'].children[0].children[0];assert(photos.children.some(x=>x.children.length===1024));
set('#variant','reverse_bands');assert.match(e['#count'].textContent,/^103 records/);
assert(e['#cases'].children[0].children[0].children.some(x=>/Full attention matrices/.test(x.textContent)));
set('#filter','flip');const base=new Map(d.cases.filter(x=>x.variant==='identity').map(x=>[x.endpoint+'|'+x.image_id,x]));
const flips=d.cases.filter(x=>x.endpoint==='selected'&&x.variant==='reverse_bands'&&x.prediction!==base.get(x.endpoint+'|'+x.image_id).prediction).length;
assert.match(e['#count'].textContent,new RegExp('^'+flips+' records'));
set('#filter','all');set('#target','0');assert.match(e['#count'].textContent,/^62 records/);
set('#target','all');set('#endpoint','final');set('#variant','reverse_columns');
set('#filter','wrong');const wrong=d.cases.filter(x=>x.endpoint==='final'&&x.variant==='reverse_columns'&&x.correct===0).length;
assert.match(e['#count'].textContent,new RegExp('^'+wrong+' records'));
set('#filter','all');set('#component','base');set('#search',d.cases.find(x=>x.endpoint==='final'&&x.variant==='reverse_columns').image_id);
assert.match(e['#count'].textContent,/^1 records/);
assert.equal(d.source_revision,'a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e');
console.log(JSON.stringify({status:'passed',case_records:1854,reference_cases:18,full_attention_cases:206,controls:['paging','eight probes','flip/wrong/target/search','base/context/combined','explained class','layer/head','attention missingness'],native_browser_render:false}));
