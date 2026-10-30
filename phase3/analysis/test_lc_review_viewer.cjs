const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),data=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1],code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.textContent=''}append(...v){this.children.push(...v)}replaceChildren(...v){this.children=v}addEventListener(k,fn){this.handlers[k]=fn}}
const elements={};for(const id of ['data','summary','endpoint','variant','target','filter','class','search','count','cases','previous','next'])elements['#'+id]=new Element();
elements['#data'].textContent=data;elements['#endpoint'].value='selected';elements['#variant'].value='identity';elements['#target'].value='all';elements['#filter'].value='all';elements['#class'].value='pred';
vm.runInNewContext(code,{document:{querySelector:s=>elements[s],createElement:()=>new Element()},console});
function set(id,value){elements[id].value=value;elements[id].handlers.input()}
assert.equal(JSON.parse(data).cases.length,1442);assert.match(elements['#count'].textContent,/^103 records/);assert.equal(elements['#cases'].children.length,8);
elements['#next'].onclick();assert.match(elements['#count'].textContent,/page 2/);elements['#previous'].onclick();
set('#variant','grayscale_both');set('#filter','flip');assert.match(elements['#count'].textContent,/^0 records/);
set('#filter','all');set('#target','0');assert.match(elements['#count'].textContent,/^62 records/);
set('#target','all');set('#endpoint','final');set('#variant','gamma_080');assert.match(elements['#count'].textContent,/^103 records/);
set('#filter','wrong');assert.match(elements['#count'].textContent,/^62 records/);
set('#filter','all');set('#class','0');assert.equal(elements['#cases'].children[0].children[0].children[2].children.length,32);
const one=JSON.parse(data).cases.find(x=>x.endpoint==='final'&&x.variant==='gamma_080').image_id;
set('#search',one);assert.match(elements['#count'].textContent,/^1 records/);assert.equal(elements['#cases'].children.length,1);
console.log(JSON.stringify({status:'passed',case_records:1442,reference_cases:18,controls:['paging','six probes','flip filter','wrong filter','target','explained class','32patches','search'],native_browser_render:false}));
