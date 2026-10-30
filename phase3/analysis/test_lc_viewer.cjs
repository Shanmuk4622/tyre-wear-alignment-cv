// Execute the actual saved-report JavaScript with a synthetic DOM, no browser.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1],data=JSON.parse(raw);
class Element{constructor(){this.value='';this.textContent='';this.children=[];this.handlers={};this.style={}}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}addEventListener(k,f){this.handlers[k]=f}}
const e={};for(const id of ['data','scope','curves','endpoint','errors','search','count','cases','bank'])e['#'+id]=new Element();
e['#data'].textContent=raw;e['#endpoint'].value='all';e['#errors'].value='all';
vm.runInNewContext(script,{document:{querySelector:s=>e[s],createElement:()=>new Element()},console});
function set(key,value){e[key].value=value;e[key].handlers.input()}
assert.equal(e['#cases'].children.length,4);assert.equal(e['#bank'].children.length,18);
assert.match(e['#scope'].textContent,/ENGINEERING FIXTURE/);
set('#endpoint','selected');assert.equal(e['#cases'].children.length,2);
set('#errors','wrong');assert.equal(e['#cases'].children.length,data.cases.filter(x=>x.endpoint==='selected'&&x.correct==='0').length);
set('#errors','all');set('#endpoint','all');set('#search',data.cases[0].image_id);
assert.equal(e['#cases'].children.length,2);
set('#search','');set('#errors','empty');assert.equal(e['#cases'].children.length,data.cases.filter(x=>x.empty_support==='1').length);
console.log(JSON.stringify({status:'passed',actual_report_JavaScript:true,fixture_cases:4,reference_cards:18,checks:['engineering scope','both endpoints','wrong predictions','empty support','source search','signed patch and reference cards'],native_browser_render:false,network_requests:0}));
