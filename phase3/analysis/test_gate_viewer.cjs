// Actual generated report code, synthetic DOM only; no browser/network.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.value='';this.textContent='';this.children=[];this.handlers={};this.style={}}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}addEventListener(k,f){this.handlers[k]=f}}
const e={};for(const key of ['data','summary','bank','cohort','class','variant','search','count','prev','next','cases'])e['#'+key]=new Element();
e['#data'].textContent=raw;e['#cohort'].value='all';e['#class'].value='all';e['#variant'].value='identity';
vm.runInNewContext(script,{document:{querySelector:s=>e[s],createElement:()=>new Element()},console});
function set(key,value){e[key].value=value;e[key].handlers.input()}
assert.match(e['#count'].textContent,/^386 images/);assert.equal(e['#cases'].children.length,8);assert.equal(e['#bank'].children.length,18);
e['#next'].onclick();assert.match(e['#count'].textContent,/page 2/);e['#prev'].onclick();
set('#cohort','v1_query');assert.match(e['#count'].textContent,/^378 images/);
set('#class','0');assert.match(e['#count'].textContent,/^66 images/);
set('#variant','gamma_120');assert.equal(e['#cases'].children.length,8);
const data=JSON.parse(raw),id=data.cases.find(x=>x.cohort==='v1_query'&&x.class_index===0).image_id;
set('#search',id);assert.match(e['#count'].textContent,/^1 images/);assert.equal(e['#cases'].children.length,1);
set('#search','');set('#class','all');set('#cohort','support_photo');assert.match(e['#count'].textContent,/^8 images/);
set('#class','2');assert.match(e['#count'].textContent,/^4 images/);
console.log(JSON.stringify({status:'passed',scope:'Actual generated JavaScript with synthetic DOM; no browser rendering claim',checks:['386 records','18 proposal cards','paging','query/support split','target classes','variant scores','search'],network_requests:0}));
