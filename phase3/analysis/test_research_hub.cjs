const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');const data=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const D=JSON.parse(data);assert.equal(D.models.length,3);assert.equal(D.metrics.length,6);assert.equal(D.publications.length,9);assert.equal(D.artifacts.length,14);
const nodes={};for(const id of ['data','models','records','caps','artifacts','model','kind'])nodes[id]={textContent:id==='data'?data:'',innerHTML:'',value:['model','kind'].includes(id)?'all':'',listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}};
vm.runInNewContext(script,{document:{getElementById:id=>{assert(nodes[id]);return nodes[id];}}});
const count=id=>(nodes[id].innerHTML.match(/<tr>/g)||[]).length-1;
assert.equal(count('models'),6);assert.equal(count('records'),9);assert.equal((nodes.artifacts.innerHTML.match(/<a /g)||[]).length,14);
for(const key of ['v1','lc','context']){nodes.model.value=key;nodes.model.listeners.change();assert.equal(count('models'),2);assert(nodes.models.innerHTML.includes('MODEL_CARD_'+key.toUpperCase()+'.md'));}
nodes.kind.value='fit';nodes.kind.listeners.change();assert.equal(count('records'),3);assert(nodes.records.innerHTML.includes('NB13/NB13A'));
nodes.kind.value='review';nodes.kind.listeners.change();assert.equal(count('records'),3);assert(nodes.records.innerHTML.includes('NB14'));assert(!nodes.records.innerHTML.includes('NB15'));
nodes.kind.value='gate';nodes.kind.listeners.change();assert.equal(count('records'),3);assert(nodes.records.innerHTML.includes('NB15'));assert(nodes.records.innerHTML.includes('NB07'));
assert(nodes.caps.innerHTML.includes('research only unpromoted'));assert(!nodes.models.innerHTML.includes('undefined'));assert(!nodes.records.innerHTML.includes('undefined'));
console.log(JSON.stringify({status:'passed',candidate_models:3,endpoint_records:6,publication_records:9,artifact_links:14,controls:['three model filters','fit/review/gate filters','model card links','capability display','pinned artifact links'],native_browser_render:false},null,2));
