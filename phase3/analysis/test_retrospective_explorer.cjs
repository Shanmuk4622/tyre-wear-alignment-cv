// Exercise actual embedded report code with a DOM stub; not a native screenshot.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
class E{constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.checked=false;}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=[...v];}addEventListener(k,f){this.handlers[k]=f;}}
const nodes={};for(const id of ['data','family','role','search','below','cards','bars','rows','count','previous','next'])nodes[id]=new E();
nodes.data.textContent=raw;nodes.family.value='all';nodes.role.value='all';
const context={document:{getElementById:id=>nodes[id],createElement:()=>new E()}};
vm.runInNewContext(script,context);const app=context.retrospectiveAudit;
assert.equal(app.data.images.length,570);assert.equal(app.data.fits.length,7);assert.equal(app.data.groups.length,10);
assert.equal(nodes.cards.children.length,7);assert.equal(nodes.rows.children.length,12);assert.equal(nodes.bars.children.length,10);
assert.equal(app.filterImages('train').length,386);assert.equal(app.filterImages('validation').length,81);assert.equal(app.filterImages('test').length,103);
assert.equal(app.filterImages('test','',true).length,5);assert.equal(app.filterImages('validation','',true).length,0);
assert(app.data.images.every(r=>app.minGain(r)>0));
nodes.role.value='test';nodes.role.handlers.change();assert.equal(nodes.bars.children.length,2);assert.match(nodes.count.textContent,/103 image records/);
nodes.below.checked=true;nodes.below.handlers.change();assert.equal(nodes.rows.children.length,5);
nodes.below.checked=false;nodes.below.handlers.change();nodes.next.onclick();assert.match(nodes.count.textContent,/page 2/);nodes.previous.onclick();assert.match(nodes.count.textContent,/page 1/);
nodes.search.value='no-such-source-id';nodes.search.handlers.input();assert.equal(nodes.rows.children.length,0);assert.equal(nodes.next.disabled,true);assert.equal(nodes.previous.disabled,true);
nodes.family.value='TRACE';nodes.family.handlers.change();assert.equal(nodes.cards.children.length,1);
assert(!/<script[^>]+src=/.test(html));
console.log(JSON.stringify({status:'passed',actual_embedded_JS:true,image_records:570,groups:10,families:7,role_counts:true,sub5pct_test_records:5,search_paging_empty_states:true,model_filter:true,external_script_dependencies:0,native_browser_render:false}));
