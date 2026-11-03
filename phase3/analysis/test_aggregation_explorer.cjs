// Functional checks of the actual report JS; canvas/DOM stub, not a screenshot.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1],script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
let unavailable=0,draws=0;
class E{constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=[...x];}addEventListener(k,f){this.handlers[k]=f;}getContext(){return {clearRect(){},fillRect(){draws++;},fillText(t){if(t==='Unavailable')unavailable++;}};}}
const nodes={};for(const id of ['data','layer','role','component','view','variant','retained','original','altered','control','scope','bars','tiles'])nodes[id]=new E();nodes.data.textContent=raw;
const context={document:{getElementById:id=>nodes[id],createElement:()=>new E()}};vm.runInNewContext(script,context);const app=context.aggregationAudit;
assert.equal(app.data.groups.length,10);assert.equal(app.data.variants.length,9);assert.equal(app.data.summary.length,108);assert.equal(nodes.bars.children.length,9);
assert(Math.abs(app.summary(app.state()).retained_gain_fraction-.611371)<1e-6);
nodes.role.value='test';nodes.role.handlers.change();assert(nodes.scope.textContent.includes('103 correlated image records'));
const low=app.data.groups.find(g=>g.role==='test'&&g.proxy===0);nodes.component.value=String(low.id);nodes.component.handlers.change();assert(Math.abs(app.summary(app.state()).retained_gain_fraction-.320336)<1e-6);
nodes.layer.value='masked_context';nodes.layer.handlers.change();assert(Math.abs(app.summary(app.state()).retained_gain_fraction-.231642)<1e-6);
nodes.role.value='validation';nodes.role.handlers.change();nodes.variant.value='train_template';nodes.variant.handlers.change();assert(app.summary(app.state()).retained_gain_fraction<0);assert(nodes.retained.textContent.startsWith('-'));
nodes.view.value='1';nodes.view.handlers.change();assert(nodes.scope.textContent.includes('recorded view1'));
nodes.view.value='0';nodes.layer.value='encoded';nodes.variant.value='identity';app.render();
const a=app.tileMeans(app.state()),b=app.tileMeans(app.state(),'identity');for(let i=0;i<32;i++)assert(a[i]===b[i]);
// Inject unavailable data only in the test's in-memory payload.
for(const row of app.data.tiles)if(row[0]===0&&app.data.groups[row[1]].role==='validation'&&row[2]===0&&row[3]===0&&row[4]===0){row[5]=0;row[6]=0;}
app.render();assert(unavailable>0);assert(draws>32);assert(!/<script[^>]+src=/.test(html));
console.log(JSON.stringify({status:'passed',actual_JS:true,roles_components_branches_views_variants:true,negative_retention_visible:true,tile_identity_zero_delta:true,unavailable_branch_fixture:true,role_images:103,external_script_dependencies:0,native_browser_render:false}));
