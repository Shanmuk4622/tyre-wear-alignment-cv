const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes={};for(const id of ['data','status','caps','support','rows','chart','epoch','klass','variant'])nodes[id]={textContent:id==='data'?data:'',innerHTML:'',value:({epoch:'60',klass:'all',variant:'full_context'})[id]||'',listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}};
const context=vm.createContext({document:{getElementById:id=>{assert(nodes[id],id);return nodes[id];}}});vm.runInContext(script,context);
assert(nodes.status.textContent.includes('2 of 6'));assert(nodes.rows.innerHTML.includes('unresolved_mid_parent_component'));assert(nodes.caps.innerHTML.includes('research_only_unpromoted'));
assert.equal((nodes.chart.innerHTML.match(/<polyline/g)||[]).length,2);
nodes.klass.value='0';nodes.klass.listeners.change();assert(nodes.rows.innerHTML.includes('new_tire__session_001'));assert(!nodes.rows.innerHTML.includes('unresolved_mid_parent_component'));
nodes.klass.value='1';nodes.klass.listeners.change();assert(nodes.rows.innerHTML.includes('unresolved_mid_parent_component'));assert(!nodes.rows.innerHTML.includes('new_tire__session_001'));
for(const variant of ['frozen_prior','bias_only','spatial_only','full_context']){nodes.variant.value=variant;nodes.variant.listeners.change();assert(nodes.chart.innerHTML.includes('<polyline'));assert(!nodes.chart.innerHTML.includes('NaN'));}
nodes.epoch.value='1';nodes.epoch.listeners.change();assert(nodes.rows.innerHTML.includes('<td>'));
nodes.epoch.value='999';nodes.epoch.listeners.change();assert.equal(nodes.epoch.value,60);
nodes.epoch.value='-2';nodes.epoch.listeners.change();assert.equal(nodes.epoch.value,1);
console.log(JSON.stringify({status:'passed',component_epoch_variant_rows:JSON.parse(data).components.length,controls:['epoch bounds','three classes/all','four algebraic variants','weighted NLL curves','capability and conditional-support tables'],native_browser_render:false},null,2));
