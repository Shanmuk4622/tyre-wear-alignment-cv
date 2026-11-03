// Execute the actual report script with a small DOM fixture; no browser-render claim.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const page=fs.readFileSync(process.argv[2],'utf8'),script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
class E{constructor(){this.children=[];this.handlers={};this.value='';this.textContent='';this.style={};this.attributes={}}append(...v){this.children.push(...v)}replaceChildren(...v){this.children=v}addEventListener(k,f){this.handlers[k]=f}setAttribute(k,v){this.attributes[k]=v}click(){this.clicked=true}}
const ids=['role','component','sort','search','previous','next','count','grid','page','csv'],elements=Object.fromEntries(ids.map(k=>[k,new E()]));
elements.role.value='all';elements.component.value='all';elements.sort.value='dice';let blob=null;
const context=vm.createContext({document:{getElementById:id=>elements[id],createElement:()=>new E(),createElementNS:()=>new E()},URL:{createObjectURL:x=>(blob=x,'fixture'),revokeObjectURL:()=>{}},Blob,setTimeout:fn=>fn()});
vm.runInContext(script,context);assert.match(elements.count.textContent,/^570 source images/);assert.equal(elements.grid.children.length,24);assert.equal(elements.component.children.length,10);
elements.next.handlers.click();assert.match(elements.page.textContent,/Page 2/);elements.previous.handlers.click();assert.match(elements.page.textContent,/Page 1/);
function change(id,value){elements[id].value=value;elements[id].handlers[id==='search'?'input':'change']()}
change('role','validation');assert.match(elements.count.textContent,/^81 source images/);change('sort','point');assert.match(elements.count.textContent,/^25 source images/);
assert(elements.grid.children.every(a=>a.children[1].children[1].children.length===12));
assert(elements.grid.children.every(a=>!a.children.at(-1).textContent.includes('Human point MAE Unavailable')));
change('role','test');assert.match(elements.count.textContent,/^25 source images/);change('sort','masked');assert.match(elements.count.textContent,/^103 source images/);
const image=elements.grid.children[0].children[0].textContent;change('search',image);assert.match(elements.count.textContent,/^1 source images/);elements.csv.handlers.click();assert(blob);
change('search','no such source');assert.match(elements.count.textContent,/^0 source images/);assert.match(elements.page.textContent,/Page 1 \/ 1/);
change('search','');change('role','all');change('component',elements.component.children[0].value);assert(elements.grid.children.length>0);
console.log(JSON.stringify({status:'passed',images:570,components:10,paging:true,role_human_sort_search_filters:true,CSV_export_invoked:true,weak_absent_point_hidden:true,native_browser_render:false}));
