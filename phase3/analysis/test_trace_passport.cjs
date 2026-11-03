// Actual generated report script; bounded DOM/canvas stub, not browser rendering.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8'),raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(raw),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];let white=0;
class E{constructor(){this.children=[];this.handlers={};this.value='';this.textContent=''}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}addEventListener(k,f){this.handlers[k]=f}getContext(){return {fillStyle:'',fillRect(){if(this.fillStyle==='#fff')white++}}}}
const nodes={};for(const id of ['data','role','search','previous','next','count','gallery','status'])nodes[id]=new E();
// Explicit missingness injection tests the renderer's unavailable branch only.
data[0].available[0][0]=false;data[0].mean_residual=null;
nodes.data.textContent=JSON.stringify(data);nodes.role.value='all';
vm.runInNewContext(script,{document:{getElementById:id=>nodes[id],createElement:()=>new E()}});
assert.equal(nodes.gallery.children.length,Math.min(8,data.length));assert.match(nodes.count.textContent,/records \/ page 1/);
for(const c of data){assert.equal(c.residual.length,32);assert.equal(c.available.length,32);assert.equal(Buffer.from(c.image.split(',')[1],'base64').subarray(0,8).toString('hex'),'89504e470d0a1a0a')}
nodes.search.value=data[0].image_id;nodes.search.handlers.input();assert.equal(nodes.gallery.children.length,1);
nodes.role.value='validation';nodes.role.handlers.change();assert.equal(nodes.gallery.children.length,0);
nodes.role.value='all';nodes.search.value='';nodes.search.handlers.input();nodes.next.onclick();nodes.previous.onclick();
assert(nodes.status.textContent.includes('Research only'));assert(nodes.status.textContent.includes('Original GEO selection remains unavailable'));assert(white>0);
console.log(JSON.stringify({status:'passed',actual_script:true,cases:data.length,role_search_paging:true,unavailable_cells_white:true,white_cells_drawn:white,PNG_signatures:true,native_browser_render:false}));
