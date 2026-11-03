// Actual report JavaScript with a bounded DOM/canvas stub; not a browser-render claim.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const raw=html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(raw);let draws=0,downloads=0;
class E{
  constructor(){this.children=[];this.handlers={};this.value='';this.textContent=''}
  append(...x){this.children.push(...x)}
  addEventListener(k,f){this.handlers[k]=f}
  getContext(){return {clearRect(){},fillRect(){draws++},beginPath(){},moveTo(){},lineTo(){},stroke(){}}}
  getBoundingClientRect(){return {left:0,top:0,width:256,height:512}}
  click(){downloads++}
}
const nodes={};for(const id of ['data','example','channel','display','previous','next','download','photo','caseid','map','readout','scale'])nodes[id]=new E();
nodes.data.textContent=raw;nodes.display.value='all';
const context={document:{getElementById:id=>nodes[id],createElement:()=>new E()},Blob,URL:{createObjectURL(){return 'blob:fixture'},revokeObjectURL(){}}};
vm.runInNewContext(script,context);
assert.equal(data.length,4);assert.equal(nodes.example.children.length,4);assert.equal(nodes.channel.children.length,6);
for(const c of data){assert.equal(c.role,'train');assert.equal(c.channels.length,6);assert.equal(Buffer.from(c.image.split(',')[1],'base64').subarray(0,8).toString('hex'),'89504e470d0a1a0a')}
for(let i=0;i<4;i++){
  nodes.example.value=String(i);nodes.example.handlers.change();
  assert.equal(context.traceDesign.state.caseIndex,i);assert(nodes.photo.src.startsWith('data:image/png;base64,'));
  assert(nodes.caseid.textContent.includes(data[i].image_id));
  for(let j=0;j<6;j++){
    nodes.channel.value=String(j);nodes.channel.handlers.change();
    assert.equal(context.traceDesign.state.channelIndex,j);
    assert.equal(context.traceDesign.csv().trim().split('\n').length,513);
    assert.equal(context.traceDesign.inspect(3,3),data[i].channels[j].available[3][3]?data[i].channels[j].values[3][3]:null);
  }
}
nodes.next.handlers.click();assert.equal(context.traceDesign.state.caseIndex,0);
nodes.previous.handlers.click();assert.equal(context.traceDesign.state.caseIndex,3);
nodes.display.value='masked';nodes.display.handlers.change();
nodes.map.handlers.mousemove({clientX:40,clientY:65});assert.match(nodes.readout.textContent,/Band 1/);
nodes.download.handlers.click();assert.equal(downloads,1);
assert.equal(context.traceDesign.inspect(-1,0),null);assert(draws>24*512);
console.log(JSON.stringify({status:'passed',actual_script:true,training_cases:4,channels:6,paging:true,hidden_overlay:true,hover:true,CSV_export_rows:512,PNG_signatures:true,native_browser_render:false}));
