// Exercise the actual generated viewer code with a bounded synthetic DOM.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const page = fs.readFileSync(process.argv[2], 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const make = () => ({children:[], value:'all', textContent:'', append(...x){this.children.push(...x)},
  replaceChildren(...x){this.children=x}, addEventListener(name,fn){this[name]=fn}});
const nodes={filter:make(), count:make(), gallery:make()};
const context = vm.createContext({document:{getElementById:id=>nodes[id],createElement:make}});
vm.runInContext(script,context);
assert.strictEqual(nodes.gallery.children.length,32);
assert.strictEqual(nodes.count.textContent,'32 engineering cases');
for(const update of ['1','2','3','4']){
  nodes.filter.value=update;
  nodes.filter.change();
  assert.strictEqual(nodes.gallery.children.length,8);
  for(const article of nodes.gallery.children){
    assert(article.children[0].textContent.startsWith('Update '+update+' · '));
    const img=article.children[1];
    assert(img.src.startsWith('data:image/png;base64,'));
    assert.strictEqual(Buffer.from(img.src.split(',')[1],'base64').subarray(0,8).toString('hex'),'89504e470d0a1a0a');
    assert(img.alt.length>0);
  }
}
nodes.filter.value='all';nodes.filter.change();
assert.strictEqual(nodes.gallery.children.length,32);
console.log(JSON.stringify({status:'passed',scope:'actual_script_synthetic_DOM_not_browser_screenshot',cases:32,update_filters:4,png_signatures_valid:true}));
