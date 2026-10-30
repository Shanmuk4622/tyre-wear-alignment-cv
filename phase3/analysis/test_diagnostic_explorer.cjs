/* Browser QA for the standalone TREAD-ONE diagnostic; no external requests. */
const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/shanm/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
let activeBrowser;
async function main(){
 const root=path.resolve(__dirname,'..');
 const target=process.argv[2]||JSON.parse(fs.readFileSync(path.join(root,'diagnostics/DIAGNOSTIC_TESTS.json'),'utf8')).actual_resume.report;
 const out=path.join(root,'diagnostics/visual_validation');fs.mkdirSync(out,{recursive:true});
 const old=fs.readFileSync(target,'utf8');
 const start=old.indexOf('const DATA=')+'const DATA='.length,end=old.indexOf('const $',start);
 const payload=JSON.parse(old.slice(start,end).trim().replace(/;$/,''));
 const preview=path.join(out,'preview.html');
 fs.writeFileSync(preview,fs.readFileSync(path.join(root,'diagnostic/explorer.html'),'utf8').replace('__TREAD_DATA__',JSON.stringify(payload).replaceAll('<','\\u003c')));
 const browser=await chromium.launchPersistentContext(path.join(out,'browser_profile'),{headless:true,channel:'msedge',viewport:{width:1280,height:960},deviceScaleFactor:1});activeBrowser=browser;
 const page=await browser.newPage();
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!r.url().startsWith('file:')&&!r.url().startsWith('data:'))requests.push(r.url())});
 await page.goto(pathToFileURL(preview).href);await page.waitForFunction(()=>document.getElementById('support').textContent!=='—');
 await page.waitForFunction(()=>document.getElementById('canvas').getContext('2d').getImageData(190,200,1,1).data[3]>0);
 await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
 const initial=await page.locator('#support').innerText();
 const canvas=()=>page.locator('#canvas').evaluate(x=>x.toDataURL());
 const before=await canvas();await page.locator('#opacity').focus();await page.locator('#opacity').press('Home');
 await page.waitForFunction(previous=>document.getElementById('canvas').toDataURL()!==previous,before);
 const opacityChanged=(await canvas())!==before;
 await page.selectOption('#variant','oracle_tread_gate');await page.waitForFunction(()=>document.getElementById('imageDetail').textContent.includes('Oracle'));
 const oracleDetail=await page.locator('#selectedDetail').innerText();
 await page.selectOption('#split','train');await page.waitForFunction(()=>document.getElementById('imageDetail').textContent.includes('train'));
 const trainSupport=await page.locator('#support').innerText();
 await page.selectOption('#layer','tread');await page.locator('#points').uncheck();
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
 const report={status:errors.length===0&&requests.length===0&&opacityChanged&&overflow<=2?'passed':'failed',
   target,errors,external_requests:requests,opacity_changes_canvas:opacityChanged,initial_support:initial,train_support:trainSupport,
   oracle_controls_update:oracleDetail.includes('probability L1 change'),mobile_horizontal_overflow_pixels:overflow};
 fs.writeFileSync(path.join(out,'VISUAL_QA.json'),JSON.stringify(report,null,2));await browser.close();console.log(JSON.stringify(report,null,2));
 if(report.status!=='passed')process.exitCode=1;
}
main().catch(async e=>{console.error(e);if(activeBrowser)await activeBrowser.close();process.exitCode=1});
