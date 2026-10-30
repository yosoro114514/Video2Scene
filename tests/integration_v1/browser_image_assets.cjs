// Read-only browser verification of all five genuinely authored GLB assets.
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const url=process.argv[2],out=path.resolve(process.argv[3]),expected=JSON.parse(fs.readFileSync(process.argv[4],'utf8'));
if(fs.existsSync(out))throw Error('Use a new browser evidence directory');fs.mkdirSync(out,{recursive:true});
(async()=>{const browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});const page=await browser.newPage({viewport:{width:1440,height:1080}});const errors=[],writes=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/commands'))writes.push(r.url())});
try{await page.goto(url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady&&window.vipeEditor.viewer?.pendingAssets===0);
await page.waitForFunction(()=>document.getElementById('evidence-image').naturalWidth>0);
const state=await page.evaluate(()=>({view:structuredClone(window.vipeEditor.state.view),modes:[...window.vipeEditor.viewer.roots].map(([id,root])=>({id,mode:root.userData.renderMode})),assetErrors:window.vipeEditor.viewer.assetErrors}));
assert.deepEqual(state.view,expected);assert.equal(state.modes.length,5);assert(state.modes.every(x=>x.mode==='asset'));assert.deepEqual(state.assetErrors,[]);
await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});await page.locator('#viewer').screenshot({path:path.join(out,'viewer.png')});
await page.reload({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady&&window.vipeEditor.viewer?.pendingAssets===0);assert.deepEqual(await page.evaluate(()=>window.vipeEditor.state.view.head),expected.head);
await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);assert.deepEqual(writes,[]);
const renderer=await page.evaluate(()=>{const gl=window.vipeEditor.viewer.renderer.getContext(),ext=gl.getExtension('WEBGL_debug_renderer_info');return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):'unavailable'});
const result={passed:true,mode:'actual_browser_all_five_image_recipe_GLB_readonly',url,head:expected.head,objects:state.modes,asset_errors:state.assetErrors,page_errors:errors,scene_write_commands:writes,source_image_decoded:true,reload_persists_head:true,mobile_no_horizontal_overflow:true,renderer};fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}catch(error){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({message:error.message,page_errors:errors},null,2));throw error}finally{await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
