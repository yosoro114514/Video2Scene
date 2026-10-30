// Actual synthetic scene/editor/Blender integration. No model service is called.
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const python=process.env.V2S_TEST_PYTHON||path.join(root,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
const output=fs.mkdtempSync(path.join(os.tmpdir(),'video2scene-demo-'));
const evidence=path.join(root,'outputs/release-validation/demo-browser');fs.mkdirSync(evidence,{recursive:true});
const child=spawn(python,['-X','utf8','-m','video2scene','demo','--output',output,'--port','0'],{cwd:root,windowsHide:true});
let logs='';child.stdout.on('data',chunk=>logs+=chunk.toString());child.stderr.on('data',chunk=>logs+=chunk.toString());
let browser;
(async()=>{
  const deadline=Date.now()+30000;
  while(!logs.includes('Synthetic demo:')&&Date.now()<deadline){if(child.exitCode!==null)throw new Error(logs);await new Promise(r=>setTimeout(r,100))}
  const url=logs.match(/Synthetic demo: (http[^\s]+)/)?.[1];assert(url,'Demo startup failed: '+logs);
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:960}});const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(url);await page.waitForFunction(()=>window.viewerReady&&window.vipeEditor?.viewer?.pendingAssets===0);
  assert.equal(await page.locator('.object-row').count(),2);
  assert.equal(await page.locator('#language-form').isVisible(),false);
  await page.locator('.object-row[data-id="t001"]').click();
  await page.locator('#position-x').fill('0.20');await page.locator('#save').click();
  await page.waitForFunction(()=>window.vipeEditor?.state?.view?.head?.head_seq===1,{},{timeout:60000});
  assert.equal(Number(await page.locator('#position-x').inputValue()),.2);
  await page.getByRole('button',{name:'导出当前 Blender / GLB'}).click();
  await page.getByRole('link',{name:'下载 Blender'}).waitFor({timeout:60000});
  for(const name of ['下载 Blender','下载 GLB']){
    const href=await page.getByRole('link',{name}).getAttribute('href');
    const response=await page.request.get(new URL(href,url).href);
    assert.equal(response.status(),200);assert((await response.body()).length>100);
  }
  await page.reload();await page.waitForFunction(()=>window.viewerReady&&window.vipeEditor?.viewer?.pendingAssets===0);
  assert.equal(await page.evaluate(()=>window.vipeEditor.state.view.head.head_seq),1);
  await page.screenshot({path:path.join(evidence,'demo.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  const report={passed:true,checks:['actual_meshes_load','numeric_edit_commits','export_downloads','reload_preserves_head','no_page_errors'],model_calls:0};
  fs.writeFileSync(path.join(evidence,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
})().catch(error=>{console.error(error);process.exitCode=1}).finally(async()=>{if(browser)await browser.close();child.kill();});
