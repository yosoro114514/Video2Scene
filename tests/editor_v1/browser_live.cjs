// Actual browser against the independently running v1 API. --read-only never writes.
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const url=process.argv[2],out=path.resolve(process.argv[3]||''),readOnly=process.argv.includes('--read-only');
if(!url||!process.argv[3])throw Error('Usage: node browser_live.cjs URL NEW_OUTPUT [--read-only]');
if(fs.existsSync(out))throw Error('Use a new evidence output directory');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});
 const errors=[],commands=[];page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>{if(request.method()==='POST'&&request.url().endsWith('/api/v1/commands'))commands.push(JSON.parse(request.postData()))});
 const current=()=>page.evaluate(()=>structuredClone(window.vipeEditor.state.view));
 const save=(name,value)=>fs.writeFileSync(path.join(out,name+'.json'),JSON.stringify(value,null,2));
 async function headAfter(headSeq){const id=commands.at(-1)?.command_id;await page.waitForFunction(({seq,id})=>{const editor=window.vipeEditor,record=editor.completedCommands.get(id);return editor.state.view.head.head_seq>seq||(record&&record.state!=='succeeded')},{seq:headSeq,id},{timeout:180000});const next=await current();if(next.head.head_seq<=headSeq)throw Error('Command did not change head: '+JSON.stringify(await page.evaluate(id=>window.vipeEditor.completedCommands.get(id),id)));return next}
 try{
  await page.goto(url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady);await page.waitForFunction(()=>document.getElementById('evidence-image').naturalWidth>0);const initial=await current();save('initial_scene',initial);
  const target=initial.snapshot.objects.find(o=>o.instance_id==='t001')||initial.snapshot.objects.find(o=>o.kind==='object');assert(target,'editable object is required');await page.locator(`[data-id="${target.instance_id}"]`).click();
  let final=initial;const checks={loads_scene:true,source_image_decoded:true};
  if(!readOnly){
   const targetHeight=target.transform.scale[1]*1.05;await page.fill('#scale-y',String(targetHeight));await page.click('#save');const edited=await headAfter(initial.head.head_seq);assert(Math.abs(edited.snapshot.objects.find(o=>o.instance_id===target.instance_id).transform.scale[1]-targetHeight)<1e-10);save('edited_scene',edited);checks.numeric_saved=true;
   await page.reload({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady);assert(Math.abs(Number(await page.inputValue('#scale-y'))-targetHeight)<1e-10);checks.reload_preserves_edit=true;
   await page.click('#refine');const refined=await headAfter(edited.head.head_seq);save('refined_scene',refined);const before=edited.snapshot.objects.find(o=>o.instance_id===target.instance_id),after=refined.snapshot.objects.find(o=>o.instance_id===target.instance_id);
   assert.notEqual(before.asset.asset_version,after.asset.asset_version);assert.notEqual(before.asset.artifact.sha256,after.asset.artifact.sha256);assert.deepEqual(before.transform,after.transform);assert.deepEqual(before.local_bounds_m,after.local_bounds_m);assert.equal(before.support_id,after.support_id);
   for(const old of edited.snapshot.objects.filter(o=>o.instance_id!==target.instance_id))assert.deepEqual(refined.snapshot.objects.find(o=>o.instance_id===old.instance_id),old);checks.refinement_new_asset_preserves_root_bounds_support_and_non_targets=true;
   await page.click('#undo');const undone=await headAfter(refined.head.head_seq);assert.equal(undone.head.revision,edited.head.revision);await page.click('#redo');final=await headAfter(undone.head.head_seq);assert.equal(final.head.revision,refined.head.revision);checks.undo_redo=true;
   await page.reload({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady);assert.deepEqual((await current()).head,final.head);checks.reload_preserves_refined_head=true;
   await page.waitForFunction(()=>window.vipeEditor.viewer?.pendingAssets===0);assert.deepEqual(await page.evaluate(()=>window.vipeEditor.viewer.assetErrors),[]);assert.equal(await page.evaluate(id=>window.vipeEditor.viewer.roots.get(id)?.userData.renderMode,target.instance_id),'asset');checks.refined_GLB_decoded_by_browser=true;
  }
  await page.waitForFunction(()=>document.getElementById('evidence-image').naturalWidth>0);await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});await page.locator('#viewer').screenshot({path:path.join(out,'viewer.png')});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});checks.mobile_no_horizontal_overflow=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);assert(checks.mobile_no_horizontal_overflow);assert.deepEqual(errors,[]);checks.no_page_errors=true;
  save('commands',commands);save('final_scene',final);const result={passed:true,mode:readOnly?'actual_browser_actual_API_read_only':'actual_browser_actual_API_edit_refine_undo_redo',url,object_count:initial.snapshot.objects.length,target_id:target.instance_id,initial_head:initial.head,final_head:final.head,checks,page_errors:errors,not_run:readOnly?['write command','Blender build','refinement','model invocation']:['independent visual quality judgment','generalization beyond this desktop scene']};save('browser_result',result);console.log(JSON.stringify(result,null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});save('failure',{message:error.message,commands,page_errors:errors,body:await page.locator('body').innerText()});throw error}finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
