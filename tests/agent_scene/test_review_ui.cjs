// Isolated browser test: all status, editor and confirmation routes are mocked.
// This never contacts a running scene API, GPT, SAM, ViPE or Blender.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const out=path.resolve(process.argv[2]||path.join(root,'outputs/redwood_agent_20261029/development/human_ui'));
fs.mkdirSync(out,{recursive:true});
const head={revision:'rev_review_1',head_seq:1};
const status={scene_id:'review_mock',phase:'preparing',message:'准备测试输入',ready_for_review:false,head:null,confirmation:null,require_edit:true,human_edit_saved:false,events:[{time:'10:00:00',message:'<script>unsafe()</script>'}]};
const posts=[];
let failConfirm=false,rejectConfirm=false,actualEditor=false;
const actualCommands=[];
const object={instance_id:'chair_01',kind:'object',category:'furniture/chair',transform:{translation_m:[1,0.5,2],scale:[1,1,1]},local_bounds_m:{min:[-0.3,0,-0.3],max:[0.3,0.9,0.3]},support_id:null};
const editorObject={...structuredClone(object),field_state:{},transform:{...structuredClone(object.transform),rotation_xyzw:[0,0,0,1]},asset:{asset_version:'mock_only',artifact:{uri:'mock:proxy',media_type:'application/json'}}};
const actualView={head:{revision:'mock_initial',head_seq:0},snapshot:{scene_id:'review_mock',revision:'mock_initial',scale_status:'estimated',objects:[editorObject,{...structuredClone(editorObject),instance_id:'coffee_table',category:'furniture/coffee_table'}]}};
const mockEditor=`<!doctype html><html><head><meta charset="utf-8"></head><body><header>mock editor</header><h1>ISOLATED MOCK · no geometry execution</h1><div class="card"><form id="language-form"></form><button id="refine">refine</button></div><form id="property-form"><input id="category" value="furniture/chair"><input id="support" value=""><input id="position-x" type="number" value="1"><input id="position-y" type="number" value="0.5"><input id="position-z" type="number" value="2"><input id="scale-x" type="number" value="1"><input id="scale-y" type="number" value="1"><input id="scale-z" type="number" value="1"></form><img id="evidence-image" src="/mock-source.svg"><div id="evidence-id">rgb-00342</div><div id="evidence-reason">mock RGB evidence</div><script>window.vipeEditor={state:{sceneId:'review_mock',selectedId:'chair_01',view:{head:${JSON.stringify(head)},snapshot:{objects:[${JSON.stringify(object)}]}}},pending:new Map(),refresh:async()=>{}};</script></body></html>`;
const server=http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/agent/status'){res.setHeader('Content-Type','application/json');return res.end(JSON.stringify(status));}
  if(url.pathname==='/agent/confirm'){
    let body='';for await(const chunk of req)body+=chunk;posts.push(JSON.parse(body));
    if(rejectConfirm){res.writeHead(409,{'Content-Type':'application/json'});return res.end(JSON.stringify({status:'error',error:{code:'STALE_REVISION',message:'隔离测试：服务端版本已变化，请重新核对。'}}));}
    if(failConfirm){res.writeHead(202,{'Content-Type':'application/json'});res.end('{incomplete');return;}
    status.confirmation=JSON.parse(body);status.phase='refining';status.ready_for_review=false;
    res.writeHead(202,{'Content-Type':'application/json'});return res.end(JSON.stringify({status:'accepted',operation_id:'mock-only'}));
  }
  if(actualEditor&&url.pathname==='/api/v1/tools/execute'){
    let body='';for await(const chunk of req)body+=chunk;const request=JSON.parse(body);
    const result=request.tool==='inspect_scene'?actualView:{frames:[{source_frame_id:'mock_actual_editor_rgb',rgb:{uri:'mock:rgb'},width:640,height:480,preview_time_s:0,source_time_s:0,selection_reason:'隔离测试源图'}]};
    res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({status:'ok',result}));
  }
  if(actualEditor&&url.pathname==='/api/v1/commands'){
    let body='';for await(const chunk of req)body+=chunk;const command=JSON.parse(body);actualCommands.push(command);
    for(const edit of command.edits){const obj=actualView.snapshot.objects.find(item=>item.instance_id===edit.instance_id);if(edit.field.startsWith('transform.'))obj.transform[edit.field.split('.')[1]]=edit.value;else obj[edit.field]=edit.value;}
    actualView.head={revision:'mock_saved_numeric',head_seq:1};actualView.snapshot.revision=actualView.head.revision;status.head={...actualView.head};status.human_edit_saved=true;
    res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({command_id:command.command_id}));
  }
  if(actualEditor&&url.pathname.startsWith('/api/v1/commands/')){res.setHeader('Content-Type','application/json');return res.end(JSON.stringify({state:'succeeded',message:'隔离测试数值保存成功'}));}
  if(actualEditor&&url.pathname.includes('/events')){res.writeHead(200,{'Content-Type':'text/event-stream'});return res.end(': mock events\n\n');}
  if(actualEditor&&url.pathname==='/api/v1/artifacts'){res.setHeader('Content-Type','image/svg+xml');return res.end('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#d9e3d4"/></svg>');}
  if(actualEditor&&!url.pathname.startsWith('/agent')){
    const relative=url.pathname==='/'?'index.html':decodeURIComponent(url.pathname.slice(1));
    const folder=path.resolve(root,'video2scene/editor_v1'),file=path.resolve(folder,relative);
    if(!file.startsWith(folder+path.sep)||!fs.existsSync(file)){res.writeHead(404);return res.end();}
    res.setHeader('Content-Type',file.endsWith('.html')?'text/html; charset=utf-8':file.endsWith('.css')?'text/css':'text/javascript');return res.end(fs.readFileSync(file));
  }
  if(url.pathname==='/'){res.setHeader('Content-Type','text/html');return res.end(mockEditor);}
  if(url.pathname==='/mock-source.svg'){res.setHeader('Content-Type','image/svg+xml');return res.end('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#d9e3d4"/><text x="50" y="100" fill="#245a47" font-size="26">Mock RGB for UI checks only</text></svg>');}
  const names={'/agent/':'review.html','/agent/review.mjs':'review.mjs','/agent/review.css':'review.css'};
  const name=names[url.pathname];if(!name){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',name.endsWith('.html')?'text/html; charset=utf-8':name.endsWith('.mjs')?'text/javascript':'text/css');
  res.end(fs.readFileSync(path.join(root,'video2scene/agent_scene',name)));
});

(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url=`http://127.0.0.1:${server.address().port}/agent/?scene_id=review_mock`;
  const browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const checks={};
  try{
    await page.goto(url,{waitUntil:'networkidle'});
    assert.equal(posts.length,0);assert(await page.locator('#editor-frame').isHidden());
    checks.no_editor_before_whitebox=true;
    assert((await page.locator('#run-events').innerText()).includes('<script>unsafe()</script>'));
    assert.equal(await page.locator('#run-events script').count(),0);checks.server_text_escaped=true;
    Object.assign(status,{phase:'needs_user',ready_for_review:true,head:{...head},message:'白模就绪，请核查。'});
    await page.click('#refresh-status');await page.waitForFunction(()=>document.querySelector('#confirm-note').textContent.includes('实际修正位置'));
    assert(await page.locator('#reviewed').isDisabled());assert.equal(posts.length,0);checks.saved_human_correction_required=true;
    status.human_edit_saved=true;await page.click('#refresh-status');await page.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    assert.equal(posts.length,0);assert(await page.locator('#confirm').isDisabled());checks.explicit_checkbox_required=true;
    const frame=page.frameLocator('#editor-frame');
    await frame.locator('#position-x').fill('1.2');
    await page.waitForFunction(()=>document.querySelector('#reviewed').disabled);
    assert((await page.locator('#confirm-note').innerText()).includes('尚未保存'));checks.unsaved_edit_blocks_confirmation=true;
    await frame.locator('#position-x').fill('1');
    await page.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    await page.check('#reviewed');assert(await page.locator('#confirm').isEnabled());
    await page.evaluate(()=>document.querySelector('#editor-frame').contentWindow.vipeEditor.pending.set('command_1',{}));
    await page.waitForFunction(()=>document.querySelector('#confirm').disabled);checks.pending_command_blocks_confirmation=true;
    await page.evaluate(()=>document.querySelector('#editor-frame').contentWindow.vipeEditor.pending.clear());
    await page.waitForFunction(()=>!document.querySelector('#confirm').disabled);
    await page.evaluate(()=>{const editor=document.querySelector('#editor-frame').contentWindow.vipeEditor;editor.state.view.head={revision:'rev_review_2',head_seq:2};});
    await page.waitForFunction(()=>!document.querySelector('#reviewed').checked&&document.querySelector('#confirm').disabled);
    checks.changed_version_clears_approval=true;
    status.head={revision:'rev_review_2',head_seq:2};await page.click('#refresh-status');
    await page.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    await page.evaluate(()=>{const editor=document.querySelector('#editor-frame').contentWindow.vipeEditor;editor.state.view.snapshot.objects[0].kind='surface';});
    await page.waitForFunction(()=>document.querySelector('#reviewed').disabled);checks.structure_not_refinable=true;
    await page.evaluate(()=>document.querySelector('#editor-frame').contentWindow.vipeEditor.state.view.snapshot.objects[0].kind='object');
    await page.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    await page.screenshot({path:path.join(out,'desktop_mock.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));checks.mobile_no_overflow=true;
    await page.screenshot({path:path.join(out,'mobile_mock.png'),fullPage:true});
    rejectConfirm=true;await page.check('#reviewed');await page.click('#confirm');
    await page.waitForFunction(()=>document.querySelector('#run-error').textContent.includes('隔离测试：服务端版本已变化'));
    await page.click('#refresh-status');await page.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    assert(await page.locator('#run-error').isVisible());assert((await page.locator('#run-error').innerText()).includes('隔离测试：服务端版本已变化'));
    assert.equal(posts.length,1);assert(await page.locator('#confirm').isDisabled());checks.definite_rejection_survives_status_refresh=true;rejectConfirm=false;
    await page.check('#reviewed');await page.click('#confirm');
    await page.waitForFunction(()=>document.querySelector('#phase-label').textContent.includes('精修'));
    assert.equal(posts.length,2);assert.deepEqual(posts[1],{scene_id:'review_mock',revision:'rev_review_2',head_seq:2,target_id:'chair_01'});checks.exact_confirm_payload_to_mock_only=true;
    assert(await page.locator('#confirm').isDisabled());
    await page.reload({waitUntil:'networkidle'});assert.equal(posts.length,2);assert(await page.locator('#confirm').isDisabled());checks.reload_does_not_repeat_confirm=true;
    // A fresh isolated browser session with an unreadable confirmation receipt.
    Object.assign(status,{phase:'needs_user',ready_for_review:true,head:{...head},confirmation:null});failConfirm=true;
    const uncertainPage=await browser.newPage({viewport:{width:1200,height:900}});uncertainPage.on('pageerror',error=>errors.push(error.message));
    await uncertainPage.goto(url,{waitUntil:'networkidle'});await uncertainPage.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    await uncertainPage.check('#reviewed');await uncertainPage.click('#confirm');
    await uncertainPage.waitForFunction(()=>document.querySelector('#confirm-note').textContent.includes('执行结果待查询'));
    assert.equal(posts.length,3);await uncertainPage.reload({waitUntil:'networkidle'});assert.equal(posts.length,3);assert(await uncertainPage.locator('#confirm').isDisabled());
    checks.uncertain_confirm_not_replayed_after_reload=true;
    await page.close();await uncertainPage.close();
    // Load production editor_v1 itself against isolated in-memory API routes.
    actualEditor=true;failConfirm=false;
    Object.assign(status,{phase:'needs_user',ready_for_review:true,head:{...actualView.head},confirmation:null,human_edit_saved:false,recommended_target_id:'coffee_table'});
    const editorPage=await browser.newPage({viewport:{width:1440,height:1000}});editorPage.on('pageerror',error=>errors.push(error.message));
    await editorPage.goto(url,{waitUntil:'domcontentloaded'});
    await editorPage.waitForFunction(()=>document.querySelector('#editor-frame').contentWindow.vipeEditor?.state?.view);
    await editorPage.click('#select-recommended');
    await editorPage.waitForFunction(()=>document.querySelector('#selected-object').textContent.includes('coffee_table'));
    const actualFrame=editorPage.frameLocator('#editor-frame');
    assert.equal(await actualFrame.locator('#category').inputValue(),'furniture/coffee_table');
    assert(await actualFrame.locator('#position-x').isVisible());assert(await actualFrame.locator('#scale-y').isVisible());assert(await actualFrame.locator('#save').isVisible());
    assert(await actualFrame.locator('#language-form').isHidden());assert(await actualFrame.locator('#refine').isHidden());
    checks.actual_editor_iframe_selects_recommended_coffee_table=true;
    await actualFrame.locator('#position-x').fill('1.15');await actualFrame.locator('#scale-y').fill('1.08');
    await editorPage.waitForFunction(()=>document.querySelector('#confirm-note').textContent.includes('尚未保存'));
    assert(await editorPage.locator('#select-recommended').isDisabled());assert.equal(actualCommands.length,0);
    await actualFrame.locator('#save').click();
    await editorPage.waitForFunction(()=>document.querySelector('#editor-frame').contentWindow.vipeEditor?.state?.view?.head?.head_seq===1);
    await editorPage.click('#refresh-status');await editorPage.waitForFunction(()=>!document.querySelector('#reviewed').disabled);
    assert.equal(actualCommands.length,1);assert.equal(actualCommands[0].kind,'patch');assert.deepEqual(actualCommands[0].selected_ids,['coffee_table']);
    assert.deepEqual(actualCommands[0].edits,[{instance_id:'coffee_table',field:'transform.translation_m',value:[1.15,0.5,2]},{instance_id:'coffee_table',field:'transform.scale',value:[1,1.08,1]}]);
    assert(await editorPage.locator('#confirm').isDisabled());assert.equal(posts.length,3);
    checks.actual_editor_numeric_save_enables_explicit_confirmation_only=true;
    await editorPage.screenshot({path:path.join(out,'actual_editor_mock_api.png'),fullPage:true});
    await editorPage.setViewportSize({width:390,height:844});
    await editorPage.waitForFunction(()=>{const frame=document.querySelector('#editor-frame');return document.documentElement.scrollWidth<=innerWidth&&frame.clientHeight>=frame.contentDocument.body.scrollHeight&&frame.contentDocument.documentElement.scrollWidth<=frame.clientWidth;});
    assert(await actualFrame.locator('#position-x').isVisible());assert(await actualFrame.locator('#save').isVisible());
    checks.actual_editor_mobile_preserves_all_numeric_controls=true;
    await editorPage.screenshot({path:path.join(out,'actual_editor_mobile_mock_api.png'),fullPage:true});
    // Read-only batch status never sends an original or a new confirmation.
    const batchIds=['coffee_table','mock_saved_02','chair_01',...Array.from({length:7},(_,i)=>`mock_waiting_${i}`)];
    Object.assign(status,{phase:'refining',message:'隔离批次正在处理 chair_01',recommended_target_id:'chair_01',ready_for_review:false,
      batch:{batch_id:'batch_furniture_001',phase:'running',editing_locked:true,completed_count:2,total:10,current_target_id:'chair_01',layout_preserved:true,
        message:'隔离批次正在处理 chair_01',targets:batchIds.map((target_id,i)=>({target_id,state:i<2?'accepted':i===2?'refining':'waiting',message:i===2?'<script>batchUnsafe()</script>':''}))}});
    await editorPage.click('#refresh-status');
    await editorPage.waitForFunction(()=>document.querySelector('#batch-targets').children.length===10);
    assert((await editorPage.locator('#batch-count').innerText()).includes('2 / 10'));
    assert.equal(await editorPage.locator('#batch-targets script').count(),0);
    assert(await editorPage.locator('#confirm').isHidden());assert.equal(posts.length,3);
    assert(await actualFrame.locator('#position-x').isDisabled());assert(await actualFrame.locator('#save').isDisabled());
    checks.batch_lists_real_counts_and_blocks_old_confirmation_and_edits=true;
    await editorPage.click('#select-recommended');
    await editorPage.waitForFunction(()=>document.querySelector('#selected-object').textContent.includes('chair_01'));
    assert((await editorPage.locator('#select-recommended').innerText()).includes('查看当前对象'));
    checks.batch_active_target_selection_tracks_server=true;
    actualView.head={revision:'mock_batch_head',head_seq:2};actualView.snapshot.revision='mock_batch_head';status.head={...actualView.head};
    status.phase='validating';status.batch.targets[2].state='validating';
    await editorPage.click('#refresh-status');
    await editorPage.waitForFunction(()=>document.querySelector('#scene-revision').textContent.includes('mock_batch_head'));
    assert((await editorPage.locator('#batch-targets').innerText()).includes('复核中'));
    assert.equal(actualCommands.length,1);assert.equal(posts.length,3);
    checks.batch_refreshes_saved_scene_without_commands=true;
    await editorPage.setViewportSize({width:1440,height:1000});
    await editorPage.screenshot({path:path.join(out,'batch_progress_mock_api.png'),fullPage:true});
    status.phase='complete';delete status.recommended_target_id;
    Object.assign(status.batch,{phase:'complete',editing_locked:false,completed_count:10,current_target_id:null,targets:batchIds.map(target_id=>({target_id,state:'accepted'}))});
    await editorPage.click('#refresh-status');
    await editorPage.waitForFunction(()=>document.querySelector('#batch-count').textContent.includes('10 / 10')&&!document.querySelector('#editor-frame').contentDocument.querySelector('#save').matches(':disabled'));
    assert(await actualFrame.locator('#save').isEnabled());assert(await editorPage.locator('#confirm').isHidden());
    assert.equal(posts.length,3);assert.equal(actualCommands.length,1);
    checks.verified_batch_completion_restores_editing_without_confirmation=true;
    const roomIds=['floor','wall_tv','wall_sofa_color','wall_sofa_green','interior_door','balcony_door','vase_tv_stand'];
    status.batch.phase='scene_changed';status.batch.history_only=true;
    Object.assign(status,{phase:'refining',message:'隔离房间任务正在构建地板',recommended_target_id:'floor',
      room_finish:{run_id:'room_finish_001',phase:'running',editing_locked:true,completed_count:0,total:7,current_target_id:'floor',layout_preserved:true,
        message:'隔离房间任务正在构建地板',targets:roomIds.map((target_id,i)=>({target_id,state:i===0?'refining':i===1?'prepared':'waiting'}))}});
    await editorPage.click('#refresh-status');
    await editorPage.waitForFunction(()=>document.querySelector('#batch-count').textContent.includes('0 / 7'));
    assert((await editorPage.locator('#page-title').innerText()).includes('房间饰面'));
    assert((await editorPage.locator('#phase-label').innerText()).includes('门 / 花瓶'));
    assert.equal(await editorPage.locator('#batch-targets li').count(),7);
    assert((await editorPage.locator('#batch-targets').innerText()).includes('待保存'));
    assert(await actualFrame.locator('#save').isDisabled());assert(await editorPage.locator('#confirm').isHidden());
    assert.equal(posts.length,3);assert.equal(actualCommands.length,1);
    checks.room_progress_supersedes_furniture_history_without_premature_completion=true;
    await editorPage.screenshot({path:path.join(out,'room_progress_mock_api.png'),fullPage:true});
    actualView.head={revision:'mock_room_finished_head',head_seq:3};actualView.snapshot.revision=actualView.head.revision;status.head={...actualView.head};
    status.phase='complete';delete status.recommended_target_id;
    Object.assign(status.room_finish,{phase:'complete',editing_locked:false,completed_count:7,current_target_id:null,
      message:'隔离房间任务已保存',targets:roomIds.map(target_id=>({target_id,state:'accepted'}))});
    await editorPage.click('#refresh-status');
    await editorPage.waitForFunction(()=>document.querySelector('#batch-count').textContent.includes('7 / 7')&&document.querySelector('#scene-revision').textContent.includes('mock_room_finished_head'));
    assert((await editorPage.locator('#phase-label').innerText()).includes('房间处理已保存'));
    assert(await actualFrame.locator('#save').isEnabled());assert.equal(posts.length,3);assert.equal(actualCommands.length,1);
    checks.room_completion_uses_latest_revision_and_restores_editing=true;
    assert.deepEqual(errors,[]);checks.no_page_errors=true;
    const result={passed:true,scope:'Isolated mock HTTP with mock editor and actual editor_v1 browser UI. All commands and confirmations use in-memory fixtures; no real confirmation, model execution, or geometry validation.',checks,errors};
    fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;server.close();});
