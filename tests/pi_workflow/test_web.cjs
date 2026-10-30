// Isolated HTTP fixtures, including production editor_v1. No model, real upload,
// real project, real scene command or human approval reaches a running service.
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'outputs/pi_workflow_v1/development/web');
fs.mkdirSync(out,{recursive:true});
const projectRows=[],creates=[],messages=[],controls=[],uploads=[],edits=[];
let rejectApproval=false,uncertainMessage=false,incompleteCreate=false,incompleteControl=false;
function cubeGLB(){
  const positions=new Float32Array([-.3,0,-.3,.3,0,-.3,.3,.9,-.3,-.3,.9,-.3,-.3,0,.3,.3,0,.3,.3,.9,.3,-.3,.9,.3]);
  const indices=new Uint16Array([0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5]);
  const binary=Buffer.concat([Buffer.from(positions.buffer),Buffer.from(indices.buffer)]),gltf={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},indices:1,material:0}]}],materials:[{pbrMetallicRoughness:{baseColorFactor:[.9,.94,.88,1],metallicFactor:0,roughnessFactor:1}}],buffers:[{byteLength:binary.length}],bufferViews:[{buffer:0,byteOffset:0,byteLength:positions.byteLength},{buffer:0,byteOffset:positions.byteLength,byteLength:indices.byteLength}],accessors:[{bufferView:0,componentType:5126,count:8,type:'VEC3',min:[-.3,0,-.3],max:[.3,.9,.3]},{bufferView:1,componentType:5123,count:36,type:'SCALAR'}]};
  const serialized=Buffer.from(JSON.stringify(gltf)),jsonChunk=Buffer.alloc(Math.ceil(serialized.length/4)*4,32);serialized.copy(jsonChunk);const header=Buffer.alloc(20),binHeader=Buffer.alloc(8);header.writeUInt32LE(0x46546c67,0);header.writeUInt32LE(2,4);header.writeUInt32LE(28+jsonChunk.length+binary.length,8);header.writeUInt32LE(jsonChunk.length,12);header.writeUInt32LE(0x4e4f534a,16);binHeader.writeUInt32LE(binary.length,0);binHeader.writeUInt32LE(0x004e4942,4);return Buffer.concat([header,jsonChunk,binHeader,binary]);
}
const glbFixture=cubeGLB();
const obj={instance_id:'chair',kind:'object',category:'furniture/chair',support_id:null,field_state:{},
  transform:{translation_m:[1,.5,2],scale:[1,1,1],rotation_xyzw:[0,0,0,1]},local_bounds_m:{min:[-.3,0,-.3],max:[.3,.9,.3]},
  asset:{asset_version:'whitebox-isolated',artifact:{uri:'mock:fixture.glb',media_type:'model/gltf-binary'},native_to_object:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}};
const view={head:{revision:'mock_whitebox',head_seq:0},snapshot:{scene_id:'pi_mock_scene',revision:'mock_whitebox',scale_status:'estimated',
  objects:[obj,{...structuredClone(obj),instance_id:'coffee_table',category:'furniture/coffee_table'}]}};
const events=[{seq:1,type:'tool_started',text:'隔离测试工具 · <script>unsafe()</script>',data:{},at:'mock time'},
  {seq:2,type:'tool_completed',text:'隔离候选',data:{result:{candidate:{candidate_id:'mock_candidate',label:'<img onerror=unsafe()>'},saved:false}},at:'mock time'}];
const json=(res,data,status=200)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8'});res.end(JSON.stringify(data));};
const server=http.createServer(async(req,res)=>{
  const url=new URL(req.url,'http://127.0.0.1');
  if(url.pathname==='/workflow/api/capabilities')return json(res,{rolling_refinement:true,default_object_concurrency:3,max_object_concurrency:4});
  const body=async()=>{const chunks=[];for await(const chunk of req)chunks.push(chunk);return Buffer.concat(chunks);};
  if(url.pathname==='/workflow/api/projects'){
    if(req.method==='GET')return json(res,{projects:projectRows});
    const value=JSON.parse(await body());creates.push(value);
    const p={id:projectRows.length?'uploaded_project':'local_project',scene_id:projectRows.length?'uploaded_scene':'pi_mock_scene',
      name:value.name,source_path:value.source_path,phase:value.instruction?'running':'imported',head:null,model:'gpt-6-astra',provider:'codex_subscription',has_scene:false,
      active_run:!!value.instruction,messages:value.instruction?[{id:'initial',role:'user',text:value.instruction,at:'mock'}]:[],frames:[]};
    projectRows.push(p);return json(res,incompleteCreate?{accepted:true}:p,201);
  }
  if(url.pathname==='/workflow/api/upload'){
    const bytes=await body();uploads.push({name:url.searchParams.get('name'),bytes:[...bytes],contentType:req.headers['content-type']});
    return json(res,{source_path:'isolated/uploads/video.mp4'});
  }
  const match=url.pathname.match(/^\/workflow\/api\/projects\/([^/]+)(\/.*)?$/);
  if(match){
    const p=projectRows.find(item=>item.id===decodeURIComponent(match[1]));if(!p)return json(res,{error:{message:'Not found'}},404);
    if(!match[2])return json(res,p);
    if(match[2]==='/events')return json(res,{events:events.filter(row=>row.seq>Number(url.searchParams.get('after')||0)),cursor:events.at(-1)?.seq||0});
    if(match[2]==='/messages'){
      const value=JSON.parse(await body());messages.push({project:p.id,...value});
      if(uncertainMessage){res.writeHead(202,{'Content-Type':'application/json'});return res.end('{incomplete');}
      p.messages.push({id:'msg_'+messages.length,role:'user',text:value.message,at:'mock'});
      return json(res,{accepted:true,run_id:'mock_run',project:p},202);
    }
    if(match[2]==='/control'){
      const value=JSON.parse(await body());controls.push({project:p.id,...value});
      if(value.action==='approve_whitebox'&&rejectApproval)return json(res,{error:{message:'隔离测试：保存版本已变化，请重新核对。',code:'STALE_REVISION'}},409);
      if(value.action==='start_refine_all')p.batch={batch_id:'fixture_batch',mode:value.mode,parallelism:value.parallelism,effective_concurrency:value.parallelism,status:'running',total:4,completed_count:1,skipped_ids:['coffee_table'],targets:[{target_id:'chair',state:'saved'},{target_id:'floor',state:'generating'},{target_id:'wall',state:'generating'},{target_id:'door',state:'package_ready'}]};
      if(value.action==='set_refinement_concurrency'){p.batch.parallelism=value.parallelism;p.batch.effective_concurrency=value.parallelism;}
      p.phase=value.action==='pause'?'paused':'running';p.active_run=p.phase==='running';return json(res,incompleteControl?{accepted:true}:p);
    }
  }
  if(url.pathname==='/api/v1/tools/execute'){
    const value=JSON.parse(await body());return json(res,{status:'ok',result:value.tool==='inspect_scene'?view:{frames:[{source_frame_id:'mock_source',rgb:{uri:'mock:rgb'},width:640,height:480,preview_time_s:0,source_time_s:0,selection_reason:'隔离 RGB'}]}});
  }
  if(url.pathname==='/api/v1/commands'){
    const value=JSON.parse(await body());edits.push(value);
    for(const edit of value.edits){const target=view.snapshot.objects.find(item=>item.instance_id===edit.instance_id);if(edit.field.startsWith('transform.'))target.transform[edit.field.split('.')[1]]=edit.value;else target[edit.field]=edit.value;}
    view.head={revision:'mock_numeric_saved',head_seq:1};view.snapshot.revision=view.head.revision;projectRows[0].head={...view.head};
    return json(res,{command_id:value.command_id});
  }
  if(url.pathname.startsWith('/api/v1/commands/'))return json(res,{state:'succeeded',message:'隔离数值保存'});
  if(url.pathname.includes('/api/v1/scenes/')&&url.pathname.endsWith('/events')){res.writeHead(200,{'Content-Type':'text/event-stream'});return res.end(': isolated event stream\n\n');}
  if(url.pathname==='/workflow/artifacts'||url.pathname==='/api/v1/artifacts'){
    if(url.searchParams.get('uri')==='mock:fixture.glb'){res.writeHead(200,{'Content-Type':'model/gltf-binary'});return res.end(glbFixture);}
    res.writeHead(200,{'Content-Type':'image/svg+xml'});return res.end('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#dce8d8"/><text x="30" y="60" font-size="24" fill="#2f654e">Isolated source fixture</text></svg>');
  }
  const workflow=url.pathname.startsWith('/workflow/'),folder=path.join(root,workflow?'video2scene/pi_workflow/web':'video2scene/editor_v1');
  const relative=workflow?url.pathname.slice('/workflow/'.length)||'index.html':url.pathname==='/'?'index.html':url.pathname.slice(1);
  const file=path.resolve(folder,relative);
  if(!file.startsWith(path.resolve(folder)+path.sep)||!fs.existsSync(file)){res.writeHead(404);return res.end();}
  res.writeHead(200,{'Content-Type':file.endsWith('.html')?'text/html; charset=utf-8':file.endsWith('.css')?'text/css':'text/javascript'});res.end(fs.readFileSync(file));
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const url=`http://127.0.0.1:${server.address().port}/workflow/`;
  const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],checks={};page.on('pageerror',error=>errors.push(error.message));
  try{
    await page.goto(url,{waitUntil:'networkidle'});assert(await page.locator('#welcome').isVisible());assert.equal(creates.length+messages.length+controls.length,0);checks.read_only_initial_load=true;
    await page.click('#new-project');await page.fill('#new-name','隔离客厅');await page.check('input[value="path"]');await page.fill('#source-path','E:\\isolated\\livingroom.mp4');await page.fill('#new-instruction','先看视频关键帧并构建白模。');await page.click('#create-project');
    await page.waitForFunction(()=>document.querySelector('#project-name').textContent==='隔离客厅');assert.equal(creates.length,1);assert(creates[0].request_id);assert.equal(messages.length,0);assert.equal(creates[0].source_path,'E:\\isolated\\livingroom.mp4');assert(await page.locator('#scene-editor').isHidden());checks.local_import_initial_instruction_starts_only_create_request=true;
    await page.click('#pause-run');await page.waitForFunction(()=>document.querySelector('#project-phase').textContent==='已暂停');await page.click('#resume-run');await page.waitForFunction(()=>document.querySelector('#project-phase').textContent==='正在运行');assert.deepEqual(controls.map(row=>row.action),['pause','resume']);checks.pause_resume_follow_server_state=true;
    Object.assign(projectRows[0],{phase:'needs_review',has_scene:true,head:{...view.head},active_run:'mock_settling',frames:[{path:'keyframes/mock.jpg',label:'关键帧 <script>frameUnsafe()</script>'},{path:'../escape.jpg',label:'invalid'}]});
    await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#approval-note').textContent.includes('结束本轮'));assert(await page.locator('#approve-whitebox').isDisabled());checks.review_waits_for_actual_run_settlement=true;projectRows[0].active_run=false;
    await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#approve-whitebox').disabled);
    const frame=page.frameLocator('#scene-editor');await frame.locator('#pi-properties-toggle').click();assert.equal(controls.length,2);assert(await frame.locator('#language-form').isHidden());assert(await frame.locator('#save').isVisible());checks.whitebox_never_adopted_automatically=true;
    await page.waitForFunction(()=>{const viewer=document.querySelector('#scene-editor').contentWindow.vipeEditor.viewer,offset=viewer.camera.position.clone().sub(viewer.controls.target);return viewer.pendingAssets===0&&[...viewer.roots.values()].every(root=>root.userData.renderMode==='asset'&&root.userData.whiteboxEdges)&&offset.y>0&&Math.abs(offset.x/offset.y)<.002&&Math.abs(offset.z/offset.y)<.002;});checks.whitebox_glbs_gain_edges_and_initial_top_view=true;
    await frame.locator('#reset-camera').click();const perspective=await page.evaluate(()=>document.querySelector('#scene-editor').contentWindow.vipeEditor.viewer.camera.position.toArray());await page.click('#refresh-state');await page.waitForTimeout(1200);const afterPoll=await page.evaluate(()=>document.querySelector('#scene-editor').contentWindow.vipeEditor.viewer.camera.position.toArray());assert(perspective.every((n,i)=>Math.abs(n-afterPoll[i])<1e-6));checks.polling_preserves_user_perspective_view=true;
    await frame.locator('#pi-top-view').click();const topState=await page.evaluate(()=>{const editor=document.querySelector('#scene-editor').contentWindow.vipeEditor,viewer=editor.viewer,offset=viewer.camera.position.clone().sub(viewer.controls.target);return {offset:offset.toArray(),head:editor.state.view.head};});assert(topState.offset[1]>0&&Math.abs(topState.offset[0]/topState.offset[1])<.002&&Math.abs(topState.offset[2]/topState.offset[1])<.002);assert.deepEqual(topState.head,view.head);assert.equal(edits.length,0);assert.equal(controls.length,2);assert.equal(messages.length,0);checks.top_view_button_changes_only_camera=true;
    await page.waitForFunction(()=>document.querySelector('#events').textContent.includes('unsafe()'));
    assert.equal(await page.locator('#events script,#frames script,#candidates img').count(),0);assert.equal(await page.locator('#frames img').count(),1);assert((await page.locator('#candidates').innerText()).includes('等待校验与保存'));checks.server_text_and_relative_artifact_paths_are_safe=true;
    await frame.locator('[data-id="coffee_table"]').click();await page.waitForFunction(()=>document.querySelector('#chat-context').textContent.includes('coffee_table'));
    await frame.locator('#position-x').fill('1.25');await page.waitForFunction(()=>document.querySelector('#approve-whitebox').disabled&&document.querySelector('#send-message').disabled);
    for(const intent of ['edit_selected','refine_selected']){await page.selectOption('#chat-intent',intent);assert(await page.locator('#send-message').isDisabled());assert((await page.locator('#chat-note').innerText()).includes('尚未保存'));}await page.selectOption('#chat-intent','chat');checks.both_object_modes_block_unsaved_changes=true;
    assert((await page.locator('#approval-note').innerText()).includes('尚未保存'));checks.unsaved_edits_block_chat_and_approval=true;
    await frame.locator('#save').click();await page.waitForFunction(()=>document.querySelector('#scene-editor').contentWindow.vipeEditor.state.view.head.head_seq===1);
    await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#approve-whitebox').disabled);
    assert.equal(edits.length,1);assert.equal(edits[0].selected_ids[0],'coffee_table');checks.production_editor_numeric_save_preserved=true;
    await page.fill('#chat-input','核对这张茶几的支撑。');await page.click('#send-message');await page.waitForFunction(()=>document.querySelector('#chat-input').value==='');
    assert.equal(messages.length,1);assert(messages[0].request_id);assert.equal(messages[0].intent,'chat');assert.deepEqual(messages[0].selected_ids,['coffee_table']);assert.equal(messages[0].base_revision,'mock_numeric_saved');assert.equal(messages[0].base_head_seq,1);checks.chat_binds_real_selection_and_saved_head=true;
    for(const intent of ['edit_selected','refine_selected']){await page.selectOption('#chat-intent',intent);await page.fill('#chat-input',`隔离模式测试 ${intent}`);await page.click('#send-message');await page.waitForFunction(()=>document.querySelector('#chat-input').value==='');assert.equal(messages.at(-1).intent,intent);assert.deepEqual(messages.at(-1).selected_ids,['coffee_table']);assert.equal(messages.at(-1).base_revision,'mock_numeric_saved');assert.equal(messages.at(-1).base_head_seq,1);}await page.selectOption('#chat-intent','chat');checks.explicit_object_modes_bind_exact_selection_and_head=true;
    assert(await page.locator('#export-files').isHidden());projectRows[0].export={head:{...view.head},artifact_refs:[{role:'scene_blend',uri:'artifact://mock/scene & export.blend'},{role:'scene_glb',uri:'artifact://mock/scene.glb'},{role:'reopen_report',uri:'artifact://mock/report.json'}]};await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#export-version').textContent==='最新版下载');assert.equal(await page.locator('#export-links a').count(),2);assert.equal(await page.locator('#export-links a').first().getAttribute('href'),'/api/v1/artifacts?uri='+encodeURIComponent(projectRows[0].export.artifact_refs[0].uri));assert.equal(await page.locator('#project-phase').innerText(),'等待采用白模');checks.export_links_use_actual_receipt_without_claiming_completion=true;
    projectRows[0].export.head={revision:'older_export',head_seq:0};await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#export-version').textContent==='旧版导出');assert((await page.locator('#export-links').innerText()).includes('下载旧版'));assert(!(await page.locator('#export-files').innerText()).includes('最新版'));checks.stale_export_is_explicitly_labeled=true;projectRows[0].export.head={...view.head};await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#export-version').textContent==='最新版下载');
    rejectApproval=true;await page.click('#approve-whitebox');await page.waitForFunction(()=>document.querySelector('#project-error').textContent.includes('保存版本已变化'));
    await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#approve-whitebox').disabled);assert(await page.locator('#project-error').isVisible());assert.equal(controls.length,3);checks.stale_approval_error_survives_polling=true;
    await page.screenshot({path:path.join(out,'desktop_mock.png')});await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,'mobile_mock.png')});checks.mobile_no_horizontal_overflow=true;
    await page.locator('#scene-editor').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'mobile_editor_mock.png')});await page.setViewportSize({width:1440,height:1000});await page.locator('#scene-editor').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'desktop_editor_mock.png')});
    rejectApproval=false;await page.click('#approve-whitebox');await page.waitForFunction(()=>document.querySelector('#whitebox-review').hidden);
    assert.deepEqual(controls.at(-1),{project:'local_project',action:'approve_whitebox',revision:'mock_numeric_saved',head_seq:1});assert.equal(controls.length,4);checks.explicit_approval_uses_exact_saved_version=true;
    for(const intent of ['edit_selected','refine_selected']){await page.selectOption('#chat-intent',intent);assert(await page.locator('#send-message').isDisabled());assert((await page.locator('#chat-note').innerText()).includes('当前运行结束'));}await page.selectOption('#chat-intent','chat');checks.object_modes_wait_for_active_run_to_end=true;
    uncertainMessage=true;await page.fill('#chat-input','这是一条隔离的不确定响应测试。');await page.click('#send-message');await page.waitForFunction(()=>document.querySelector('#chat-note').textContent.includes('执行结果待核对'));
    const count=messages.length;await page.reload({waitUntil:'networkidle'});assert.equal(messages.length,count);assert.equal(controls.length,4);assert(await page.locator('#send-message').isDisabled());checks.uncertain_message_not_replayed_after_reload=true;
    await page.click('#new-project');await page.fill('#new-name','隔离上传');await page.locator('#video-file').setInputFiles({name:'test.mp4',mimeType:'video/mp4',buffer:Buffer.from([0,1,2,3,4,5])});await page.click('#create-project');
    await page.waitForFunction(()=>document.querySelector('#project-name').textContent==='隔离上传');assert.equal(uploads.length,1);assert.equal(uploads[0].name,'test.mp4');assert.deepEqual(uploads[0].bytes,[0,1,2,3,4,5]);assert.equal(creates.length,2);assert.equal(creates[1].source_path,'isolated/uploads/video.mp4');assert.equal(messages.length,count);assert(await page.locator('#scene-editor').isHidden());assert(await page.locator('#export-files').isHidden());assert(!(await page.locator('#chat-context').innerText()).includes('coffee_table'));checks.raw_video_upload_and_project_context_isolation=true;
    assert.equal(await page.locator('#chat-intent').inputValue(),'chat');for(const intent of ['edit_selected','refine_selected']){await page.selectOption('#chat-intent',intent);assert(await page.locator('#send-message').isDisabled());assert((await page.locator('#chat-note').innerText()).includes('选中一个对象'));}await page.selectOption('#chat-intent','chat');checks.object_modes_require_selection=true;
    for(const phase of ['failed','needs_auth']){Object.assign(projectRows[1],{phase,active_run:false,error:'隔离恢复前的错误提示'});await page.click('#refresh-state');await page.waitForFunction(phase=>document.querySelector('#project-phase').dataset.phase===phase,phase);assert(await page.locator('#resume-run').isVisible());assert((await page.locator('#project-error').innerText()).includes('隔离恢复前'));const before=controls.length;await page.click('#resume-run');await page.waitForFunction(()=>document.querySelector('#project-phase').dataset.phase==='running');assert.equal(controls.length,before+1);assert.equal(controls.at(-1).action,'resume');}checks.failed_and_needs_auth_expose_resume=true;projectRows[1].error=null;
    Object.assign(projectRows[1],{phase:'running',active_run:true});incompleteControl=true;await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#pause-run').hidden);await page.click('#pause-run');await page.waitForFunction(()=>document.querySelector('#project-error').textContent.includes('操作结果待核对'));assert(await page.evaluate(()=>!!sessionStorage.getItem('pi_workflow_uncertain_uploaded_project_pause')));const controlCount=controls.length;await page.reload({waitUntil:'networkidle'});assert.equal(controls.length,controlCount);checks.incomplete_control_receipt_remains_unresolved=true;
    incompleteCreate=true;await page.click('#new-project');await page.fill('#new-name','隔离缺失回执');await page.check('input[value="path"]');await page.fill('#source-path','E:\\isolated\\receipt.mp4');await page.click('#create-project');await page.waitForFunction(()=>document.querySelector('#create-error').textContent.includes('创建结果待核对'));assert(await page.locator('#create-project').isDisabled());const createCount=creates.length;await page.reload({waitUntil:'networkidle'});await page.click('#new-project');assert(await page.locator('#create-project').isDisabled());assert.equal(creates.length,createCount);checks.incomplete_create_receipt_not_replayed=true;
    await page.goto(`${url}?project_id=uploaded_project`,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.querySelector('#project-name').textContent==='隔离上传');assert.equal(new URL(page.url()).searchParams.get('project'),'uploaded_project');await page.goto(`${url}?project=local_project&project_id=uploaded_project`,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>document.querySelector('#project-name').textContent==='隔离客厅');assert.equal(new URL(page.url()).searchParams.get('project'),'local_project');checks.project_id_alias_and_project_precedence_select_correct_project=true;
    incompleteControl=false;projectRows[0].batch=null;Object.assign(projectRows[0],{phase:'needs_review',active_run:false,head:{...view.head}});
    view.snapshot.objects[1].asset.asset_version='gpt-direct-explicit-fixture';
    for(const [id,kind] of [['floor','floor'],['wall','wall'],['door','opening']])view.snapshot.objects.push({...structuredClone(obj),instance_id:id,kind});
    await page.reload({waitUntil:'networkidle'});await page.waitForFunction(()=>!document.querySelector('#start-refine-all').disabled);await page.click('#open-activity');await frame.locator('#pi-properties-toggle').click();
    assert((await page.locator('#start-refine-all').innerText()).includes('（4）'));assert((await page.locator('#batch-note').innerText()).includes('点击即采用'));checks.batch_includes_structures_skips_refined=true;
    const batchSavedX=await frame.locator('#position-x').inputValue();await frame.locator('#position-x').fill('2.75');await page.waitForFunction(()=>document.querySelector('#start-refine-all').disabled);assert((await page.locator('#batch-note').innerText()).includes('未保存'));await frame.locator('#position-x').fill(batchSavedX);await page.waitForFunction(()=>!document.querySelector('#start-refine-all').disabled);checks.batch_blocks_unsaved_layout=true;
    const batchBefore=controls.length;await page.click('#start-refine-all');await page.waitForFunction(()=>!document.querySelector('#batch-progress-area').hidden);assert.equal(controls.length,batchBefore+1);const batchRequest=controls.at(-1);assert.equal(batchRequest.action,'start_refine_all');assert(batchRequest.request_id);assert.equal(batchRequest.revision,view.head.revision);assert.equal(batchRequest.head_seq,view.head.head_seq);assert(await page.locator('#start-refine-all').isDisabled());assert((await page.locator('#batch-progress-label').innerText()).includes('1 / 4'));assert.equal(await page.locator('#batch-targets li').count(),4);checks.batch_one_click_exact_head_and_progress=true;
    assert.equal(batchRequest.parallelism,3);assert.equal(batchRequest.mode,'rolling');assert((await page.locator('#batch-progress-label').innerText()).includes('对象运行 2/3'));assert.equal(await page.locator('#batch-targets li[data-state="generating"]').count(),2);assert((await page.locator('#batch-targets li[data-state="package_ready"]').innerText()).includes('等待合并'));checks.parallel_workers_and_waiting_package_visible=true;
    Object.assign(projectRows[0],{phase:'idle',active_run:null});await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#project-phase').textContent==='后台精修中');assert(await page.locator('#pause-run').isVisible());assert(!(await page.locator('#pause-run').isDisabled()));assert.equal(await page.locator('#run-status').textContent(),'后台对象正在精修');assert(await page.locator('#resume-run').isHidden());checks.rolling_background_stays_visible_and_pausable_between_pi_turns=true;
    await page.selectOption('#batch-concurrency','4');await page.click('#apply-concurrency');await page.waitForFunction(()=>document.querySelector('#batch-progress-label').textContent.includes('请求 4'));assert.equal(controls.at(-1).action,'set_refinement_concurrency');assert.equal(controls.at(-1).parallelism,4);checks.rolling_concurrency_config=true;
    await page.locator('#batch-targets li[data-state="package_ready"] button').click();assert.equal(controls.at(-1).action,'pause_object');assert.equal(controls.at(-1).target_id,'door');checks.object_pause_targets_one_worker=true;
    Object.assign(projectRows[0],{phase:'paused',active_run:false});projectRows[0].batch.status='paused';await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#resume-run').hidden);assert(await page.locator('#start-refine-all').isDisabled());await page.click('#resume-run');assert.equal(controls.at(-1).action,'resume');assert.equal(controls.at(-1).parallelism,undefined);assert.equal(projectRows[0].batch.parallelism,4);checks.batch_resume_uses_existing_queue=true;
    projectRows[0].batch=null;Object.assign(projectRows[0],{phase:'idle',active_run:false});incompleteControl=true;await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#start-refine-all').disabled);await page.click('#start-refine-all');await page.waitForFunction(()=>document.querySelector('#project-error').textContent.includes('操作结果待核对'));const batchUncertainCount=controls.length;await page.reload({waitUntil:'networkidle'});assert.equal(controls.length,batchUncertainCount);assert(await page.locator('#start-refine-all').isDisabled());checks.batch_uncertain_start_not_replayed=true;
    await page.click('#toggle-chat');await page.fill('#chat-input','仅验证抽屉草稿，不发送。');
    const frameSource=await page.locator('#scene-editor').getAttribute('src'),beforeRequests=messages.length+controls.length+edits.length;
    const openWidth=(await page.locator('#scene-editor').boundingBox()).width;
    await page.keyboard.press('Escape');await page.waitForTimeout(250);assert.equal(await page.locator('#assistant-drawer').getAttribute('aria-hidden'),'true');assert((await page.locator('#scene-editor').boundingBox()).width>openWidth+250);
    await page.click('#toggle-chat');assert.equal(await page.inputValue('#chat-input'),'仅验证抽屉草稿，不发送。');assert.equal(await page.locator('#scene-editor').getAttribute('src'),frameSource);assert.equal(messages.length+controls.length+edits.length,beforeRequests);checks.drawer_preserves_draft_scene_and_has_no_write_side_effects=true;
    await page.locator('#tab-chat').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#tab-progress').getAttribute('aria-selected'),'true');await page.keyboard.press('ArrowRight');assert(await page.locator('.evidence-card').isVisible());checks.drawer_keyboard_tabs=true;
    await page.setViewportSize({width:390,height:844});await page.waitForTimeout(250);assert(await page.locator('#drawer-backdrop').isVisible());assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth&&document.documentElement.scrollHeight<=innerHeight+1));await page.keyboard.press('Escape');await page.click('#toggle-projects');assert(await page.locator('#new-project').isVisible());await page.click('#close-projects');checks.mobile_drawer_and_project_navigation=true;
    await page.setViewportSize({width:1440,height:1000});
    const currentProject=projectRows.find(p=>p.id===new URL(page.url()).searchParams.get('project'));
    currentProject.phase='running';currentProject.active_run='fixture-reconnect';
    currentProject.model_retry={status:'waiting',attempt:2,max_attempts:5,delay_ms:4000,at:new Date().toISOString()};
    await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#model-retry').hidden);
    assert((await page.locator('#model-retry').innerText()).includes('第 2/5 次重连'));assert(await page.locator('#pause-run').isVisible());
    currentProject.phase='idle';currentProject.active_run=null;currentProject.model_retry.status='recovered';
    await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#model-retry').hidden);checks.model_reconnect_status_and_recovery=true;
    delete currentProject.model_retry;currentProject.phase='running';currentProject.active_run='fixture-legacy-reconnect';
    events.push({seq:3,type:'auto_retry_start',at:new Date().toISOString(),data:{run_id:'fixture-legacy-reconnect',data:{attempt:1,maxAttempts:5,delayMs:2000}}});
    await page.click('#refresh-state');await page.waitForFunction(()=>!document.querySelector('#model-retry').hidden);
    assert((await page.locator('#model-retry').innerText()).includes('第 1/5 次重连'));
    events.push({seq:4,type:'auto_retry_end',at:new Date().toISOString(),data:{run_id:'fixture-legacy-reconnect',data:{success:true,attempt:1}}});
    await page.click('#refresh-state');await page.waitForFunction(()=>document.querySelector('#model-retry').hidden);
    checks.model_reconnect_compatible_with_running_host=true;currentProject.phase='idle';currentProject.active_run=null;
    await page.click('#toggle-chat');await page.fill('#chat-input','断线期间保留的草稿');
    const reconnectFrame=await page.locator('#scene-editor').getAttribute('src'),beforeReconnectWrites=messages.length+controls.length+edits.length;
    let offlineReads=0;
    const offlineRoute=async route=>{if(route.request().method()==='GET'){offlineReads++;await route.abort('failed');}else await route.continue();};
    await page.route('**/workflow/api/**',offlineRoute);await page.click('#refresh-state');
    await page.waitForFunction(()=>document.querySelector('#global-error').textContent.includes('自动重连'));
    await page.waitForFunction(()=>document.querySelector('#connection-text').textContent.includes('4 秒后重连'));
    assert(offlineReads>=4&&offlineReads<=6);
    await page.unroute('**/workflow/api/**',offlineRoute);await page.evaluate(()=>window.dispatchEvent(new Event('online')));
    await page.waitForFunction(()=>document.querySelector('#global-error').hidden&&document.querySelector('#connection-dot').classList.contains('online'));
    assert.equal(await page.inputValue('#chat-input'),'断线期间保留的草稿');
    assert.equal(await page.locator('#scene-editor').getAttribute('src'),reconnectFrame);
    assert.equal(messages.length+controls.length+edits.length,beforeReconnectWrites);checks.local_reconnect_backoff_preserves_draft_scene_and_never_replays_writes=true;
    const progressWrites=creates.length+messages.length+controls.length+edits.length;
    const probe={id:'progress_probe',scene_id:'pi_mock_scene',name:'进度条隔离验证',provider:'codex_subscription',phase:'running',active_run:'probe_run',active_tool:'inspect_video',has_scene:false,head:null,messages:[],frames:[]};
    projectRows.push(probe);await page.goto(url+'?project=progress_probe',{waitUntil:'networkidle'});
    const stage=async n=>page.waitForFunction(n=>document.querySelector('#workflow-progress').dataset.stage===String(n),n);
    const refresh=async()=>{await page.click('#refresh-state');await page.waitForTimeout(100);};
    await stage(1);assert(await page.locator('#workflow-progress').isVisible());
    probe.active_tool='segment_instances';await refresh();await stage(2);assert((await page.locator('#workflow-current-task').innerText()).includes('SAM'));
    probe.active_tool='build_whitebox';await refresh();await stage(3);
    Object.assign(probe,{active_tool:null,active_run:null,phase:'needs_review',has_scene:true,head:{...view.head}});await refresh();await stage(4);
    assert((await page.locator('#workflow-progress-state').innerText()).includes('等待你核对'));checks.workflow_stages_follow_real_tools_and_human_review=true;
    Object.assign(probe,{phase:'idle',acceptance:{acceptance_id:'fixture'},batch:{mode:'rolling',status:'running',parallelism:3,total:3,completed_count:0,targets:[{target_id:'floor',state:'generating',worker:{stage:'independent_review'}},{target_id:'sofa',state:'pending',depends_on:['floor']},{target_id:'door',state:'package_ready'}]}});
    await refresh();await stage(5);await page.keyboard.press('Escape');
    assert(await page.locator('#workflow-progress').isVisible());assert((await page.locator('#workflow-current-task').innerText()).includes('地板 · 独立视觉复核'));assert((await page.locator('#workflow-counts').innerText()).includes('等待支撑 1'));
    assert.equal(await page.locator('#workflow-object-progress').getAttribute('value'),'0');assert.equal(await page.locator('#workflow-object-progress').getAttribute('max'),'3');
    assert.equal(await page.locator('#workflow-steps [aria-current=step]').count(),1);checks.workflow_background_workers_and_saved_counts_visible_without_drawer=true;
    await page.click('#workflow-progress-details');assert.equal(await page.locator('#tab-progress').getAttribute('aria-selected'),'true');await page.keyboard.press('Escape');
    probe.active_tool='view_frames';probe.active_run='probe_run';probe.phase='running';await refresh();await stage(5);probe.active_tool=null;probe.active_run=null;probe.phase='idle';
    probe.batch.targets[0].state='saved';probe.batch.completed_count=1;await refresh();await page.waitForFunction(()=>document.querySelector('#workflow-object-progress').value===1);checks.workflow_counts_only_saved_objects_and_retains_stage_when_revisiting_frames=true;
    probe.phase='paused';probe.batch.status='paused';await refresh();await page.waitForFunction(()=>document.querySelector('#workflow-progress').dataset.state==='paused');
    assert.equal(await page.locator('#workflow-object-progress').getAttribute('value'),'1');
    probe.phase='failed';probe.batch.status='waiting';await refresh();await page.waitForFunction(()=>document.querySelector('#workflow-progress').dataset.state==='error');checks.workflow_pause_and_failure_keep_saved_progress=true;
    probe.phase='running';probe.active_run='retry_probe';probe.model_retry={status:'waiting',attempt:1,max_attempts:5,delay_ms:2000,at:new Date().toISOString()};await refresh();await page.waitForFunction(()=>document.querySelector('#workflow-progress-state').textContent==='模型重连中');delete probe.model_retry;
    probe.batch.targets.forEach(r=>r.state='saved');probe.batch.completed_count=3;probe.batch.status='running';probe.active_tool='export_scene';await refresh();await stage(6);
    assert.notEqual(await page.locator('#workflow-progress').getAttribute('data-state'),'complete');
    Object.assign(probe,{phase:'complete',active_run:null,active_tool:null,export:{head:{revision:'older',head_seq:0},artifact_refs:[{role:'scene_glb',uri:'mock:fixture.glb'},{role:'scene_blend',uri:'mock:scene.blend'}]}});probe.batch.status='complete';await refresh();
    assert.notEqual(await page.locator('#workflow-progress').getAttribute('data-state'),'complete');
    probe.export.head={...probe.head};await refresh();await page.waitForFunction(()=>document.querySelector('#workflow-progress').dataset.state==='complete');
    assert.equal(await page.locator('#workflow-steps [data-state=done]').count(),6);checks.workflow_delivery_requires_final_phase_and_current_export=true;
    await page.setViewportSize({width:390,height:844});assert(await page.locator('#workflow-progress').isVisible());assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth&&document.documentElement.scrollHeight<=innerHeight+1));
    assert((await page.locator('#scene-editor').boundingBox()).height>200);checks.workflow_progress_mobile_keeps_scene_visible=true;
    assert.equal(creates.length+messages.length+controls.length+edits.length,progressWrites);checks.workflow_progress_and_details_never_write_scene_or_start_work=true;
    assert.deepEqual(errors,[]);checks.no_page_errors=true;
    const result={passed:true,scope:'Ephemeral mock HTTP and in-memory fixtures; real editor_v1 UI only. No real model, project, scene edit or human approval.',checks,errors};fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;server.close();});
