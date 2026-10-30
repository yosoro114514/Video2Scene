// Actual browser, mock API. This is UI verification, not G3 model/Blender proof.
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),staticRoot=path.join(root,'video2scene/editor_v1');
const out=path.resolve(process.argv[2]||path.join(root,'outputs/parallel_dev/T05/attempt_001/browser_mock_001'));
if(fs.existsSync(out))throw Error('Use a new evidence output directory');fs.mkdirSync(out,{recursive:true});
const examples=JSON.parse(fs.readFileSync(path.join(root,'docs/development/contracts/examples.json'),'utf8'));
const value=name=>structuredClone(examples.positive.find(x=>x.name===name).value);
let view=value('model_SceneView');const sceneId=view.snapshot.scene_id;const jobs=new Map();
for(const obj of view.snapshot.objects){obj.asset.artifact.uri=`artifact://${sceneId}/assets/${obj.instance_id}.blend`;obj.asset.artifact.media_type='application/x-blender'}
const initial=structuredClone(view),commands=new Map(),events=[],requests=[],sse=new Set();let seq=0,forceConflict=false;
const evidence=value('model_EvidenceBundle');evidence.scene_revision=view.head.revision;
const image=fs.readFileSync(path.join(root,'outputs/uw_whitebox_loop_20261028/source_frames/000.png'));
const send=(res,data,status=200)=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(data))};
const error=(code,message)=>({contract_version:'video2scene/1.0',error:{code,message,retryable:false,details:{}}});
function emit(type,data){const event={contract_version:'video2scene/1.0',event_id:`evt_${++seq}`,scene_id:sceneId,seq,type,occurred_at:new Date().toISOString(),request_id:null,data};events.push(event);for(const res of sse)res.write(`id: ${seq}\ndata: ${JSON.stringify(event)}\n\n`);return event}
const server=http.createServer(async(req,res)=>{try{
 const url=new URL(req.url,'http://localhost');
 if(url.pathname.endsWith('/events')){res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache','Connection':'keep-alive'});res.write(': connected\n\n');sse.add(res);req.on('close',()=>sse.delete(res));return}
 if(url.pathname==='/api/v1/artifacts'){res.writeHead(200,{'Content-Type':'image/png'});res.end(image);return}
 if(req.method==='POST'){
  let body='';for await(const chunk of req)body+=chunk;const data=JSON.parse(body);requests.push(data);
  if(url.pathname==='/api/v1/tools/execute'){
   let result=data.tool==='inspect_scene'?structuredClone(view):data.tool==='select_evidence'?{...structuredClone(evidence),scene_revision:data.params.revision}:null;
   if(data.tool==='job_control'){const job=jobs.get(data.params.job_id);if(data.params.action==='cancel')job.state='cancel_requested';result=structuredClone(job)}
   send(res,{contract_version:'video2scene/1.0',request_id:data.request_id,scene_id:sceneId,tool:data.tool,status:'ok',result,error:null,warnings:[]});return;
  }
  if(url.pathname==='/api/v1/commands'){
   if(forceConflict){forceConflict=false;view.head={revision:'external',head_seq:view.head.head_seq+1};view.snapshot.revision='external';send(res,error('STALE_REVISION','另一会话已更新场景'),409);return}
   assert.equal(data.base_head_seq,view.head.head_seq);assert.equal(data.base_revision,view.head.revision);assert(!('decision_ref' in data));
   const receipt={command_id:data.command_id,operation_id:`op_${data.command_id}`,state:'accepted'};
   const record={receipt,state:'succeeded',request_ids:[],message:'测试 API 已保存修改（mock）',error:null};
   if(data.kind==='patch')for(const edit of data.edits){const object=view.snapshot.objects.find(o=>o.instance_id===edit.instance_id);if(edit.field.startsWith('transform.'))object.transform[edit.field.split('.')[1]]=edit.value;else object[edit.field]=edit.value}
   else if(data.text==='undo')view.snapshot=structuredClone(initial.snapshot);
   else if(data.text==='redo')view.snapshot.objects[1].transform.scale[1]=1.1;
   else if(data.text.includes('那里')){record.state='needs_clarification';record.message='请明确目标位置。'}
   if(record.state==='succeeded'){const previous=view.head.revision;view.head={revision:`mock_${view.head.head_seq+1}`,head_seq:view.head.head_seq+1};view.snapshot.revision=view.head.revision;emit('scene.head_changed',{revision:view.head.revision,previous_revision:previous,head_seq:view.head.head_seq,event_id:`head_${seq+1}`})}
   commands.set(data.command_id,record);send(res,receipt,202);return;
  }
 }
 if(url.pathname.startsWith('/api/v1/commands/')){const record=commands.get(url.pathname.split('/').pop());return record?send(res,record):send(res,error('NOT_FOUND','unknown'),404)}
 const relative=url.pathname==='/'?'index.html':decodeURIComponent(url.pathname).replace(/^\//,'');const file=path.resolve(staticRoot,relative);if(!file.startsWith(staticRoot+path.sep)&&file!==staticRoot)return send(res,error('NOT_FOUND','not found'),404);if(!fs.existsSync(file)||!fs.statSync(file).isFile())return send(res,error('NOT_FOUND','not found'),404);
 res.writeHead(200,{'Content-Type':file.endsWith('.mjs')||file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html'});res.end(fs.readFileSync(file));
 }catch(error){send(res,{error:{message:error.stack}},500)}});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const base=`http://127.0.0.1:${server.address().port}/?scene_id=${sceneId}`;
 const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady===true);await page.waitForFunction(()=>document.getElementById('evidence-image').naturalWidth>0);
  assert.equal(await page.locator('.object-row').count(),2);await page.locator('[data-id="t001"]').click();await page.fill('#scale-y','1.1');await page.click('#save');await page.waitForFunction(()=>window.vipeEditor.state.view.head.head_seq===1);
  assert.equal(view.snapshot.objects[1].transform.scale[1],1.1);await page.reload({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.viewerReady);assert.equal(Number(await page.inputValue('#scale-y')),1.1);
  await page.click('#undo');await page.waitForFunction(()=>window.vipeEditor.state.view.head.head_seq===2);assert.equal(Number(await page.inputValue('#scale-y')),1);
  await page.click('#redo');await page.waitForFunction(()=>window.vipeEditor.state.view.head.head_seq===3);assert.equal(Number(await page.inputValue('#scale-y')),1.1);
  await page.fill('#language','把它移到那里');await page.click('#send');await page.waitForFunction(()=>document.getElementById('activity').textContent.includes('请明确目标位置'));assert.equal(view.head.head_seq,3);
  forceConflict=true;await page.fill('#scale-y','1.2');await page.click('#save');await page.waitForFunction(()=>window.vipeEditor.state.view.head.revision==='external');assert.equal(Number(await page.inputValue('#scale-y')),1.1);
  const frozenHead=structuredClone(view.head);const job={job_id:'job_fixture',scene_id:sceneId,base_revision:view.head.revision,base_head_seq:view.head.head_seq,request_id:'job_request_fixture',state:'running',progress:.35,result:null,error:null};jobs.set(job.job_id,job);emit('job.updated',structuredClone(job));await page.waitForFunction(()=>document.getElementById('jobs').textContent.includes('running'));assert.equal(await page.locator('#save').isEnabled(),true);await page.locator('#jobs button').click();await page.waitForFunction(()=>document.getElementById('jobs').textContent.includes('cancel_requested'));assert.deepEqual(view.head,frozenHead);assert.equal(await page.locator('#jobs button').isDisabled(),true);job.state='cancelled';emit('job.updated',structuredClone(job));await page.waitForFunction(()=>document.getElementById('jobs').textContent.includes('cancelled'));assert.deepEqual(view.head,frozenHead);
  emit('agent.clarification',{operation_id:'other_op',session_id:'other_session',command_id:'other_command',question:'STALE_OTHER_SESSION_QUESTION'});await page.waitForFunction(expected=>window.vipeEditor.state.lastEventSeq>=expected,seq);assert(!(await page.locator('body').innerText()).includes('STALE_OTHER_SESSION_QUESTION'));
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile horizontal overflow');
  await page.locator('#show-assets').uncheck();assert.equal(await page.evaluate(()=>window.vipeEditor.state.selectedId),'t001');
  assert.deepEqual(errors,[]);const report={passed:true,mode:'actual_browser_mock_API',checks:['property_command_and_reload','undo_redo','clarification_keeps_head','conflict_refresh_without_replay','selection_stable','job_cancel_waits_confirmation','job_events_keep_head','editing_available_during_job','other_session_clarification_not_displayed','desktop_mobile_no_overflow','no_page_errors'],request_count:requests.length,not_run:['actual API integration','Blender build','model invocation','G3 closed loop'],page_errors:errors};fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});fs.writeFileSync(path.join(out,'failure.json'),JSON.stringify({error:error.message,requests,page_errors:errors,body:await page.locator('body').innerText()},null,2));throw error}finally{await browser.close();for(const res of sse)res.end();server.close()}
})().catch(error=>{console.error(error);server.closeAllConnections();server.close();process.exitCode=1});
