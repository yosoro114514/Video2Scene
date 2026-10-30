const $=id=>document.getElementById(id);
const API='/workflow/api';
const PHASE={imported:'已导入视频',idle:'等待下一条指令',running:'正在运行',needs_review:'等待采用白模',paused:'已暂停',complete:'已完成',failed:'执行失败',needs_auth:'需要登录 Codex'};
const EVENT_NAMES={'run.started':'运行开始','run.completed':'运行完成','run.failed':'运行失败','tool.started':'工具开始','tool.completed':'工具完成','tool.failed':'工具失败','scene.head_changed':'场景已保存','whitebox.ready':'白模就绪','whitebox.approved':'布局已采用','job.updated':'作业进度','message':'消息',project_created:'项目已创建',tool_started:'工具开始',tool_completed:'工具完成',tool_failed:'工具失败',tool_receipt:'工具结果已交给 Pi',tool_execution_start:'Pi 开始调用工具',tool_execution_end:'工具调用返回',user_message:'收到指令',human_pause:'请求暂停',human_accepted:'布局已采用',ready:'Pi 已就绪',prompt_started:'开始处理指令',turn_start:'开始推理',assistant_message:'Pi 回复',settled:'本轮运行结束',run_error:'运行失败',aborted:'已暂停',process_exit:'Pi 进程结束',fatal:'运行错误'};
const TOOL_NAMES={inspect_video:'查看视频关键帧',view_frames:'补充查看关键帧',record_prior:'记录场景先验',estimate_geometry:'ViPE 相机与深度',segment_instances:'SAM 实例分割',measure:'融合场景测量',build_whitebox:'构建白模',request_human_review:'等待人工核对',inspect_scene:'核对保存场景',refine_object:'精修对象',validate_candidate:'校验候选',commit_candidate:'保存候选',export_scene:'导出场景',finish_workflow:'核对闭环结果',workflow_status:'核对工作流状态'};
Object.assign(TOOL_NAMES,{start_refinement_jobs:'安排并发精修',refinement_jobs_status:'核对对象进度',refine_objects_parallel:'并发精修对象',stage_refinement:'合并对象到场景',save_refinement:'合并、验证并保存对象',repair_whitebox_supports:'校正白模支撑'});
const WORKFLOW_STAGES=['视频观察','空间分析','构建白模','人工核对','对象精修','导出交付'];
const TOOL_STAGES={inspect_video:0,view_frames:0,record_prior:0,estimate_geometry:1,segment_instances:1,measure:1,build_whitebox:2,repair_whitebox_supports:2,request_human_review:3,refine_object:4,refine_objects_parallel:4,start_refinement_jobs:4,refinement_jobs_status:4,stage_refinement:4,save_refinement:4,export_scene:5,finish_workflow:5};
const OBJECT_NAMES={floor:'地板',sofa:'沙发',armchair:'扶手椅',coffee_table:'茶几',cabinet:'柜子',tv_stand:'电视柜',floor_lamp:'落地灯',rug:'地毯',door:'门',window:'窗户',wall:'墙体',wall_sofa:'沙发侧墙',wall_cabinet:'柜子侧墙',wall_door:'门侧墙',wall_door_left:'门侧左墙',wall_door_header:'门楣',wall_window:'窗侧墙',wall_window_sofa_side:'窗边沙发侧墙',wall_window_door_side:'窗边门侧墙',wall_window_header:'窗楣'};
const OBJECT_STATES={registered:'排队准备',preparing:'准备证据',generating:'建模与复核',package_ready:'等待合并',staging:'合并到场景',candidate_ready:'等待验证',validating:'验证模型',validated:'等待保存',saving:'保存中',needs_review:'待核对',failed:'需要处理',paused:'已暂停'};
const WORKER_STAGES={waiting_resource:'等待可用资源',resource_acquired:'建模与复核',authoring:'建模中',validating_geometry:'检查模型几何',independent_review:'独立视觉复核',candidate_built:'整理复核结果'};
Object.assign(EVENT_NAMES,{assistant_attempt_error:'本次模型请求未完成',auto_retry_start:'模型连接中断，准备重试',auto_retry_end:'模型重连结果',summarization_retry_scheduled:'上下文压缩连接重试',summarization_retry_attempt_start:'正在重试上下文压缩',summarization_retry_finished:'上下文压缩重试结束'});
let projects=[],project=null,projectId=new URLSearchParams(location.search).get('project')||new URLSearchParams(location.search).get('project_id')||'',generation=0;
let fresh=false,polling=false,timer=null,editorTimer=null,loadedScene='',messageBusy=false,controlBusy=false,createBusy=false;
let connectionFailures=0;
let capabilities={},concurrencyEdited=false;
const reconnectDelay=()=>Math.min(30000,1000*2**Math.min(connectionFailures,5));
let actionError='',globalError='',messageSignature='',frameSignature='',candidateSignature='',lastList=0,refreshingEditor=false;
const eventCache=new Map();
const viewerDisplay=new WeakMap();
let uploaded=null;
let drawerOpen=false,drawerTab='chat',drawerProject='',projectsOpen=false;
const drawerTabs=['chat','progress','evidence'];
function syncDrawer(){
  $('scene-layout').classList.toggle('drawer-open',drawerOpen);
  $('assistant-drawer').inert=!drawerOpen;
  $('assistant-drawer').setAttribute('aria-hidden',String(!drawerOpen));
  $('drawer-backdrop').hidden=!drawerOpen||innerWidth>900;
  $('toggle-chat').setAttribute('aria-expanded',String(drawerOpen&&drawerTab==='chat'));
  $('open-activity').setAttribute('aria-expanded',String(drawerOpen&&drawerTab==='progress'));
  for(const name of drawerTabs){
    const selected=name===drawerTab;
    $(`tab-${name}`).setAttribute('aria-selected',String(selected));$(`tab-${name}`).tabIndex=selected?0:-1;
    $(`panel-${name}`).hidden=!selected;
  }
}
function setDrawer(open,tab=drawerTab,{focus=false,persist=true}={}){
  const hadFocus=$('assistant-drawer').contains(document.activeElement);
  drawerOpen=open;drawerTab=drawerTabs.includes(tab)?tab:'chat';syncDrawer();
  if(open&&drawerTab==='chat'){$('chat-unread').hidden=true;requestAnimationFrame(()=>{$('messages').scrollTop=$('messages').scrollHeight;});}
  if(persist&&projectId)try{localStorage.setItem(`pi_workbench_drawer_${projectId}`,JSON.stringify({open,tab:drawerTab}));}catch{}
  if(open&&focus)$(drawerTab==='chat'?'chat-input':`tab-${drawerTab}`).focus({preventScroll:true});
  if(!open&&(focus||hadFocus))$('toggle-chat').focus({preventScroll:true});
}
function restoreDrawer(){
  if(drawerProject===projectId)return;
  drawerProject=projectId;$('chat-unread').hidden=true;
  let saved;try{saved=JSON.parse(localStorage.getItem(`pi_workbench_drawer_${projectId}`));}catch{}
  setDrawer(saved?.open??!project.has_scene,saved?.tab||'chat',{persist:false});
}
function setProjects(open){
  projectsOpen=open;document.body.classList.toggle('projects-open',open);
  $('projects-backdrop').hidden=!open;$('toggle-projects').setAttribute('aria-expanded',String(open));
}
function closePanels(event){
  if(event.key!=='Escape'||$('project-dialog').open)return;
  if(projectsOpen){setProjects(false);$('toggle-projects').focus();event.preventDefault();}
  else if(drawerOpen){setDrawer(false,drawerTab,{focus:true});event.preventDefault();}
}
function text(id,value){$(id).textContent=String(value??'');}
function setError(id,value){text(id,value||'');$(id).hidden=!value;}
function uuid(){return crypto.randomUUID();}
function phaseLabel(value){return PHASE[value]||value||'等待状态';}
function sameHead(a,b){return !!a&&!!b&&a.revision===b.revision&&a.head_seq===b.head_seq;}
function piActive(p=project){if(!p||p.phase==='paused')return false;return p.phase==='running'||p.run?.active===true||p.active_run===true||typeof p.active_run==='string'||p.active_run?.active===true;}
function backgroundActive(p=project){return !!p&&p.batch?.mode==='rolling'&&p.batch.status==='running'&&!['paused','failed','needs_auth'].includes(p.phase);}
function active(p=project){return piActive(p)||backgroundActive(p);}
function projectLabel(p){return p.phase==='idle'&&backgroundActive(p)?'后台精修中':phaseLabel(p.phase);}
function runLabel(){return piActive()?'Pi 正在处理':backgroundActive()?'后台对象正在精修':phaseLabel(project?.phase);}
function uncertainKey(id,kind){return `pi_workflow_uncertain_${id||'create'}_${kind}`;}
function uncertain(id,kind){try{return JSON.parse(sessionStorage.getItem(uncertainKey(id,kind))||'null');}catch{return null;}}
function remember(id,kind,value){try{if(value)sessionStorage.setItem(uncertainKey(id,kind),JSON.stringify(value));else sessionStorage.removeItem(uncertainKey(id,kind));}catch{}}
function errorText(value,fallback){return typeof value?.error==='string'?value.error:value?.error?.message||value?.message||fallback;}
async function request(url,body){
  let response,value;
  const abort=body===undefined?new AbortController():null;
  const timeout=abort?setTimeout(()=>abort.abort(),10000):null;
  try{
  try{response=await fetch(url,body===undefined?{cache:'no-store',signal:abort.signal}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
  catch(error){error.uncertain=body!==undefined;throw error;}
  try{value=await response.json();}catch{const error=new Error(`服务响应不完整 (${response.status})，请查看实际运行状态。`);error.uncertain=body!==undefined;throw error;}
  if(!value||typeof value!=='object'||Array.isArray(value)){const error=new Error('服务响应格式不完整，请核对实际运行状态。');error.uncertain=body!==undefined;throw error;}
  if(!response.ok||value.status==='error'||(value.error&&!value.id&&!value.project&&!value.phase)){
    const error=new Error(errorText(value,`请求失败 (${response.status})`));error.httpStatus=response.status;
    error.uncertain=body!==undefined&&(response.status>=500||response.status===408||value.error?.details?.outcome==='uncertain');throw error;
  }
  return value;
  }finally{clearTimeout(timeout);}
}
function endpoint(suffix='',identity=projectId){return `${API}/projects/${encodeURIComponent(identity)}${suffix}`;}
function unwrap(value){return value.project||value;}
function requireProject(value,identity){
  const result=unwrap(value);
  if(!result?.id||(identity&&result.id!==identity)){const error=new Error('服务返回的项目信息不完整或身份不匹配，请核对实际状态。');error.uncertain=true;throw error;}
  return result;
}
function applyProject(value,identity=projectId){
  const next=unwrap(value);
  if(!next?.id||next.id!==identity)throw new Error('服务返回的项目身份不匹配。');
  project=next;fresh=true;
  const index=projects.findIndex(row=>row.id===next.id);if(index>=0)projects[index]=next;else projects.unshift(next);
  renderProject();
}
function renderProjects(){
  $('project-list').replaceChildren();
  if(!projects.length){const p=document.createElement('p');p.className='empty';p.textContent='还没有项目。导入一个视频开始。';$('project-list').append(p);return;}
  for(const item of projects){
    const button=document.createElement('button');button.type='button';button.className='project-item'+(item.id===projectId?' active':'');button.dataset.projectId=item.id;
    button.setAttribute('aria-current',item.id===projectId?'page':'false');
    const title=document.createElement('strong'),small=document.createElement('small');title.textContent=item.name||item.id;small.textContent=projectLabel(item);
    button.append(title,small);button.addEventListener('click',()=>selectProject(item.id));$('project-list').append(button);
  }
}
async function loadProjects(){
  const value=await request(`${API}/projects`);if(!Array.isArray(value.projects))throw new Error('项目列表格式无效。');
  projects=value.projects;lastList=Date.now();renderProjects();
  if(!projectId&&projects.length)await selectProject(projects[0].id);
}
async function selectProject(identity){
  if(identity===projectId&&project)return;
  setProjects(false);
  generation++;projectId=identity;project=null;fresh=false;actionError='';loadedScene='';messageSignature='';frameSignature='';candidateSignature='';
  concurrencyEdited=false;
  $('chat-intent').value='chat';
  $('scene-editor').src='about:blank';$('scene-editor').hidden=true;$('editor-empty').hidden=false;$('open-editor').hidden=true;
  $('messages').replaceChildren();$('events').replaceChildren();$('frames').replaceChildren();$('candidates').replaceChildren();
  $('export-files').hidden=true;$('export-links').replaceChildren();
  text('selected-object','尚未选择对象');text('editor-save-status','等待场景');text('project-name','读取项目…');
  $('welcome').hidden=true;$('project-workspace').hidden=false;
  const url=new URL(location.href);url.searchParams.set('project',identity);history.replaceState(null,'',url);renderProjects();
  const current=generation;
  try{const value=await request(endpoint());if(current!==generation)return;applyProject(value,identity);setConnection(true);}
  catch(error){if(current!==generation)return;setConnection(false);globalError=error.message;setError('global-error',globalError);}
  updateContext();schedulePoll(0);
}
function setConnection(connected){$('connection-dot').classList.toggle('online',connected);text('connection-text',connected?'已连接本地服务':`连接中断 · ${reconnectDelay()/1000} 秒后重连`);}
function renderMessages(){
  const messages=Array.isArray(project.messages)?project.messages:[];
  const signature=JSON.stringify(messages);if(signature===messageSignature)return;
  if(messageSignature&&(!drawerOpen||drawerTab!=='chat')&&messages.at(-1)?.role==='assistant')$('chat-unread').hidden=false;
  messageSignature=signature;
  const box=$('messages'),nearBottom=box.scrollHeight-box.scrollTop-box.clientHeight<80;box.replaceChildren();
  if(!messages.length){const p=document.createElement('p');p.className='empty';p.textContent='描述想构建的空间，或指定需要核对的对象。';box.append(p);return;}
  for(const item of messages){
    const row=document.createElement('article');row.className='message '+(item.role==='user'?'user':item.role==='assistant'?'assistant':'system');
    const role=document.createElement('span');role.className='role';role.textContent=item.role==='user'?'你':item.role==='assistant'?'Pi':'运行记录';
    const content=document.createElement('div');content.className='content';content.textContent=item.text||'';row.append(role,content);box.append(row);
  }
  if(nearBottom||messages.length<=2)box.scrollTop=box.scrollHeight;
}
function mergeEvents(identity,value){
  const current=eventCache.get(identity)||{rows:[],cursor:0};
  const bySeq=new Map(current.rows.map(row=>[row.seq,row]));
  for(const row of value.events||[])if(Number.isInteger(row.seq))bySeq.set(row.seq,row);
  // Keep milestones when the visible event list is trimmed. No elapsed-time estimate.
  for(const row of value.events||[]){
    if(['tool_started','tool_completed','tool_failed'].includes(row.type)){
      const name=Object.keys(TOOL_STAGES).find(name=>new RegExp(`\\b${name}\\b`).test(row.text||row.data?.text||''));
      let step=TOOL_STAGES[name];
      if(row.type==='tool_completed'&&name==='record_prior')step=1;
      if(row.type==='tool_completed'&&name==='measure')step=2;
      if(Number.isInteger(step))current.workflowStage=Math.max(current.workflowStage||0,step);
    }
  }
  current.rows=[...bySeq.values()].sort((a,b)=>a.seq-b.seq).slice(-100);
  if(Number.isInteger(value.cursor))current.cursor=Math.max(current.cursor,value.cursor);
  eventCache.set(identity,current);
}
function renderEvents(){
  const rows=eventCache.get(projectId)?.rows||[];$('events').replaceChildren();
  if(!rows.length){const li=document.createElement('li');li.className='empty';li.textContent='尚无工具调用。';$('events').append(li);}
  for(const row of [...rows].reverse().slice(0,60)){
    const li=document.createElement('li');if(/fail|error/.test(row.type||''))li.className='error-event';
    const time=document.createElement('time');time.textContent=row.at||'';
    const label=document.createElement('strong');label.textContent=EVENT_NAMES[row.type]||row.type||'状态更新';
    const data=row.data?.data||row.data;
    const retryDetail=row.type==='auto_retry_start'?`第 ${data?.attempt}/${data?.maxAttempts} 次 · ${Math.ceil((data?.delayMs||0)/1000)} 秒后重试`:row.type==='auto_retry_end'?(data?.success?'连接已恢复':data?.finalError==='Retry cancelled'?'已取消重试':'重试已停止'):'';
    const detail=document.createElement('span');detail.textContent=row.text||retryDetail||row.data?.message||row.data?.stage||row.data?.status||'';
    li.append(time,label,detail);$('events').append(li);
  }
  text('event-count',rows.length?`已读取 ${rows.length} 条事件`:'实际运行记录');renderCandidates();renderModelRetry();renderWorkflowProgress();
}
function renderWorkflowProgress(){
  if(!project)return;
  const p=project,b=p.batch,rows=b?.targets||[],tool=typeof p.active_tool==='string'?p.active_tool:p.active_tool?.name||p.active_tool?.tool;
  const saved=rows.length?rows.filter(r=>r.state==='saved').length:Math.max(0,b?.completed_count||0),total=b?.total||rows.length;
  const hasBatch=!!b&&total>0,allSaved=hasBatch&&saved===total;
  const currentExport=sameHead(p.export?.head,p.head)&&['scene_blend','scene_glb'].every(role=>p.export?.artifact_refs?.some(r=>r.role===role));
  const done=p.phase==='complete'&&!active(p)&&currentExport&&(!hasBatch||allSaved);
  let step=p.acceptance||hasBatch?4:p.has_scene?3:eventCache.get(projectId)?.workflowStage||0;
  if(allSaved)step=5;
  // Evidence can be revisited without resetting the overall workflow.
  if(Number.isInteger(TOOL_STAGES[tool]))step=Math.max(step,TOOL_STAGES[tool]);
  if(['refine_object','refine_objects_parallel','stage_refinement','save_refinement'].includes(tool))step=4;
  if(p.phase==='needs_review'&&!p.acceptance&&!hasBatch)step=3;
  if(done)step=5;
  const live=rows.filter(r=>['registered','preparing','generating','staging','validating','saving'].includes(r.state));
  const ready=rows.filter(r=>['package_ready','candidate_ready','validated'].includes(r.state));
  const issues=rows.filter(r=>['failed','needs_review'].includes(r.state));
  const blocked=rows.filter(r=>r.state==='pending'&&(r.depends_on||[]).some(id=>rows.some(d=>d.target_id===id&&d.state!=='saved')));
  const pausedRows=rows.filter(r=>r.state==='paused');
  let state='等待继续',tone='waiting';
  if(done){state='已完成';tone='complete';}
  else if(active(p)||live.length||b?.status==='running'){state='进行中';tone='running';}
  if(!done&&step===3&&!p.acceptance){state='等待你核对';tone='waiting';}
  if(['failed','needs_auth'].includes(p.phase)||['failed','stale'].includes(b?.status)||issues.length&&!live.length&&!ready.length){state=p.phase==='needs_auth'?'需要登录':'需要处理';tone='error';}
  if(p.phase==='paused'||b?.status==='paused'){state='已暂停';tone='paused';}
  if(!$('model-retry').hidden){state='模型重连中';tone='waiting';}
  if(!fresh){state='连接中断 · 显示上次状态';tone='offline';}
  const descriptions=['查看视频关键帧，识别房间形状和主要对象。','结合分割、相机和深度，分析空间布局。','生成房间结构与家具白模。','请核对布局，修改后保存并采用，再进入精修。','按采用的布局建模、独立复核并逐件保存。','核对最终保存版本，导出 Blender 与 GLB 文件。'];
  let task=TOOL_NAMES[tool]||descriptions[step];
  if(step===4&&hasBatch){
    const focus=[...live,...ready].slice(0,3);
    task=focus.length?focus.map(r=>`${r.label||OBJECT_NAMES[r.target_id]||r.target_id} · ${r.state==='generating'&&WORKER_STAGES[r.worker?.stage]||OBJECT_STATES[r.state]||r.state}`).join('；'):issues.length?'有对象需要核对，可打开详情查看原因。':blocked.length?'等待支撑对象保存后继续。':pausedRows.length?'剩余对象已暂停，可打开详情继续。':'正在安排下一件对象。';
    if(live.length+ready.length>3)task+='；更多见详情';
  }
  if(allSaved&&step===5&&!done)task='对象已全部保存，正在等待或执行整屋导出与最终核对。';
  if(done)task='当前保存版本已完成核对，Blender 与 GLB 文件可在详情中下载。';
  if(p.phase==='paused'||b?.status==='paused')task='已保留当前进度；继续后从保存状态恢复。'+(live.length?' 已启动的任务正在收尾。':'');
  if(p.phase==='failed'||p.phase==='needs_auth')task=p.phase==='needs_auth'?'登录 Codex 后继续，已有结果保留。':'当前步骤未完成，请查看错误详情；已有保存结果保留。';
  if(b?.status==='stale')task='布局已发生修改，请核对当前保存版本后重新安排精修。';
  const panel=$('workflow-progress');panel.dataset.state=tone;panel.dataset.stage=String(step+1);
  text('workflow-step-count',done?'6 / 6 步':`第 ${step+1} / 6 步`);text('workflow-progress-title',WORKFLOW_STAGES[step]);text('workflow-progress-state',state);
  [...$('workflow-steps').children].forEach((li,index)=>{
    li.dataset.state=done||index<step?'done':index===step?'current':'pending';
    if(!done&&index===step)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current');
    li.setAttribute('aria-label',`${WORKFLOW_STAGES[index]}：${done||index<step?'已完成':index===step?state:'尚未开始'}`);
  });
  if($('workflow-current-task').textContent!==task)text('workflow-current-task',task);
  $('workflow-counts').hidden=!hasBatch;$('workflow-object-progress').hidden=!hasBatch;
  if(hasBatch){
    const counts=[`已保存 ${saved} / ${total} 个对象`,live.length?`进行中 ${live.length}`:'',ready.length?`待保存 ${ready.length}`:'',blocked.length?`等待支撑 ${blocked.length}`:'',issues.length?`待处理 ${issues.length}`:'',pausedRows.length?`暂停 ${pausedRows.length}`:''].filter(Boolean).join(' · ');
    if($('workflow-counts').textContent!==counts)text('workflow-counts',counts);
    $('workflow-object-progress').max=total;$('workflow-object-progress').value=saved;$('workflow-object-progress').setAttribute('aria-valuetext',`已保存 ${saved} / ${total} 个对象`);
  }
}
function renderModelRetry(){
  // Also understand bridge events from an already running host. Updating the
  // browser and bridge does not require stopping a project to update Python.
  let retry=project?.model_retry;
  if(!retry&&active()){
    const row=[...(eventCache.get(projectId)?.rows||[])].reverse().find(e=>
      ['auto_retry_start','auto_retry_end'].includes(e.type)&&e.data?.run_id===project.active_run);
    if(row){const data=row.data.data||{};retry={status:row.type==='auto_retry_start'?'waiting':data.success?'recovered':'exhausted',
      attempt:data.attempt,max_attempts:data.maxAttempts,delay_ms:data.delayMs,at:row.at};}
  }
  const reconnecting=active()&&['waiting','retrying'].includes(retry?.status);
  $('model-retry').hidden=!reconnecting;
  if(reconnecting){
    const seconds=Math.max(0,Math.ceil(((Date.parse(retry.at)||Date.now())+(retry.delay_ms||0)-Date.now())/1000));
    text('model-retry',`模型连接暂时中断 · 第 ${retry.attempt}/${retry.max_attempts} 次重连${retry.status==='waiting'&&seconds>0?`，约 ${seconds} 秒后重试`:'，正在连接'}。已完成的工作已保留，可点击暂停停止重试。`);
    text('run-status','模型正在重连');
  }else if(project)text('run-status',runLabel());
}
function artifactURL(relative){
  if(typeof relative!=='string'||!relative||relative.startsWith('/')||/^[A-Za-z]:|^[a-z]+:\/\//i.test(relative)||relative.split(/[\\/]/).includes('..'))return null;
  return `/workflow/artifacts?project_id=${encodeURIComponent(projectId)}&path=${encodeURIComponent(relative)}`;
}
function renderFrames(){
  const frames=Array.isArray(project.frames)?project.frames:[],signature=JSON.stringify([projectId,frames]);if(signature===frameSignature)return;frameSignature=signature;
  $('frames').replaceChildren();
  for(const frame of frames){
    const url=artifactURL(frame.path);if(!url)continue;
    const figure=document.createElement('figure');figure.className='frame';const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener';
    const img=document.createElement('img');img.src=url;img.loading='lazy';img.alt=frame.label||frame.source_frame_id||'项目关键帧';
    const caption=document.createElement('figcaption');caption.textContent=frame.label||frame.source_frame_id||frame.path;
    img.addEventListener('error',()=>{caption.textContent=`${frame.label||frame.source_frame_id||'关键帧'} · 预览暂不可用`;});
    link.append(img);figure.append(link,caption);$('frames').append(figure);
  }
  if(!$('frames').children.length){const p=document.createElement('p');p.className='empty';p.textContent='读取视频后会显示关键帧。';$('frames').append(p);}
}
function renderCandidates(){
  if(!project)return;
  const supplied=Array.isArray(project.candidates)?project.candidates:(eventCache.get(projectId)?.rows||[]).flatMap(row=>{
    const result=row.data?.result;
    if(result?.candidate)return [{...result.candidate,status:result.status==='complete'&&result.head?'已保存':result.saved===false?'等待校验与保存':result.status||''}];
    return /candidate/.test(row.type||'')?[row.data?.candidate||row.data||{}]:[];
  });
  const candidates=[...new Map(supplied.map((item,index)=>[item.candidate_id||item.instance_id||index,item])).values()];
  const signature=JSON.stringify(candidates);if(signature===candidateSignature)return;candidateSignature=signature;$('candidates').replaceChildren();
  for(const item of candidates.slice(-16)){
    const div=document.createElement('div');div.className='candidate';const name=document.createElement('strong');name.textContent=item.label||item.target_id||item.instance_id||item.candidate_id||'场景候选';
    div.append(name);const summary=item.summary||item.state||item.status;if(summary)div.append(document.createTextNode(` · ${summary}`));$('candidates').append(div);
  }
  if(!candidates.length){const p=document.createElement('p');p.className='empty';p.textContent='几何与对象候选会在生成后显示。';$('candidates').append(p);}
}
function renderExports(){
  const exported=project?.export,refs=Array.isArray(exported?.artifact_refs)?exported.artifact_refs.filter(ref=>['scene_blend','scene_glb'].includes(ref.role)&&typeof ref.uri==='string'&&ref.uri):[];
  $('export-files').hidden=refs.length===0;$('export-links').replaceChildren();if(!refs.length)return;
  const current=sameHead(exported.head,project.head);$('export-files').dataset.current=String(current);
  text('export-version',current?'最新版下载':'旧版导出');
  const head=exported.head,version=head?`head ${head.head_seq} · ${head.revision}`:'未记录导出版本';
  text('export-note',current?`对应当前保存版本：${version}`:`${version}。这份导出与当前保存版本不一致。`);
  for(const role of ['scene_blend','scene_glb']){
    const ref=refs.find(item=>item.role===role);if(!ref)continue;
    const link=document.createElement('a');link.href=`/api/v1/artifacts?uri=${encodeURIComponent(ref.uri)}`;link.className='download-link';
    const extension=role==='scene_blend'?'blend':'glb';link.download=`${project.scene_id||project.id}.${extension}`;
    link.textContent=`${current?'下载':'下载旧版'} ${role==='scene_blend'?'Blender 场景 (.blend)':'通用模型 (.glb)'}`;$('export-links').append(link);
  }
}
function renderProject(){
  if(!project)return;
  restoreDrawer();
  $('welcome').hidden=true;$('project-workspace').hidden=false;
  text('project-name',project.name||project.id);text('project-source',project.source_path||project.source?.path||'');
  $('project-name').title=project.name||project.id;$('project-source').title=project.source_path||project.source?.path||'';
  text('project-phase',projectLabel(project));$('project-phase').dataset.phase=project.phase==='idle'&&backgroundActive()?'running':project.phase;
  text('provider-label',project.provider==='codex_subscription'?'Codex 订阅':project.provider||'Codex 订阅');text('model-label',project.model||'模型由服务配置');
  text('saved-head',project.head?`保存版本 · head ${project.head.head_seq} · ${project.head.revision}`:'尚未建立场景');
  text('run-status',runLabel());
  $('pause-run').hidden=!active();$('resume-run').hidden=!['paused','failed','needs_auth'].includes(project.phase)&&!['paused','failed','waiting'].includes(project.batch?.status);
  $('pause-run').disabled=controlBusy||!fresh||!!uncertain(projectId,'pause');$('resume-run').disabled=controlBusy||!fresh||!!uncertain(projectId,'resume');
  $('resume-run').title=['failed','needs_auth'].includes(project.phase)?'修复当前错误或登录状态后，继续会重新读取保存状态。':'继续当前项目';
  $('whitebox-review').hidden=project.phase!=='needs_review'||!!project.batch&&project.batch.status!=='stale';
  const serverError=project.error?(typeof project.error==='string'?project.error:project.error.message||'运行出现错误，请查看进度记录。'):'';
  setError('project-error',[actionError,serverError].filter(Boolean).join(' · '));
  renderModelRetry();
  const work=project.active_tool,job=project.job;
  $('active-work').hidden=!work&&!job;
  const toolName=typeof work==='string'?work:work?.name||work?.tool;
  text('active-tool',TOOL_NAMES[toolName]||toolName||'作业');text('active-job',job?.status||job?.state||work?.status||'');
  const progress=job?.progress;$('job-progress').hidden=typeof progress!=='number';if(typeof progress==='number')$('job-progress').value=Math.max(0,Math.min(1,progress));
  renderMessages();renderFrames();renderCandidates();renderExports();renderProjects();
  if(project.has_scene&&project.scene_id&&loadedScene!==project.scene_id){loadedScene=project.scene_id;const url=`/?scene_id=${encodeURIComponent(project.scene_id)}`;$('scene-editor').src=url;$('scene-editor').hidden=false;$('editor-empty').hidden=true;$('open-editor').href=url;$('open-editor').hidden=false;}
  if(!project.has_scene){$('scene-editor').hidden=true;$('editor-empty').hidden=false;$('open-editor').hidden=true;}
  if(Array.isArray(project.events))mergeEvents(projectId,{events:project.events});renderEvents();updateContext();
}
function editorContext(){
  try{const frame=$('scene-editor'),editor=frame.contentWindow?.vipeEditor,doc=frame.contentDocument;
    if(!project?.has_scene||!editor?.state?.view||editor.state.sceneId!==project.scene_id)return null;
    const view=editor.state.view,obj=view.snapshot.objects.find(item=>item.instance_id===editor.state.selectedId);return {editor,doc,view,obj};
  }catch{return null;}
}
function dirty(context){
  if(!context?.obj)return false;const {doc,obj}=context;
  const category=doc.getElementById('category'),support=doc.getElementById('support');if(!category||!support)return true;
  if(category.value.trim()!==obj.category||(support.value||null)!==obj.support_id)return true;
  for(const [prefix,values] of [['position',obj.transform.translation_m],['scale',obj.transform.scale]])for(const [index,axis] of ['x','y','z'].entries()){
    const input=doc.getElementById(`${prefix}-${axis}`);if(!input||input.value.trim()===''||!input.validity.valid||!Number.isFinite(Number(input.value))||Math.abs(Number(input.value)-values[index])>Math.max(1e-9,Math.abs(values[index])*1e-10))return true;
  }return false;
}
function configureEditor(){
  try{
    const doc=$('scene-editor').contentDocument;if(!doc?.head)return;
    if(!doc.getElementById('pi-wrapper-style')){
      const style=doc.createElement('style');style.id='pi-wrapper-style';style.textContent=`
        html,body{height:100%;overflow:hidden}header{display:none!important}
        main{display:block!important;max-width:none!important;padding:12px!important;height:100%;position:relative;margin:0!important}
        .workspace{height:100%;display:flex;flex-direction:column;min-height:0}
        .toolbar{flex:none;flex-direction:row!important;align-items:center!important;margin:0 0 10px!important;gap:8px}
        .toolbar>div:first-child{display:none}.view-actions{width:100%;justify-content:flex-start!important;gap:7px;align-items:center}
        .view-actions button{padding:6px 9px;font-size:11px}.view-actions .switch{font-size:10px;margin:0 0 0 auto}
        #viewer{flex:1;height:auto!important;min-height:0!important;border-radius:9px 9px 0 0}
        .viewer-footer{flex:none;padding:8px 11px;gap:8px;font-size:10px}.viewer-hint{left:10px;bottom:10px;font-size:10px;padding:5px 8px}
        main>aside{display:none;position:absolute;z-index:5;right:12px;top:57px;bottom:12px;width:min(300px,calc(100% - 24px));overflow:auto;background:#fff;border:1px solid #dce5de;border-radius:10px;box-shadow:0 8px 35px #173e2d25;padding:14px;overscroll-behavior:contain}
        body.pi-properties-open main>aside{display:block}
        main>aside .card{padding:0 0 14px;border:0;border-bottom:1px solid #dce5de;border-radius:0;margin:0 0 16px}
        #objects{max-height:155px;overflow:auto}.object-row{font-size:11px;padding:8px}
        #refine,#language-form{display:none!important}.evidence-panel{margin:0;padding:0;border:0}
        .evidence-content{grid-template-columns:1fr;gap:8px}.evidence-info p{font-size:11px}.evidence-panel .section-heading{flex-wrap:wrap}
        .evidence-panel select{max-width:100%}.viewer-footer span:nth-child(2){margin-left:auto}
        @media(max-width:500px){main{padding:8px!important}.view-actions{gap:5px}.view-actions button{font-size:10px;padding:5px 6px}.view-actions .switch{font-size:9px;margin-left:0}.viewer-footer span:nth-child(2){display:none}.viewer-hint{max-width:calc(100% - 20px)}main>aside{right:8px;bottom:8px;top:75px}}
      `;doc.head.append(style);
      const language=doc.getElementById('language-form');if(language?.closest('.card'))language.closest('.card').hidden=true;
      const evidence=doc.querySelector('.evidence-panel'),properties=doc.querySelector('main>aside');if(evidence&&properties)properties.append(evidence);
      doc.addEventListener('keydown',event=>{
        if(event.key==='Escape'&&doc.body.classList.contains('pi-properties-open')){doc.body.classList.remove('pi-properties-open');doc.getElementById('pi-properties-toggle')?.setAttribute('aria-expanded','false');doc.getElementById('pi-properties-toggle')?.focus();event.preventDefault();}
        else closePanels(event);
      });
    }
    const reset=doc.getElementById('reset-camera');
    if(reset&&!doc.getElementById('pi-top-view')){const button=doc.createElement('button');button.id='pi-top-view';button.type='button';button.className='quiet';button.textContent='俯视布局';button.title='从上方查看当前布局；重置视角可返回透视';button.addEventListener('click',()=>topView(editorContext()?.editor.viewer));reset.after(button);}
    if(reset&&!doc.getElementById('pi-properties-toggle')){
      const button=doc.createElement('button');button.id='pi-properties-toggle';button.type='button';button.className='quiet';button.textContent='对象与属性';button.setAttribute('aria-expanded','false');button.setAttribute('aria-controls','pi-object-properties');
      const properties=doc.querySelector('main>aside');if(properties)properties.id='pi-object-properties';
      button.addEventListener('click',()=>{const open=doc.body.classList.toggle('pi-properties-open');button.setAttribute('aria-expanded',String(open));});
      doc.getElementById('pi-top-view').after(button);
    }
  }catch{}
}
function topView(viewer){
  if(!viewer?.camera||!viewer?.controls||!viewer.roots?.size)return;
  const {camera,controls}=viewer,damping=controls.enableDamping;
  controls.enableDamping=false;controls.update();
  const target=controls.target,distance=Math.max(camera.position.distanceTo(target),.25);
  const floor=editorContext()?.view.snapshot.objects.find(obj=>obj.kind==='floor');
  const root=viewer.roots.get(floor?.instance_id||'floor'),direction=camera.position.clone().set(0,0,1);
  if(root)direction.applyQuaternion(root.getWorldQuaternion(root.quaternion.clone()));
  direction.y=0;if(direction.lengthSq()<1e-12)direction.set(0,0,1);direction.normalize();
  // A small offset avoids the orbit-control pole while keeping the floor axes
  // aligned with the screen. This changes only the camera, never scene records.
  camera.position.set(target.x+direction.x*distance*.001,target.y+distance,target.z+direction.z*distance*.001);
  camera.lookAt(target);controls.update();controls.enableDamping=damping;viewer.renderer.render(viewer.scene,camera);
}
function updateWhiteboxDisplay(context){
  const viewer=context.editor.viewer,snapshot=context.view.snapshot;
  if(!viewer||viewer.pendingAssets||!viewer.roots?.size)return;
  const whitebox=snapshot.objects.some(obj=>obj.asset?.asset_version?.startsWith('whitebox-'));
  if(!whitebox)return;
  let state=viewerDisplay.get(viewer);if(!state){state={autoTopApplied:false};viewerDisplay.set(viewer,state);}
  if(state.generation!==viewer.generation||state.selectedId!==context.editor.state.selectedId){viewer.addWhiteboxEdges(snapshot);state.generation=viewer.generation;state.selectedId=context.editor.state.selectedId;}
  if(!state.autoTopApplied){topView(viewer);state.autoTopApplied=true;}
}
function approvalReason(context){
  if(!project||project.phase!=='needs_review')return '白模准备好后，可以采用当前保存布局。';
  if(active())return '白模已经保存，正在等待 Pi 结束本轮工作后开放采用。';
  if(controlBusy)return '正在提交操作，请稍候。';
  if(uncertain(projectId,'approve_whitebox'))return '上一条采用请求结果待核对；页面没有重复发送。请查看运行记录。';
  if(!fresh)return '项目状态尚未同步。';if(!context)return '等待当前项目的场景载入。';
  if(!sameHead(project.head,context.view.head))return '项目状态与编辑器版本尚未同步，请稍候。';
  if(context.editor.pending?.size)return '场景保存或撤销仍在进行，请等待完成。';
  if(dirty(context))return '存在尚未保存的属性修改，请先在编辑器点击“保存修改”。';
  if(!context.view.snapshot.objects.length)return '场景没有可采用的对象。';return '';
}
function messageReason(context){
  if(!project||!fresh)return '等待当前项目状态同步。';
  if(messageBusy)return '正在提交指令，请稍候。';
  if(uncertain(projectId,'message'))return '上一条指令的执行结果待核对，页面不会自动重发。';
  if(context?.editor.pending?.size)return '等待场景操作完成后再发送。';
  if(dirty(context))return '属性修改尚未保存。先保存，再发送带场景上下文的指令。';
  if($('chat-intent').value!=='chat'){
    if(active())return '等待当前运行结束后，再发送对象修改或精修指令。';
    if(!context?.obj)return '请先在三维编辑器中选中一个对象。';
    if(!sameHead(project.head,context.view.head))return '当前编辑器与项目保存版本尚未同步，请稍候。';
  }
  return '';
}
function batchReason(context){
  if(!project?.has_scene||!fresh||!context)return '等待保存场景载入。';
  if(controlBusy||messageBusy)return '正在提交操作，请稍候。';
  if(uncertain(projectId,'start_refine_all'))return '上次启动结果待核对，页面不会重复提交。';
  if(active())return '当前任务正在运行。';
  if(project.batch&&!['complete','stale'].includes(project.batch.status))return '已有精修批次；暂停或失败后请使用上方“继续”。';
  if(context.editor.pending?.size||dirty(context))return '存在未保存修改，请先保存当前布局。';
  if(!sameHead(project.head,context.view.head))return '编辑器与保存版本尚未同步。';
  if(!remainingTargets(context).length)return '全部对象已精修，无需重复生成。';
  return '';
}
function remainingTargets(context){return context?.view.snapshot.objects.filter(o=>['object','support','wall','floor','opening','ceiling'].includes(o.kind)&&o.asset?.asset_version?.startsWith('whitebox-'))||[];}
function renderBatch(context){
  $('batch-refinement').hidden=!project?.has_scene;
  const batch=project?.batch,reason=batchReason(context),targets=remainingTargets(context);
  $('batch-config').hidden=!capabilities.rolling_refinement;
  const adjustable=batch?.mode==='rolling'&&!['complete','stale'].includes(batch.status);
  if(!concurrencyEdited)$('batch-concurrency').value=String(adjustable?batch.parallelism:3);
  $('apply-concurrency').hidden=!adjustable;$('apply-concurrency').disabled=controlBusy||!fresh||!concurrencyEdited;
  $('start-refine-all').disabled=!!reason;
  text('start-refine-all',`一键精修全部剩余对象${targets.length?`（${targets.length}）`:''}`);
  text('batch-note',reason||`点击即采用当前 head ${context.view.head.head_seq} 的已保存布局，精修 ${targets.length} 个对象；固定布局与门洞，颜色纹理允许适度差异。可随时暂停。`);
  $('batch-progress-area').hidden=!batch;if(!batch)return;
  const label={running:'进行中',paused:'已暂停',failed:'需要处理',waiting:'等待继续',stale:'布局已修改，请重新开始',complete:'已完成'}[batch.status]||batch.status;
  const running=batch.targets.filter(r=>['registered','preparing','generating'].includes(r.state)).length;
  const ready=batch.targets.filter(r=>['package_ready','candidate_ready','validated'].includes(r.state)).length;
  const mode=batch.mode==='rolling'?`持续并发 · 对象运行 ${running}/${batch.effective_concurrency??batch.parallelism}（请求 ${batch.parallelism}） · 待保存 ${ready}`:batch.parallelism===2?'双对象并发 · 串行保存':'逐对象处理';
  text('batch-progress-label',`${label} · ${mode} · 已保存 ${batch.completed_count} / ${batch.total} · 跳过 ${batch.skipped_ids?.length||0} 个已精修对象`);
  $('batch-progress').max=batch.total||1;$('batch-progress').value=batch.completed_count||0;
  $('batch-targets').replaceChildren();
  const labels={pending:'排队／等待依赖',registered:'已登记',preparing:'准备证据',needs_review:'待核对 · 不自动重做',paused:'已暂停',generating:'建模与视觉复核中',package_ready:'已复核 · 等待合并',staging:'合并到最新场景',candidate_ready:'待验证',validating:'验证中',validated:'待保存',saving:'保存中',saved:'已保存',failed:'需要处理'};
  const priority={generating:0,staging:0,validating:0,saving:0,package_ready:1,candidate_ready:1,validated:1,failed:2,paused:3,pending:4,saved:5};
  for(const row of [...(batch.targets||[])].sort((a,b)=>(priority[a.state]??4)-(priority[b.state]??4))){
    const li=document.createElement('li'),name=document.createElement('span'),status=document.createElement('strong');
    const workerLabel={waiting_resource:'等待可用资源',resource_acquired:'执行中',authoring:'建模中',validating_geometry:'检查模型几何',independent_review:'独立视觉复核',candidate_built:'整理已复核资产'}[row.worker?.stage];
    name.textContent=row.target_id;status.textContent=row.pause_requested&&row.state==='generating'?'正在复核，完成后暂停':row.state==='generating'&&workerLabel?workerLabel:labels[row.state]||row.state;li.dataset.state=row.state;
    li.append(name,status);if(row.error)li.title=row.error;
    if((batch.mode==='rolling'||batch.parallelism===2)&&['pending','registered','generating','paused','failed','package_ready'].includes(row.state)){
      const button=document.createElement('button');button.type='button';button.className='quiet';
      const resume=['paused','failed'].includes(row.state);button.textContent=resume?'继续此项':row.state==='generating'?'复核后暂停':'暂停此项';
      button.disabled=controlBusy||!fresh||(resume&&batch.mode!=='rolling'&&!!project.active_run)||(!resume&&row.pause_requested);
      button.addEventListener('click',()=>control(resume?'resume_object':'pause_object',{target_id:row.target_id}));li.append(button);
    }
    $('batch-targets').append(li);
  }
  const batchError=['running','complete'].includes(batch.status)?null:batch.error;
  setError('batch-error',batchError||batch.status!=='complete'&&batch.throttle_reason&&`连接异常，暂缓派发并降低并发：${batch.throttle_reason}`||'');
}
function updateContext(){
  configureEditor();const context=editorContext();
  if(context){const {editor,doc,view,obj}=context;
    updateWhiteboxDisplay(context);
    text('selected-object',obj?`${obj.category} · ${obj.instance_id}`:'尚未选择对象');
    text('editor-save-status',editor.pending?.size?'保存处理中':dirty(context)?'修改未保存':`已保存 · head ${view.head.head_seq}`);
    text('chat-context',`${obj?obj.instance_id:'未选对象'} · head ${view.head.head_seq}`);
    if(fresh&&project.head&&!sameHead(project.head,view.head)&&!dirty(context)&&!editor.pending?.size&&!refreshingEditor&&project.head.head_seq>view.head.head_seq){refreshingEditor=true;Promise.resolve(editor.refresh()).catch(()=>{}).finally(()=>{refreshingEditor=false;});}
  }else{text('chat-context',project?.head?`当前保存 head ${project.head.head_seq} · 未选对象`:'当前没有场景选择');}
  renderBatch(context);
  renderWorkflowProgress();
  const reason=approvalReason(context);text('approval-note',reason||`将采用 head ${context.view.head.head_seq}（${context.view.head.revision}），点击后精修一个推荐对象。`);$('approve-whitebox').disabled=!!reason;
  const messageBlock=messageReason(context),intent=$('chat-intent').value;$('send-message').disabled=!!messageBlock;
  text('chat-note',messageBlock||(intent==='chat'?'讨论模式用于说明与核对问题。Ctrl / ⌘ + Enter 发送。':`${intent==='edit_selected'?'将允许 Pi 修改':'将允许 Pi 精修'}当前选中对象，指令绑定已保存版本。Ctrl / ⌘ + Enter 发送。`));
}
async function poll(){
  if(polling)return;polling=true;const identity=projectId,current=generation;
  try{
    if(identity){
      const cursor=eventCache.get(identity)?.cursor||0;
      const values=await Promise.allSettled([request(endpoint('',identity)),request(endpoint(`/events?after=${cursor}`,identity))]);
      if(current!==generation)return;
      if(values[0].status==='rejected')throw values[0].reason;applyProject(values[0].value,identity);
      if(values[1].status==='fulfilled'){mergeEvents(identity,values[1].value);renderEvents();}
    }
    if(Date.now()-lastList>5000)await loadProjects();
    connectionFailures=0;setConnection(true);globalError='';setError('global-error','');
  }catch(error){if(current===generation){fresh=false;connectionFailures++;setConnection(false);globalError=`连接本地服务失败，${reconnectDelay()/1000} 秒后自动重连。场景和输入草稿已保留。`;setError('global-error',globalError);}}
  finally{polling=false;updateContext();schedulePoll(connectionFailures?reconnectDelay():document.hidden?2500:1000);}
}
function schedulePoll(delay=1000){clearTimeout(timer);timer=setTimeout(poll,delay);}
async function sendMessage(event){
  event?.preventDefault();updateContext();if($('send-message').disabled)return;
  const message=$('chat-input').value.trim();if(!message)return;
  const identity=projectId,current=generation,context=editorContext();
  const payload={request_id:uuid(),message,intent:$('chat-intent').value,selected_ids:context?.obj?[context.obj.instance_id]:[]};
  const head=context?.view.head||project.head;if(head){payload.base_revision=head.revision;payload.base_head_seq=head.head_seq;}
  messageBusy=true;actionError='';remember(identity,'message',payload);updateContext();
  try{const value=await request(endpoint('/messages',identity),payload);if(value.accepted!==true){const error=new Error('服务没有返回指令接收结果，请核对实际运行状态。');error.uncertain=true;throw error;}if(value.project)requireProject(value,identity);remember(identity,'message',null);if(current!==generation)return;$('chat-input').value='';if(value.project)applyProject(value,identity);else schedulePoll(0);}
  catch(error){if(!error.uncertain)remember(identity,'message',null);if(current===generation){actionError=error.message+(error.uncertain?' 执行结果待核对，未自动重发。':'');setError('project-error',actionError);}}
  finally{messageBusy=false;if(current===generation)updateContext();schedulePoll(0);}
}
async function control(action,extra={}){
  if(!project||!fresh||controlBusy||uncertain(projectId,action))return;
  const context=editorContext();if(action==='approve_whitebox'&&approvalReason(context))return;
  if(action==='start_refine_all'&&batchReason(context))return;
  const identity=projectId,current=generation,payload={action,...extra};
  if(action==='start_refine_all'){payload.parallelism=capabilities.rolling_refinement?Number($('batch-concurrency').value):2;if(capabilities.rolling_refinement)payload.mode='rolling';}
  if(action==='start_refine_all')payload.request_id=uuid();
  if(action==='approve_whitebox'||action==='start_refine_all'){payload.revision=context.view.head.revision;payload.head_seq=context.view.head.head_seq;}
  controlBusy=true;actionError='';remember(identity,action,payload);updateContext();renderProject();
  try{const value=await request(endpoint('/control',identity),payload);requireProject(value,identity);remember(identity,action,null);if(current===generation)applyProject(value,identity);}
  catch(error){if(!error.uncertain)remember(identity,action,null);if(current===generation){actionError=error.message+(error.uncertain?' 操作结果待核对，未自动重发。':'');setError('project-error',actionError);}}
  finally{controlBusy=false;if(current===generation){renderProject();updateContext();}schedulePoll(0);}
}
function uploadVideo(file){
  return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST',`${API}/upload?name=${encodeURIComponent(file.name)}`);xhr.setRequestHeader('Content-Type',file.type||'application/octet-stream');
    xhr.upload.onprogress=event=>{if(event.lengthComputable){$('upload-progress').value=event.loaded/event.total;text('upload-text',`正在上传 ${Math.round(event.loaded/event.total*100)}% · ${(event.loaded/1024/1024).toFixed(1)} MB`);}};
    xhr.onload=()=>{let value;try{value=JSON.parse(xhr.responseText);}catch{return reject(new Error('上传响应不完整，请核对本地服务。'));}if(xhr.status<200||xhr.status>=300||!value.source_path)return reject(new Error(errorText(value,`上传失败 (${xhr.status})`)));resolve(value.source_path);};
    xhr.onerror=()=>reject(new Error('上传连接中断，请核对本地服务后重新选择视频。'));xhr.send(file);
  });
}
function sourceMode(){return document.querySelector('input[name="source-mode"]:checked').value;}
function updateSourceMode(){const upload=sourceMode()==='upload';$('file-field').hidden=!upload;$('path-field').hidden=upload;}
async function createProject(event){
  event.preventDefault();if(createBusy||uncertain('','create'))return;
  const name=$('new-name').value.trim();if(!name)return;
  createBusy=true;$('create-project').disabled=true;$('close-dialog').disabled=true;setError('create-error','');
  try{
    let source_path=$('source-path').value.trim();
    if(sourceMode()==='upload'){
      const file=$('video-file').files[0];if(!file)throw new Error('请先选择一个视频文件。');if(file.size>2*1024**3)throw new Error('视频文件超过 2 GB，请选择较小文件或使用已准备的本地路径。');
      const key=[file.name,file.size,file.lastModified].join(':');
      if(uploaded?.key===key)source_path=uploaded.path;else{$('upload-status').hidden=false;$('upload-progress').value=0;text('upload-text','正在上传视频…');source_path=await uploadVideo(file);uploaded={key,path:source_path};text('upload-text','视频已上传，正在创建项目。');}
    }
    if(!source_path)throw new Error('请填写本地视频路径。');
    const payload={request_id:uuid(),name,source_path};const instruction=$('new-instruction').value.trim();if(instruction)payload.instruction=instruction;
    remember('','create',payload);
    let created;try{created=requireProject(await request(`${API}/projects`,payload));remember('','create',null);}catch(error){if(!error.uncertain)remember('','create',null);throw error;}
    $('project-dialog').close();$('project-form').reset();uploaded=null;$('upload-status').hidden=true;updateSourceMode();
    projects.unshift(created);await selectProject(created.id);
  }catch(error){setError('create-error',error.message+(error.uncertain?' 创建结果待核对，请查看项目列表；页面不会重复创建。':''));}
  finally{createBusy=false;$('create-project').disabled=!!uncertain('','create');$('close-dialog').disabled=false;}
}
function openCreate(){setError('create-error',uncertain('','create')?'上一条创建请求结果待核对，请刷新项目列表查看；未重复创建。':'');$('create-project').disabled=!!uncertain('','create');$('project-dialog').showModal();$('new-name').focus();}
$('toggle-chat').addEventListener('click',()=>setDrawer(!(drawerOpen&&drawerTab==='chat'),'chat',{focus:true}));
$('open-activity').addEventListener('click',()=>setDrawer(true,'progress',{focus:true}));
$('close-drawer').addEventListener('click',()=>setDrawer(false,drawerTab,{focus:true}));
$('drawer-backdrop').addEventListener('click',()=>setDrawer(false,drawerTab,{focus:true}));
for(const name of drawerTabs){
  $(`tab-${name}`).addEventListener('click',()=>setDrawer(true,name));
  $(`tab-${name}`).addEventListener('keydown',event=>{
    const index=drawerTabs.indexOf(name);let next;
    if(event.key==='ArrowRight')next=(index+1)%drawerTabs.length;
    if(event.key==='ArrowLeft')next=(index+drawerTabs.length-1)%drawerTabs.length;
    if(event.key==='Home')next=0;if(event.key==='End')next=drawerTabs.length-1;
    if(next!==undefined){event.preventDefault();setDrawer(true,drawerTabs[next]);$(`tab-${drawerTabs[next]}`).focus();}
  });
}
$('toggle-projects').addEventListener('click',()=>setProjects(!projectsOpen));
$('close-projects').addEventListener('click',()=>setProjects(false));$('projects-backdrop').addEventListener('click',()=>setProjects(false));
document.addEventListener('keydown',closePanels);window.addEventListener('resize',()=>{syncDrawer();if(innerWidth>900)setProjects(false);});
$('new-project').addEventListener('click',openCreate);$('welcome-create').addEventListener('click',openCreate);$('close-dialog').addEventListener('click',()=>$('project-dialog').close());
$('project-dialog').addEventListener('cancel',event=>{if(createBusy)event.preventDefault();});$('project-form').addEventListener('submit',createProject);
document.querySelectorAll('input[name="source-mode"]').forEach(input=>input.addEventListener('change',updateSourceMode));
$('chat-form').addEventListener('submit',sendMessage);$('chat-input').addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();sendMessage();}});
$('chat-intent').addEventListener('change',updateContext);
$('pause-run').addEventListener('click',()=>control('pause'));$('resume-run').addEventListener('click',()=>control('resume'));$('approve-whitebox').addEventListener('click',()=>control('approve_whitebox'));
$('start-refine-all').addEventListener('click',()=>control('start_refine_all'));
$('batch-concurrency').addEventListener('change',()=>{concurrencyEdited=true;updateContext();});
$('apply-concurrency').addEventListener('click',async()=>{await control('set_refinement_concurrency',{parallelism:Number($('batch-concurrency').value)});concurrencyEdited=false;updateContext();});
$('refresh-state').addEventListener('click',()=>poll());$('refresh-projects').addEventListener('click',async()=>{try{await loadProjects();}catch(error){setError('global-error',error.message);}});
$('workflow-progress-details').addEventListener('click',()=>setDrawer(true,'progress',{focus:true}));
$('scene-editor').addEventListener('load',()=>{configureEditor();updateContext();});document.addEventListener('visibilitychange',()=>{if(!document.hidden)schedulePoll(0);});
window.addEventListener('online',()=>schedulePoll(0));
window.addEventListener('pagehide',()=>{clearTimeout(timer);clearInterval(editorTimer);});editorTimer=setInterval(updateContext,400);
try{capabilities=await request(`${API}/capabilities`);}catch{} // An older running host keeps its existing mode until upgraded.
try{await loadProjects();if(projectId&&!project)await selectProject(projectId);setConnection(true);}catch(error){setConnection(false);setError('global-error',error.message);}schedulePoll(0);
