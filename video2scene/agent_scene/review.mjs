// Same-origin wrapper around editor_v1. No scene mutations occur until an explicit
// user click; saving edits remains the existing editor's responsibility.
const $=id=>document.getElementById(id);
const phases={preparing:['准备输入',0],prior:['GPT 查看关键帧',0],segmentation:['SAM 分割与核查',1],geometry:['相机、深度与白模拟合',1],whitebox:['构建白模',1],needs_user:['等待你核查白模',2],refining:['GPT 正在精修对象',3],validating:['检查精模与约束',3],complete:['本次闭环已完成',4],failed:['运行需要处理',-1]};
let sceneId=new URLSearchParams(location.search).get('scene_id')||'';
let status=null,statusFresh=false,statusPending=false,confirmPending=false,lastSelectionToken='',lastSource='',loadedScene='',statusTimer=null,editorTimer=null,actionError='',batchRefreshing=false;
const savedKey=()=>`vipe_agent_confirmation_${sceneId}`;
let uncertain=null;
function recoverPending(){try{uncertain=JSON.parse(sessionStorage.getItem(savedKey())||'null');}catch{uncertain=null;}}
function rememberPending(value){uncertain=value;try{if(value)sessionStorage.setItem(savedKey(),JSON.stringify(value));else sessionStorage.removeItem(savedKey());}catch{}}
if(sceneId)recoverPending();

function text(id,value){$(id).textContent=String(value??'—');}
function sameHead(a,b){return !!a&&!!b&&a.revision===b.revision&&a.head_seq===b.head_seq;}
function refinableObject(obj){return obj&&(obj.kind==='object'||(obj.kind==='support'&&obj.category.startsWith('furniture/')));}
function currentWorkflow(){return status?.room_finish||status?.batch;}
function batchLocksEditing(){return !!currentWorkflow()&&currentWorkflow().editing_locked!==false;}
function exactConfirmation(value,payload){return !!value&&value.revision===payload.revision&&value.head_seq===payload.head_seq&&value.target_id===payload.target_id;}
function errorMessage(value,fallback){return value?.error?.message||value?.message||fallback;}
function editorContext(){
  try{
    const frame=$('editor-frame'),win=frame.contentWindow,doc=frame.contentDocument,editor=win?.vipeEditor;
    if(!editor?.state?.view||editor.state.sceneId!==sceneId)return null;
    const view=editor.state.view,obj=view.snapshot.objects.find(item=>item.instance_id===editor.state.selectedId);
    return {editor,doc,view,obj};
  }catch{return null;}
}
function formIsDirty(context){
  const {doc,obj}=context;if(!obj)return false;
  const category=doc.getElementById('category'),support=doc.getElementById('support');
  if(!category||!support)return true;
  if(category.value.trim()!==obj.category||(support.value||null)!==obj.support_id)return true;
  for(const [prefix,original] of [['position',obj.transform.translation_m],['scale',obj.transform.scale]]){
    for(const [i,axis] of ['x','y','z'].entries()){
      const input=doc.getElementById(`${prefix}-${axis}`);
      if(!input||input.value.trim()===''||!input.validity.valid)return true;
      const value=Number(input.value);
      if(!Number.isFinite(value)||Math.abs(value-original[i])>Math.max(1e-9,Math.abs(original[i])*1e-10))return true;
    }
  }
  return false;
}
function dimensions(obj){
  try{return ['x','y','z'].map((_,i)=>((obj.local_bounds_m.max[i]-obj.local_bounds_m.min[i])*obj.transform.scale[i]).toFixed(3)).join(' × ');}catch{return '未知';}
}
function setupEditor(){
  if(!sceneId||loadedScene===sceneId)return;
  const ready=status?.ready_for_review===true||!!status?.head||['needs_user','refining','validating','complete'].includes(status?.phase);
  if(!ready)return;
  loadedScene=sceneId;
  const url=`/?scene_id=${encodeURIComponent(sceneId)}`;
  $('editor-frame').src=url;$('open-editor').href=url;$('editor-frame').hidden=false;$('editor-loading').hidden=true;
}
function configureIframe(){
  try{
    const doc=$('editor-frame').contentDocument;
    if(!doc||doc.getElementById('agent-review-wrapper-style'))return;
    // The host must also gate refinement server-side. This removes duplicate
    // refinement entry points from the human review flow, not an access control.
    const style=doc.createElement('style');style.id='agent-review-wrapper-style';
    style.textContent='header{display:none!important}main{padding:20px!important;gap:20px!important}#refine{display:none!important}#language-form{display:none!important}#language-form~p{display:none!important}';doc.head.append(style);
    const language=doc.getElementById('language-form');if(language){const card=language.closest('.card');if(card)card.hidden=true;}
    doc.addEventListener('submit',event=>{if(batchLocksEditing()){event.preventDefault();event.stopImmediatePropagation();}},true);
    doc.addEventListener('click',event=>{if(batchLocksEditing()&&event.target.closest('#save,#undo,#redo,#send,#refine')){event.preventDefault();event.stopImmediatePropagation();}},true);
  }catch{}
}
function selectBatchTarget(identity){
  const context=editorContext();
  if(!context||context.editor.pending?.size||formIsDirty(context)||!context.view.snapshot.objects.some(item=>item.instance_id===identity))return;
  context.editor.select?.(identity);refreshEditorState();
}
function renderBatch(){
  const batch=currentWorkflow(),room=!!status?.room_finish;
  $('batch-progress').hidden=!batch;document.body.classList.toggle('batch-mode',!!batch);
  if(!batch)return;
  if(room)text('phase-label',['complete','scene_changed'].includes(batch.phase)?'房间处理已保存':['failed','invalid'].includes(batch.phase)?'房间处理需要核对':'房间饰面与门 / 花瓶');
  text('page-title',room?'房间饰面与门 / 花瓶':'保留布局，逐个精修家具');
  text('page-lead',room?'家具精模继续保留。本阶段为地板与墙体补充几何和颜色，并新增两扇门及电视柜花瓶；保存前核对已有布局与家具。':'按本次授权逐个生成家具几何与材质。已修正的墙体、位置、尺寸与支撑关系保持锁定，完成的对象随场景版本更新。');
  text('confirm-heading',room?'本次房间处理已授权':'本次批量精修已授权');
  text('confirm-description',room?'墙体、地板、门与花瓶作为后续任务处理。家具批次和原茶几确认记录保留，界面显示当前任务的真实阶段。':'原茶几确认与修复记录继续保留。当前批次按已授权的对象清单执行，进度以通过校验并保存的对象为准。');
  text('editor-heading','查看当前保存场景');
  text('editor-guidance',batchLocksEditing()?'可选择对象查看当前已保存的场景与源图。任务期间位置、缩放和撤销操作已锁定，视角与墙体显隐仍可调整。':'当前任务已经保存，可继续选择对象核对源图，或编辑位置、缩放并保存。后续修改会产生新的场景版本。');
  text('batch-heading',room?'墙面、地板、门与花瓶':'家具逐对象精修');
  text('batch-count',`${batch.completed_count??0} / ${batch.total??(room?7:10)} 已验证保存`);
  text('batch-message',batch.message||status.message);
  $('batch-meter').max=Math.max(1,batch.total||10);$('batch-meter').value=batch.completed_count||0;
  text('batch-lock',room?(batch.phase==='scene_changed'?'房间处理的完成版本已验证；当前场景包含后续修改。':batch.layout_preserved?'既有家具、根变换和支撑关系已核验保持不变。新增门与花瓶按本次授权单独记录。':'正在核对房间任务的授权、保存版本和资产记录。'):(batch.phase==='scene_changed'?'批次完成版本已验证；当前场景包含后续人工修改。':batch.layout_preserved?'相连墙体与布局已核验保持不变。颜色允许适度调整；对象结构仍须通过独立复核。':'正在核对批次授权、锁定布局和资产记录。'));
  $('batch-targets').replaceChildren();
  const labels={waiting:'等待',refining:room?'建模中':'精修中',validating:'复核中',prepared:'待保存',accepted:'已保存',failed:'失败',needs_reconciliation:'待核对'};
  for(const item of batch.targets||[]){
    const li=document.createElement('li');li.dataset.state=item.state;
    const head=document.createElement('div');head.className='target-head';
    const button=document.createElement('button');button.type='button';button.textContent=item.target_id;button.addEventListener('click',()=>selectBatchTarget(item.target_id));
    const state=document.createElement('span');state.className='target-state';state.textContent=labels[item.state]||item.state;
    head.append(button,state);li.append(head);
    if(item.message){const detail=document.createElement('small');detail.textContent=item.message;li.append(detail);}
    $('batch-targets').append(li);
  }
}
function renderStatus(){
  if(!status)return;
  const [label,index]=phases[status.phase]||[status.phase||'等待状态',-1];
  text('phase-label',label);text('phase-message',status.message||'等待新的运行记录。');
  renderBatch();
  text('layout-summary',status.layout_summary||'按源图核查房间形状、家具尺寸和位置，未见部分包含建模假设。');
  $('layout-uncertainties').replaceChildren();
  const notes=[...(status.uncertainties||[]),...(status.rejected_objects||[]).map(o=>`未纳入白模：${o.object_id} · ${o.reason}`)];
  for(const note of notes){const li=document.createElement('li');li.textContent=note;$('layout-uncertainties').append(li);}
  document.querySelectorAll('.milestones li').forEach((item,i)=>{item.classList.toggle('current',i===index);item.classList.toggle('done',index>i);if(i===index)item.setAttribute('aria-current','step');else item.removeAttribute('aria-current');});
  const events=Array.isArray(status.events)?status.events.slice(-10):[];
  $('run-events').replaceChildren();
  for(const event of events){const li=document.createElement('li');if(event.time){const time=document.createElement('time');time.textContent=event.time;li.append(time);}li.append(document.createTextNode(typeof event==='string'?event:event.message||event.phase||'状态已更新'));$('run-events').append(li);}
  if(!events.length){const li=document.createElement('li');li.textContent=status.message||'尚无阶段事件。';$('run-events').append(li);}
  if(status.confirmation){text('confirmed-version',`已确认 ${status.confirmation.revision} / head ${status.confirmation.head_seq} / ${status.confirmation.target_id}`);$('confirmed-version').hidden=false;if(uncertain&&exactConfirmation(status.confirmation,uncertain)){rememberPending(null);actionError='';}}
  else $('confirmed-version').hidden=true;
  const serverError=status.error?(typeof status.error==='string'?status.error:status.error.message||'运行出现错误，请查看服务日志。'):'';
  const errors=[actionError,serverError].filter(Boolean);
  $('run-error').hidden=!errors.length;text('run-error',[...new Set(errors)].join(' · '));
}
function eligibility(context){
  if(confirmPending)return '正在提交确认，请稍候。';
  if(uncertain)return '上一条确认的执行结果待查询；页面不会重复发送。请刷新状态或检查服务运行记录。';
  if(!statusFresh)return '运行状态尚未同步，暂不能确认。';
  if(currentWorkflow())return ['complete','scene_changed'].includes(currentWorkflow().phase)?'当前任务已完成；可继续编辑场景。原单对象确认记录保持不变。':'当前任务已获授权，正在按对象清单推进。原单对象确认入口已关闭。';
  if(status?.phase==='complete')return '该运行已完成；结果和版本以服务记录为准。';
  if(status?.confirmation||['refining','validating'].includes(status?.phase))return '当前版本已交给精修流程，正在等待结果。';
  if(!context)return '等待白模工作台载入。';
  if(!(status?.ready_for_review===true||status?.phase==='needs_user'))return '白模准备尚未完成。';
  if(!sameHead(status.head,context.view.head))return '工作台与运行记录正在同步版本，请稍候。';
  if(context.editor.pending?.size)return '工作台仍有待完成的保存或撤销操作，请等待完成。';
  if(formIsDirty(context))return '属性中有尚未保存的修改。请先点击工作台中的“保存修改”。';
  if(status.require_edit===true&&status.human_edit_saved!==true)return '请先根据源图实际修正位置或缩放，并点击工作台中的“保存修改”。保存后的人工修正通过服务核验后才能确认。';
  if(!context.obj)return '请在工作台中选择一个对象。';
  if(!refinableObject(context.obj))return '当前选择是房间结构，请选择一个家具或物体进行精修。';
  if(Array.isArray(status.refinable_target_ids)&&!status.refinable_target_ids.includes(context.obj.instance_id))return '当前对象尚未具备完整的精修输入，请选择可精修对象。';
  return '';
}
function refreshEditorState(){
  configureIframe();
  const context=editorContext();
  if(context){
    const {view,obj,editor,doc}=context;
    const batchLocked=batchLocksEditing();
    for(const id of ['property-fields','save','undo','redo']){
      const control=doc.getElementById(id);if(!control)continue;
      if(batchLocked){control.dataset.batchLocked='true';control.disabled=true;}
      else if(control.dataset.batchLocked){delete control.dataset.batchLocked;control.disabled=false;}
    }
    if(currentWorkflow()&&status.head&&!sameHead(status.head,view.head)&&!batchRefreshing&&!formIsDirty(context)){
      batchRefreshing=true;Promise.resolve(editor.refresh()).catch(()=>{}).finally(()=>{batchRefreshing=false;});
    }
    if(editor.viewer&&!editor.viewer.reviewDisplay){
      const viewer=editor.viewer;viewer.reviewDisplay={loading:true};
      import('/vendor/three/build/three.module.js').then(({ACESFilmicToneMapping})=>{
        viewer.renderer.toneMapping=ACESFilmicToneMapping;viewer.renderer.toneMappingExposure=.8;
        viewer.reviewDisplay={ready:true};
      }).catch(()=>{viewer.reviewDisplay={unavailable:true};});
    }
    $('room-top').disabled=!editor.viewer;$('room-perspective').disabled=!editor.viewer;
    if(editor.viewer){for(const item of view.snapshot.objects){if(['wall','ceiling'].includes(item.kind)){const root=editor.viewer.roots.get(item.instance_id);if(root)root.visible=$('room-walls').checked;}}}
    editor.viewer?.addWhiteboxEdges?.(view.snapshot);
    if(editor.viewer&&!editor.viewer.reviewOverviewReady&&editor.viewer.pendingAssets===0){
      roomView(false);editor.viewer.reviewOverviewReady=true;
    }
    const contentHeight=Math.max(720,Math.min(6000,(doc.body?.scrollHeight||0)+16));
    if(Math.abs($('editor-frame').getBoundingClientRect().height-contentHeight)>4)$('editor-frame').style.height=`${contentHeight}px`;
    text('scene-id',sceneId);text('scene-revision',`${view.head.revision} · head ${view.head.head_seq}`);
    text('selected-object',obj?`${obj.category} · ${obj.instance_id}`:'未选择对象');text('object-dimensions',obj?dimensions(obj):'—');
    const support=view.snapshot.objects.find(item=>item.instance_id===obj?.support_id);
    text('object-support',support?`${support.category} · ${support.instance_id}`:obj?.support_id||'未指定 / 未确认');
    const token=JSON.stringify([view.head,editor.state.selectedId]);
    if(token!==lastSelectionToken){$('reviewed').checked=false;lastSelectionToken=token;}
    const dirty=formIsDirty(context),pending=!!editor.pending?.size;
    const needsEdit=status?.require_edit===true&&status?.human_edit_saved!==true;
    text('saved-indicator',batchLocked?'布局已锁定':pending?'操作处理中':dirty?'修改未保存':needsEdit?'尚未保存人工修正':'已读取保存版本');$('saved-indicator').classList.toggle('warning',!batchLocked&&(dirty||pending||needsEdit));
    text('scene-note',batchLocked?`当前共 ${view.snapshot.objects.length} 个对象。显示最新保存版本，当前任务保持已有布局与家具。`:dirty?'属性已变化。请先保存修改，再确认精修。':pending?'等待工作台操作完成；确认按钮暂不可用。':`当前共 ${view.snapshot.objects.length} 个对象。保存修改会生成新的场景版本。`);
    const recommendation=view.snapshot.objects.find(item=>item.instance_id===status?.recommended_target_id&&(currentWorkflow()||refinableObject(item)));
    $('select-recommended').hidden=!recommendation;
    if(recommendation){text('select-recommended',`${currentWorkflow()?'查看当前对象':'选择建议对象'} · ${recommendation.instance_id}`);$('select-recommended').disabled=dirty||pending||typeof editor.select!=='function';}
    const image=doc.getElementById('evidence-image');
    if(image&&!image.hidden&&image.complete&&image.naturalWidth>0){
      try{const url=new URL(image.src,location.href);if(url.origin===location.origin){if(url.href!==lastSource){$('source-image').src=url.href;lastSource=url.href;}$('source-image').hidden=false;$('source-empty').hidden=true;text('source-frame',doc.getElementById('evidence-id')?.textContent||'源图');text('source-reason',doc.getElementById('evidence-reason')?.textContent||'');}}catch{}
    }else{$('source-image').hidden=true;$('source-empty').hidden=false;text('source-frame','等待当前对象证据');text('source-reason','');}
  }
  const reason=eligibility(context);
  $('reviewed').disabled=!!reason;
  $('confirm').disabled=!!reason||!$('reviewed').checked;
  text('confirm-note',reason||($('reviewed').checked?'将按上方保存版本启动精修；只在点击按钮后发送。':'核对完成后勾选确认，再点击按钮开始精修。'));
}
async function fetchStatus(){
  if(statusPending)return;statusPending=true;
  try{
    const response=await fetch(`/agent/status${sceneId?'?scene_id='+encodeURIComponent(sceneId):''}`,{cache:'no-store'});
    const value=await response.json();
    if(!response.ok||value.error&&!value.phase)throw new Error(errorMessage(value,`状态读取失败 (${response.status})`));
    if(typeof value.scene_id!=='string'||!value.scene_id)throw new Error('运行状态没有有效 scene_id');
    if(sceneId&&value.scene_id!==sceneId)throw new Error('运行状态属于另一个场景，未加载。');
    if(!sceneId){sceneId=value.scene_id;recoverPending();const url=new URL(location.href);url.searchParams.set('scene_id',sceneId);history.replaceState(null,'',url);}
    status=value;statusFresh=true;text('connection-text','已连接本地运行');$('connection-dot').classList.add('online');text('updated-at',`状态更新 ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`);renderStatus();setupEditor();
  }catch(error){statusFresh=false;text('connection-text','连接未就绪');$('connection-dot').classList.remove('online');text('phase-message',error.message);}
  finally{statusPending=false;refreshEditorState();clearTimeout(statusTimer);statusTimer=setTimeout(fetchStatus,document.hidden?5000:2000);}
}
async function confirmScene(){
  const context=editorContext();const reason=eligibility(context);
  if(reason||!$('reviewed').checked){refreshEditorState();return;}
  const payload={scene_id:sceneId,revision:context.view.head.revision,head_seq:context.view.head.head_seq,target_id:context.obj.instance_id};
  confirmPending=true;actionError='';rememberPending(payload);renderStatus();refreshEditorState();
  try{
    const response=await fetch('/agent/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    let result;
    try{result=await response.json();}catch{throw new Error(`确认响应不完整 (${response.status})；正在查询原请求结果。`);}
    if(!response.ok||result.status==='error'){
      const error=new Error(errorMessage(result,`确认失败 (${response.status})`));
      error.definite=response.status>=400&&response.status<500&&response.status!==408&&result.error?.details?.outcome!=='uncertain';
      if(error.definite)rememberPending(null);throw error;
    }
    $('reviewed').checked=false;
    text('confirm-note','确认已提交，正在读取精修状态。');
  }catch(error){
    actionError=error.message;renderStatus();
    if(error.definite){$('reviewed').checked=false;try{await context.editor.refresh();}catch{}}
  }finally{confirmPending=false;await fetchStatus();refreshEditorState();}
}
function roomView(top){const context=editorContext(),viewer=context?.editor.viewer;if(!viewer)return;
  viewer.camera.up.set(0,1,0);viewer.reset();
  const floor=context.view.snapshot.objects.find(o=>o.kind==='floor'),root=floor&&viewer.roots.get(floor.instance_id);
  if(top){const target=viewer.controls.target,distance=viewer.camera.position.distanceTo(target)*.65;
    if(root){viewer.camera.up.set(0,0,-1).applyQuaternion(root.quaternion);}
    viewer.camera.position.set(target.x,target.y+distance,target.z+.001);viewer.controls.update();
  }else if(root){
    // Face an open side when the observed wall shell is incomplete. This is a
    // viewing choice only; saved object/camera transforms are never modified.
    const walls=context.view.snapshot.objects.filter(o=>o.kind==='wall'),direction=root.position.clone().set(0,0,0);
    for(const wall of walls){const wallRoot=viewer.roots.get(wall.instance_id);if(wallRoot)direction.add(wallRoot.position.clone().sub(root.position));}
    direction.y=0;
    if(walls.length&&direction.lengthSq()>1e-6){
      direction.negate().normalize();const side=direction.clone().set(-direction.z,0,direction.x);
      direction.addScaledVector(side,.18);direction.y=1.15;direction.normalize();
      const target=viewer.controls.target,distance=viewer.camera.position.distanceTo(target)*.8;
      viewer.camera.position.copy(target).addScaledVector(direction,distance);viewer.controls.update();
    }
  }
}
$('room-top').addEventListener('click',()=>roomView(true));
$('room-perspective').addEventListener('click',()=>roomView(false));
$('room-walls').addEventListener('change',refreshEditorState);
$('reviewed').addEventListener('change',refreshEditorState);
$('confirm').addEventListener('click',confirmScene);
$('select-recommended').addEventListener('click',()=>{
  const context=editorContext();
  if(!context||context.editor.pending?.size||formIsDirty(context)||!context.view.snapshot.objects.some(item=>item.instance_id===status?.recommended_target_id&&(currentWorkflow()||refinableObject(item))))return;
  context.editor.select?.(status.recommended_target_id);refreshEditorState();
  $('editor-heading').scrollIntoView({behavior:'auto',block:'start'});
});
$('refresh-status').addEventListener('click',()=>fetchStatus());
$('editor-frame').addEventListener('load',()=>{configureIframe();refreshEditorState();});
$('source-image').addEventListener('error',()=>{$('source-image').hidden=true;$('source-empty').hidden=false;text('source-empty','源图读取失败，请在工作台刷新证据。');});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)fetchStatus();});
window.addEventListener('pagehide',()=>{clearTimeout(statusTimer);clearInterval(editorTimer);});
editorTimer=setInterval(refreshEditorState,500);
fetchStatus();
