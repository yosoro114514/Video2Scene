export const VERSION = 'video2scene/1.0';
export const uid = (prefix='req') => `${prefix}_${crypto.randomUUID().replaceAll('-','')}`;
export const initialState = sceneId => ({sceneId, view:null, lastEventSeq:0, announcedHeadSeq:-1, selectedId:null, jobs:new Map()});

export function acceptView(state, view) {
  if (!view || view.snapshot.scene_id !== state.sceneId) return false;
  const latest = Math.max(state.announcedHeadSeq, state.view?.head.head_seq ?? -1);
  if (view.head.head_seq < latest || view.snapshot.revision !== view.head.revision) return false;
  state.view = view;
  if (!view.snapshot.objects.some(o=>o.instance_id===state.selectedId)) state.selectedId = view.snapshot.objects.find(o=>o.kind==='object')?.instance_id || view.snapshot.objects[0].instance_id;
  return true;
}

export function consumeEvent(state, event) {
  if (event.scene_id !== state.sceneId || !Number.isInteger(event.seq) || event.seq<=state.lastEventSeq) return {accepted:false,refresh:false};
  state.lastEventSeq = event.seq;
  if(event.type==='scene.head_changed') {
    const current = Math.max(state.announcedHeadSeq,state.view?.head.head_seq ?? -1);
    if(event.data.head_seq<=current) return {accepted:true,refresh:false};
    state.announcedHeadSeq=event.data.head_seq;
    return {accepted:true,refresh:true};
  }
  if(event.type==='job.updated') state.jobs.set(event.data.job_id,event.data);
  return {accepted:true,refresh:false};
}

export function makeCommand(state, sessionId, intent, payload) {
  if(!state.view) throw new Error('场景尚未载入');
  return {contract_version:VERSION,command_id:uid('cmd'),session_id:sessionId,scene_id:state.sceneId,
    base_revision:state.view.head.revision,base_head_seq:state.view.head.head_seq,
    selected_ids:state.selectedId?[state.selectedId]:[],intent,...payload};
}

export const toolRequest = (sceneId,tool,params) => ({contract_version:VERSION,request_id:uid(),scene_id:sceneId,tool,params});
export const artifactURL = ref => `/api/v1/artifacts?uri=${encodeURIComponent(ref.uri)}`;

export function propertyEdits(obj, values) {
  const edits=[];
  for(const [field,value] of Object.entries(values)) {
    const original = field.startsWith('transform.')?obj.transform[field.split('.')[1]]:obj[field];
    const same=Array.isArray(value)&&Array.isArray(original)?value.length===original.length&&value.every((n,i)=>Math.abs(n-original[i])<=1e-10):JSON.stringify(value)===JSON.stringify(original);
    if(!same) edits.push({instance_id:obj.instance_id,field,value});
  }
  return edits;
}
