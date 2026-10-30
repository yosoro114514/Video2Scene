/** A persistent JSONL bridge around Pi's real SDK agent loop. Python owns jobs
 * and scene transactions; Pi owns every model/tool turn and persisted context. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {callHost} from './host-transport.mjs';
import {createAgentSession,createExtensionRuntime,ModelRuntime,SessionManager,SettingsManager} from '@earendil-works/pi-coding-agent';
import {Type} from 'typebox';
import {CodexSubscriptionCredentials,safeError} from './codex-subscription.mjs';
import {runtimeSettings} from './runtime-settings.mjs';

const config=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const emit=value=>process.stdout.write(JSON.stringify(value)+'\n');
let cleanupSession;
process.on('uncaughtException',error=>{emit({type:'fatal',error:safeError(error)});cleanupSession?.dispose();process.exitCode=1;process.stdin.destroy()});
const sha=data=>crypto.createHash('sha256').update(data).digest('hex');
let activeRun=null,turn=0,toolCalls=0,failedAssistantMessage=null;
const pendingTools=new Set();
const imageRecord=c=>({type:'image',mimeType:c.mimeType,sha256:sha(Buffer.from(c.data,'base64'))});
const compact=value=>Array.isArray(value)?value.map(compact):value&&typeof value==='object'?
  value.type==='image'&&value.data?imageRecord(value):Object.fromEntries(Object.entries(value).filter(([k])=>!['thinking','signature'].includes(k)).map(([k,v])=>[k,compact(v)])):value;
const credentials=new CodexSubscriptionCredentials();
const modelRuntime=await ModelRuntime.create({credentials,modelsPath:null,refreshOnCreate:false,allowModelNetwork:false});
const model=modelRuntime.getModel('openai-codex',config.model||'gpt-6-astra');
if(!model||model.provider!=='openai-codex')throw new Error('Requested Codex subscription model is not in the installed Pi catalog');
await credentials.read('openai-codex');
const customTools=(config.tools||[]).map(definition=>({
  name:definition.name,label:definition.name,description:definition.description,parameters:Type.Unsafe(definition.parameters),
  async execute(toolCallId,args,signal,onUpdate){
    if(++toolCalls>(config.max_tool_calls||64))throw new Error('Tool budget reached; save progress and wait for continuation');
    pendingTools.add(toolCallId);
    try{
      if(config.smoke){
        const bytes=fs.readFileSync(config.smoke_tool_image);
        const result={content:[{type:'text',text:'Probe succeeded. Describe this second reference image too.'},{type:'image',data:bytes.toString('base64'),mimeType:'image/jpeg'}],details:{smoke:true}};
        emit({type:'tool_receipt',run_id:activeRun,turn_id:turn,toolCallId,name:definition.name,args,result:compact(result)});return result;
      }
      const {ok,result}=await callHost(config.tool_endpoint,{project_id:config.project_id,run_id:activeRun,turn_id:turn,tool_call_id:toolCallId,name:definition.name,args},signal);
      if(!ok||result.error)throw new Error(result.error?.message||result.error||'Workflow tool failed');
      emit({type:'tool_receipt',run_id:activeRun,turn_id:turn,toolCallId,name:definition.name,args,result:compact(result)});
      return result;
    }finally{pendingTools.delete(toolCallId)}
  }
}));
const resourceLoader={
 getExtensions:()=>({extensions:[],errors:[],runtime:createExtensionRuntime()}),getSkills:()=>({skills:[],diagnostics:[]}),
 getPrompts:()=>({prompts:[],diagnostics:[]}),getThemes:()=>({themes:[],diagnostics:[]}),getAgentsFiles:()=>({agentsFiles:[]}),
 getSystemPrompt:()=>config.system_prompt,getSystemPromptSource:()=>undefined,getAppendSystemPrompt:()=>[],getAppendSystemPromptSources:()=>[],
 extendResources:()=>{},reload:async()=>{}
};
fs.mkdirSync(config.session_dir,{recursive:true});
const manager=config.session_file&&fs.existsSync(config.session_file)?SessionManager.open(config.session_file,config.session_dir,config.cwd):SessionManager.continueRecent(config.cwd,config.session_dir);
const {session,modelFallbackMessage}=await createAgentSession({cwd:config.cwd,agentDir:config.session_dir,modelRuntime,model,
 thinkingLevel:config.effort||'high',resourceLoader,customTools,tools:customTools.map(t=>t.name),noTools:'builtin',
 sessionManager:manager,settingsManager:SettingsManager.inMemory(runtimeSettings())});
cleanupSession=session;
if(modelFallbackMessage||session.model?.provider!=='openai-codex'||session.model?.id!==model.id)throw new Error('Model fallback is forbidden for this workflow');
session.subscribe(event=>{
 if(event.type==='turn_start'){turn++;emit({type:'turn_start',run_id:activeRun,turn_id:turn})}
 else if(event.type==='message_update'&&event.assistantMessageEvent.type==='text_delta')emit({type:'text_delta',run_id:activeRun,text:event.assistantMessageEvent.delta});
 else if(event.type==='message_end'&&event.message?.role==='assistant'){
   const m=event.message,record={type:'assistant_message',run_id:activeRun,turn_id:turn,text:(m.content||[]).filter(x=>x.type==='text').map(x=>x.text).join('\n'),
     tool_calls:(m.content||[]).filter(x=>x.type==='toolCall').map(compact),usage:m.usage,stopReason:m.stopReason,model:m.model,provider:m.provider,error:m.errorMessage?safeError(m.errorMessage):undefined};
   // A failed attempt is not yet a failed run. Publish the terminal error only
   // after Pi exhausts recovery; this also works with already running hosts.
   failedAssistantMessage=m.stopReason==='error'?record:null;
   emit(m.stopReason==='error'?{...record,type:'assistant_attempt_error'}:record);
 }else if(['auto_retry_start','auto_retry_end','summarization_retry_scheduled','summarization_retry_attempt_start','summarization_retry_finished'].includes(event.type)){
   const data=compact(event);
   for(const key of ['errorMessage','finalError'])if(data[key])data[key]=safeError(data[key]);
   emit({type:event.type,run_id:activeRun,turn_id:turn,data});
 }else if(['tool_execution_start','tool_execution_end','compaction_start','compaction_end','queue_update'].includes(event.type))emit({type:event.type,run_id:activeRun,turn_id:turn,data:compact(event)});
});
emit({type:'ready',project_id:config.project_id,provider:'openai-codex',model:model.id,subscription:true,pi_version:'1.4.0',transport:'sse',
 retry:runtimeSettings().retry,session_file:manager.getSessionFile(),tools:session.getActiveToolNames(),persisted_messages:session.messages?.length??session.state?.messages?.length??0});

async function command(value){
 try{
  if(value.type==='prompt'){
   if(session.isStreaming){emit({type:'command_error',request_id:value.id,error:'Run is active; use steer or wait'});return}
   activeRun=value.run_id||value.id;toolCalls=0;failedAssistantMessage=null;
   const images=(value.images||[]).map(p=>({type:'image',data:fs.readFileSync(p).toString('base64'),mimeType:/\.png$/i.test(p)?'image/png':'image/jpeg'}));
   emit({type:'prompt_started',request_id:value.id,run_id:activeRun,images:images.map(imageRecord)});
   await session.prompt(value.message,{images});
   if(failedAssistantMessage){emit(failedAssistantMessage);failedAssistantMessage=null;}
   emit({type:'settled',run_id:activeRun,aborted:false,tool_calls:toolCalls,session_file:manager.getSessionFile()});activeRun=null;
  }else if(value.type==='steer'){await session.steer(value.message);emit({type:'steering_queued',request_id:value.id,run_id:activeRun})}
  else if(value.type==='abort'){await session.abort();emit({type:'aborted',request_id:value.id,run_id:activeRun})}
  else if(value.type==='close'){await session.abort();session.dispose();process.exitCode=0;process.stdin.destroy()}
  else if(value.type==='state')emit({type:'state',request_id:value.id,run_id:activeRun,streaming:session.isStreaming,tools_pending:[...pendingTools],session_file:manager.getSessionFile()});
  else throw new Error('Unknown bridge command');
 }catch(error){emit({type:'run_error',request_id:value.id,run_id:activeRun,error:safeError(error)});activeRun=null}
}
// Split JSONL on LF only: Unicode line separators can appear in model/user text.
let buffer='';
process.stdin.setEncoding('utf8');
process.stdin.on('data',chunk=>{
  buffer+=chunk;
  let at;
  while((at=buffer.indexOf('\n'))>=0){
    const line=buffer.slice(0,at).replace(/\r$/,'');buffer=buffer.slice(at+1);
    if(line.trim()){
      try{void command(JSON.parse(line))}
      catch(error){emit({type:'command_error',error:safeError(error)})}
    }
  }
});
process.stdin.on('end',()=>{session.dispose()});
