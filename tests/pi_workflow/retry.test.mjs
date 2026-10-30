// Real Pi SDK recovery loop; only model transport and a tool are deterministic fixtures.
// No real credentials, model requests, scene writes, or provider billing.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createAgentSession,createExtensionRuntime,ModelRuntime,SessionManager,SettingsManager} from '../../tools/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/index.js';
import {AssistantMessageEventStream} from '../../tools/pi-runtime/node_modules/@earendil-works/pi-ai/dist/utils/event-stream.js';
import {runtimeSettings} from '../../tools/pi-runtime/runtime-settings.mjs';

globalThis.fetch=()=>{throw new Error('Network forbidden in retry fixtures');};
const resources=()=>({getExtensions:()=>({extensions:[],errors:[],runtime:createExtensionRuntime()}),
  getSkills:()=>({skills:[],diagnostics:[]}),getPrompts:()=>({prompts:[],diagnostics:[]}),getThemes:()=>({themes:[],diagnostics:[]}),
  getAgentsFiles:()=>({agentsFiles:[]}),getSystemPrompt:()=> 'Isolated retry fixture.',getSystemPromptSource:()=>undefined,
  getAppendSystemPrompt:()=>[],getAppendSystemPromptSources:()=>[],extendResources:()=>{},reload:async()=>{}});
async function fixture(respond,run,{delay=1}={}){
  const folder=await fs.mkdtemp(path.join(os.tmpdir(),'pi-retry-fixture-'));
  const runtime=await ModelRuntime.create({credentials:{read:async()=>({type:'oauth',access:'fixture',refresh:'',expires:Date.now()+3600000,accountId:'fixture'}),list:async()=>[{providerId:'openai-codex',type:'oauth'}],modify:async()=>{throw Error('No auth writes');}},modelsPath:null,refreshOnCreate:false,allowModelNetwork:false});
  const model=runtime.getModel('openai-codex','gpt-6-astra');
  let calls=0,writes=0;const contexts=[],options=[],events=[];
  runtime.streamSimple=async(m,context,opts)=>{
    contexts.push(structuredClone(context));options.push(opts);calls++;
    const response=respond(calls),stream=new AssistantMessageEventStream();
    const message={role:'assistant',content:[],api:model.api,provider:model.provider,model:model.id,timestamp:Date.now(),
      usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},...response};
    stream.push(message.stopReason==='error'?{type:'error',reason:'error',error:message}:{type:'done',reason:message.stopReason,message});
    stream.end();return stream;
  };
  const settings=runtimeSettings();settings.compaction.enabled=false;settings.retry.baseDelayMs=delay;
  const {session}=await createAgentSession({cwd:folder,agentDir:folder,modelRuntime:runtime,model,thinkingLevel:'off',
    resourceLoader:resources(),sessionManager:SessionManager.create(folder,folder),settingsManager:SettingsManager.inMemory(settings),
    noTools:'builtin',tools:['fixture_save'],customTools:[{name:'fixture_save',label:'fixture',description:'Count an isolated save',
      parameters:{type:'object',properties:{}},execute:async()=>{writes++;return {content:[{type:'text',text:'saved fixture once'}],details:{}};}}]});
  session.subscribe(event=>events.push(event));
  try{await run({session,events,contexts,options,stats:()=>({calls,writes})});}
  finally{session.dispose();await fs.rm(folder,{recursive:true,force:true});}
}
const error=text=>({stopReason:'error',errorMessage:text});
const done={stopReason:'stop',content:[{type:'text',text:'fixture recovered'}]};
const tool={type:'toolCall',id:'fixture_save_1',name:'fixture_save',arguments:{}};

test('production recovery budget is five retries, 2/4/8/16/30 seconds, no nested provider retry',()=>{
  const {retry}=runtimeSettings();assert.equal(retry.enabled,true);assert.equal(retry.maxRetries,5);
  assert.deepEqual(Array.from({length:5},(_,i)=>Math.min(retry.maxAgentDelayMs,retry.baseDelayMs*2**i)),[2000,4000,8000,16000,30000]);
  assert.equal(retry.provider.maxRetries,0);
});
test('fetch/stream failures resume the same turn, preserve completed tool result and never replay partial tool calls',async()=>{
  await fixture(n=>n===1?{stopReason:'toolUse',content:[tool]}:n===2?{...error('fetch failed'),content:[{...tool,id:'partial_do_not_execute'}]}:n===3?error('terminated'):done,async({session,events,contexts,options,stats})=>{
    await session.prompt('Save the fixture once, then report.');
    assert.deepEqual(stats(),{calls:4,writes:1});
    assert.equal(contexts.at(-1).messages.filter(m=>m.role==='user').length,1);
    assert.equal(contexts.at(-1).messages.filter(m=>m.role==='toolResult'&&m.toolCallId==='fixture_save_1').length,1);
    assert(!contexts.at(-1).messages.some(m=>m.stopReason==='error'));
    assert(events.some(e=>e.type==='auto_retry_end'&&e.success));
    assert(options.every(o=>o.maxRetries===0));
    const persisted=await fs.readFile(session.sessionManager.getSessionFile(),'utf8');assert(persisted.includes('fetch failed'));
  });
});
test('persistent network failure exhausts exactly five retries and settles',async()=>{
  await fixture(()=>error('fetch failed'),async({session,events,stats})=>{
    await session.prompt('Fixture persistent failure.');assert.equal(stats().calls,6);
    assert.equal(events.filter(e=>e.type==='auto_retry_start').length,5);
    assert(events.some(e=>e.type==='auto_retry_end'&&!e.success&&e.attempt===5));
  });
});
for(const message of ['CODEX_LOGIN_REQUIRED','401 Unauthorized','403 Forbidden','429 insufficient_quota','subscription_sharing_usage_limit_exceeded','This operation was aborted']){
  test(`non-transient error does not retry: ${message}`,async()=>{
    await fixture(()=>error(message),async({session,events,stats})=>{
      await session.prompt('Fixture non-transient failure.');assert.equal(stats().calls,1);
      assert(!events.some(e=>e.type==='auto_retry_start'));
    });
  });
}
test('user pause aborts retry backoff without another model call',async()=>{
  await fixture(()=>error('fetch failed'),async({session,events,stats})=>{
    session.subscribe(event=>{if(event.type==='auto_retry_start')setTimeout(()=>session.abort(),10);});
    await session.prompt('Fixture pause.');assert.equal(stats().calls,1);
    assert(events.some(e=>e.type==='auto_retry_end'&&e.finalError==='Retry cancelled'));
  },{delay:2000});
});
