import {test} from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {callHost} from '../../tools/pi-runtime/host-transport.mjs';

async function localServer(handler,run){
 const server=http.createServer(handler);
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{await run(`http://127.0.0.1:${server.address().port}/tool`)}
 finally{server.closeAllConnections();await new Promise(resolve=>server.close(resolve))}
}
test('tool result preserves image attachment and Unicode JSON across chunks',async()=>{
 await localServer((request,response)=>{
  let received='';request.on('data',data=>received+=data);request.on('end',()=>{
   assert.equal(JSON.parse(received).message,'场景');response.writeHead(200,{'Content-Type':'application/json'});
   response.write('{"content":');setTimeout(()=>response.end('[{"type":"image","mimeType":"image/png","data":"fixture"}]}'),25);
  });
 },async url=>{const result=await callHost(url,{message:'场景'},undefined,1000);assert.equal(result.ok,true);assert.equal(result.result.content[0].data,'fixture')});
});
test('abort cancels a pending tool transport',async()=>{
 await localServer(()=>{},async url=>{const abort=new AbortController();const request=callHost(url,{},abort.signal,1000);abort.abort();await assert.rejects(request,{name:'AbortError'})});
});
test('timeout and HTTP error are distinct, with no automatic retry',async()=>{
 let count=0;await localServer((request,response)=>{count++;if(count===1){response.writeHead(422);response.end('{"error":{"message":"invalid"}}')}},async url=>{
  assert.deepEqual(await callHost(url,{},undefined,1000),{ok:false,result:{error:{message:'invalid'}}});
  await assert.rejects(callHost(url,{},undefined,30),/transport timeout/);assert.equal(count,2);
 });
});
