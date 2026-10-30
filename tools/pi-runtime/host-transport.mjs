import http from 'node:http';

// Native GPU/Blender tools can legitimately exceed fetch's default header limit.
export const callHost=(url,payload,signal,timeoutMs=1900_000)=>new Promise((resolve,reject)=>{
 const body=JSON.stringify(payload);
 const request=http.request(url,{method:'POST',signal,headers:{'Content-Type':'application/json','Content-Length':Buffer.byteLength(body),'X-Workflow-Token':process.env.V2S_PI_TOOL_TOKEN||''}},response=>{
  const chunks=[];let bytes=0;
  response.on('data',chunk=>{bytes+=chunk.length;if(bytes>48*1024*1024)response.destroy(new Error('Tool response exceeds size budget'));else chunks.push(chunk)});
  response.on('error',reject);
  response.on('end',()=>{try{resolve({ok:response.statusCode>=200&&response.statusCode<300,result:JSON.parse(Buffer.concat(chunks).toString('utf8'))})}catch(error){reject(error)}});
 });
 request.setTimeout(timeoutMs,()=>request.destroy(new Error('Workflow tool transport timeout; inspect the recorded job before retry')));
 request.on('error',reject);request.end(body);
});
