import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** Read the user's existing Codex login in memory. Never creates Pi credentials,
 * refreshes shared tokens, writes the original login or falls back to an API key. */
export class CodexSubscriptionCredentials {
  constructor(authPath=path.join(process.env.CODEX_HOME||path.join(os.homedir(),'.codex'),'auth.json')){this.authPath=authPath}
  async read(providerId){
    if(providerId!=='openai-codex')return undefined;
    let value;
    try{value=JSON.parse(fs.readFileSync(this.authPath,'utf8'))}
    catch{throw new Error('CODEX_LOGIN_REQUIRED: sign in to the original Codex application before continuing')}
    const tokens=value.tokens;
    if(value.auth_mode!=='chatgpt'||!tokens?.access_token)throw new Error('CODEX_LOGIN_REQUIRED: existing Codex ChatGPT subscription login is unavailable');
    let claims;
    try{claims=JSON.parse(Buffer.from(tokens.access_token.split('.')[1],'base64url').toString('utf8'))}
    catch{throw new Error('CODEX_LOGIN_REQUIRED: the existing Codex login is not readable; sign in again')}
    const expires=Number(claims.exp)*1000;
    if(!Number.isFinite(expires)||expires<Date.now()+600000)throw new Error('CODEX_LOGIN_REFRESH_REQUIRED: refresh the original Codex login before continuing; no billing fallback');
    const accountId=tokens.account_id||claims['https://api.openai.com/auth']?.chatgpt_account_id;
    if(!accountId)throw new Error('CODEX_LOGIN_REQUIRED: subscription account context is missing');
    return {type:'oauth',access:tokens.access_token,refresh:tokens.refresh_token||'',expires,accountId};
  }
  async list(){return await this.read('openai-codex')?[{providerId:'openai-codex',type:'oauth'}]:[]}
  async modify(){throw new Error('CODEX_LOGIN_REFRESH_REQUIRED: this adapter reads Codex authorization without rotating or writing shared credentials')}
  async delete(){throw new Error('Shared Codex authorization is read-only')}
}

export function safeError(error){
  return String(error?.message||error).replace(/\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,'[redacted token]').replace(/\bsk-[A-Za-z0-9_-]+\b/g,'[redacted secret]').slice(0,4000);
}
