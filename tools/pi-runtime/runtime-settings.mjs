// Pi owns retries of a failed assistant turn, retaining completed tool results.
// Disable the provider's nested retries so the visible five-retry budget is real.
export function runtimeSettings(){
  return {transport:'sse',compaction:{enabled:true,reserveTokens:24000,keepRecentTokens:24000},
    retry:{enabled:true,maxRetries:5,baseDelayMs:2000,maxAgentDelayMs:30000,
      provider:{maxRetries:0}}};
}
