"""Recognize a terminal capacity rejection before any assistant work started."""
import json
from pathlib import Path


CAPACITY_MESSAGES=('selected model is at capacity','model is currently at capacity')


def capacity_rejection(output):
    output=Path(output)
    if (output/'response.json').exists():return None
    try:
        lines=(output/'events.jsonl').read_text(encoding='utf-8').splitlines()
        events=[json.loads(line) for line in lines if line.strip()]
    except (OSError,ValueError):return None
    # Any assistant item, tool, completed turn, unknown event or partial line
    # makes execution uncertain. Never infer absence of effects from stderr.
    allowed={'thread.started','turn.started','error','turn.failed'}
    if not events or any(not isinstance(e,dict) or e.get('type') not in allowed for e in events):return None
    if events[-1].get('type')!='turn.failed':return None
    if any(e['type']=='turn.failed' and not isinstance(e.get('error'),dict) for e in events):return None
    errors=[e.get('message') if e['type']=='error' else e.get('error',{}).get('message')
            for e in events if e['type'] in ('error','turn.failed')]
    if not errors or any(not isinstance(e,str) or not any(m in e.lower() for m in CAPACITY_MESSAGES) for e in errors):return None
    return {'code':'MODEL_CAPACITY','message':errors[-1],'execution_started':False,
            'events':str(output/'events.jsonl')}
