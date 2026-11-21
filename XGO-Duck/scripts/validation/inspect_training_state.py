"""Read live records through Linux to avoid Windows read-sharing conflicts.

No simulation, training, writes or CUDA imports. Safe to use while the single
visible task is running; keep control decisions in the frozen run protocol.
"""
from pathlib import Path
import json
import time

ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    record=Path(read(ROOT/'records/training-current.json')['record'])
    state=read(record/'state.json')
    result={k:state.get(k) for k in ('status','completed_updates_this_run','iteration','common_step',
        'elapsed_learning_seconds','reason','guard')}
    result['record']=str(record)
    result['heartbeat_age_seconds']=round(time.time()-state.get('heartbeat_unix',time.time()),1)
    if (record/'clips.json').exists():
        clips=read(record/'clips.json')
        result['clips']=[{k:c[k] for k in ('file','complete_clip','frames','captured_at')} for c in clips]
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
