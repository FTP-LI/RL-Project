"""Read current goal, real project worker PIDs and latest records; no compute.

Do not infer an active job from a stale status file or lock. Historical model
records remain visible separately from the currently running evaluation.
"""
from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    project=read(ROOT/'project.json');stage=project['staged_training_request']
    workers=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except (OSError,ProcessLookupError):continue
        files=[Path(a) for a in args if a.startswith(str(ROOT/'scripts')) and a.endswith('.py')]
        if files:workers.append(dict(pid=int(entry.name),script=str(files[0].relative_to(ROOT)),arguments=args[args.index(str(files[0]))+1:]))
    result=dict(current_stage=stage['current_stage'],current_protocol=stage.get('current_protocol'),
        next_gate=stage['next_gate'],learning_running=any(x['script'].startswith('scripts/training/') or '--train' in x['arguments'] for x in workers),
        workers=workers,hardware_deployment_authorized=stage['deployment_authorized_now'])
    if (ROOT/'records/training-current.json').exists():
        pointer=read(ROOT/'records/training-current.json')
        record=ROOT/(pointer.get('record') or pointer['path']);state_path=record/'state.json'
        state=read(state_path) if state_path.exists() else {}
        # A reviewed archive can retain its last live state verbatim.
        status=pointer.get('status') if pointer.get('live') is False else state.get('status')
        result['last_training']=dict(record=str(record),status=status or state.get('status','initializing'),
            completed_updates=state.get('completed_updates_this_run',state.get('completed_updates')),
            actor_updates=state.get('actor_updates'),critic_updates=state.get('critic_updates'),
            purpose=pointer.get('purpose'),clips=state.get('clips',[]))
        for key in ('view','curve','comparison'):
            if pointer.get(key):result['last_training'][key]=pointer[key]
    if (ROOT/'records/evaluations/current-evaluation.json').exists():
        record=Path(read(ROOT/'records/evaluations/current-evaluation.json')['path']);state=read(record/'evaluation.json')
        result['last_evaluation']=dict(record=str(record),status=state['status'],completed_conditions=[x['condition'] for x in state['conditions']])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
