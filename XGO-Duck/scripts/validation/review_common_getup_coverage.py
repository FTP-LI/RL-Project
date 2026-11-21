"""Diagnose requested-versus-actual release coverage from existing traces.

Never overwrite the original wrong-release failures or use this post-hoc
grouping as a new pass. It separates sample coverage from the ability to rise
and hold, so the next independent protocol can require enough real releases
in all four directions without repeatedly training on a mislabeled failure.
"""
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from standup_metrics import summarize
from record_io import write


def main():
    record=Path(read(ROOT/'records/reports/common-plant-screen/getup/progress.json')['path'])
    p=read(record/'protocol.json');report=read(record/'evaluation.json')
    groups={k:[] for k in ('prone','supine','left-side','right-side','other')};requested={};sources={}
    for pose in p['poses']:
        path=record/f'{pose}-trace.npz'
        with np.load(path) as f:t={k:f[k] for k in f.files}
        result=summarize(t,p['criteria']);old=next(c for c in report['conditions'] if c['condition']==pose)
        assert result['successes']==old['successes']
        requested[pose]=dict(original_successes=result['successes'],trials=32,wrong_release=result['wrong_release_pose'],
            timely_holds=sum(x['hold_start_seconds'] is not None for x in result['episodes']))
        for row in result['episodes']:groups[row['release_pose']].append(dict(requested_pose=pose,environment_index=row['env_index'],
            timely_hold=row['hold_start_seconds'] is not None,original_success=row['success']))
        sources[str(path.relative_to(ROOT))]=sha(path)
    observed={k:dict(episodes=len(v),timely_holds=sum(x['timely_hold'] for x in v),
        rows=v,coverage_at_least32=len(v)>=32) for k,v in groups.items()}
    out=ROOT/'records/reports/getup-release-coverage';out.mkdir(exist_ok=True)
    result=dict(status='diagnosis-complete-no-new-admission',requested=requested,observed=observed,
        original_successes=sum(x['original_successes'] for x in requested.values()),
        total_timely_holds=sum(x['timely_holds'] for x in requested.values()),total_episodes=128,
        original_gate_unchanged=True,posthoc_grouping_not_a_new_pass=True,new_physics_steps=0,new_learning_updates=0,
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),
        next_protocol_question='Require>=32actual release episodes per class plus unchanged timely3s hold and all-first-episode success constraints; freeze before new data. Keep original requested-stratum result separately.')
    write(out/'diagnosis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('observed','source_trace_sha256')},indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in observed.items()},indent=2))


if __name__=='__main__':main()
