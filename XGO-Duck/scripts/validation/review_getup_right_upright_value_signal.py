"""Measure signal size and its tilt/support decomposition before value fitting."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'records/reports/getup-right-upright-clock'
PLAN = ROOT/'configs/experiments/getup_right_upright_clock.json'
PROOF = OUT/'analysis.json'
DEST = OUT/'value-signal-review.json'


def sha(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not DEST.exists()
    proof,spec = read(PROOF),read(PLAN)
    assert proof['status']=='passed-actual-cuda-reward-clock' and not proof['heldout_used']
    assert sha(PLAN)==proof['protocol_sha256'] and not proof['learning_updates']
    record = ROOT/proof['record']
    trace = record/'reward-trace.npz'
    target = ROOT/proof['value_target']['path']
    assert sha(trace)==proof['source_sha256'][trace.relative_to(ROOT).as_posix()]
    assert sha(target)==proof['value_target']['sha256']
    with np.load(trace) as a:
        d = {key:a[key] for key in ('z','feet','raw','base_reward','reward_age','active','done')}
    with np.load(target) as a:
        kept,control,delta = a['worlds'],a['control_returns'],a['reward_delta_returns']
        np.testing.assert_array_equal(a['candidate_returns'],control+delta)
    horizon = spec['horizon']
    rows = []
    incomplete = 0
    for event in proof['trigger_events']:
        start = event['control']-1
        end = start+horizon
        for world in event['worlds']:
            ids = np.arange(start,end+1)
            if end>=len(d['z']) or not d['active'][ids,world].all() or not np.all(np.diff(d['reward_age'][ids,world])==1):
                incomplete += 1
                continue
            angle = np.rad2deg(np.arccos(np.clip(-d['z'][ids,world].astype(float),-1.,1.)))
            progress = np.clip((45.-angle)/30.,0.,1.)
            support = d['feet'][ids,world].mean(axis=1)
            physical_tilt = support[1:]*np.diff(progress)
            support_change = progress[:-1]*np.diff(support)
            raw = d['raw'][ids[1:],world].astype(float)
            np.testing.assert_allclose(raw,physical_tilt+support_change,atol=2e-6,rtol=0)
            old = d['base_reward'][ids[1:],world].astype(float)
            rows.append(dict(control=start+1,world=world,in_complete_first_episode=bool(world in kept and end<400),
                absolute_peak_raw_increment=float(abs(raw).max()),raw_sum=float(raw.sum()),
                tilt_component_sum=float(physical_tilt.sum()),support_component_sum=float(support_change.sum()),
                raw_absolute_sum=float(abs(raw).sum()),base_reward_absolute_sum=float(abs(old).sum())))
    assert rows and sum(row['in_complete_first_episode'] for row in rows)==proof['complete_first_episode_triggers']
    kept_raw = d['raw'][:400,kept].astype(float)
    existing = d['base_reward'][:400,kept].astype(float)
    weights = {}
    for coefficient in (1,10,50):
        applied = kept_raw*(.02*coefficient)
        ratio = abs(applied)/(abs(existing)+1e-8)
        active = kept_raw!=0.
        weights[str(coefficient)] = dict(peak_absolute_step_reward=float(abs(applied).max()),
            peak_absolute_return_delta=float(abs(coefficient*delta).max()),
            contribution_larger_than_existing_count=int((ratio[active]>1.).sum()),
            nonzero_step_contribution_quantiles=np.quantile(abs(applied[active]),[.5,.9,.99]).tolist(),
            base_reward_quantiles_same_steps=np.quantile(abs(existing[active]),[.5,.9,.99]).tolist())
    result = dict(status='verified-value-signal-scale',learning_updates=0,heldout_used=False,physical_substeps=0,
        base_return_std=float(control.std()),record=proof['record'],complete_windows=len(rows),incomplete_windows=incomplete,
        complete_first_episode_triggers=proof['complete_first_episode_triggers'],weights=weights,rows=rows,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (PLAN,PROOF,trace,target,Path(__file__))},
        scope='Exact offline rescaling of saved reward only. Does not choose a new coefficient or show trained behavior.')
    DEST.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('RIGHT_UPRIGHT_VALUE_SIGNAL',result['complete_windows'],result['base_return_std'],weights,flush=True)


if __name__=='__main__':
    main()
