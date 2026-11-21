"""Bound the highest possible original roll score as a branch runs."""
from pathlib import Path

import numpy as np

from roll_landing_latch import RollLandingLatch
from roll_stand_walk_sequence import STATE_FIELDS
from run_continuous_roll_stand_walk import OUT, PLAN, checked
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def maximum_possible(latch, horizon):
    certified = latch.first_ready >= 0
    potential = certified | (latch.intact & (latch.step < horizon))
    return int(potential.sum()), int(certified.sum())


def main():
    p = checked()
    evidence = OUT/'stopped-review.json'
    stopped = read(evidence)
    assert stopped['status']=='verified-stopped-development-attempt'
    record = ROOT/stopped['record']
    raw = record/'trace.npz'
    assert sha(raw)==stopped['sources'][raw.relative_to(ROOT).as_posix()]
    start, end = p['initial_stand_controls'],p['roll_controls']
    with np.load(raw) as values:
        states = {field:values[field][start:start+end+1] for field in STATE_FIELDS}
    latch = RollLandingLatch(p['roll']['criteria'],p['worlds'])
    results = []
    for step in range(end+1):
        latch.advance({field:states[field][step] for field in STATE_FIELDS})
        possible, passed = maximum_possible(latch,end)
        results.append(dict(roll_control=step,global_control=start+step,upper_bound=possible,certified=passed))
    upper = np.array([row['upper_bound'] for row in results])
    passed = np.array([row['certified'] for row in results])
    assert (np.diff(upper)<=0).all() and (np.diff(passed)>=0).all() and (passed<=upper).all()
    gate = p['minimum_roll_completion']
    stop = next(row for row in results if row['upper_bound']<gate)
    assert stop['global_control'] < stopped['completed_control_steps'] and results[-1]['certified']==0
    output = OUT/'roll-entry-impossibility-replay.json'
    assert not output.exists()
    write(output,dict(status='passed-saved-impossibility-replay',source_sha256={path.relative_to(ROOT).as_posix():sha(path)
        for path in (PLAN,evidence,raw,Path(__file__),ROOT/'scripts/validation/roll_landing_latch.py')},
        gate=gate,first_stop=stop,verified_actual_completed_controls=stopped['completed_control_steps'],
        controls_avoided_on_same_trace=stopped['completed_control_steps']-stop['global_control'],timeline=results,
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Upper bound only. The historical trial is unchanged and no new rollout was performed.'))
    print('ROLL_ENTRY_IMPOSSIBILITY',stop,'actual',stopped['completed_control_steps'],flush=True)


if __name__=='__main__':
    main()
