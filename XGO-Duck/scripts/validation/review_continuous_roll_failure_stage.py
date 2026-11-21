"""Locate the continuous trial's failed roll gates using the original scorer."""
from collections import Counter
from pathlib import Path

import numpy as np

from roll_metrics import summarize
from roll_stand_walk_sequence import ROLL
from run_continuous_roll_stand_walk import OUT, PLAN, checked
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def checked_fields(path,proof,fields):
    assert sha(path)==proof['sources'][path.relative_to(ROOT).as_posix()]
    with np.load(path) as data:
        return {field:data[field] for field in fields}


def main():
    p = checked()
    proof_path = OUT/'stopped-review.json'
    proof = read(proof_path)
    assert proof['status']=='verified-stopped-development-attempt' and not proof['heldout_used']
    folder = ROOT/proof['record']
    fields = ('active','quaternion','head_quaternion','height','angular_velocity','ground_contact',
              'head_contact','foot_contact','head_penetration_m','joint_envelope_ok','phase_next')
    trace = checked_fields(folder/'trace.npz',proof,fields)
    commands = checked_fields(folder/'commands.npz',proof,('phase_used','requested_targets','applied_targets','bridge'))
    start, duration = p['initial_stand_controls'],p['roll_controls']
    stop = start+duration
    episodes = trace['phase_next'][start]==ROLL
    assert int(episodes.sum())==proof['counts']['entered_roll']==p['worlds']
    assert (commands['phase_used'][start:stop]==ROLL).all()
    traces = {name:values[start:stop+1] for name,values in trace.items() if name!='phase_next'}
    traces['time'] = np.arange(duration+1)*p['roll']['criteria']['step_dt']
    result = summarize(traces,p['roll']['criteria'])
    assert result['attempts']==p['worlds'] and result['successes']==0
    original = read(ROOT/'records/reports/roll-endpoint-confirmation/development-b-readback.json')
    assert original['status']=='passed-independent-verification' and not original['heldout_used']
    assert original['fresh_successes']==117
    info = {}
    for name,rows in (('isolated',original['fresh_rows']),('continuous',result['episodes'])):
        info[name] = dict(with_head_support=sum(row['head_contact_seconds'] is not None for row in rows),
            with_supported_completion=sum(row['completion_seconds'] is not None for row in rows),
            with_landing_hold=sum(row['hold_end_seconds'] is not None for row in rows),
            failures=dict(Counter(cause for row in rows for cause in row['reasons'])))
    low,high = np.asarray(p['walking']['joint_limits'],np.float32).T
    raw = commands['requested_targets'][start:stop]
    bounded = np.clip(raw,low,high)
    sent = commands['applied_targets'][start:stop]
    bridge = commands['bridge'][start:stop]
    np.testing.assert_allclose(sent[~bridge],bounded[~bridge],atol=2e-7,rtol=0)
    assert int(bridge.sum())==p['worlds']*p['cross_model_bridge_controls']
    values = dict(roll_world_controls=int(raw.shape[0]*raw.shape[1]),bridge_world_controls=int(bridge.sum()),
        max_bridge_error_vs_projected_rad=float(abs(sent-bounded).max()),
        max_projection_error_vs_raw_rad=float(abs(bounded-raw).max()),
        worlds_with_nonzero_projection=int((abs(bounded-raw).max(axis=(0,2))>1e-7).sum()),
        worlds_with_nonzero_slew=int((abs(sent-bounded).max(axis=(0,2))>1e-7).sum()))
    output = OUT/'roll-failure-stage.json'
    assert not output.exists()
    write(output,dict(status='passed-saved-failure-localization',scope='Different development seeds and entry conditions. Comparison is descriptive.',
        roll=info,targets=values,continuous_episodes=result['episodes'],heldout_used=False,
        physical_substeps=0,learning_updates=0,stopped_review_sha256=sha(proof_path),
        protocol_sha256=sha(PLAN),source_sha256=sha(Path(__file__))))
    print('CONTINUOUS_ROLL_FAILURE_STAGE',info,values,flush=True)


if __name__=='__main__':
    main()
