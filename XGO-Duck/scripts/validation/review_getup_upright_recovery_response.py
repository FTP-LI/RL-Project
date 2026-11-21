"""Compare actual near-trigger motion against the two original-policy traces."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from standup_metrics import projected_gravity


OUT = ROOT/'records/reports/getup-upright-recovery-pair'
ROLES = ('reference-a','reference-b','bounded-legs')


def main(seed):
    source = OUT/f'seed{seed}-analysis.json'
    proof = read(source)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    record = ROOT/proof['record']
    event_path = record/'upright-trigger-events.json'
    assert sha(event_path)==proof['event_sha256']
    nominated = [event for event in read(event_path)[ROLES[-1]] if event['geometry']['feasible']]
    offsets = (0,5,11,22)
    fields = ('quaternion','foot_contact','native_head_leg_contact','applied_targets')
    branches = {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        assert sha(path)==proof['arms'][role]['trace_sha256']
        with np.load(path) as arrays:
            branches[role] = {key:arrays[key] for key in fields}
    rows = []
    for event in nominated:
        world,trigger = event['world'],event['decision_control']
        index = trigger-101
        result = {}
        for role in ROLES:
            trace = branches[role]
            frames = index+np.asarray(offsets)
            gravity = projected_gravity(trace['quaternion'][frames,world])
            tilt = np.rad2deg(np.arccos(np.clip(-gravity[:,2],-1.,1.)))
            contact = trace['native_head_leg_contact'][index+1:index+23,world]
            feet = trace['foot_contact'][index+1:index+23,world]
            result[role] = dict(tilt_deg=tilt.tolist(),head_leg_samples=int(contact.sum()),
                bilateral_samples=int(feet.all(axis=1).sum()),
                timed_and_final=bool(proof['arms'][role]['timing']['episodes'][world]['hold_start_seconds'] is not None
                    and proof['arms'][role]['end_hold']['stable_by_environment'][world]))
        a,b,c = (np.asarray(result[role]['tilt_deg']) for role in ROLES)
        rows.append(dict(world=world,control=trigger,measured=result,
            candidate_minus_reference_average_tilt_deg=(c-.5*(a+b)).tolist(),
            reference_repeat_tilt_spread_deg=np.abs(a-b).tolist()))
    median = np.median([row['candidate_minus_reference_average_tilt_deg'] for row in rows],axis=0).tolist() if rows else []
    output = OUT/f'seed{seed}-response.json'
    assert not output.exists()
    write(output,dict(status='passed-saved-trigger-response-review',source_sha256=sha(source),
        script_sha256=sha(Path(__file__)),offsets_controls=list(offsets),rows=rows,
        median_candidate_minus_reference_average_tilt_deg=median,
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Descriptive differences from two reference trajectories at the same elapsed times.'))
    print('GETUP_UPRIGHT_RESPONSE',seed,len(rows),median,flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed',type=int)
    main(parser.parse_args().seed)
