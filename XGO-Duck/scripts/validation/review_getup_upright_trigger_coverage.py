"""Count which saved episodes a sustained-contact recovery trigger would affect."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from review_getup_nominal_supine import REPORTS
from standup_metrics import projected_gravity


OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'
FIELDS = ('time','quaternion','foot_contact','native_head_leg_contact')
WINDOW = 5


def first(mask):
    present = mask.any(axis=0)
    return np.where(present,np.argmax(mask,axis=0),-1)


def review(folder):
    path = ROOT/'records/reports'/folder/'analysis.json'
    proof = read(path)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    arm = proof['arms']['fixed-weighted']
    prefix,branch = (record/name for name in ('trace-shared-prefix.npz','trace-fixed-weighted.npz'))
    assert sha(prefix)==proof['prefix_sha256'] and sha(branch)==arm['trace_sha256']
    with np.load(prefix) as shared,np.load(branch) as individual:
        trace = {key:np.concatenate((shared[key],individual[key])) for key in FIELDS}
    episodes = arm['timing']['episodes']
    released = np.array([row['release_pose']=='supine' for row in episodes])
    nominal = np.array([row['requested_pose']=='supine' for row in episodes])
    successes = np.array([row['hold_start_seconds'] is not None for row in episodes])
    successes &= np.array(arm['end_hold']['stable_by_environment'],dtype=bool)
    time = trace['time']
    gravity = projected_gravity(trace['quaternion'])
    tilt = np.rad2deg(np.arccos(np.clip(-gravity[...,2],-1.,1.)))
    hit = (released[None] & (time[:,None]>2.) & (time[:,None]<=3.6)
           & (tilt>15.) & (tilt<=45.) & trace['foot_contact'].all(axis=-1)
           & trace['native_head_leg_contact'])
    windows = np.lib.stride_tricks.sliding_window_view(hit,WINDOW,axis=0).all(axis=-1)
    dwell = np.pad(windows,((WINDOW-1,0),(0,0)))
    # Compare a fixed 80 ms trend, with no future observations.
    slow = np.pad(tilt[:-WINDOW+1]-tilt[WINDOW-1:]<3.,((WINDOW-1,0),(0,0)))
    results = {}
    for label,selected in (('persistent_contact',dwell),('persistent_contact_and_low_rotation_progress',dwell&slow)):
        start = first(selected)
        reaches = start>=0
        results[label] = dict(
            candidate_failures=np.flatnonzero(reaches&~successes&released).tolist(),
            protected_successes=np.flatnonzero(reaches&successes&released).tolist(),
            missed_failures=np.flatnonzero(~reaches&~successes&released).tolist(),
            candidate_nominal_failures=np.flatnonzero(reaches&~successes&nominal).tolist(),
            trigger_time_seconds=[float(time[k]) if k>=0 else None for k in start])
    return dict(record=proof['record'],source_sha256=sha(path),actual_supine=int(released.sum()),
                nominal_supine=int(nominal.sum()),failed_supine=int((~successes&released).sum()),results=results)


def main():
    result = {name:review(name) for name in REPORTS}
    geometry = read(OUT/'nominal-supine-upright-target-projection.json')
    assert geometry['status']=='passed-static-target-projection' and not geometry['heldout_used']
    out = OUT/'nominal-supine-upright-trigger-coverage.json'
    assert not out.exists()
    write(out,dict(status='passed-saved-trigger-coverage',cohorts=result,source_sha256=sha(Path(__file__)),
        required_consecutive_samples=WINDOW,tilt_range_deg=[15.,45.],minimum_progress_deg=3.,
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Causal trigger arithmetic on development traces. Counts are not intervention outcomes.'))
    print('GETUP_UPRIGHT_TRIGGER', {name:{k:(len(x['candidate_failures']),len(x['protected_successes']))
        for k,x in v['results'].items()} for name,v in result.items()},flush=True)


if __name__=='__main__':
    main()
