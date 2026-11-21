"""Check the signed lateral load-transfer geometry in saved backward starts."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from review_walk_backward_support_transfer import OUT as TRANSFER


OUT = TRANSFER.with_name('lateral-loading-review.json')
FIELDS = ('xy','qpos','root_quaternion','foot_positions','active')


def arrays(path, checksum):
    assert sha(path)==checksum
    with np.load(path) as source:
        return {name:source[name] for name in FIELDS}


def geometry(trace, event_control, world, side):
    # Match the previously frozen 140 ms pre-event interval, ending before
    # the first recorded contact of the single-support sequence.
    index = event_control-1
    selection = slice(index-8,index)
    assert trace['active'][selection,world].all()
    base = trace['qpos'][selection,world,:2]
    feet = trace['foot_positions'][selection,world,:,:2]
    quat = trace['root_quaternion'][selection,world]
    np.testing.assert_allclose(base,trace['xy'][selection,world],atol=2e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(quat,axis=1),1.,atol=2e-5,rtol=0)
    w,x,y,z = quat.T
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    lateral = np.stack((-np.sin(yaw),np.cos(yaw)),axis=-1)
    offset = np.sum((base-feet.mean(axis=1))*lateral,axis=1)
    separation = np.sum((feet[:,side]-feet[:,1-side])*lateral,axis=1)
    assert np.isfinite(np.stack((offset,separation))).all()
    signed = np.sign(separation[-1])*offset
    span_valid = np.min(np.abs(separation))>=.015 and np.sign(separation).min()==np.sign(separation).max()
    return dict(toward_event_foot_m=float(signed[-1]-signed[0]), toward_foot_at_end_m=float(signed[-1]),
        minimum_lateral_span_m=float(np.min(np.abs(separation))), lateral_span_valid=bool(span_valid))


def describe(rows):
    return dict(n=len(rows),valid_span=sum(row['lateral_span_valid'] for row in rows),
        toward_median_m=float(np.median([row['toward_event_foot_m'] for row in rows])) if rows else None,
        sign_positive=sum(row['toward_event_foot_m']>0 for row in rows),
        at_end_median_m=float(np.median([row['toward_foot_at_end_m'] for row in rows])) if rows else None)


def review(cohort):
    source = ROOT/cohort['source']
    assert sha(source)==cohort['source_sha256']
    proof = read(source)
    assert not proof['heldout_used'] and proof['learning_updates']==0
    folder = ROOT/proof['record']
    if 'shared_prefix_sha256' in proof:
        trace = arrays(folder/'trace-backward.npz',proof['trace_sha256']['backward'])
        prefix = arrays(folder/'trace-shared-prefix.npz',proof['shared_prefix_sha256'])
        trace = {key:np.concatenate((prefix[key],value)) for key,value in trace.items()}
    else:
        trace = arrays(folder/'trace-backward.npz',proof['source_sha256'])
    prior = cohort['roles']['backward']
    rows = []
    for row in prior['rows']:
        event = row['support']
        if event is None:
            continue
        side = ('left','right').index(event['side'])
        step = event['control']
        natural = geometry(trace,step,row['world'],side)
        stalled = [dict(world=world,**geometry(trace,step,world,side)) for world in prior['stall_worlds']]
        rows.append(dict(world=row['world'],support_side=event['side'],control=step,natural=natural,stalled=stalled))
    assert len(rows)==prior['sustained_support_by_velocity_block']
    groups = {}
    for name in ('left','right'):
        local = [row for row in rows if row['support_side']==name]
        groups[name] = dict(natural=describe([row['natural'] for row in local]),
                            stalled=describe([item for row in local for item in row['stalled']]))
    return dict(seed=cohort['seed'],groups=groups,rows=rows,source_sha256=sha(source))


def main():
    prior = read(TRANSFER)
    assert not OUT.exists() and prior['status']=='passed-saved-development-review' and not prior['heldout_used']
    cohorts = [review(cohort) for cohort in prior['cohorts']]
    assert sum(len(cohort['rows']) for cohort in cohorts)==59
    write(OUT,dict(status='passed-saved-geometry-review',cohorts=cohorts,transfer_review_sha256=sha(TRANSFER),
        source_sha256=sha(Path(__file__)),physical_substeps=0,heldout_used=False,
        scope='Root-to-midfoot lateral projection in body yaw frame, signed by the eventual support side. Geometric proxy; no contact force, causality, or runtime side prediction. Stalled windows may reuse the same world.'))
    print('BACKWARD_LATERAL_LOADING',[(cohort['seed'],cohort['groups']) for cohort in cohorts],flush=True)


if __name__=='__main__':
    main()
