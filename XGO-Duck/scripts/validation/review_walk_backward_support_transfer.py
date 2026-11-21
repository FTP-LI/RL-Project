"""Align saved walking targets to sustained single-foot support in development only."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


DRAFT = ROOT/'configs/experiments/common_walk_backward_support_transfer_draft.json'
ONSET = ROOT/'configs/experiments/common_walk_backward_onset.json'
PULSE = ROOT/'configs/experiments/common_walk_backward_start_pulse.json'
OUT = ROOT/'records/reports/common-walk-backward-start-pulse/support-transfer-review.json'
KEYS = ('contact','active','foot_positions','joint_positions','requested_joint_targets','applied_joint_targets')
AXES = ('hip_roll','hip_pitch','knee','ankle')


def archive(path):
    with np.load(path) as data:
        return {key:data[key] for key in KEYS}


def first_support(trace,world,begin,stop):
    """Return the first onset of four consecutive samples on the same foot."""
    contact = trace['contact'][begin:stop,world]
    active = trace['active'][begin:stop,world]
    for offset in range(len(active)-3):
        if active[offset:offset+4].all() and contact[offset].sum() == 1 and (contact[offset:offset+4] == contact[offset]).all():
            return begin+offset,int(np.flatnonzero(contact[offset])[0])
    return None


def before_event(trace,world,event,side,indices):
    """Use seven completed states, ending 20 ms before the support marker."""
    early, late = event-8,event-1
    assert early >= 0 and trace['active'][early:late+1,world].all()
    other = 1-side
    result = {}
    for label,key in (('actual','joint_positions'),('requested','requested_joint_targets'),('applied','applied_joint_targets')):
        values = trace[key][[early,late],world]
        delta = (values[1]-values[0])[indices]
        result[label] = dict(stance_delta_rad=delta[side].tolist(),swing_delta_rad=delta[other].tolist(),
                             asymmetry_excursion_rad=float(np.max(np.abs(np.abs(delta[side])-np.abs(delta[other])))))
    heights = trace['foot_positions'][[early,late],world,:,2]
    result['relative_swing_rise_m'] = float((heights[1,other]-heights[1,side])-(heights[0,other]-heights[0,side]))
    return result


def inspect(trace,worlds,stalls,indices):
    rows = []
    for item in worlds:
        onset = item['onset_block_end_seconds'] or 0
        if onset <= 2 or not item['survived']:
            continue
        world = item['world']
        earlier = first_support(trace,world,10,100)
        event = first_support(trace,world,100,onset*50)
        if event is None:
            rows.append(dict(world=world,qualifying_block_end=onset,support=None,earlier_support=earlier is not None))
            continue
        step,side = event
        before = before_event(trace,world,step,side,indices)
        matched = [before_event(trace,index,step,side,indices) for index in stalls]
        rows.append(dict(world=world,qualifying_block_end=onset,support=dict(control=step+1,time=(step+1)*.02,
            side=('left','right')[side],confirmed_control=step+4),earlier_support=earlier is not None,
            preceding=before,aligned_stalls=matched))
    by_side = {}
    for side in ('left','right'):
        selected = [row for row in rows if row['support'] and row['support']['side']==side]
        reference = [item for row in selected for item in row['aligned_stalls']]
        def median(values,key):
            if not values:
                return None
            features = [v[key]['asymmetry_excursion_rad'] if key != 'relative_swing_rise_m' else v[key] for v in values]
            return float(np.median(features))
        keys = ('actual','requested','applied','relative_swing_rise_m')
        by_side[side] = dict(starters=len(selected),aligned_stall_windows=len(reference),
            starter_medians={k:median([row['preceding'] for row in selected],k) for k in keys},
            stall_medians={k:median(reference,k) for k in keys})
    return dict(delayed=len(rows),sustained_support_by_velocity_block=sum(row['support'] is not None for row in rows),
                support_before_two_seconds=sum(row['earlier_support'] for row in rows),
                stall_worlds=stalls,by_side=by_side,rows=rows)


def review(source):
    proof = read(source)
    assert proof['status']=='passed-integrity' and not proof['heldout_used'] and proof['learning_updates']==0
    paired = 'scores' in proof
    plan = PULSE if paired else ONSET
    spec = read(plan)
    assert proof['protocol_sha256']==sha(plan)
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==sha(plan)
    names = spec['environment']['joint_names']
    indices = np.asarray([[names.index(f'{side}_{axis}') for axis in AXES] for side in ('left','right')])
    if paired:
        prefix = record/'trace-shared-prefix.npz'
        assert sha(prefix)==proof['shared_prefix_sha256']
        prefix = archive(prefix)
        roles = ('backward','reference-repeat')
        stalls = proof['gate']['common_near_static']
    else:
        roles = ('backward',)
        stalls = proof['analysis']['near_static_worlds']
    result = {}
    for role in roles:
        path = record/f'trace-{role}.npz'
        assert sha(path)==(proof['trace_sha256'][role] if paired else proof['source_sha256'])
        trace = archive(path)
        if paired:
            trace = {key:np.concatenate((prefix[key],trace[key])) for key in KEYS}
        assert trace['active'].shape[0] == 300 and trace['contact'].shape[-1]==2
        assert all(trace['active'][:,world].all() for world in stalls)
        scores = proof['scores'][role] if paired else proof['analysis']
        result[role] = inspect(trace,scores['worlds'],stalls,indices)
    return dict(seed=proof['seed'],source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),roles=result)


if __name__ == '__main__':
    draft = read(DRAFT)
    assert not draft['heldout_used_to_select_design'] and not OUT.exists()
    rows = [review(ROOT/path) for path in draft['development_sources']]
    assert [row['seed'] for row in rows]==[1071,1072,1073,1074]
    write(OUT,dict(status='passed-saved-development-review',cohorts=rows,draft_sha256=sha(DRAFT),
         script_sha256=sha(Path(__file__)),axes=list(AXES),physical_substeps=0,heldout_used=False,
         scope='Support defined by simulator contact flags. Matched-time stalled windows are descriptive. The two references of one seed are not independent.'))
    print('BACKWARD_SUPPORT_TRANSFER',[(r['seed'],{k:(v['delayed'],v['sustained_support_by_velocity_block'],v['by_side'])
          for k,v in r['roles'].items()}) for r in rows],flush=True)
