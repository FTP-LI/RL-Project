"""Schedule a small neck-pitch offset over the measured support transfer."""

import json

import numpy as np

from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_stand_walk_sequence import ROLL


REPORT = ROOT/'records/reports/roll-forward-input-integration'
DRAFT = ROOT/'configs/experiments/roll_forward_input_integration/neck_support_probe_draft.json'


def neck_pitch_offset(controls, schedule):
    """Ramp in before the gap and return to the actor target after support transfer."""
    a,b,c,d = (schedule[key] for key in ('last_zero_before_active_control','full_offset_first_control',
                                         'full_offset_last_control','first_zero_after_active_control'))
    return np.clip(np.minimum((controls-a)/(b-a),(d-controls)/(d-c)),0.,1.)


def apply_neck_pitch(targets, controls, phase, joint_index, limits, schedule, magnitude):
    result = targets.copy()
    desired = targets[...,joint_index]
    offset = neck_pitch_offset(controls,schedule)*magnitude
    legal = np.clip(desired+offset,*limits)
    result[...,joint_index] = np.where((phase==ROLL)&(offset!=0.),legal,desired)
    return result


def main():
    draft = read(DRAFT)
    source = ROOT/draft['source']
    assert sha(source)==draft['source_sha256'] and not any(draft[key] for key in (
        'simulation_authorized_by_draft','reserved_seed_used','learning_updates','hardware_io'))
    case_paths = {
        1247: REPORT/'s1247-friction-low-low-friction/trajectory-readback.json',
        1248: REPORT/'support-window-force/force-readback.json',
    }
    results = {}
    for seed,evidence in case_paths.items():
        proof = read(evidence)
        folder = ROOT/'records/evaluations'/proof['run_id']
        protocol = folder/'protocol.json'
        plan = read(protocol)
        assert sha(protocol)==proof['protocol_sha256'] and plan['seed']==seed and plan['worlds']==draft['plant']['worlds_per_seed']
        if seed==1247:
            receipt = ROOT/proof['source_receipt']
            assert sha(receipt)==proof['source_receipt_sha256']
            files = {folder/'shared'/name:digest for name,digest in read(receipt)['prefix_sha256'].items()}
        else:
            receipt = folder/'guarded-evaluation.json'
            saved = read(receipt)
            files = {folder/'trace.npz':saved['trace_sha256'],folder/'commands.npz':saved['command_sha256']}
        assert all(sha(path)==digest for path,digest in files.items())
        data, trace = [arrays(next(path for path in files if path.name==name)) for name in ('commands.npz','trace.npz')]
        original = data['actual_motor_target']
        phase = data['phase_used']
        np.testing.assert_array_equal(phase,trace['phase_next'][:-1])
        order = plan['walking']['joint_names']
        joint = order.index(draft['offset']['joint'])
        limits = plan['walking']['joint_limits'][joint]
        controls = np.arange(1,len(phase)+1)[:,None]
        original_roles = [apply_neck_pitch(original,controls,phase,joint,limits,draft['offset'],draft['roles'][role])
                          for role in ('reference-a','reference-b')]
        for values in original_roles:
            np.testing.assert_array_equal(values,original)
        candidate = apply_neck_pitch(original,controls,phase,joint,limits,draft['offset'],draft['roles']['neck-pitch-positive'])
        other = np.arange(len(order))!=joint
        np.testing.assert_array_equal(candidate[...,other],original[...,other])
        delta = candidate[...,joint]-original[...,joint]
        times = np.flatnonzero((delta!=0.).any(axis=1))+1
        mask = delta!=0.
        assert times.min()==74 and times.max()==99 and not mask[phase!=ROLL].any()
        assert delta.min()>=0. and delta.max()<=.020001
        selected = candidate[...,joint][mask]
        assert selected.min()>=limits[0]-1e-6 and selected.max()<=limits[1]+1e-6
        results[seed] = dict(affected_worlds=int(mask.any(axis=0).sum()),affected_control_worlds=int(mask.sum()),
            first_changed_control=int(times.min()),last_changed_control=int(times.max()),maximum_target_offset_rad=float(delta.max()),
            maximum_original_head_depth_m=float(trace['head_penetration_m'][:-1][mask].max()),
            original_head_depth_limit_m=plan['roll']['criteria']['head_penetration_max_m'],
            source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (evidence,protocol,receipt,*files)})
    target = REPORT/'support-window-force/neck-support-target-preflight.json'
    target.write_text(json.dumps(dict(draft_sha256=sha(DRAFT),readback=results,additional_physical_steps=0,
        learning_updates=0,heldout_used=False,hardware_io=False,scope='Archived target projection before dynamic test'),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ROLL_NECK_TARGET_PREFLIGHT', {seed:(row['affected_worlds'],row['affected_control_worlds']) for seed,row in results.items()})


if __name__=='__main__':
    main()
