"""Audit the fixed forward controller's yaw command and per-step joint correction limits."""

from collections import Counter

import numpy as np

from run_roll_stand_forward_diagnostic import MASTER, PLANS, REPORT, ROOT, put, read, require, sha
from roll_stand_walk_sequence import FORWARD


def check_route(run_id, evidence):
    folder = ROOT/'records/evaluations'/run_id
    plan, receipt, scoring = (read(folder/name) for name in ('protocol.json','guarded-evaluation.json','raw-score.json'))
    path = folder/'commands.npz'
    require(read(evidence)['run_id']==run_id and receipt['status']=='collected-review-required'
            and sha(path)==receipt['command_sha256'], 'Route command record changed')
    with np.load(path,allow_pickle=False) as source:
        commands = {name:source[name] for name in ('applied_targets','requested_targets','alternate_action','feedback_yaw','phase_used')}
    with np.load(folder/'startup.npz',allow_pickle=False) as source:
        home = source['walk_home']
    lo,hi = np.asarray(plan['walking']['joint_limits'],np.float32).T
    require(home.shape==lo.shape, 'Saved policy joint offset differs')
    cap,limit = plan['feedback']['maximum_yaw_input_rad_s'],plan['feedback']['maximum_joint_correction_rad']
    counts = Counter()
    details = []
    incomplete = 0
    steps = plan['forward_controls']
    for row in scoring['forward']:
        if not row['entered_forward']:
            continue
        world,begin = row['world'],row['forward_start']
        window = slice(begin,begin+steps)
        flags = commands['phase_used'][window,world] == FORWARD
        if len(flags)!=steps or not flags.all():
            incomplete += 1
            continue
        reference = commands['requested_targets'][window,world].clip(lo,hi)
        proposal = (commands['alternate_action'][window,world]+home).clip(lo,hi)
        diff = proposal-reference
        max_joint = np.abs(diff).argmax(axis=1)
        extreme = np.abs(diff).max(axis=1)
        multiplier = np.minimum(1.,limit/np.maximum(extreme,1e-8))
        applied = commands['applied_targets'][window,world]
        np.testing.assert_allclose(applied,reference+diff*multiplier[:,None],atol=2e-7,rtol=0)
        limited = extreme > limit+1e-7
        names = plan['walking']['joint_names']
        counts.update(names[i] for i in max_joint[limited])
        yaw_command = commands['feedback_yaw'][window,world]
        require(np.all(np.abs(yaw_command)<=cap+1e-6),'Saved alternate actor yaw exceeds the fixed bound')
        requested = np.quantile(extreme,[.5,.90,.95])
        details.append(dict(world=world, yaw_mean_abs_rad_s=row['yaw_error_rad_s'],
            yaw_cap_fraction=float(np.isclose(np.abs(yaw_command),cap,atol=1e-6,rtol=0).mean()),
            joint_correction_limited_fraction=float(limited.mean()),
            median_requested_max_joint_delta_rad=float(requested[0]),
            p90_requested_max_joint_delta_rad=float(requested[1]),
            p95_requested_max_joint_delta_rad=float(requested[2])))
    median = {key:float(np.median([item[key] for item in details])) if details else None for key in
        ('yaw_mean_abs_rad_s','yaw_cap_fraction','joint_correction_limited_fraction',
         'median_requested_max_joint_delta_rad','p90_requested_max_joint_delta_rad','p95_requested_max_joint_delta_rad')}
    return dict(run_id=run_id,evidence_sha256=sha(evidence),command_sha256=receipt['command_sha256'],
                assigned_worlds=plan['worlds'],entered=len(details)+incomplete,complete_forward=len(details),median=median,
                maximum_yaw_input_rad_s=cap,maximum_joint_correction_rad=limit,
                binding_joint_step_counts=dict(counts),worlds=details)


def main():
    master = read(MASTER)
    tracked = read(REPORT/'paired-analysis.json')
    require(set(master['order'])=={row['case'] for row in tracked['rows']},'Diagnostic cases differ')
    summary = {}
    for row in tracked['rows']:
        case = row['case']
        plan = read(PLANS/(case+'.json'))
        direct, routed = ROOT/row['evidence'],ROOT/row['post_roll_evidence']
        old = read(routed)
        proof = dict(direct=check_route(plan['run_id'],direct),post_roll=check_route(old['run_id'],routed))
        summary[case] = proof
        print('FORWARD_CONTROL_LIMITS',case,{key:value['median'] for key,value in proof.items()},flush=True)
    output = REPORT/'feedback-limit-readback.json'
    proof = dict(cases=summary,learning_updates=0,new_physics_substeps=0,hardware_io=False)
    if output.exists():
        require(read(output)==proof,'Saved limit readback differs')
    else:
        put(output,proof)


if __name__=='__main__':
    main()
