"""Read back trigger decisions, projected targets and original recovery gates."""
from argparse import ArgumentParser
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from check_getup_supine_upright_geometry import targets
from common_scene_contract import compare_arrays, model_arrays
from getup_upright_intervention import Trigger
from record_io import write
from run_getup_upright_recovery import OUT, PLAN, ROLES, checked, evaluator, page
from standup_metrics import classify_release, projected_gravity
import verify_getup_supine_evaluation as verifier


def selection(reports,spec):
    outcome = {}
    for role,arm in reports.items():
        timely = [row['hold_start_seconds'] is not None for row in arm['timing']['episodes']]
        outcome[role] = np.asarray(timely)&np.asarray(arm['end_hold']['stable_by_environment'])
    left,right,candidate = (outcome[role] for role in ROLES)
    rescued = np.flatnonzero(~left&~right&candidate)
    lost = np.flatnonzero(left&right&~candidate)
    base = reports[ROLES[-1]]
    nominal = [row['requested_pose']=='supine' for row in base['timing']['episodes']]
    nominal_success = int((candidate&nominal).sum())
    gate = spec['intervention']['gate']
    passed = (base['all_motion_gates_passed'] and len(rescued)>=gate['minimum_recovered_joint_reference_failures_per_seed']
              and len(lost)<=gate['maximum_losses_among_joint_reference_successes_per_seed']
              and nominal_success>=gate['minimum_success_in_nominal_supine'])
    return dict(passed_paired_development_gate=bool(passed),common_failures_rescued=rescued.tolist(),
        common_successes_lost=lost.tolist(),original_pose_and_hold_gates=base['all_motion_gates_passed'],
        nominal_supine_timed_and_final=nominal_success)


def projection(trace,role,recorded,model,spec):
    p,design = spec['environment'],spec['intervention']
    geometry = read(ROOT/design['geometry_protocol'])
    names = spec['model_interface']['joint_names'].split(',')
    release = spec['prefix_controls']
    q = trace['quaternion']
    pose = classify_release(projected_gravity(q[release-1]),p['criteria']['release_alignment_min'])
    trigger = Trigger(pose,design['entry'])
    decisions = []
    for k in range(release+1,round(design['entry']['last_trigger_seconds']/.02)+1):
        selected = trigger.observe(k,q[k-1],trace['foot_contact'][k-1],trace['native_head_leg_contact'][k-1])
        decisions.extend((int(world),k) for world in selected)
    assert decisions==[(r['world'],r['decision_control']) for r in recorded],role
    if role!=ROLES[-1]:
        assert all('geometry' not in event for event in recorded)
        return dict(nominated=len(recorded),modified=0,maximum_target_delta_rad=0.)
    expected = trace['requested_targets'].copy()
    low,high = np.asarray(p['joint_limits'],np.float32).T
    weights = np.asarray(design['weights'],np.float32)
    for event in recorded:
        world,step = event['world'],event['decision_control']
        check = targets(model,dict(qpos=trace['qpos'][step-1,world]),geometry,False)
        original = event['geometry']
        assert check['feasible']==original['feasible']
        if not check['feasible']:
            continue
        assert check['joint_names']==original['joint_names']
        np.testing.assert_allclose(check['joint_delta_rad'],original['joint_delta_rad'],atol=1e-7,rtol=0)
        axes = [names.index(name) for name in check['joint_names']]
        window = slice(step,step+len(weights))
        current = expected[window,world]
        base = current[:,axes]
        desired = base+weights[:,None]*np.asarray(original['joint_delta_rad'],np.float32)
        current[:,axes] = np.where((base>=low[axes])&(base<=high[axes]),
            np.clip(desired,low[axes],high[axes]),base)
    slew = round(p['handoff_settings']['start_seconds']/.02)
    np.testing.assert_allclose(trace['applied_targets'][:slew],expected[:slew],rtol=0,atol=1e-6)
    changed = np.abs(expected[:slew]-trace['requested_targets'][:slew])
    maximum = float(changed.max())
    assert maximum<=design['target']['maximum_joint_delta_rad']+1e-7
    return dict(nominated=len(recorded),modified=int((changed.max(axis=(0,2))>1e-7).sum()),
        maximum_target_delta_rad=maximum)


def main(seed):
    from mjlab.scene import Scene
    from getup_knee_support_supply_recovery import shared

    spec,record = checked(seed)
    state = read(record/'evaluation.json')
    assert state['status']=='collected-review-required'
    events = read(record/'upright-trigger-events.json')
    assert tuple(events)==ROLES
    source_plant = read(ROOT/'configs/experiments/getup_knee_support_supply_recovery.json')['environment']
    cfg,_,_ = shared.prepare(source_plant)
    model = Scene(cfg.env.scene,device='cpu').compile()
    with np.load(record/'scene-static-fields.npz') as data:
        compare_arrays(model_arrays(model),{key:data[key] for key in data.files})
    original = verifier.check_trace
    replay = {}

    def check_trace(trace,p,addresses):
        role = ROLES[len(replay)]
        replay[role] = projection(trace,role,events[role],model,spec)
        if role!=ROLES[-1]:
            return original(trace,p,addresses)
        # Candidate targets have already been checked above. Check the shared
        # input and late slew invariants using the reconstructed nominal prefix.
        applied = trace['applied_targets'].copy()
        slew = round(p['handoff_settings']['start_seconds']/.02)
        applied[:slew] = trace['requested_targets'][:slew]
        return original(dict(trace,applied_targets=applied),p,addresses)

    with patch.object(verifier,'check_trace',check_trace):
        result = verifier.verify(spec,record,state,minimum_gain=spec['actual_supine_minimum_gain'])
    assert set(replay)==set(ROLES) and not torch.cuda.is_initialized()
    decision = selection(result['arms'],spec)
    write(OUT/f'seed{seed}-analysis.json',dict(result,selection=decision,intervention=replay,
        protocol_sha256=sha(PLAN),event_sha256=sha(record/'upright-trigger-events.json')))
    state['status'] = 'completed'
    write(record/'evaluation.json',state)
    view = read(record/'state.json')
    view.update(status='completed',purpose='仰卧转正配对试验 · 独立评分已完成',updated_at=datetime.now(evaluator.TZ).isoformat())
    with patch.object(live,'PAGE',page(seed)):
        live.publish(record,view)
    print('GETUP_UPRIGHT_RECOVERY_VERIFIED',seed,decision,replay,flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed',type=int)
    main(parser.parse_args().seed)
