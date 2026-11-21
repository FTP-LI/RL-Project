"""Check the prospective neck cue on saved roll development observations."""
from pathlib import Path

import numpy as np
from mjlab.scene import Scene

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from review_roll_head_motor_envelope import reconstruct_internal_target
from review_roll_head_support_geometry import Geometry, OUT, prepare
from roll_metrics import progress


DRAFT = ROOT / 'configs/experiments/roll_neck_support_draft.json'


def check(trace, old_geometry, geometry, motor, names, draft):
    phase, _, rotations = progress(trace)
    arm = draft['intervention']
    angle = np.rad2deg(phase)
    both_feet = np.all(trace['foot_contact'], axis=-1)
    prior_feet = np.zeros_like(both_feet)
    prior_feet[1:] = np.logical_or.accumulate(both_feet[:-1], axis=0)
    neck = names.index('neck_pitch')
    velocity = trace['qvel'][:, :, geometry.dofs[0]]
    lower, upper = geometry.ranges[0]
    requests = trace['applied_targets'][:, :, neck]
    upcoming = np.vstack([requests[1:], requests[-1:]])
    lo, hi = arm['forward_phase_degrees']
    selected = (trace['active'] & prior_feet & trace['head_contact']
                & ~np.any(trace['foot_contact'], axis=-1)
                & (angle >= lo) & (angle <= hi)
                & (np.abs(rotations[..., 2, 1]) <= arm['maximum_body_lateral_projection'])
                & (trace['angular_velocity'][..., 1] > arm['forward_body_pitch_rate_above_rad_s'])
                & (velocity < arm['neck_rate_below_rad_s']) & (upcoming >= lower) & (upcoming <= upper))
    selected[-1] = False  # The final state has no following request.
    k, world = np.where(selected)
    assert len(k) and np.all(k > 0)
    baseline = upcoming[k, world].astype(np.float64)
    candidate = np.minimum(baseline + arm['maximum_positive_target_offset_rad'], upper)
    assert ((candidate >= baseline) & (candidate <= upper)).all()

    dt = draft['physics_dt']
    max_change = dt * motor.model.max_velocity.value
    smooth = reconstruct_internal_target(requests, max_change)[k, world].astype(np.float64)
    # Compare the first available request at a fixed archived pose, after one
    # unchanged substep with the previous request. The pose itself is not integrated.
    smooth = np.clip(requests[k, world], smooth - max_change, smooth + max_change)
    q, dq = trace['qpos'][k, world, geometry.addresses[0]], velocity[k, world]
    torque_delta = {}
    for voltage in (7.0, 7.7):
        motor.vin = voltage
        torques = []
        for target in (baseline, candidate):
            motor.q_target_smooth = smooth.copy()
            control = motor.compute_control(target, q, dq, dt)
            torques.append(motor.compute_torque(control, True, q, dq))
        torque_delta[str(voltage)] = torques[1] - torques[0]
        assert (torque_delta[str(voltage)] >= -1e-12).all()

    earliest = []
    for row in old_geometry['rows']:
        if row['success']:
            continue
        i = row['world']
        cutoff = round(row['before_air']['time_s'] / draft['control_dt'])
        hits = np.flatnonzero(selected[:cutoff+1, i])
        if not hits.size:
            earliest.append(dict(world=i, reached=False))
            continue
        j = int(hits[0])
        parts = geometry.measure(trace['qpos'][j, i], trace['qvel'][j, i])
        earliest.append(dict(world=i, reached=True, first_seconds=float(trace['time'][j]),
            before_departure_seconds=float(trace['time'][cutoff+1]-trace['time'][j]),
            neck_velocity_rad_s=float(velocity[j, i]), neck_lift_mm_s=parts['neck_contribution_mm_s'],
            increasing_neck_angle_reduces_clearance=parts['neck_contribution_mm_s'] > 0))
    floor, nominal = torque_delta['7.0'], torque_delta['7.7']
    passed_effect = np.minimum(floor, nominal) >= draft['offline_gate']['positive_neck_torque_increase_nm']
    return dict(affected_worlds=len(np.unique(world)), affected_samples=len(k),
        effective_samples=int(passed_effect.sum()), earliest_head_departure=earliest,
        median_increment_nm={'7.0': float(np.median(floor)), '7.7': float(np.median(nominal))},
        maximum_offset_rad=float((candidate-baseline).max()), first_selected_seconds=float(trace['time'][k].min()))


def main():
    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and not draft['heldout_use']
    evidence_path = ROOT / draft['development_evidence']
    evidence = read(evidence_path)
    assert evidence['delay_and_target_replay_exact'] and not evidence['heldout_used']
    for name, digest in evidence['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    old = read(OUT / 'development-head-support-geometry.json')
    cohorts = {}
    for role in ('development-a', 'development-b'):
        folder = ROOT / read(OUT / f'{role}-analysis.json')['evaluator']
        cfg = prepare(read(folder / 'protocol.json'))
        scene = Scene(cfg.scene, device='cpu')
        geometry = Geometry(scene.compile())
        (actuator,) = scene['robot'].actuators
        names = read(folder / 'evaluation.json')['joint_names']
        with np.load(folder / 'forward-roll-trace.npz') as stored:
            trace = {key: stored[key] for key in ('active', 'time', 'quaternion', 'angular_velocity', 'qpos', 'qvel',
                     'foot_contact', 'head_contact', 'ground_contact', 'applied_targets')}
        cohorts[role] = check(trace, old['cohorts'][role], geometry, actuator._bam_model.actuator, names, draft)

    rows = [row for values in cohorts.values() for row in values['earliest_head_departure']]
    reached = sum(row['reached'] for row in rows)
    lever = sum(row.get('increasing_neck_angle_reduces_clearance', False) for row in rows)
    affected = sum(values['affected_samples'] for values in cohorts.values())
    effect = sum(values['effective_samples'] for values in cohorts.values())
    gate = draft['offline_gate']
    proceed = (reached >= gate['minimum_reached_head_departure_failures']
        and lever / max(reached, 1) >= gate['minimum_first_event_support_lever_fraction']
        and effect / affected >= gate['minimum_fraction_samples_increasing_positive_neck_torque'])
    result = dict(status='passed-saved-data-only', offline_gate_passed=proceed,
        head_departure_failures=len(rows), reached_before_departure=reached, support_lever=lever,
        total_affected=affected, positive_torque_increment_gate_count=effect, cohorts=cohorts,
        draft_sha256=sha(DRAFT), motor_envelope_sha256=sha(evidence_path), script_sha256=sha(Path(__file__)),
        heldout_used=False, new_physical_steps=0, learning_updates=0,
        scope='Causal state cue and fixed-state motor calculation only. Exact simulated contact is required.')
    write(OUT / 'neck-support-timing-preflight.json', result)
    print({key: value for key, value in result.items() if key != 'cohorts'}, flush=True)


if __name__ == '__main__':
    main()
