"""Align saved official recovery trajectories before first head-leg contact.

Keep outcome labels, compare only actual supine releases, and match early
physical state without using the next target. CPU geometry locates possible
pairs at the saved poses. It does not estimate GPU contact forces or integrate.
"""
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from getup_coverage_metrics import evaluate
from recovery_end_hold import evaluate_end_hold
from standup_metrics import projected_gravity


REPORTS = ROOT / 'records/reports'
OUT = REPORTS / 'getup-contact-entry'
PARENTS = {
    '1016': 'getup-late-feasibility/original-analysis.json',
    '1050': 'getup-persistent-contact/pair/original-analysis.json',
    '1054': 'getup-knee-support/supply-recovery/analysis.json',
}
FIELDS = ('time', 'active', 'height', 'quaternion', 'angular_velocity', 'foot_contact',
          'joint_envelope_ok', 'native_head_leg_contact', 'joint_positions', 'qpos', 'qvel', 'requested_targets')
PAIRS = ('left_foot_collision', 'right_foot_collision', 'left_hip_yaw_collision', 'right_hip_yaw_collision')
RELEASE = 99  # Saved index 99 is control 100 and physical time 2.00 s.
MATCH = 114
LAST_EARLY = 250  # Exclusive saved index for the 5.00 s event window.


def read_arrays(path, expected, keys):
    assert sha(path) == expected, path
    with np.load(path) as stored:
        return {key: stored[key] for key in keys}


def source(seed, branch):
    proof_path = REPORTS / PARENTS[seed]
    proof = read(proof_path)
    record = ROOT / proof['record']
    protocol = record / 'protocol.json'
    plan = read(protocol)
    sources = {proof_path: sha(proof_path), protocol: sha(protocol)}
    if seed == '1054':
        assert proof['status'] == 'passed-integrity' and sha(protocol) == proof['protocol_sha256']
        plan = plan['environment']
        prefix, post = (record / name for name in ('trace-shared-prefix.npz', f'trace-{branch}.npz'))
        prefix_data = read_arrays(prefix, proof['shared_prefix_sha256'], FIELDS)
        post_data = read_arrays(post, proof['trace_sha256'][branch], FIELDS)
        trace = {key: np.concatenate((prefix_data[key], post_data[key])) for key in FIELDS}
        initial = record / 'initial.npz'
        with np.load(initial) as saved:
            trace['requested_pose'] = saved['requested_pose']
        sources.update({prefix: proof['shared_prefix_sha256'], post: proof['trace_sha256'][branch], initial: sha(initial)})
        scored = proof['scores'][branch]
        rows, hold = scored['rows'], scored['final_hold_by_world']
    else:
        assert proof['status'] == 'passed' and proof['continuous_no_physical_reset']
        path = record / 'coverage-trace.npz'
        trace = read_arrays(path, proof['trace_sha256'], (*FIELDS, 'requested_pose'))
        original = record / 'coverage.json'
        rows, hold = read(original)['episodes'], proof['end_hold']['stable_by_environment']
        sources.update({path: proof['trace_sha256'], original: sha(original)})
    assert plan['seed'] == int(seed) and trace['active'].all()
    assert plan['criteria']['step_dt'] == .02 and plan['criteria']['freeze_seconds'] == 2.
    recomputed = evaluate(trace, plan)['episodes']
    assert [(row['release_pose'], row['hold_start_seconds']) for row in rows] == [
        (row['release_pose'], row['hold_start_seconds']) for row in recomputed]
    assert evaluate_end_hold(trace, plan, recomputed)['stable_by_environment'] == hold
    scene = record / 'scene-static-fields.npz'
    sources[scene] = sha(scene)
    return plan, trace, rows, np.asarray(hold), scene, sources


def first_run(mask, samples):
    windows = np.lib.stride_tricks.sliding_window_view(mask[RELEASE:LAST_EARLY], samples)
    values = np.flatnonzero(windows.all(axis=1))
    return None if len(values) == 0 else RELEASE + int(values[0])


def match_before_contact(trace, gravity, eligible, cases, passed, neck):
    """One-to-one matches at fixed age using pose and velocity alone."""
    failures = np.flatnonzero(eligible & ~passed)
    successes = np.flatnonzero(eligible & passed)
    if not len(failures) or not len(successes):
        return [], int(len(failures))
    t = MATCH
    q = trace['joint_positions'][t]
    v = trace['qvel'][t, :, 6:]
    height = trace['height'][t]
    omega = trace['angular_velocity'][t]

    def delta(values):
        return values[failures, None] - values[successes][None, :]

    def rms(values):
        return np.sqrt(np.mean(delta(values) ** 2, axis=-1))

    scales = np.stack((abs(delta(height)) / .012,
                       np.linalg.norm(delta(gravity[t]), axis=-1) / (2*np.sin(np.deg2rad(8)/2)),
                       rms(q) / .16, abs(delta(q[:, neck])) / .12,
                       rms(v) / 2.5, abs(delta(omega[:, 1])) / 1.0), axis=-1)
    allowed = (scales <= 1.).all(axis=-1)
    ii, jj = linear_sum_assignment(np.where(allowed, np.linalg.norm(scales, axis=-1), 1e6))
    matches = []
    for i, j in zip(ii, jj, strict=True):
        if not allowed[i, j]:
            continue
        f, s = int(failures[i]), int(successes[j])
        requested = trace['requested_targets'][t + 1, [f, s]]
        matches.append(dict(failed_world=f, successful_world=s, scaled_pose_differences=scales[i, j].tolist(),
            failed_head_step=cases[f]['first_head_contact_step'], successful_head_step=cases[s]['first_head_contact_step'],
            next_target_failed_minus_success_deg=np.rad2deg(requested[0] - requested[1]).tolist()))
    return matches, int(len(failures))


def geom_reader(model):
    data = mujoco.MjData(model)
    head, *legs = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'robot/{name}')
                   for name in ('head_collision', *PAIRS))
    assert min(head, *legs) >= 0 and data.time == 0.

    def measure(qpos, qvel):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, qpos)
        assert data.time == 0.
        active = set()
        for contact in data.contact:
            if contact.dist >= 0 or head not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == head else contact.geom[0])
            if other in legs:
                active.add(PAIRS[legs.index(other)])
        distance = [mujoco.mj_geomDistance(model, data, head, leg, .2, None) for leg in legs]
        assert all(np.isfinite(distance))
        return dict(contact_pairs=sorted(active), signed_separation_m=dict(zip(PAIRS, distance, strict=True)))

    return measure


def summary(rows):
    def values(key):
        return [row[key] for row in rows if row[key] is not None]

    def stats(key):
        a = values(key)
        return dict(observed=len(a), median=None if not a else float(np.median(a)))

    pairs = Counter(pair for row in rows for pair in row['pairs_at_head_entry'])
    return dict(cases=len(rows), first_head_contact_step=stats('first_head_contact_step'),
        sustained_head_step=stats('sustained_head_step'), first_both_feet_step=stats('first_both_feet_step'),
        first_upright_step=stats('first_upright_step'), head_entry_minus_feet_step=stats('head_entry_minus_feet_step'),
        head_before_both=sum(k < 0 for k in values('head_entry_minus_feet_step')),
        geometric_agreement_at_entry_or_preceding_sample=sum(bool(row['pairs_at_head_entry']) for row in rows),
        pairs_at_head_entry=dict(pairs), neck_actual_at_match_deg=stats('neck_actual_at_match_deg'),
        neck_next_target_deg=stats('neck_next_target_deg'), pitch_speed_at_match_rad_s=stats('pitch_speed_at_match_rad_s'),
        trunk_tilt_at_match_deg=stats('trunk_tilt_at_match_deg'),
        closest_head_leg_gap_at_match_mm=stats('closest_head_leg_gap_at_match_mm'))


def analyze(plan, trace, rows, hold, measure, joints):
    gravity = projected_gravity(trace['quaternion'])
    tilt = np.rad2deg(np.arccos((-gravity[..., 2]).clip(-1, 1)))
    contact = trace['native_head_leg_contact']
    both = trace['foot_contact'].all(axis=-1)
    upright = both & (tilt <= plan['criteria']['tilt_max_deg'])
    supine = np.asarray([row['release_pose'] == 'supine' for row in rows])
    timely = np.asarray([row['hold_start_seconds'] is not None for row in rows])
    passed = timely & hold
    assert abs(trace['time'][MATCH] - 2.3) < 1e-8
    neck = joints.index('neck_pitch')
    cases = {}
    for world in np.flatnonzero(supine):
        head = first_run(contact[:, world], 1)
        long_head = first_run(contact[:, world], 3)
        feet = first_run(both[:, world], 5)
        stand = first_run(upright[:, world], 5)
        frames = {'match': MATCH}
        if head is not None:
            frames.update(before_contact=max(RELEASE, head - 1), contact=head)
        if feet is not None:
            frames['both_feet'] = feet
        geom = {phase: measure(trace['qpos'][k, world], trace['qvel'][k, world]) for phase, k in frames.items()}
        present = set().union(*(geom[phase]['contact_pairs'] for phase in ('before_contact', 'contact') if phase in geom))
        cases[int(world)] = dict(world=int(world), timely_and_final=bool(passed[world]),
            first_head_contact_step=None if head is None else head + 1,
            sustained_head_step=None if long_head is None else long_head + 1,
            first_both_feet_step=None if feet is None else feet + 1,
            first_upright_step=None if stand is None else stand + 1,
            head_entry_minus_feet_step=None if head is None or feet is None else head - feet,
            pairs_at_head_entry=sorted(present), kinematic_geometry=geom,
            neck_actual_at_match_deg=float(np.rad2deg(trace['joint_positions'][MATCH, world, neck])),
            neck_next_target_deg=float(np.rad2deg(trace['requested_targets'][MATCH + 1, world, neck])),
            pitch_speed_at_match_rad_s=float(trace['angular_velocity'][MATCH, world, 1]),
            trunk_tilt_at_match_deg=float(tilt[MATCH, world]),
            closest_head_leg_gap_at_match_mm=1000*min(geom['match']['signed_separation_m'].values()))
    eligible = supine & ~contact[RELEASE:MATCH + 1].any(axis=0)
    matches, available = match_before_contact(trace, gravity, eligible, cases, passed, neck)
    differences = [row['next_target_failed_minus_success_deg'] for row in matches]
    stats = {label: summary([row for row in cases.values() if row['timely_and_final'] == state])
             for label, state in (('success', True), ('failure', False))}
    return dict(by_outcome=stats, matched_pairs=matches, uncontacted_failure_candidates=available,
                median_next_target_failed_minus_success_deg=None if not differences else
                dict(zip(joints, np.median(differences, axis=0).tolist(), strict=True)), cases=list(cases.values()))


def main():
    import onnx
    import torch
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays, compare_arrays
    from getup_knee_support_supply_recovery import shared

    latest = read(ROOT / 'configs/experiments/getup_knee_support_supply_recovery.json')['environment']
    cfg, _, _ = shared.prepare(latest)
    model = Scene(cfg.env.scene, device='cpu').compile()
    measure = geom_reader(model)
    cohorts, sources = {}, {}
    for seed, branch in (('1016', 'original'), ('1050', 'original'), ('1054', 'reference-a'), ('1054', 'reference-b')):
        plan, trace, rows, hold, scene, fingerprints = source(seed, branch)
        with np.load(scene) as stored:
            compare_arrays(model_arrays(model), {key: stored[key] for key in stored.files})
        model_path = ROOT / plan['model']
        assert sha(model_path) == plan['model_sha256'] == latest['model_sha256']
        joints = next(item.value for item in onnx.load(model_path).metadata_props if item.key == 'joint_names').split(',')
        assert len(joints) == 14
        for name in ('neck_pitch', 'left_knee', 'right_knee', 'left_hip_pitch', 'right_hip_pitch'):
            address = model.jnt_qposadr[model.joint(f'robot/{name}').id]
            np.testing.assert_array_equal(trace['qpos'][MATCH, :, address], trace['joint_positions'][MATCH, :, joints.index(name)])
        key = seed if branch != 'reference-b' else '1054-repeat'
        cohorts[key] = analyze(plan, trace, rows, hold, measure, joints)
        sources.update(fingerprints)
        compact = {k: {name: row[name] for name in ('cases', 'first_head_contact_step', 'first_both_feet_step',
                   'head_before_both', 'pairs_at_head_entry', 'neck_next_target_deg', 'pitch_speed_at_match_rad_s')}
                   for k, row in cohorts[key]['by_outcome'].items()}
        print('CONTACT_ENTRY', key, compact, 'matches', len(cohorts[key]['matched_pairs']), flush=True)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'analysis.json', dict(status='passed-saved-data-only', seeds=[1016, 1050, 1054],
        repeat='1054-repeat shares seed and prefix, excluded from independent cohort counts',
        source_sha256={p.relative_to(ROOT).as_posix(): h for p, h in sources.items()},
        source_actor_sha256=latest['model_sha256'], script_sha256=sha(Path(__file__)), cohorts=cohorts,
        matching_time_seconds=2.3, following_control_time_seconds=2.32,
        matching_limits=dict(height_m=.012, gravity_angle_deg=8, joint_rms_rad=.16, neck_rad=.12,
                             joint_velocity_rms_rad_s=2.5, pitch_speed_rad_s=1.),
        event_start_times='Saved step k is end of control interval, .02*k seconds; sustained head=3, feet/upright=5 samples',
        geometry_scope='Verified common CPU model at exact archived qpos/qvel. Contact sensor is last substep before post-control qpos. No load or integrated dynamics inferred.',
        new_physical_steps=0, learning_updates=0, heldout_used=False))


if __name__ == '__main__':
    main()
