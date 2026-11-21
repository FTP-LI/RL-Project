"""Find and replicate a pre-contact action difference on immutable saved releases."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha


PLAN = ROOT / 'configs/experiments/getup_archived_policy_teacher_timing.json'
OUT = ROOT / 'records/reports/getup-archived-policy-comparison/teacher-timing.json'
GROUPS = ('teacher_only', 'parent_only', 'both_pass', 'both_fail')


def success(arm, expected_pose=None):
    episodes = arm['timely']['episodes']
    actual = np.array([row['release_pose'] for row in episodes])
    if expected_pose is not None:
        np.testing.assert_array_equal(actual, expected_pose)
    assert [row['env_index'] for row in episodes] == list(range(len(episodes)))
    passed = np.array([row['hold_start_seconds'] is not None for row in episodes]) & np.asarray(arm['final']['stable_by_environment'])
    assert int(passed.sum()) == arm['combined']['overall']['timely_holds']
    return passed, actual


def native_trace(path, proof, controls, worlds):
    assert sha(path) == proof
    fields = ('foot_contact', 'native_head_leg_contact', 'joint_positions', 'quaternion', 'actor_input',
              'raw_policy_actions', 'requested_targets', 'applied_targets')
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['time'][:len(controls)], controls * .02, rtol=0, atol=1e-12)
        trace = {name: archive[name][:len(controls), worlds] for name in fields}
    assert all(np.isfinite(a).all() for a in trace.values())
    return trace


def actor(spec, role):
    p = spec['policies'][role]
    path = ROOT / p['model']
    assert sha(path) == p['model_sha256']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    assert session.get_inputs()[0].shape == [1, 61] and float(meta['action_scale']) == 1.
    assert session.get_inputs()[0].name == 'obs'
    return session, meta['joint_names'].split(',')


def predict(actor, inputs):
    flat = inputs.reshape(-1, 61)
    return np.concatenate([actor.run(None, {'obs': row[None]})[0] for row in flat]).reshape(*inputs.shape[:2], 14)


def upright_support(trace, degrees, width):
    quat = trace['quaternion'].astype(np.float64)
    vertical = 1 - 2 * (quat[..., 1]**2 + quat[..., 2]**2)
    stable = (trace['foot_contact'].all(axis=-1) & ~trace['native_head_leg_contact'] &
              (vertical >= np.cos(np.deg2rad(degrees))))
    runs = np.lib.stride_tricks.sliding_window_view(stable, width, axis=0).all(axis=-1)
    return np.where(runs.any(axis=0), np.argmax(runs, axis=0), -1)


def prepare(seed, plan, pool):
    result = pinned(plan['development_results'][str(seed)])
    assert result['seed'] == seed and not result['heldout_used'] and result['learning_updates'] == 0
    record = ROOT / result['record']
    path = record / 'protocol.json'
    assert sha(path) == result['protocol_sha256']
    spec = read(path)
    assert spec['prefix_controls'] == 100 and spec['environment']['criteria']['step_dt'] == .02
    roles = (plan['parent_role'], plan['teacher_role'])
    arms = [result['arms'][role] for role in roles]
    parent_pass, actual = success(arms[0])
    teacher_pass, _ = success(arms[1], actual)
    worlds = np.flatnonzero(actual == 'supine')
    groups = dict(zip(GROUPS, (teacher_pass & ~parent_pass, parent_pass & ~teacher_pass, teacher_pass & parent_pass,
                              ~parent_pass & ~teacher_pass), strict=True))
    groups = {name: values[worlds] for name, values in groups.items()}
    assert np.stack(list(groups.values())).sum(axis=0).min() == np.stack(list(groups.values())).sum(axis=0).max() == 1
    controls = np.arange(101, round(plan['support_window_end_seconds'] / .02) + 1)
    traces = [pool.submit(native_trace, record / f'trace-{role}.npz', arm['trace_sha256'], controls, worlds)
              for role, arm in zip(roles, arms, strict=True)]
    parent, teacher = (job.result() for job in traces)
    prefix_path = record / 'trace-shared-prefix.npz'
    release_paths = [record / f'release-{role}.npz' for role in roles]
    with np.load(prefix_path, allow_pickle=False) as prefix, np.load(release_paths[0], allow_pickle=False) as a, \
            np.load(release_paths[1], allow_pickle=False) as b:
        for key in ('qpos', 'qvel', 'actor'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(prefix['qpos'][-1], a['qpos'])
        initial_head = prefix['native_head_leg_contact'][-1, worlds]
    np.testing.assert_array_equal(parent['actor_input'][0], teacher['actor_input'][0])
    head = parent['native_head_leg_contact'] | teacher['native_head_leg_contact']
    feet = (parent['foot_contact'] != teacher['foot_contact']).any(axis=-1)
    width = plan['minimum_consecutive_command_controls']
    assert controls[width - 1] * .02 <= plan['command_window_end_seconds']
    eligible = ~initial_head & ~(head[:width] | feet[:width]).any(axis=0)
    parent_session, joints = actor(spec, roles[0])
    teacher_session, teacher_joints = actor(spec, roles[1])
    assert joints == teacher_joints and len(joints) == 14
    chosen = np.flatnonzero(eligible)
    x, y = parent['actor_input'][:width, chosen], teacher['actor_input'][:width, chosen]
    a, b = predict(parent_session, x), predict(teacher_session, x)
    teacher_actual = predict(teacher_session, y)
    tolerance = plan['joint_gate']['maximum_same_input_reproduction_error_rad']
    err = max(float(abs(a - parent['raw_policy_actions'][:width, chosen]).max()),
              float(abs(teacher_actual - teacher['raw_policy_actions'][:width, chosen]).max()))
    assert err <= tolerance
    sent = np.asarray([trace['applied_targets'][:width, chosen] for trace in (parent, teacher)])
    raw = np.asarray([trace['requested_targets'][:width, chosen] for trace in (parent, teacher)])
    qdelta = (teacher['joint_positions'] - parent['joint_positions'])[:, chosen]
    limits = np.asarray(spec['environment']['joint_limits'])
    legal = ((sent >= limits[:, 0] - 1e-7) & (sent <= limits[:, 1] + 1e-7)).all(axis=(0, 1))
    cut = plan['contact_confirmation_controls']
    theta = spec['environment']['criteria']['tilt_max_deg']
    support = [upright_support(trace, theta, cut)[chosen] for trace in (parent, teacher)]
    return dict(seed=seed, worlds=worlds, eligible=eligible, groups=groups, joints=joints, err=err, support=support,
                sent=sent[1] - sent[0], requested=raw[1] - raw[0], same_input=b - a, legal=legal, response=qdelta,
                trace_sha256={role: arm['trace_sha256'] for role, arm in zip(roles, arms, strict=True)},
                shared_prefix_sha256=sha(prefix_path), command_controls=controls[:width].tolist())


def metric(data, joint, sign, plan):
    gate = plan['joint_gate']
    groups = {name: mask[data['eligible']] for name, mask in data['groups'].items()}
    delta = sign * data['sent'][..., joint].mean(axis=0)
    same = sign * data['same_input'][..., joint].mean(axis=0)
    parent_support, support = data['support']
    width = len(data['command_controls'])
    response = sign * data['response'][width:width + plan['maximum_followup_controls'], :, joint]
    response_hit = response >= gate['minimum_signed_measured_response_rad']
    response_at = width + response_hit.argmax(axis=0)
    earlier = response_hit.any(axis=0) & (support >= response_at + plan['minimum_response_lead_controls'])
    earlier &= (parent_support < 0) | (support < parent_support)
    samples = {}
    for name, mask in groups.items():
        samples[name] = dict(total=int(data['groups'][name].sum()), eligible=int(mask.sum()),
            median_target_rad=float(np.median(delta[mask])) if mask.any() else None,
            median_same_input_rad=float(np.median(same[mask])) if mask.any() else None,
            consistent=int((delta[mask] > 0).sum()), early_response_and_support=int(earlier[mask].sum()))
    selected = groups['teacher_only']
    primary = samples['teacher_only']
    other_medians = [samples[name]['median_target_rad'] for name in GROUPS[1:]]
    gap = min(primary['median_target_rad'] - value for value in other_medians) if selected.any() and None not in other_medians else None
    sent, requested = (data[key][:, selected, joint] for key in ('sent', 'requested'))
    max_delta = float(np.max(abs(sent))) if sent.size else None
    projection = float(np.sum(abs(sent - requested)) / max(1e-12, np.sum(abs(requested)))) if sent.size else None
    n = primary['eligible']
    tests = dict(coverage=n >= gate['minimum_teacher_only_examples'],
        sent_delta=bool(n and primary['median_target_rad'] >= gate['minimum_median_signed_target_delta_rad']),
        same_input=bool(n and primary['median_same_input_rad'] >= gate['minimum_median_signed_same_input_delta_rad']),
        sign=bool(n and primary['consistent'] / n >= gate['minimum_signed_target_consistency']),
        support_lead=bool(n and primary['early_response_and_support'] / n >= gate['minimum_positive_response_and_support_fraction']),
        specificity=bool(gap is not None and gap >= gate['minimum_teacher_vs_each_control_median_gap_rad']),
        bounds=bool(n and max_delta <= gate['maximum_absolute_sent_delta_rad'] and data['legal'][selected, joint].all()),
        projection=bool(n and projection <= gate['maximum_parent_teacher_projection_loss_fraction']))
    return dict(joint=data['joints'][joint], joint_index=joint, sign=sign, groups=samples, minimum_control_gap_rad=gap,
                maximum_target_delta_rad=max_delta, projection_loss_fraction=projection, tests=tests, passed=all(tests.values()))


if __name__ == '__main__':
    assert not OUT.exists()
    p = read(PLAN)
    assert not any(p[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    pinned(p['draft'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        discovery, replication = [prepare(p[key], p, pool) for key in ('discovery_seed', 'replication_seed')]
    assert discovery['joints'] == replication['joints']
    candidates = [metric(discovery, j, sign, p) for j in range(14) for sign in (1, -1)]
    ranked = sorted(candidates, key=lambda row: (row['passed'],
        row['minimum_control_gap_rad'] if row['minimum_control_gap_rad'] is not None else -np.inf,
        row['groups']['teacher_only']['median_target_rad'] or 0, -row['joint_index']), reverse=True)
    chosen = ranked[0]
    confirm = metric(replication, chosen['joint_index'], chosen['sign'], p)
    result = dict(status='completed-saved-teacher-timing', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        admitted_to_learning_preparation=chosen['passed'] and confirm['passed'], discovery=chosen, replication=confirm,
        all_discovery_candidates=ranked, source={d['seed']: {key: d[key] for key in
            ('err', 'trace_sha256', 'shared_prefix_sha256', 'command_controls')} for d in (discovery, replication)},
        physical_substeps=0, learning_updates=0, heldout_used=False)
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('ARCHIVED_GETUP_TEACHER_TIMING', chosen['joint'], chosen['sign'], result['admitted_to_learning_preparation'], flush=True)
    for label, value in (('discovery', chosen), ('replication', confirm)):
        print(label, value['groups']['teacher_only'], 'gap', value['minimum_control_gap_rad'], value['tests'], flush=True)
