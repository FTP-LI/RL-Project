"""Independently score the three same-release policies on saved CPU data."""
import numpy as np

from getup_coverage_metrics import POSES, evaluate
from getup_release_fork_state import digest
from recovery_end_hold import evaluate_end_hold
from s1_reference_inputs import ROOT, read, sha
from verify_getup_anchor_same_release import check_trace, independent_holds, paired, verify_policy, verify_video
from verify_getup_shared_release import load_arrays, same_bytes


def select(roles, reports, comparisons, minimum_gain):
    parent, control, candidate = roles
    parent_cmp, control_cmp = comparisons[parent], comparisons[control]
    checks = dict(
        all_original_gates=reports[candidate]['all_motion_gates_passed'],
        actual_supine_gain_against_control=control_cmp['actual_supine']['net'],
        minimum_actual_supine_gain=minimum_gain,
        no_aggregate_loss_against_parent=parent_cmp['all']['net'] >= 0,
        no_actual_supine_loss_against_parent=parent_cmp['actual_supine']['net'] >= 0,
    )
    checks['passes_admission'] = bool(checks['all_original_gates'] and
        checks['actual_supine_gain_against_control'] >= minimum_gain and
        checks['no_aggregate_loss_against_parent'] and checks['no_actual_supine_loss_against_parent'])
    chosen = next((role for role, eligible in (
        (control, reports[control]['all_motion_gates_passed']),
        (parent, reports[parent]['all_motion_gates_passed']),
        (candidate, checks['passes_admission']),
    ) if eligible), None)
    return dict(weighted_candidate=checks, selected_for_independent_confirmation=chosen)


def verify(spec, record, state, minimum_gain):
    import onnxruntime as ort
    import torch

    assert not torch.cuda.is_initialized() and state['actor_weights_unchanged']
    controls, substeps = spec['maximum_unique_controls'], spec['maximum_physics_substeps']
    assert state['actual'] == dict(controls=controls, attempted_controls=controls,
                                  substeps=substeps, attempted_substeps=substeps, resets=1)
    assert state['contact_peak'] < state['contact_capacity']
    gpu = read(record/'zero-readback.json')
    assert gpu['status'] == 'passed' and gpu['actual_physics_steps'] == 0
    assert gpu['frozen_protocol_sha256'] == sha(record/'protocol.json')
    p, roles = spec['environment'], spec['branches']
    release = spec['prefix_controls']
    receipt, inventory = (read(record/name) for name in ('release-readback.json', 'release-state-fields.json'))
    assert set(receipt) == set(roles)
    baseline = receipt[roles[0]]
    assert baseline['state_sha256'] == digest(inventory) and baseline['counters'] == [release, 5*release]
    initial = load_arrays(record/'initial.npz')
    prefix_path = record/'trace-shared-prefix.npz'
    prefix = load_arrays(prefix_path)
    physical = load_arrays(record/f'release-{roles[0]}.npz')
    initial_action = load_arrays(record/f'first-control-{roles[0]}.npz')
    np.testing.assert_array_equal(prefix['qpos'][-1], physical['qpos'])
    np.testing.assert_array_equal(prefix['qvel'][-1], physical['qvel'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reports, outcomes, videos = {}, {}, []
    release_classes = None
    assert len(state['conditions']) == sum(len(cameras) for cameras in spec['cameras'].values())
    for role in roles:
        assert receipt[role] == baseline
        same_bytes(physical, load_arrays(record/f'release-{role}.npz'))
        first = load_arrays(record/f'first-control-{role}.npz')
        np.testing.assert_array_equal(first['actor_input'], initial_action['actor_input'])
        np.testing.assert_array_equal(first['motor'][0], initial_action['motor'][0])
        assert first['motor'].shape[:2] == (5, p['num_envs']) and np.isfinite(first['motor']).all()
        path = record/f'trace-{role}.npz'
        arm = load_arrays(path)
        assert arm.keys() == prefix.keys()
        t = {key: np.concatenate((prefix[key], arm[key])) for key in prefix}
        t['requested_pose'] = initial['requested_pose']
        for key in ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets'):
            np.testing.assert_array_equal(arm[key][0], first[key])
        maximum_step = check_trace(t, p, state['qpos_joint_addresses'])
        plan = spec['policies'][role]
        model = ROOT/plan['model']
        assert sha(model) == plan['model_sha256']
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        numeric = verify_policy(t, session)
        timely = evaluate(t, p)
        final = evaluate_end_hold(t, p, timely['episodes'])
        early_ok, final_ok = independent_holds(t, p['criteria'])
        np.testing.assert_array_equal(early_ok, [row['hold_start_seconds'] is not None for row in timely['episodes']])
        np.testing.assert_array_equal(final_ok, final['stable_by_environment'])
        classes = np.array([row['release_pose'] for row in timely['episodes']])
        if release_classes is None:
            release_classes = classes
        np.testing.assert_array_equal(classes, release_classes)
        outcomes[role] = early_ok & final_ok & np.isin(classes, POSES)
        items = {item['condition']: item['video'] for item in state['conditions'] if item['role'] == role}
        assert items.keys() == spec['cameras'][role].keys()
        for pose, camera in spec['cameras'][role].items():
            clip = verify_video(record, items[pose], t, initial, camera)
            videos.append(dict(clip, role=role, requested_pose=pose, actual_release_pose=str(classes[camera])))
        audits = read(record/f'contact-audits-{role}.json')
        assert audits and all(row['matched'] for row in audits)
        for row in audits:
            assert int(t['native_head_leg_contact'][row['control_step']-1].sum()) == row['contact_worlds']
        reports[role] = dict(
            timing=timely, end_hold=final, combined_successes=int(outcomes[role].sum()),
            all_motion_gates_passed=timely['functional_coverage_gate'] and final['functional_coverage_gate'],
            trace_sha256=sha(path), onnx=numeric, maximum_late_target_step_rad=maximum_step,
            independent_holds_exact=True,
        )
    assert len(videos) == len(state['conditions']) and not torch.cuda.is_initialized()
    supine = release_classes == 'supine'
    all_worlds = np.ones(len(release_classes), dtype=bool)
    comparisons = {role: dict(all=paired(outcomes[role], outcomes[roles[-1]], all_worlds),
                             actual_supine=paired(outcomes[role], outcomes[roles[-1]], supine))
                   for role in roles[:-1]}
    return dict(
        status='passed-independent-verification', arms=reports, comparisons=comparisons,
        selection=select(roles, reports, comparisons, minimum_gain), release=receipt, videos=videos,
        first_inputs_and_delayed_motor_exact=True, record=record.relative_to(ROOT).as_posix(),
        prefix_sha256=sha(prefix_path), actual_control_transitions=controls*p['num_envs'],
        actual_physics_substeps=substeps, physical_resets=1, learning_updates=0,
        heldout_used=False, verification_physics_substeps=0,
    )
