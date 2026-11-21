"""Score fixed archived policies from recorded arrays with an independent hold count."""

from argparse import ArgumentParser

import numpy as np
import onnxruntime as ort

from getup_coverage_metrics import POSES, evaluate, group_rows
from getup_release_fork_state import digest
from getup_shared_release import assert_sources
from record_io import write
from recovery_end_hold import evaluate_end_hold
from run_getup_archived_policy_comparison import ROOT, files, read, sha
from verify_getup_anchor_same_release import check_trace, independent_holds, paired, verify_policy, verify_video
from verify_getup_shared_release import load_arrays, same_bytes


def verify(seed):
    plan, out, _ = files(seed)
    spec = read(plan)
    assert_sources(spec)
    p, roles = spec['environment'], spec['branches']
    record = ROOT / spec['record']
    state = read(record / 'evaluation.json')
    observer = read(record / 'windows-view-verification.json')
    assert observer['status'] == 'passed' and observer['running_snapshots_with_original_frame'] >= 3
    assert state['status'] == 'completed' and state['actor_weights_unchanged']
    assert state['protocol_sha256'] == sha(plan) == sha(record / 'protocol.json')
    assert state['learning_updates'] == 0 and not spec['heldout_used']
    controls, steps = spec['maximum_unique_controls'], spec['maximum_physics_substeps']
    assert state['actual'] == dict(controls=controls, attempted_controls=controls, substeps=steps,
                                   attempted_substeps=steps, resets=1)
    assert state['contact_peak'] < state['contact_capacity']
    assert read(record / 'zero-readback.json') == read(out / 'native-zero.json')['proof']['zero_readback']
    receipts = read(record / 'release-readback.json')
    inventory = read(record / 'release-state-fields.json')
    shared_receipt = receipts[roles[0]]
    assert set(receipts) == set(roles) and all(value == shared_receipt for value in receipts.values())
    assert shared_receipt['state_sha256'] == digest(inventory)
    assert shared_receipt['counters'] == [spec['prefix_controls'], 5 * spec['prefix_controls']]
    source_release = load_arrays(record / f'release-{roles[0]}.npz')
    source_first = load_arrays(record / f'first-control-{roles[0]}.npz')
    initial = load_arrays(record / 'initial.npz')
    prefix = load_arrays(record / 'trace-shared-prefix.npz')
    np.testing.assert_array_equal(prefix['qpos'][-1], source_release['qpos'])
    np.testing.assert_array_equal(prefix['qvel'][-1], source_release['qvel'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reports, classes, combined = {}, {}, {}
    for role in roles:
        same_bytes(source_release, load_arrays(record / f'release-{role}.npz'))
        first = load_arrays(record / f'first-control-{role}.npz')
        np.testing.assert_array_equal(first['actor_input'], source_first['actor_input'])
        np.testing.assert_array_equal(first['motor'][0], source_first['motor'][0])
        assert first['motor'].shape[:2] == (5, p['num_envs']) and np.isfinite(first['motor']).all()
        if role == roles[1]:
            same_bytes(first, source_first)
        path = record / f'trace-{role}.npz'
        branch = load_arrays(path)
        assert branch.keys() == prefix.keys()
        trace = {key: np.concatenate((prefix[key], branch[key])) for key in prefix}
        trace['requested_pose'] = initial['requested_pose']
        for key in ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets'):
            np.testing.assert_array_equal(branch[key][0], first[key])
        del branch
        step = check_trace(trace, p, state['qpos_joint_addresses'])
        session = ort.InferenceSession(str(ROOT / spec['policies'][role]['model']), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        numeric = verify_policy(trace, session)
        timely = evaluate(trace, p)
        final = evaluate_end_hold(trace, p, timely['episodes'])
        t, f = independent_holds(trace, p['criteria'])
        np.testing.assert_array_equal(t, [episode['hold_start_seconds'] is not None for episode in timely['episodes']])
        np.testing.assert_array_equal(f, final['stable_by_environment'])
        classes[role] = np.array([item['release_pose'] for item in timely['episodes']])
        np.testing.assert_array_equal(classes[role], classes[roles[0]])
        combined[role] = t & f & np.isin(classes[role], POSES)
        combo = group_rows([dict(episode, hold_start_seconds=episode['hold_start_seconds'] if good else None)
                            for episode, good in zip(timely['episodes'], combined[role], strict=True)],
                           p['success_rate_min'], p['actual_release_minimum'])
        info, = [item for item in state['conditions'] if item['role'] == role]
        clip = verify_video(record, info['video'], trace, initial, spec['cameras'][role][info['condition']])
        audits = read(record / f'contact-audits-{role}.json')
        assert audits and all(item['matched'] for item in audits)
        for audit in audits:
            assert int(trace['native_head_leg_contact'][audit['control_step'] - 1].sum()) == audit['contact_worlds']
        reports[role] = dict(combined=combo, timely=timely, final=final, onnx=numeric, trace_sha256=sha(path),
                             video=clip, maximum_late_target_step_rad=step)
        del trace
    group = classes[roles[0]]
    all_worlds = np.ones(len(group), dtype=bool)
    repeat = paired(combined[roles[0]], combined[roles[1]], all_worlds)
    comparisons = {}
    selected = []
    gate = spec['draft']['development_gate']
    for role in roles[2:]:
        comparisons[role] = {ref: {'overall': paired(combined[ref], combined[role], all_worlds),
            'by_pose': {pose: paired(combined[ref], combined[role], group == pose) for pose in POSES}}
            for ref in roles[:2]}
        if reports[role]['combined']['functional_coverage_gate'] and all(
                row['by_pose']['supine']['net'] >= gate['minimum_actual_supine_net_gain_per_reference']
                for row in comparisons[role].values()):
            selected.append(role)
    result = dict(status='passed-independent-archived-policy-verification', record=spec['record'], seed=seed,
                  protocol_sha256=sha(plan), windows_observer_sha256=sha(record / 'windows-view-verification.json'),
                  arms=reports, repeat=repeat, comparisons=comparisons, candidates_for_second_development=selected,
                  physical_substeps=steps, transitions=controls * p['num_envs'], gpu=state['gpu'],
                  sampling_wall_seconds=state['sampling_wall_seconds'], learning_updates=0, heldout_used=False)
    write(out / 'analysis.json', result)
    print('ARCHIVED_POLICY_COMPARISON_VERIFIED', {role: value['combined']['overall']['timely_holds']
          for role, value in reports.items()}, 'advance', selected, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int)
    verify(parser.parse_args().seed)
