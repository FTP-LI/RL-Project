"""Recheck rolling reasons, release subgroups and sampling cost from saved traces."""

from collections import Counter
from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


KEY = 'nominal_roll_timestep_closed_loop'
OUT = ROOT / 'records/reports/nominal-skill-composition/closed-loop-timestep/support-and-throughput.json'


def matched_effect(candidate, reference, subset):
    gain = subset & candidate['passed'] & ~reference['passed']
    loss = subset & reference['passed'] & ~candidate['passed']
    delta = candidate['peak_supported_deg'] - reference['peak_supported_deg']
    return dict(worlds=int(subset.sum()), gained=int(gain.sum()), lost=int(loss.sum()), net=int(gain.sum()-loss.sum()),
        gained_world_ids=np.flatnonzero(gain).tolist(), lost_world_ids=np.flatnonzero(loss).tolist(),
        gained_reference_failure_reasons=dict(Counter(reference['reason'][gain])),
        lost_candidate_failure_reasons=dict(Counter(candidate['reason'][loss])),
        median_paired_peak_supported_deg_delta_gain=float(np.median(delta[gain])) if gain.any() else None,
        median_paired_peak_supported_deg_delta_loss=float(np.median(delta[loss])) if loss.any() else None)


def throughput(updates, world_count, start, stop):
    """Use the observer's measured intervals, including the recording overhead."""
    samples = [x for x in updates if start < x['control'] <= stop]
    expected = list(range(start + 50, stop + 1, 50))
    assert [row['control'] for row in samples] == expected and expected[-1] == stop
    transitions = 50 * world_count
    elapsed = sum(transitions / row['transitions_per_second'] for row in samples)
    return dict(first_cumulative_control=start, exclusive_end_cumulative_control=stop,
        samples=len(samples), elapsed_seconds=elapsed, control_transitions=transitions * len(samples),
        overall_transitions_per_second=transitions * len(samples) / elapsed)


def forward_admission(plan, folder, branch, passed, reason, poses, peak):
    if branch['controls'] < plan['composition']['command_entry_control']:
        return None
    entry = plan['composition']['command_entry_control']
    chunk, = (item for item in branch['chunks'] if item['first'] <= entry <= item['last'])
    path = folder / chunk['file']
    assert sha(path) == chunk['sha256']
    with np.load(path, allow_pickle=False) as trajectory:
        k = entry - chunk['first']
        roll_ready, stand_reads = trajectory['roll_ready'][k], trajectory['base_stand_reads'][k]
    cohort = np.asarray(plan['seven_command']['groups']['forward'])
    reached = np.isin(cohort, branch['command_entry_worlds'])
    assert int(reached.sum()) == branch['command_admission']['forward']
    dropped = []
    for world in cohort[~reached]:
        if not passed[world]:
            blocker = reason[world]
        elif not roll_ready[world]:
            blocker = 'post-roll-stand-handoff'
        else:
            assert stand_reads[world] < 25
            blocker = 'base-stand-dwell'
        dropped.append(dict(world=int(world), actual_release_pose=str(poses[world]), stage=str(blocker),
            peak_supported_deg=float(peak[world]), roll_completed=bool(passed[world]), roll_ready=bool(roll_ready[world]),
            consecutive_base_stand_reads=int(stand_reads[world])))
    return dict(assigned=len(cohort), admitted=int(reached.sum()), minimum=plan['seven_command']['minimum_by_command']['forward'],
        blocked_scenes=dropped, blocked_by_stage=dict(Counter(row['stage'] for row in dropped)))


def main():
    project = read(ROOT / 'project.json')
    state = project['goal_execution'][KEY]
    assert state['status'] == 'reviewed'
    analysis = read(ROOT / state['result'], state['result_sha256'])
    plan = read(ROOT / state['protocol'], state['protocol_sha256'])
    collection = ROOT / state['collection']
    ledger = read(collection, state['collection_sha256'])
    assert analysis['collection_sha256'] == state['collection_sha256']
    assert analysis['first_policy_inputs_output_and_motor_exact'] and not analysis['learning_updates']
    folder = collection.parent
    poses_file = folder / 'release-poses.npz'
    assert sha(poses_file) == ledger['release_sha256']
    with np.load(poses_file, allow_pickle=False) as data:
        poses = data['poses']
    snapshot = folder / 'state.json'
    view = read(snapshot)
    assert view['run_id'] == ledger['run_id'] and view['controls'] == ledger['controls']
    roles = plan['solver_comparison']['roles']
    start, stop = plan['solver_comparison']['fine_solver_interval']
    shared = [dict(row, file='shared/' + row['file']) for row in ledger['prefix']['chunks']]
    offset = start
    outcomes, performance, reasons, admission = {}, {}, {}, {}
    for role in roles:
        branch = ledger['branches'][role]
        chunks = shared + [dict(row, file=role + '/' + row['file']) for row in branch['chunks']]
        _, entered, passed, reason, peak, _ = replay(plan, folder, dict(chunks=chunks))
        assert int(passed.sum()) == analysis['results'][role]['roll_complete']
        outcomes[role] = dict(entered=entered, passed=passed, reason=reason, peak_supported_deg=peak[1])
        reasons[role] = dict(Counter(reason))
        admission[role] = forward_admission(plan, folder / role, branch, passed, reason, poses, peak[1])
        performance[role] = throughput(view['updates'], len(poses), offset, offset + stop - start)
        offset += branch['controls'] - start
    assert offset == ledger['controls']
    comparisons = {}
    for left, right in ((roles[1], roles[0]), (roles[-1], roles[0]), (roles[-1], roles[1])):
        name = f'{left}-vs-{right}'
        all_worlds = matched_effect(outcomes[left], outcomes[right], np.ones(len(poses), dtype=bool))
        assert all_worlds['net'] == analysis['comparisons'][name]['roll']['net']
        by_pose = {pose: matched_effect(outcomes[left], outcomes[right], poses == pose) for pose in np.unique(poses)}
        assert {k: v['net'] for k, v in by_pose.items()} == analysis['comparisons'][name]['roll_net_by_actual_release_pose']
        comparisons[name] = dict(all_worlds=all_worlds, by_actual_release_pose=by_pose)
    assert not OUT.exists()
    write(OUT, dict(protocol_sha256=state['protocol_sha256'], result_sha256=state['result_sha256'],
        collection_sha256=state['collection_sha256'], observer_state_sha256=sha(snapshot), seed=analysis['seed'],
        script_sha256=sha(Path(__file__).resolve()), scoring_clock_s=plan['composition']['control_period_s'],
        contact_measurement='Original 20 ms endpoint replay of rolling latch. No inner-step force reconstruction.',
        failure_reasons=reasons, paired_roll_changes=comparisons, forward_admission=admission,
        sampling_throughput_in_roll_window=performance,
        peak_cuda_memory_mib=max(row['used_mib'] for row in ledger['gpu_samples']), generated_physical_steps=0))
    print('ROLL_TIMESTEP_SUPPORT_READBACK', sha(OUT), flush=True)


if __name__ == '__main__':
    main()
