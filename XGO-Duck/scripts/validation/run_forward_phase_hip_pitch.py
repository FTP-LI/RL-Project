"""Compare the observed-phase hip-pitch schedule on the original 20-second route."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_roll_stand_forward_diagnostic as direct
from forward_phase_hip_pitch import HipPitchSchedule, KEY, REPORT, STAGE as PREFLIGHT
from native_nominal_baseline import verify_sources
from roll_stand_walk_sequence import FORWARD


ROOT, P, read, put, rel, sha, require = (direct.ROOT, direct.P, direct.read, direct.put, direct.rel, direct.sha, direct.require)
ENTRY = Path(__file__).resolve()
STAGE = 'forward-phase-hip-pitch'
PLANS = ROOT/'configs/experiments/forward_phase_hip_pitch'
MASTER = PLANS/'master.json'
ORDER = ('s1247-friction-low', 's1247-nominal', 's1248-friction-low', 's1248-nominal')


def freeze():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, official = goals[KEY], goals[direct.KEY]
    dry, draft_path, original_master = ROOT/state['result'], ROOT/state['draft'], ROOT/official['master']
    require(direct.pair.trial.allowed(project, PREFLIGHT) and state['status'] == 'archived-schedule-verified'
            and official['status'] == 'reviewed' and not task['simulation_running'] and not task['learning_running']
            and not PLANS.exists() and sha(dry) == state['result_sha256'] and sha(original_master) == official['master_sha256'],
            'The completed preflight and original forward comparison are required')
    proof, schedule, reference = read(dry), read(draft_path), read(original_master)
    require(proof['target_and_timing_verified'] and proof['schedule_ready_for_route_design']
            and not any(proof[flag] for flag in ('physical_steps', 'learning_updates', 'heldout_used', 'hardware_io'))
            and set(proof['cases']) == set(reference['protocols']) == set(ORDER), 'The archived verification scope changed')
    support_sources = [ROOT/val for val in schedule['observer'].values()]
    expected = [draft_path, ENTRY.with_name('forward_phase_hip_pitch.py'), *support_sources]
    require(all(sha(path) == proof['source_sha256'][rel(path)] for path in expected), 'The audited schedule or sensors changed')
    common = {rel(path): sha(path) for path in (ENTRY, dry, original_master, *expected)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plans = {}
    for key in ORDER:
        original = original_master.parent/f'{key}.json'
        marker = official['scored'][key]
        base_receipt = ROOT/marker['path']
        plan, baseline = read(original), read(base_receipt)
        folder = ROOT/'records/evaluations'/baseline['run_id']
        raw, trace, startup, manifest = (folder/name for name in ('raw-score.json', 'trace.npz', 'startup.npz', 'guarded-evaluation.json'))
        saved_sources = {rel(path): sha(path) for path in (original, base_receipt, raw, trace, startup, manifest)}
        require(saved_sources[rel(original)] == reference['protocols'][key] == baseline['protocol_sha256']
                and saved_sources[rel(base_receipt)] == marker['sha256'] and saved_sources[rel(raw)] == baseline['raw_score_sha256']
                and saved_sources[rel(trace)] == read(manifest)['trace_sha256'] == proof['source_sha256'][rel(trace)]
                and baseline['full_route'] and not baseline['holdout_used'] and not any(baseline['safety'].values())
                and plan['run_id'] == baseline['run_id'] and plan['maximum_controls'] == 1050 and plan['worlds'] == 128
                and plan['initial_stand_controls'] == 50 and plan['forward_controls'] == schedule['full_forward_controls'] == 1000
                and plan['feedback']['maximum_joint_correction_rad'] == schedule['maximum_joint_correction_rad'] == .03
                and plan['walking']['criteria']['yaw_error_max'] == schedule['forward_yaw_limit_rad_s'] == .20
                and plan['minimum_continuous_successes'] == schedule['minimum_forward_successes'] == 116,
                'Original actor, reset, route or gate changed')
        out = REPORT/key
        require(not out.exists(), 'Preserve the earlier output')
        plan.update(stage=STAGE, run_id=f'duck_forward_phase_hip_pitch_{key}_{stamp}', phase_hip_pitch=schedule,
                    guarded_output=rel(out), page=rel(out/'route-view.html'),
                    phase_hip_pitch_reference=dict(raw=rel(raw), trace=rel(trace), startup=rel(startup), score=rel(base_receipt)),
                    route_scope='One second standing and twenty seconds forward with prior actor and early-support hip-pitch pulse')
        plan['frozen_sha256'].update(common)
        plan['frozen_sha256'].update(saved_sources)
        verify_sources(plan)
        plans[key] = plan
    PLANS.mkdir()
    checks = {}
    for key, plan in plans.items():
        out, path = REPORT/key, PLANS/f'{key}.json'
        out.mkdir()
        put(path, plan)
        checks[key] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=checks[key], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=checks, archived_preflight=rel(dry),
                     archived_preflight_sha256=sha(dry), reference_master=rel(original_master), original_yaw_limit_rad_s=.2,
                     minimum_forward_successes=116, controls_per_case=1050, worlds_per_case=128,
                     paired_bootstrap_seed=341247, maximum_transitions=4*1050*128, heldout_used=False, hardware_io=False,
                     learning_updates=0, automatic_retry=False, automatic_extension=False))
    state.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={},
                 archived_result=rel(dry), archived_result_sha256=sha(dry))
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='hip-pitch-route-ready', current_evidence=rel(dry),
                next_gate='Evaluate the four matched 20-second routes using the original movement and contact gates.')
    direct.pair.trial.write(P, project)
    print('FORWARD_PHASE_HIP_PITCH_FROZEN', state['master_sha256'], flush=True)


def collect(key):
    import run_continuous_roll_stand_walk as base

    class TimedHipPitchControl(base.PolicyControl):
        def __init__(self, plan, networks, homes, sim_home, limits):
            super().__init__(plan, networks, homes, sim_home, limits)
            self.schedule = HipPitchSchedule(plan, plan['phase_hip_pitch'])

        def next(self, step, observations, sequence):
            target, row = super().next(step, observations, sequence)
            # The official actor retains its original input, raw history and
            # projected targets. Only this control's final motor request can change.
            changed, side = self.schedule.apply(step-1, row['requested_targets'], row['applied_targets'],
                                                base.cpu(observations), row['phase_used'] == FORWARD)
            if side.any():
                target = base.torch.as_tensor(changed, device=target.device)
            return target, dict(row, hip_pitch_pulse_side=side)

    with patch.object(base, 'PolicyControl', TimedHipPitchControl):
        direct.collect(key)


def score(key):
    import verify_continuous_roll_stand_walk as verifier

    official_check = verifier.check_policy

    def check(commands, scene, plan, phase):
        original, actual = commands['applied_targets'], commands['actual_motor_target']
        result = official_check(dict(commands, actual_motor_target=original), scene, plan, phase)
        schedule = HipPitchSchedule(plan, plan['phase_hip_pitch'])
        sides = commands['hip_pitch_pulse_side']
        require(sides.shape == phase.shape and np.isin(sides, (0, 1, 2)).all(), 'Invalid recorded trigger label')
        maximum = 0.
        for k, target in enumerate(original):
            expected, selected = schedule.apply(k, commands['requested_targets'][k], target, commands['observed'][k],
                                                phase[k] == FORWARD)
            np.testing.assert_array_equal(sides[k], selected)
            np.testing.assert_allclose(actual[k], expected, rtol=0, atol=1e-6)
            change = np.zeros_like(target)
            for label, (axis, sign) in enumerate(zip(schedule.pitch, schedule.sign, strict=True), 1):
                change[selected == label, axis] = sign*schedule.delta
            np.testing.assert_allclose(actual[k]-target, change, rtol=0, atol=1e-6)
            requested = np.clip(commands['requested_targets'][k], schedule.lower, schedule.upper)
            require(np.abs(actual[k]-requested).max() <= schedule.cap+1e-6 and (actual[k] >= schedule.lower-1e-6).all()
                    and (actual[k] <= schedule.upper+1e-6).all(), 'The actual applied target left the original envelope')
            maximum = max(maximum, float(np.abs(actual[k]-target).max()))
        return dict(result, actual_motor_pulse_verified=True, maximum_pulse_rad=maximum,
                    left_pulse_controls=int((sides == 1).sum()), right_pulse_controls=int((sides == 2).sum()))

    with patch.object(verifier, 'check_policy', check):
        return direct.score_case(key)


def summary(raw):
    metrics, yaw = direct.forward_summary(raw)
    for key in ('signed_speed_m_s', 'contact_slip_m_s', 'block_error_m_s'):
        values = [row[key] for row in raw if row['entered_forward'] and row[key] is not None]
        metrics['world_median_'+key] = float(np.median(values)) if values else None
    return metrics, yaw


def compare_case(plan, current, seed):
    reference = {key: ROOT/path for key, path in plan['phase_hip_pitch_reference'].items()}
    require(all(sha(path) == plan['frozen_sha256'][rel(path)] for path in reference.values()), 'The paired official sources changed')
    folder = ROOT/'records/evaluations'/plan['run_id']
    meta = read(folder/'guarded-evaluation.json')
    raw, trace, command = (folder/name for name in ('raw-score.json', 'trace.npz', 'commands.npz'))
    require(sha(raw) == current['raw_score_sha256'] and sha(trace) == meta['trace_sha256'] and
            sha(command) == meta['command_sha256'] and current['policy_readback']['actual_motor_pulse_verified'],
            'An independently reviewed candidate file changed')
    with np.load(reference['startup'], allow_pickle=False) as old, np.load(folder/'startup.npz', allow_pickle=False) as new:
        require(set(old.files) == set(new.files), 'The startup layouts differ')
        for name in old.files:
            np.testing.assert_array_equal(old[name], new[name], err_msg=name)
    with np.load(reference['trace'], allow_pickle=False) as old, np.load(trace, allow_pickle=False) as new:
        before = new['qpos'][:plan['phase_hip_pitch']['first_zero_based_control']+1]
        prefix = float(np.max(np.abs(before-old['qpos'][:len(before)])))
        matches = np.array_equal(before[0], old['qpos'][0])
        contact = new['foot_contact'][:-1]
    require(matches, 'The two physical zero states differ')
    actual_support = contact[..., 0].astype(np.int8)+2*contact[..., 1].astype(np.int8)
    with np.load(command, allow_pickle=False) as controls:
        triggered = controls['hip_pitch_pulse_side']
    events = {}
    for label, side in ((1, 'left'), (2, 'right')):
        mask = triggered == label
        n = int(mask.sum())
        agrees = int((mask & (actual_support == label)).sum())
        events[side] = dict(pulses=n, matching_physical_support=agrees, precision=agrees/n if n else None)

    baseline, old_yaw = summary(read(reference['raw'])['forward'])
    candidate, new_yaw = summary(read(raw)['forward'])
    common = sorted(old_yaw.keys() & new_yaw.keys())
    require(common, 'There are no paired evaluable 20-second scenes')
    change = np.asarray([new_yaw[i]-old_yaw[i] for i in common])
    indices = np.random.default_rng(seed).integers(0, len(change), (4000, len(change)))
    bounds = np.quantile(np.median(change[indices], axis=1), (.025, .975)).tolist()
    return dict(seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'], official=baseline,
                timed_hip_pitch=candidate, paired_scenes=len(common), paired_yaw_median_change_rad_s=float(np.median(change)),
                paired_yaw_median_bootstrap_95_rad_s=bounds, exact_zero_state=matches, maximum_pre_pulse_qpos_difference=prefix,
                pulse_events=events, passes_all_original_gates=current['passed_gate'], official_run_id=read(reference['score'])['run_id'],
                candidate_run_id=current['run_id'], clips=current['clips'])


def compare():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready'
            and sha(MASTER) == state['master_sha256'] and set(state['scored']) == set(ORDER), 'The four cases need independent review')
    cases = {}
    for index, key in enumerate(ORDER):
        protocol = PLANS/f'{key}.json'
        receipt = state['scored'][key]
        proof = ROOT/receipt['path']
        row = read(proof)
        require(sha(protocol) == master['protocols'][key] == row['protocol_sha256'] and sha(proof) == receipt['sha256']
                and row['full_route'] and row['controls'] == 1050 and not any(row['safety'].values()), 'The verified route changed')
        cases[key] = compare_case(read(protocol), row, master['paired_bootstrap_seed']+index)

    passed = all(row['passes_all_original_gates'] for row in cases.values())
    output = REPORT/'closed-loop-comparison.json'
    put(output, dict(cases=cases, all_original_gates_passed=passed, reused_verified_official_controls=True,
                     bootstrap_unit='paired scene within each seed and friction cohort', physical_transitions=4*1050*128,
                     learning_updates=0, hardware_io=False, heldout_used=False))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output), all_original_gates_passed=passed,
                 physical_steps=4*1050*5, physical_transitions=4*1050*128)
    project['staged_training_request'].update(current_evidence=rel(output), current_protocol_execution='hip-pitch-route-reviewed',
        next_gate='Confirm this controller in the combined route and on fresh development conditions.' if passed else
        'Use the complete closed-loop changes to choose a new sustained-gait intervention without extending this batch.')
    direct.pair.trial.write(P, project)
    rows = ['# 髋俯仰闭环前进对照', '', '偏航为各场景前进20秒的平均绝对偏航速度，再取场景中位数。', '',
            '| 种子与脚底摩擦系数 | 官方通过数 | 髋俯仰通过数 | 官方偏航（rad/s） | 髋俯仰偏航（rad/s） |',
            '| --- | ---: | ---: | ---: | ---: |']
    for row in cases.values():
        old, new = row['official'], row['timed_hip_pitch']
        rows.append(f'| {row["seed"]}／{row["foot_friction"]:.1f} | {old["passed"]}/128 | {new["passed"]}/128 | '
                    f'{old["median_abs_yaw_rad_s"]:.6f} | {new["median_abs_yaw_rad_s"]:.6f} |')
    rows.extend(('', '[配对偏航差、其他动作指标、触发和原片](closed-loop-comparison.json)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows), encoding='utf-8')
    print('FORWARD_PHASE_HIP_PITCH_REVIEWED', passed, {key: row['timed_hip_pitch']['passed'] for key, row in cases.items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'compare'))
    cli.add_argument('case', nargs='?', choices=ORDER)
    args = cli.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY, compare=compare):
            if args.action in ('collect', 'score'):
                if args.case is None:
                    cli.error('Select a frozen case')
                raise SystemExit((collect if args.action == 'collect' else score)(args.case))
            raise SystemExit(direct.batch() if args.action == 'batch' else compare())
