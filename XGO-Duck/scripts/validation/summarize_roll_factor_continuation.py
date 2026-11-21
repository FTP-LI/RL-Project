"""Compare three verified runs at the same control step; preserve a guarded stop."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from roll_screen_stop_guard import check_online
from roll_stand_walk_sequence import STATE_FIELDS, Sequence

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'records/reports'
OUT = REPORTS / 'roll-matched-factor-low-continuation-s1242'
CONFIG = ROOT / 'configs/experiments'
MASTER = {
    'nominal-a': CONFIG / 'roll_factor_report_capture_s1242/master.json',
    'nominal-b': CONFIG / 'roll_factor_report_capture_s1242/master.json',
    'friction-low': CONFIG / 'roll_factor_low_continuation_s1242/master.json',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_run(role):
    master = read(MASTER[role])
    path = ROOT / master['plans'][role]
    plan = read(path)
    report = ROOT / plan['guarded_output']
    analysis = read(report / 'independent-analysis.json')
    folder = ROOT / 'records/evaluations' / plan['run_id']
    manifest = read(folder / 'guarded-evaluation.json')
    receipt = read(report / 'execution-receipt.json')
    match = read(folder / 'matched-zero-runtime.json')
    require(analysis['run_id'] == plan['run_id'] and plan['comparison_role'] == role
            and sha(path) == master['plan_sha256'][role] == analysis['protocol_sha256'], 'Run provenance: ' + role)
    require((plan['seed'], plan['worlds'], plan['maximum_controls'], plan['minimum_roll_completion'],
             plan['minimum_continuous_successes'], plan['parameter_factors']['vin_range_v']) == (1242, 128, 1550, 116, 116, 7.7)
            and plan['heldout_used'] is False and analysis['learning_updates'] == 0 and analysis['hardware_io'] is False,
            'Frozen scope: ' + role)
    require(match['status'] == 'passed-input-and-rng-check' and match['actuator_arrays_exact'] and match['rng_exact']
            and not match['unexpected_field_names'] and match['physical_substeps'] == 0, 'Zero-step evidence: ' + role)
    require(analysis['safety_summary'] == manifest['safety_summary']
            and manifest['safety_summary'] == dict(trace_valid=True, nonfinite_samples=0,
                                                    head_over_limit_worlds=0, joint_over_limit_worlds=0), 'Safety: ' + role)
    require(not manifest['artifact_errors'] and len(analysis['clips']) == 1
            and analysis['clips'][0]['sha256'] == sha(ROOT / analysis['clips'][0]['path']), 'Media: ' + role)
    return path, plan, analysis, folder, manifest, receipt, match


def ids(mask):
    return np.flatnonzero(mask).tolist()


def pair(a, b):
    gained, lost = sorted(set(b)-set(a)), sorted(set(a)-set(b))
    return dict(gained_worlds=gained, lost_worlds=lost, net=len(gained)-len(lost))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Confirm saved results without writing new files')
    args = parser.parse_args()
    target = OUT / 'matched-prefix-analysis.json'
    if not args.check_only:
        require(not target.exists(), 'Preserve the existing immutable analysis')
    runs = {role: find_run(role) for role in MASTER}
    _, low_plan, low, _, low_manifest, _, _ = runs['friction-low']
    horizon = low['controls']
    require(low['status'] == 'stopped-baseline-independently-reviewed' and low['counts'] is None
            and low['stop_reason'] == 'roll-gate-impossible' and 0 < horizon < 1550, 'Expected original guard stop')
    rows = {}
    for role, (path, plan, analysis, folder, manifest, receipt, matched) in runs.items():
        trace_path = folder / 'trace.npz'
        require(sha(trace_path) == manifest['trace_sha256'], 'Saved trace digest: ' + role)
        require(analysis['controls'] == manifest['actual_control_steps']
                and analysis['control_transitions'] == 128 * analysis['controls']
                and analysis['physics_substeps'] == 5 * analysis['controls'], 'Actual sample counts: ' + role)
        if role == 'friction-low':
            require(plan['parameter_factors']['foot_friction'] == .8 and receipt['exit_code'] == 3, 'Low-friction stop receipt')
        else:
            require(analysis['status'] == 'completed-baseline-independently-verified'
                    and analysis['passed_development_gate'] and receipt['exit_code'] == 0, 'Nominal completion: ' + role)
        replay = Sequence(plan)
        with np.load(trace_path, allow_pickle=False) as data:
            trace = {field: data[field][:horizon+1] for field in (*STATE_FIELDS, 'phase_next', 'guard_phase_valid')}
        decisions = []
        for step in range(horizon+1):
            require(bool(trace['guard_phase_valid'][step]), f'Invalid phase at {role} step {step}')
            state = {field: trace[field][step] for field in STATE_FIELDS}
            replay.update(step, state)
            np.testing.assert_array_equal(replay.phase, trace['phase_next'][step])
            decision = check_online(replay, state, step, plan)
            if decision['stop']:
                decisions.append(decision)
        latch = replay.roll_latch
        certified = latch.first_ready >= 0
        possible = certified | (latch.intact & (latch.step < plan['roll_controls']))
        terminal = decisions[0] if decisions else None
        if role == 'friction-low':
            require(decisions == [low_manifest['guard_result']['decision']] and terminal['control'] == horizon
                    and terminal['possible'] == int(possible.sum()), 'Stop condition differs on replay')
        else:
            require(not decisions, 'Nominal prefix was not complete: ' + role)
        final = None
        if analysis['counts'] is not None:
            saved = read(folder / 'raw-score.json')
            require([r['world'] for r in saved['forward']] == list(range(plan['worlds'])), 'World order changed')
            final = dict(roll_completed=sum(s >= 0 for s in saved['roll_switch_at']),
                         entered_forward=sum(s >= 0 for s in saved['forward_at']),
                         continuous_successes=sum(r['success'] for r in saved['forward']))
            require(final == {k: analysis['counts'][k] for k in final}, 'Complete nominal score differs: ' + role)
        rows[role] = dict(run_id=plan['run_id'], collected_at=receipt['started_at'], protocol=rel(path),
            protocol_sha256=sha(path), trace_sha256=sha(trace_path), foot_friction=plan['parameter_factors']['foot_friction'],
            actual_controls=analysis['controls'], transitions=analysis['control_transitions'], safety=analysis['safety_summary'],
            roll_certificate_worlds=ids(certified), still_possible_worlds=ids(possible),
            certified_while_entering_forward_worlds=ids(replay.forward_at >= 0),
            phase_counts={str(int(k)): v for k,v in sorted(Counter(replay.phase).items())},
            impossible_worlds=ids(~possible), impossible_seen_head_worlds=ids(~possible & latch.seen_head),
            impossible_supported_rotation_worlds=ids(~possible & latch.finished),
            impossible_physically_active_worlds=ids(~possible & trace['active'][-1]),
            allowed_internal_fields_differing=len(matched['differing_field_names']), stop=terminal, final=final, clip=analysis['clips'][0])
    paired = {f'{b}_vs_{a}': {field: pair(rows[a][field], rows[b][field]) for field in
        ('roll_certificate_worlds', 'still_possible_worlds', 'certified_while_entering_forward_worlds')}
        for a,b in (('nominal-a','nominal-b'), ('nominal-a','friction-low'), ('nominal-b','friction-low'))}
    result = dict(status='independently-replayed-common-prefix', control_step=horizon,
        simulation_seconds=horizon*low_plan['roll']['criteria']['step_dt'], seed=1242, worlds=128,
        comparison_scope='A common development prefix. Nominal references on an earlier day; known solver internals differ. No complete low-friction route rate.',
        frozen_rolling_gate=116, arms=rows, paired=paired, new_physics_substeps=0,
        new_learning_updates=0, hardware_io=False)
    if args.check_only:
        require(read(target) == result, 'The read-only replay differs from the saved comparison')
    else:
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
    print('ROLL_PREFIX_REPLAY', horizon, {role: (len(r['roll_certificate_worlds']), len(r['still_possible_worlds']),
          len(r['certified_while_entering_forward_worlds']), r['final']) for role, r in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
