"""Screen prospective support-gap alerts using only recorded roll actor inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from roll_metrics import progress, rotation
from summarize_roll_factor_continuation import ROOT, OUT, read, rel, require, sha


EVIDENCE = OUT / 'matched-prefix-analysis.json'
PLAN = ROOT / 'configs/experiments/roll_policy_visible_support_screen.json'
OUTPUT = OUT / 'policy-visible-support-screen.json'
RUNTIME = ROOT / 'official_resources/xgoduck_runtime_arduino/python/rl_core.py'
ROLES = ('nominal-a', 'nominal-b', 'friction-low')
SENSORS = {'IMU': 6, 'IMU-and-joints': 34}
DECLARED = dict(
    alert_offsets=[16, 18, 20, 22, 24], history_controls=4, minimum_lead_controls=4,
    minimum_gap_controls=2, minimum_gap_degrees=10.0, false_alarm_maximum=0.05,
    failure_coverage_minimum=0.75, same_world_groups=4, ridge=1.0,
)


def put(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def source_inventory():
    prefix = read(EVIDENCE)
    require(prefix['status'] == 'independently-replayed-common-prefix' and prefix['worlds'] == 128
            and prefix['control_step'] == 500 and prefix['seed'] == 1242, 'Expected the verified development prefix')
    sources = [Path(__file__), ROOT / 'scripts/validation/roll_metrics.py', RUNTIME, EVIDENCE,
               OUT / 'support-deficit-audit.json']
    arms = {}
    for role in ROLES:
        arm = prefix['arms'][role]
        folder = ROOT / 'records/evaluations' / arm['run_id']
        guard = folder / 'guarded-evaluation.json'
        manifest = read(guard)
        require(arm['trace_sha256'] == manifest['trace_sha256'] and arm['safety']['trace_valid']
                and manifest['actual_control_steps'] == arm['actual_controls'] and not manifest['learning_updates'],
                'Missing verified source: ' + role)
        path = ROOT / arm['protocol']
        require(sha(path) == arm['protocol_sha256'], 'Execution protocol changed: ' + role)
        plans = read(path)
        require(plans['heldout_used'] is False and plans['seed'] == 1242 and plans['worlds'] == 128
                and plans['initial_stand_controls'] == 50 and plans['roll']['criteria']['step_dt'] == .02,
                'Run does not match declared clock')
        sources.extend((path, guard, folder / 'env.yaml'))
        arms[role] = dict(protocol=rel(path), trace=rel(folder / 'trace.npz'), trace_sha256=arm['trace_sha256'],
                          commands=rel(folder / 'commands.npz'), commands_sha256=manifest['command_sha256'],
                          collected_at=arm['collected_at'])
    return {rel(p): sha(p) for p in sources}, arms


def freeze():
    current = read(ROOT / 'project.json')
    state, goal = current['staged_training_request'], current['goal_execution']
    require(state['current_stage'] == 'roll-policy-visible-support-screen-preflight' and goal['status'] == 'active'
            and not any(state[k] for k in ('simulation_running', 'learning_running', 'pause_requested'))
            and goal['roll_policy_visible_support']['learning_authorized'] is False,
            'Screen is not the current read-only stage')
    sources, arms = source_inventory()
    put(PLAN, dict(status='frozen-saved-data-only', frozen_at=datetime.now(timezone.utc).isoformat(),
        purpose='Forecast the start of a consequential post-head-contact support gap before action; no policy is modified.',
        input_layout={'gyro':[0,3], 'projected_gravity':[3,6], 'joint_position':[6,20], 'joint_velocity':[20,34]},
        input_clock='Controller roll entry onward. Actor input for action k contains pre-action state k.',
        target='Among roll failures, first post-head-contact no-ground-contact bout with at least two samples and 10 degrees of positive rotation before geometric threshold. Contact is evaluation truth only.',
        predictors='Ridge linear score. Both models have current sensor values and changes over 4 elapsed controls. Phase-wise normalization is fitted to training successes only.',
        evaluation='Hold all of one run out, and for each of four groups hold that world-ID residue class out of both fitting runs. Report each run after pooling its disjoint test subsets.',
        decision='Set the fold threshold from training successes only at the false-alarm budget. Every failed roll stays in the test denominator. At least one variant must pass coverage and false alarms in every run before a new trigger may be considered.',
        declared=DECLARED, source_sha256=sources, arms=arms, new_physics_steps=0,
        new_policy_updates=0, hardware_io=False, final_holdout_used=False))
    print('SUPPORT_SCREEN_FROZEN', sha(PLAN), flush=True)


def first_gap(ground, positive_degrees, start, end, criteria):
    """Return the first boundary of the first qualifying sampled gap."""
    k = start
    while k <= end:
        if ground[k]:
            k += 1
            continue
        back = k
        while k <= end and not ground[k]:
            k += 1
        if k - back >= criteria['minimum_gap_controls'] and positive_degrees[back:k].sum() >= criteria['minimum_gap_degrees']:
            return back
    return None


def prepare(role, arm, prefix, criteria):
    plan = read(ROOT / arm['protocol'])
    start, end = plan['initial_stand_controls'], prefix['control_step']
    offsets = np.asarray(criteria['alert_offsets'])
    k = start + offsets
    commands, trace = (ROOT / arm[x] for x in ('commands', 'trace'))
    require(sha(commands) == arm['commands_sha256'] and sha(trace) == arm['trace_sha256'],
            'Raw command or trace digest differs: ' + role)
    with np.load(commands, allow_pickle=False) as saved:
        raw = saved['actor_input']
        native = saved['observed']
        phases = saved['phase_used'][:end]
        # Verify unit-free sensor layout data came from the actual actor, including the applied HOME shift for joint angle.
        np.testing.assert_array_equal(raw[k, :, :6], native[k, :, :6])
        np.testing.assert_array_equal(raw[k, :, 20:34], native[k, :, 20:34])
        candidate_input = raw[:k[-1]+1, :, :34]
    with np.load(trace, allow_pickle=False) as saved:
        np.testing.assert_array_equal(phases, saved['phase_next'][:end])
        d = {key: saved[key][start:end+1] for key in
             ('quaternion', 'head_quaternion', 'foot_contact', 'head_contact', 'ground_contact', 'active')}
    require(candidate_input.shape[1:] == (128,34) and np.isfinite(candidate_input).all()
            and np.all(phases[start:k[-1]+1] == 1), 'A check window contains inactive or non-roll inputs')
    g, _, _ = progress(d)
    positive = np.rad2deg(np.maximum(np.diff(g, axis=0, prepend=0.), 0.))
    feet = d['active'] & d['foot_contact'].all(axis=-1)
    prior_feet = np.pad(np.logical_or.accumulate(feet[:-1], axis=0), ((1,0),(0,0)))
    head = (prior_feet & d['active'] & d['head_contact'] & (rotation(d['head_quaternion'])[...,2,1] < -.3)
            & (g > np.deg2rad(20.)) & (g < np.deg2rad(170.)))
    geom_min = np.deg2rad(360 - plan['roll']['criteria']['completion_tolerance_deg'])
    fail = np.ones(128, dtype=bool)
    fail[prefix['arms'][role]['roll_certificate_worlds']] = False
    rows, event = [], []
    for world in range(128):
        hits, crossings = np.flatnonzero(head[:,world]), np.flatnonzero(g[:,world] >= geom_min)
        h = int(hits[0]) if len(hits) else None
        c = int(crossings[0]) if len(crossings) else None
        gap = first_gap(d['ground_contact'][:,world], positive[:,world], h+1, c, criteria) if h is not None and c is not None else None
        event.append(-1 if gap is None else gap)
        rows.append(dict(world=world, failed_roll=bool(fail[world]), first_qualifying_head_offset=h,
                         first_geometric_threshold_offset=c, first_roll_gap_offset=gap))
    x = candidate_input[k].transpose(1,0,2).astype(float)
    previous = candidate_input[k-criteria['history_controls']].transpose(1,0,2).astype(float)
    return x, x-previous, fail, np.asarray(event), rows


def score_fit(train, fail, lead, budget, ridge):
    """Balance total failure/success weights; split each world's weight across its usable times."""
    good = ~fail
    require(good.sum() and (fail & lead.any(axis=1)).any(), 'Fold has no usable labelled examples')
    center = train[good].mean(axis=0)
    scale = np.maximum(train[good].std(axis=0), 0.001)
    norm = (train-center) / scale
    eligible = good[:,None] | (fail[:,None] & lead)
    nf = int((fail & eligible.any(axis=1)).sum())
    weight = np.broadcast_to(np.where(fail, .5/nf, .5/int(good.sum()))[:,None], eligible.shape).copy()
    weight *= eligible / np.maximum(eligible.sum(axis=1, keepdims=True), 1)
    a = norm[eligible]
    w = weight[eligible]
    target = np.broadcast_to(np.where(fail, 1., -1.)[:,None], eligible.shape)[eligible]
    linear = np.linalg.solve(a.T @ (w[:,None]*a) + ridge*np.eye(a.shape[-1]), a.T @ (w*target))
    max_good = np.sort(np.max(norm[good] @ linear, axis=1))[::-1]
    allowed = int(np.floor(budget * len(max_good)))
    threshold = float(np.nextafter(max_good[allowed], np.inf))
    return center, scale, linear, threshold


def as_list(mask):
    return np.flatnonzero(mask).tolist()


def run():
    spec, prefix = read(PLAN), read(EVIDENCE)
    require(spec['status'] == 'frozen-saved-data-only' and spec['declared'] == DECLARED and not spec['final_holdout_used'],
            'Frozen screen differs')
    require(all(sha(ROOT/name) == digest for name, digest in spec['source_sha256'].items()), 'Frozen source changed')
    criteria = spec['declared']
    arms = {r:prepare(r,spec['arms'][r],prefix,criteria) for r in ROLES}
    widths = SENSORS
    offsets = np.asarray(criteria['alert_offsets'])
    world_ids = np.arange(prefix['worlds'])
    result = {}
    for variant,width in widths.items():
        sensor = {r:np.concatenate((arm[0][...,:width],arm[1][...,:width]),axis=-1) for r,arm in arms.items()}
        per_arm = {}
        for role, (_, _, fail, event, rows) in arms.items():
            first = np.full(len(world_ids), -1)
            other = tuple(a for a in ROLES if a != role)
            folds = []
            for fold in range(criteria['same_world_groups']):
                test = (world_ids % criteria['same_world_groups']) == fold
                x = np.concatenate([sensor[a][~test] for a in other])
                y = np.concatenate([arms[a][2][~test] for a in other])
                ev = np.concatenate([arms[a][3][~test] for a in other])
                lead = ev[:,None] - offsets[None,:] >= criteria['minimum_lead_controls']
                center, scale, linear, cut = score_fit(x, y, lead, criteria['false_alarm_maximum'], criteria['ridge'])
                scores = ((sensor[role][test]-center)/scale) @ linear
                alarm = scores >= cut
                first[test] = np.where(alarm.any(axis=1), offsets[alarm.argmax(axis=1)], -1)
                folds.append(dict(residue=fold, fit_runs=other, fit_worlds_per_run=int((~test).sum()), scored_worlds=int(test.sum()),
                                  threshold=cut, feature_coefficients=linear.tolist()))
            alerts = first >= 0
            timely = fail & alerts & (event-first >= criteria['minimum_lead_controls'])
            false = (~fail) & alerts
            possible = fail & (event-offsets[0] >= criteria['minimum_lead_controls'])
            coverage, false_rate = int(timely.sum())/int(fail.sum()), int(false.sum())/int((~fail).sum())
            per_arm[role] = dict(
                failures=int(fail.sum()), successes=int((~fail).sum()), timely_failure_alerts=int(timely.sum()),
                timely_failure_worlds=as_list(timely), late_or_gapless_failure_alerts=int((fail & alerts & ~timely).sum()),
                failure_worlds_with_observable_timing=as_list(possible), false_success_alerts=int(false.sum()),
                false_success_worlds=as_list(false), failure_coverage=coverage, success_false_alarm_rate=false_rate,
                qualifies=coverage >= criteria['failure_coverage_minimum'] and false_rate <= criteria['false_alarm_maximum'],
                lead_ms=[int((event[i]-first[i])*20) for i in np.flatnonzero(timely)],
                alert_offsets=[None if i<0 else int(i) for i in first], folds=folds,
            )
        result[variant] = dict(arms=per_arm, passes_all_development_arms=all(a['qualifies'] for a in per_arm.values()))
    return dict(status='verified-crossed-development-screen', protocol_sha256=sha(PLAN),
        positives='Roll failures; qualifying post-head-contact unsupported turn is only a forecast timing reference.',
        raw_worlds=128, recorded_episodes=sum(len(a[4]) for a in arms.values()), one_seed=1242,
        causal_sensor_definition='Current or previously observed raw actor channels; no simulator contact, friction or future channels enter the predictor.',
        note='An observed gap is an endpoint sequence diagnostic, not exact contact force or an independently validated hardware event.',
        screen=result, events={r:a[4] for r,a in arms.items()},
        new_physics_steps=0, new_policy_updates=0, hardware_io=False, final_holdout_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        return freeze()
    if not args.check_only:
        require(not OUTPUT.exists(), 'Keep the original screen; use --check-only')
    info = run()
    if args.check_only:
        # Float roundoff across CPU BLAS libraries may change coefficient bytes. Test every scored decision.
        saved = read(OUTPUT)
        def scored(data):
            return {**data, 'screen':{
                name: {**variant, 'arms': {r:{k:v for k,v in arm.items() if k != 'folds'}
                       for r,arm in variant['arms'].items()}} for name,variant in data['screen'].items()}}
        require(scored(info) == scored(saved), 'Saved screen changed on independent calculation')
    else:
        put(OUTPUT, info)
    print('SUPPORT_SCREEN', {k:{r:(a['timely_failure_alerts'],a['failures'],a['false_success_alerts'],a['successes'])
          for r,a in v['arms'].items()} for k,v in info['screen'].items()}, flush=True)


if __name__ == '__main__':
    main()
