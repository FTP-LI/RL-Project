"""Test whether the IMU identifies difficult recoveries before first foot contact."""

from itertools import product
from pathlib import Path

import numpy as np

from preflight_nominal_supine_unloaded_foot import chunk_at, PROJECT, ROOT, allowed, read, rel, sha
from record_io import write
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from verify_nominal_skill_composition import read_trace


KEY = 'nominal_supine_pre_first_touch'
FIELDS = ('quaternion', 'angular_velocity', 'foot_contact', 'active')


def span(folder, chunks, lo, hi, worlds):
    selected = [row for row in chunks if row['last'] >= lo and row['first'] <= hi]
    start, end = selected[0]['first'], selected[-1]['last']
    trace = read_trace(folder, selected, start, end, FIELDS, sha)
    return {name:trace[name][lo-start:hi-start+1, worlds] for name in FIELDS}


def source_arrays(project, cohort, design):
    entry = project['goal_execution'][cohort['key']]
    collection = ROOT/entry['collection']
    raw = read(collection, cohort['collection_sha256'])
    protocol = read(ROOT/entry['protocol'], entry['protocol_sha256'])
    assert raw['protocol_sha256'] == entry['protocol_sha256'] and entry['status'] == 'reviewed'
    assert protocol['environment']['seed'] == cohort['seed'] not in design['reserved_seeds']
    release = protocol['composition']['release_control']
    end = protocol['composition']['roll_start_control']
    assert release == 100 and end == 600 and protocol['composition']['control_period_s'] == .02
    assert protocol['roll_route']['standing_dwell_controls'] == 25
    assert protocol['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
    folder = collection.parent
    pose = folder/'release-poses.npz'
    assert sha(pose) == raw['release_sha256']
    with np.load(pose, allow_pickle=False) as data:
        worlds = np.flatnonzero(data['poses'] == 'supine')
    assert len(worlds) == raw['actual_release_counts']['supine']
    deadline = design['latest_touch_control']
    if 'role' in cohort:
        role = cohort['role']
        split = raw['prefix']['controls']
        series = raw['branches'][role]['chunks']
        early = span(folder/'shared', raw['prefix']['chunks'], release, min(split, deadline), worlds)
        folder /= role
        if split < deadline:
            late = span(folder, series, split+1, deadline, worlds)
            early = {key:np.concatenate((early[key], late[key])) for key in FIELDS}
    else:
        series = raw['chunks']
        early = span(folder, series, release, deadline, worlds)
    final, source = chunk_at(folder, series, end, ('roll_phase',))
    return early, final['roll_phase'][worlds] == ROLL, worlds, source


def features(trace, spec):
    # For body-frame gyro, d(world up dot body up)/dt = gx*wy - gy*wx.
    up = rotation(trace['quaternion'])[..., 2, :]
    sine = np.linalg.norm(up[..., :2], axis=-1)
    gyro = trace['angular_velocity']
    c_dot = up[..., 0]*gyro[..., 1] - up[..., 1]*gyro[..., 0]
    upright = np.rad2deg(c_dot/np.maximum(sine, spec['gravity_projection_min_sine']))
    tilt = np.rad2deg(np.arccos(np.clip(up[..., 2], -1., 1.)))
    contact = trace['foot_contact'].any(axis=-1) & trace['active']
    trace['touched_by'] = np.logical_or.accumulate(contact, axis=0)
    trace['first_touch'] = np.where(contact.any(axis=0), 100+contact.argmax(axis=0), -1)
    valid = trace['active'] & (sine >= spec['gravity_projection_min_sine'])
    return tilt, upright, valid


def scores(batch, control, rule, lead):
    trace, passed, worlds, (tilt, upright, valid) = batch
    k = control-100
    triggered = valid[k] & ~trace['touched_by'][k]
    if 'minimum_tilt_deg' in rule:
        triggered &= tilt[k] >= rule['minimum_tilt_deg']
    if 'maximum_upright_rate_deg_s' in rule:
        triggered &= upright[k] <= rule['maximum_upright_rate_deg_s']
    after = k+lead-1
    eligible_lead = trace['active'][after] & ~trace['touched_by'][after]
    hits = triggered & ~passed & eligible_lead
    failures, successes = int((~passed).sum()), int(passed.sum())
    observed_leads = trace['first_touch'][hits] - control
    return dict(control=control, failures=failures, successes=successes, early_hits=int(hits.sum()),
                original_success_exposure=int((triggered & passed).sum()), coverage=float(hits.sum()/failures),
                exposure=float((triggered & passed).sum()/successes), hit_scene_indices=worlds[hits].tolist(),
                observed_lead_controls=observed_leads[observed_leads>=0].tolist(),
                lead_right_censored=int((observed_leads<0).sum()))


def select(batch, spec):
    trace, passed, _, (tilt, upright, valid) = batch
    ranked = []
    for cutoff in spec['feature_controls']:
        index = cutoff-100
        cases = passed & valid[index] & ~trace['touched_by'][index]
        if not cases.any():
            continue
        qs = np.asarray(spec['threshold_quantiles'])
        limits = (np.unique(np.quantile(tilt[index,cases], 1.-qs)), np.unique(np.quantile(upright[index,cases], qs)))
        for x, y in product(*([None,*values] for values in limits)):
            rule = {}
            if x is not None:
                rule['minimum_tilt_deg'] = float(x)
            if y is not None:
                rule['maximum_upright_rate_deg_s'] = float(y)
            if not rule:
                continue
            result = scores(batch, cutoff, rule, spec['minimum_lead_to_first_touch_controls'])
            if result['exposure'] <= spec['maximum_original_success_exposure']:
                ranked.append((rule, result))
    return min(ranked, key=lambda item:(-item[1]['early_hits'], item[1]['original_success_exposure'],
               len(item[0]), item[1]['control'], tuple(item[0].items())), default=None)


def census(batch, controls, lead):
    trace, entered, _, (tilt, upright, valid) = batch
    rows = []
    for control in controls:
        k = control-100
        pre = valid[k] & ~trace['touched_by'][k]
        j = k+lead-1
        row = {'control':control}
        for label, mask in (('admitted', entered), ('missed', ~entered)):
            cohort = mask & pre
            allowed_lead = cohort & trace['active'][j] & ~trace['touched_by'][j]
            row[label] = dict(eligible_at_feature=int(cohort.sum()), eligible_with_lead=int(allowed_lead.sum()),
                tilt_median_deg=float(np.median(tilt[k,cohort])) if cohort.any() else None,
                upright_rate_median_deg_s=float(np.median(upright[k,cohort])) if cohort.any() else None)
        rows.append(row)
    return rows


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    design_path = ROOT/node['design']
    spec = read(design_path)
    assert allowed(project,spec['stage']) and node['status'] == 'archive-audit-ready'
    assert not project['staged_training_request']['simulation_running']
    prior = project['goal_execution']['nominal_supine_foot_authority']
    source = ROOT/prior['effect_readback']
    baseline = read(source,spec['preceding_response_sha256'])
    assert baseline['common_unloaded_reads_at_most_one'] > 0 and not prior['advance_to_integrated_trial']
    batches, telemetry = [], []
    for entry in spec['sources']:
        trace, passed, worlds, boundary = source_arrays(project,entry,spec)
        data = (trace, passed, worlds, features(trace,spec))
        batches.append(data)
        telemetry.append(dict(seed=entry['seed'], collection_sha256=entry['collection_sha256'], boundary=boundary,
            actual_supine=len(worlds), missed=int((~passed).sum()),
            reads=census(data,spec['feature_controls'],spec['minimum_lead_to_first_touch_controls'])))
    selected = select(batches[0],spec)
    if selected is None:
        rule, results = None, []
    else:
        rule, fit = selected
        results = [fit, *(scores(data, fit['control'], rule, spec['minimum_lead_to_first_touch_controls']) for data in batches[1:])]
    passed = bool(results) and all(row['coverage'] >= spec['minimum_original_miss_coverage'] and
                                  row['exposure'] <= spec['maximum_original_success_exposure'] for row in results)
    output = source.parent.parent/'pre-first-touch-imu.json'
    assert not output.exists()
    write(output,dict(protocol_sha256=sha(design_path),script_sha256=sha(Path(__file__)),
        preceding_response_sha256=sha(source),cohorts=telemetry,selected_imu_rule=rule,
        rule_fit_and_transfer=results,admitted_causal_trigger=passed,new_physics_steps=0,learning_updates=0,hardware_io=False))
    node.update(status='archive-audit-reviewed',design_sha256=sha(design_path),analysis=rel(output),
                analysis_sha256=sha(output),admitted_causal_trigger=passed)
    task = project['staged_training_request']
    task.update(current_protocol_execution='nominal-supine-pre-first-touch-reviewed',current_evidence=rel(output))
    project.update(current_protocol_execution=task['current_protocol_execution'],current_evidence=rel(output))
    write(PROJECT,project)
    print('SUPINE_PRE_FIRST_TOUCH_IMU',sha(output),rule,results,passed,flush=True)


if __name__ == '__main__':
    main()
