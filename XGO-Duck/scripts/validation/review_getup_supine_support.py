"""Compare first sustained two-foot contact on saved development trajectories.

Align at confirmation of five ground-contact samples, then separate torso
alignment, contact and joint requests. The saved forces are unavailable.
"""
from pathlib import Path
import sys

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write
from s1_reference_inputs import read, sha
from recovery_end_hold import evaluate_end_hold
from standup_metrics import projected_gravity

REPORTS = ROOT / 'records/reports'
PROTOCOLS = ROOT / 'configs/experiments'
STUDIES = {1050: ('getup-persistent-contact/pair', 'getup_persistent_contact_pair.json'),
           1016: ('getup-late-feasibility', 'getup_late_feasibility_pair.json')}
OUTPUT = ROOT / 'records/reports/getup-shared-release/fixed-terrain/supine-support-review.json'
JOINTS = ('left_hip_pitch', 'right_hip_pitch', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'neck_pitch')


def first_window(mask, controls):
    windows = np.lib.stride_tricks.sliding_window_view(mask, controls)
    good = np.flatnonzero(windows.all(axis=1))
    return int(good[0]) if len(good) else None


def median(values):
    return float(np.median(values)) if len(values) else None


def support_state(trace, index, i, tilt, gravity, limits, joints):
    """Capture the joint data and contact state from the confirmation sample."""
    pitch = np.rad2deg(np.arctan2(gravity[index, i, 0], -gravity[index, i, 2]))
    position, request, target = (trace[name][index, i] for name in ('joint_positions', 'requested_targets', 'applied_targets'))
    excess = np.maximum(np.maximum(limits[:, 0] - target, target - limits[:, 1]), 0)
    return dict(confirmation_seconds=float(trace['time'][index]), tilt_deg=float(tilt[index, i]),
                signed_sagittal_gravity_deg=float(pitch), height_m=float(trace['height'][index, i]),
                body_pitch_rate_rad_s=float(trace['angular_velocity'][index, i, 1]),
                simultaneous_head_leg_contact=bool(trace['native_head_leg_contact'][index, i]),
                actual_deg=np.rad2deg(position[joints]).tolist(), requested_deg=np.rad2deg(request[joints]).tolist(),
                applied_deg=np.rad2deg(target[joints]).tolist(), applied_outside=np.flatnonzero(excess > 1e-6).tolist())


def summarize(rows, joint_names, measured):
    result = dict(episodes=len(rows))
    for name in ('head_leg_contact', 'double_support', 'upright_double_support'):
        observed = [row[name] for row in rows if row[name] is not None]
        result[name] = dict(observed=len(observed), first_sample_time_median_s=median(observed))
    result['double_support_without_upright'] = sum(
        row['double_support'] is not None and row['upright_double_support'] is None for row in rows)
    events = [row['first_double_support_confirmation'] for row in rows if row['first_double_support_confirmation'] is not None]
    result['confirmed_support'] = dict(episodes=len(events),
        medians={name: median([event[name] for event in events]) for name in ('tilt_deg', 'signed_sagittal_gravity_deg',
                  'height_m', 'body_pitch_rate_rad_s')},
        head_leg_contact=sum(event['simultaneous_head_leg_contact'] for event in events),
        applied_outside={name: sum(j in event['applied_outside'] for event in events)
                         for j, name in enumerate(joint_names)},
        joint_median_deg={name: {kind: median([event[kind][j] for event in events])
                                for kind in ('actual_deg', 'requested_deg', 'applied_deg')}
                          for j, name in enumerate(measured)})
    return result


def review_study(seed, pair, master_path, sources):
    master = read(master_path)
    master_hash = sha(master_path)
    sources[str(master_path.relative_to(ROOT))] = master_hash
    scores, all_rows = {}, {}
    for role in ('original', 'control', 'candidate'):
        proof_path = pair / f'{role}-analysis.json'
        proof = read(proof_path)
        record = ROOT / proof['record']
        protocol, coverage, hold_path = (record / name for name in ('protocol.json', 'coverage.json', 'end-hold.json'))
        plan = read(protocol)
        assert plan['seed'] == seed and proof['status'] == 'passed' and proof['continuous_no_physical_reset']
        cases = read(coverage)['episodes']
        assert [row['env_index'] for row in cases] == list(range(320))
        hold = read(hold_path)['stable_by_environment']
        assert hold == proof['end_hold']['stable_by_environment']
        path = record / 'coverage-trace.npz'
        model = ROOT / plan['model']
        assert sha(path) == proof['trace_sha256'] and master_hash == proof['protocol_sha256']
        assert plan == dict(master['arms'][role], model=plan['model'], model_sha256=plan['model_sha256'])
        assert sha(model) == plan['model_sha256']
        metadata = {p.key:p.value for p in onnx.load(model).metadata_props}
        joint_names = metadata['joint_names'].split(',')
        measured = [name for name in JOINTS if name in joint_names]
        assert len(measured) == 7 and len(joint_names) == 14
        joints = [joint_names.index(name) for name in measured]
        limits = np.asarray(plan['joint_limits'])
        window = (2., 5.)
        controls = 5
        assert plan['criteria']['step_dt'] * controls == .1
        with np.load(path) as saved:
            keys = ('time', 'foot_contact', 'native_head_leg_contact', 'quaternion', 'height', 'angular_velocity',
                    'active', 'joint_envelope_ok', 'joint_positions', 'requested_targets', 'applied_targets')
            trace = {key: saved[key] for key in keys}
        assert evaluate_end_hold(trace, plan, cases)['stable_by_environment'] == hold
        select = (trace['time'] > window[0]) & (trace['time'] <= window[1])
        early = {name: value[select] for name, value in trace.items()}
        time = early['time']
        support = early['foot_contact'].all(axis=-1)
        contact = early['native_head_leg_contact']
        gravity = projected_gravity(early['quaternion'])
        tilt = np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1, 1)))
        assert len(time) == 150
        rows = []
        for i, case in enumerate(cases):
            if case['release_pose'] != 'supine':
                continue
            head = first_window(contact[:, i], 1)
            both = first_window(support[:, i], controls)
            upright = first_window(support[:, i] & (tilt[:, i] <= plan['criteria']['tilt_max_deg']), controls)
            rows.append(dict(environment=i, timely_and_final=bool(case['hold_start_seconds'] is not None and hold[i]),
                head_leg_contact=None if head is None else float(time[head]),
                double_support=None if both is None else float(time[both]),
                upright_double_support=None if upright is None else float(time[upright]),
                first_double_support_confirmation=None if both is None else support_state(early, both + controls - 1,
                                                                                           i, tilt, gravity, limits, joints)))
        assert len(rows) == proof['end_hold']['actual']['supine']['episodes']
        assert sum(row['timely_and_final'] for row in rows) == proof['end_hold']['actual']['supine']['timely_holds']
        scores[role] = {name: summarize([row for row in rows if row['timely_and_final'] == value], joint_names, measured)
                        for name, value in [('successful', True), ('failed', False)]}
        all_rows[role] = dict(joint_names=joint_names, measured=measured, rows=rows)
        sources.update({str(p.relative_to(ROOT)): sha(p) for p in (path, protocol, coverage, hold_path, proof_path, model)})
    return dict(scores=scores, joint_trace=all_rows)


def main():
    sources = {}
    studies = {seed: review_study(seed, REPORTS / pair, PROTOCOLS / protocol, sources)
               for seed, (pair, protocol) in STUDIES.items()}
    result = dict(status='passed-saved-data-only', development_seeds=list(STUDIES), window_seconds=[2., 5.],
                  sustained_support_seconds=.1, original_score_unchanged=True, studies=studies,
                  source_sha256=sources, script_sha256=sha(Path(__file__)),
                  heldout_used=False, new_physical_steps=0, learning_updates=0,
                  scope='Early event association within actual supine strata; independently sampled physical releases.',
                  sensor_timing='Head-leg reward and feet sensing occur in the same control interval; contact does not measure load.')
    write(OUTPUT, result)
    for seed, study in studies.items():
        for role, groups in study['scores'].items():
            for name, score in groups.items():
                print('SUPINE_SUPPORT', seed, role, name, score['episodes'], score['confirmed_support']['medians'])


if __name__ == '__main__':
    main()
