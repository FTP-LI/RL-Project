"""Summarize saved learning drift and evaluation failures, with no simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np

from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from review_getup_anchor_same_release import describe
from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/getup-supine-weighted-learning'
PLAN = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
FIGURE = ROOT/'docs/reports/assets/getup-supine-learning-drift.png'
ARMS = ('control', 'candidate')


def failure_details(proof):
    record = ROOT/proof['record']
    spec = read(record/'protocol.json')
    keys = ('time', 'native_head_leg_contact', 'quaternion', 'joint_envelope_ok', 'foot_contact',
            'height', 'angular_velocity')
    prefix_path = record/'trace-shared-prefix.npz'
    assert sha(prefix_path) == proof['prefix_sha256']
    with np.load(prefix_path) as data:
        prefix = {key: data[key] for key in keys}
    stats, outcomes = {}, {}
    groups = None
    for role, arm in proof['arms'].items():
        path = record/f'trace-{role}.npz'
        assert sha(path) == arm['trace_sha256']
        with np.load(path) as data:
            trace = {key: np.concatenate((prefix[key], data[key])) for key in keys}
        cases = arm['timing']['episodes']
        current = {
            'actual-supine': np.array([case['release_pose'] == 'supine' for case in cases]),
            'requested-supine': np.array([case['requested_pose'] == 'supine' for case in cases]),
            'actual-right-side': np.array([case['release_pose'] == 'right-side' for case in cases]),
        }
        if groups is None:
            groups = current
        for name in groups:
            np.testing.assert_array_equal(current[name], groups[name])
        timely = np.array([case['hold_start_seconds'] is not None for case in cases])
        final = np.array(arm['end_hold']['stable_by_environment'])
        outcomes[role] = timely & final
        stats[role] = {name: describe(trace, group & ~outcomes[role], spec['environment']['criteria'])
                       for name, group in groups.items()}
    roles = spec['branches']
    chosen = outcomes[roles[-1]]
    changed = {role: {name: dict(
        wins=np.flatnonzero(group & chosen & ~outcomes[role]).tolist(),
        losses=np.flatnonzero(group & ~chosen & outcomes[role]).tolist(),
    ) for name, group in groups.items()} for role in roles[:-1]}
    return dict(failure_telemetry=stats, paired_subgroup_changes=changed)


def main():
    coordinator = read(OUT/'coordinator.json')
    assert coordinator['status'] in ('completed', 'stopped-review-required')
    assert coordinator['protocol_sha256'] == sha(PLAN)
    spec = read(PLAN)
    result = dict(status='passed-saved-data-only', protocol_sha256=sha(PLAN),
                  source_sha256=sha(Path(__file__)), physical_substeps=0, learning_updates=0)
    stats = {}
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=.17, right=.95, bottom=.19, top=.88)
    colors = dict(control='#6b7784', candidate='#147b9e')
    labels = dict(control='等权对照', candidate='仰卧加权')
    threshold = spec['training']['control']['tripwire']['success_replay_rms_rad_max']
    ax.axhline(threshold, color='#815230', linewidth=1., linestyle='--', label='停止阈值')
    for role in ARMS:
        record = ROOT/coordinator['training_records'][role]
        file = record/'learning-metrics.jsonl'
        if not file.exists() or not file.stat().st_size:
            continue
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        state = read(record/'state.json')
        assert [row['update'] for row in rows] == list(range(1, state['completed_updates']+1))
        initial = read(record/'parent-readback.json')['parent_drift_rms_rad']
        rms = [initial, *(row['drift_rms'] for row in rows)]
        ax.plot(range(len(rms)), rms, color=colors[role], label=labels[role],
                linewidth=2.3 if role == 'candidate' else 1.6)
        stats[role] = dict(updates=len(rows), initial_drift_rms_rad=initial, final_drift_rms_rad=rms[-1],
                           peak_drift_rms_rad=max(rms), peak_absolute_drift_rad=max(row['drift_max'] for row in rows),
                           actual_focus_samples=sum(row['early_supine_samples'] for row in rows),
                           training_elapsed_at_final_update_s=rows[-1]['elapsed'], metrics_sha256=sha(file))
    if stats:
        upper = max(threshold, *(arm['peak_drift_rms_rad'] for arm in stats.values()))
        lower = min(.020, *(arm['initial_drift_rms_rad'] for arm in stats.values()))-.002
        ax.set(xlabel='PPO 更新次数', ylabel='成功样本动作偏差 RMS（rad）', ylim=(lower, upper+.005),
               xlim=(-1, 51), xticks=np.arange(0, 51, 10))
        ax.set_title('成功样本动作偏差', loc='left')
        ax.legend(loc='lower right', ncol=1)
        result['figure'] = checked_save(fig, FIGURE)
    else:
        plt.close(fig)
    result['training'] = stats
    path = OUT/'analysis.json'
    if coordinator['status'] == 'completed':
        proof = read(path)
        assert coordinator['analysis_sha256'] == sha(path)
        result.update(evaluation_sha256=sha(path), **failure_details(proof))
    write(OUT/'mechanism-review.json', result)
    print('SUPINE_LEARNING_REVIEW', {role: (arm['updates'], arm['final_drift_rms_rad'])
                                    for role, arm in stats.items()}, flush=True)


if __name__ == '__main__':
    main()
