"""Summarize the frozen command-magnitude diagnostic from verified trajectories.

Why: keyboard defaults differ from the low-speed acceptance commands. Measure
whether larger requested velocities change motion, without changing weights,
physics or the low-speed acceptance definition. Each mean first averages time
within an episode, then weights the 32 first episodes equally. It is not a
multi-seed training comparison. No simulator, optimizer or robot connection.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from figure_style import apply_figure_style

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run/'evaluation.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    plan = json.loads((run/'protocol.json').read_text())
    assert report['command_amplitude_diagnostic'] and report['status'] == 'completed'
    assert verification['status'] == 'passed' and verification['run_id'] == report['run_id']
    assert report['num_envs'] == 32 and report['episode_seconds'] == 20 and report['seed'] == 1001
    assert not report['training_started'] and not report['meets_all_reference_criteria']
    assert all(not r['formal_acceptance_eligible'] for r in report['conditions'])
    assert digest(run/'protocol.json') == report['protocol_sha256']
    reference = ROOT/'records/evaluations/official_walk_diagnostic_reference_repeat_20261101T012547_s1001'
    assert (run/'env.yaml').read_bytes() == (reference/'env.yaml').read_bytes(), 'Physics/config differs from prior raw-action reference'
    assert report['source_sha256'] == plan['training_source_sha256']
    traces, rows = {}, []
    initial = None
    for item, check in zip(report['conditions'], verification['checks'], strict=True):
        name = item['condition']
        trace = run/f'{name}-trace.npz'
        assert check['condition'] == name and digest(trace) == check['trace_sha256']
        with np.load(trace) as data:
            active = data['active']
            assert np.isfinite(data['observations'][active]).all()
            observed = data['velocity'][:, :, 0] if item['command'][0] else data['yaw_rate']
            q0, xy0 = data['initial_joint_positions'], data['initial_xy']
            if initial is None:
                initial = (q0, xy0)
            else:
                np.testing.assert_array_equal(q0, initial[0])
                np.testing.assert_array_equal(xy0, initial[1])
            actual = (observed*active).sum(axis=0)/active.sum(axis=0)
            target = item['command'][0] or item['command'][2]
            rows.append(dict(condition=name, target=float(target), units='m/s' if item['command'][0] else 'rad/s',
                             measured_mean=float(actual.mean()), measured_min=float(actual.min()), measured_max=float(actual.max()),
                             mean_response_fraction=float(actual.mean()/target),
                             linear_xy_mae_m_s=item['linear_xy_mae_m_s'], yaw_mae_rad_s=item['yaw_mae_rad_s'],
                             full_episode_survival=item['full_episode_survival'], falls=item['fall_count_excluding_timeouts'],
                             final_displacement_m_median=item['final_displacement_m_median'],
                             meets_numeric_criteria=item['meets_numeric_criteria']))
            traces[name] = (np.where(active, observed, np.nan), target)
    shutil.copy2(__file__, run/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('verify_fixed_evaluation.py'), run/'verify_fixed_evaluation.py')
    with (run/'amplitude-summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    apply_figure_style(sizes=(11, 10, 9), font='DejaVu Sans')
    # Keep the saved image in the same canvas coordinates as bbox QA/crops.
    plt.rcParams['savefig.bbox'] = None
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.94, hspace=0.32)
    groups = [('forward_low', 'forward_keyboard', 'forward_max'), ('turn_low', 'turn_keyboard')]
    colors = ('#0072B2', '#D55E00', '#009E73')
    for index, (ax, names) in enumerate(zip(axes, groups)):
        for name, color in zip(names, colors):
            measured, target = traces[name]
            t = (np.arange(len(measured))+1)*report['step_dt']
            ax.plot(t, measured, color=color, alpha=0.12, linewidth=0.4)
            ax.plot(t, np.nanmean(measured, axis=1), color=color, linewidth=1.8, label=f'Command {target:g}')
            ax.axhline(target, color=color, linestyle='--', linewidth=1)
        ax.set_title('Forward response' if index == 0 else 'Turning response', pad=10)
        ax.set_ylabel('Body vx (m/s)' if index == 0 else 'Yaw rate (rad/s)')
        ax.legend(loc='upper right', ncol=len(names), frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
        ax.grid(axis='y', alpha=0.15)
        ax.margins(x=0.01, y=0.20)
    axes[-1].set_xlabel('Simulation time (s)')
    axes[-1].set_xticks(np.arange(0, 21, 5))
    fig.text(0.13, 0.065, '32 first episodes per command; seed 1001; fixed weights and dynamics.', fontsize=10)
    fig.text(0.13, 0.035, 'Thin: episodes; bold: active-episode mean; dashed: command.', fontsize=10)
    figures = run/'figures'; figures.mkdir(exist_ok=True)
    png = figures/'command-amplitude.png'
    fig.savefig(png, dpi=300, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(text, text.get_window_extent(renderer)) for text in fig.findobj(matplotlib.text.Text)
             if text.get_visible() and text.get_text().strip()]
    collisions = []
    for i, (text, box) in enumerate(texts):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), text.get_text()
        for other, other_box in texts[i+1:]:
            if box.overlaps(other_box):
                collisions.append([text.get_text(), other.get_text()])
    for ax in axes:
        own_ticks = set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                spine_box = spine.get_window_extent(renderer)
                for text, box in texts:
                    if text not in own_ticks and box.overlaps(spine_box):
                        collisions.append([text.get_text(), 'spine'])
    assert not collisions, collisions
    with Image.open(png) as image:
        # Crop from actual axis/title/label bounds, not an arbitrary halfway cut.
        # This preserves both panel titles in the visual QA artifacts.
        factor = image.width/fig.bbox.width
        for ax, name in zip(axes, ('forward-panel.png', 'turn-panel.png')):
            box = ax.get_tightbbox(renderer).padded(8)
            bounds = (max(0, int(box.x0*factor)), max(0, int(image.height-box.y1*factor)),
                      min(image.width, int(box.x1*factor)+1), min(image.height, int(image.height-box.y0*factor)+1))
            image.crop(bounds).save(figures/name)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), rows=rows,
                  scope='Verified fixed-weight command magnitude diagnostic; original low-speed failures remain',
                  protocol_sha256=digest(run/'protocol.json'), analysis_sha256=digest(Path(__file__)),
                  verification_sha256=digest(run/'verification.json'),
                  same_env_yaml_as_reference=True, identical_reset_joint_positions_and_xy=True,
                  text_collisions=collisions, figure_sha256=digest(png), visual_inspection='pending',
                  training_started=False, hardware_verified=False)
    (run/'amplitude-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
