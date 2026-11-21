"""Recheck the stopped control from saved checkpoints and trajectory, without physics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np
import torch

from figure_style import apply_figure_style
from getup_recovery_retention_pair import OUT, PLAN, checked
from getup_supine_learning import load_parent
from getup_value_bootstrap import td
from record_io import write
from render_roll_angle_results import checked_save
from run_getup_recovery_retention import verify_retention
from run_getup_supine_training import check_matched_start
from s1_reference_inputs import ROOT, read, sha
from verify_getup_supine_training import verify_clips, verify_trace
from verify_getup_teacher_anchor_train import exact


def checkpoint_drift(path, plan, anchor, retention, strata):
    learner, saved = load_parent(path, plan, anchor, 'cpu')
    with torch.inference_mode():
        anchor_error = (learner.actor(td('actor', learner.anchor_inputs))-learner.anchor_targets).numpy()
        retention_error = learner.actor(td('actor', torch.from_numpy(retention[0]))).numpy()-retention[1]
    assert np.isfinite(anchor_error).all() and np.isfinite(retention_error).all()
    return saved, dict(anchor_rms_rad=float(np.sqrt(np.mean(anchor_error**2))),
                       anchor_max_rad=float(np.abs(anchor_error).max()),
                       retention_rms_rad={name: float(np.sqrt(np.mean(retention_error[indices]**2)))
                                          for name, indices in strata.items()}, checkpoint_sha256=sha(path))


def render_drift(rms, cutoff):
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=.16, right=.95, bottom=.19, top=.88)
    ax.axhline(cutoff, color='#815230', linewidth=1., linestyle='--', label='停止阈值')
    ax.plot(range(len(rms)), rms, color='#147b9e', linewidth=1.8, label='对照组')
    ax.scatter(len(rms)-1, rms[-1], color='#815230', s=28, zorder=3)
    lower, upper = min(rms)-.003, max(rms)+.005
    ticks = np.arange(np.ceil(lower/.005), np.floor(upper/.005)+1)*.005
    ax.set(xlabel='PPO 更新次数', ylabel='成功样本动作偏差 RMS（rad）',
           ylim=(lower, upper), yticks=ticks, xlim=(-1, 34), xticks=np.arange(0, 34, 5))
    ax.set_title('成功样本动作偏差', loc='left')
    ax.legend(loc='lower right')
    return checked_save(fig, ROOT/'docs/reports/assets/getup-recovery-retention-drift.png')


def main():
    torch.set_num_threads(4)
    spec, coordinator = read(PLAN), read(OUT/'coordinator.json')
    digest = sha(PLAN)
    assert coordinator['protocol_sha256'] == digest == sha(OUT/'protocol.json')
    assert coordinator['status'] == 'stopped-review-required' and coordinator['phase'] == 'training-control'
    assert coordinator['visible_tty'] and coordinator['ended_at']
    assert not (ROOT/coordinator['training_records']['candidate']).exists()
    assert not (ROOT/coordinator['evaluation_record']).exists() and not (OUT/'analysis.json').exists()
    checked(spec)
    source_hashes = spec['evaluation']['source_sha256']
    for filename, checksum in source_hashes.items():
        assert sha(OUT/'sources'/filename) == checksum, filename

    zero = read(OUT/'zero-step/verification.json')
    assert zero['physical_substeps'] == 0 and zero['status'] == 'passed' and zero['protocol_sha256'] == digest
    check_matched_start(OUT/'zero-step/candidate', OUT/'zero-step/control')
    plan = spec['training']['control']
    record = ROOT/coordinator['training_records']['control']
    state, limit = read(record/'state.json'), spec['training_limit']
    assert state['paired_zero_start_verified'] and state['protocol_sha256'] == digest and read(record/'protocol.json') == plan
    assert state['status'] == coordinator['status'] and state['error'] == coordinator['error']
    assert state['error'] == "AssertionError('Successful-state drift guard')"
    assert not (record/'model-final.pt').exists() and not (ROOT/'artifacts'/plan['run_id']).exists()
    check_matched_start(record, OUT/'zero-step/control')
    metric_path = record/'learning-metrics.jsonl'
    rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
    updates, controls = len(rows), state['controls']
    assert 25 < updates == state['completed_updates'] < plan['training_updates']
    assert state['metrics'] == rows[-1] and [r['update'] for r in rows] == list(range(1, updates+1))
    assert [r['controls'] for r in rows] == list(range(limit['rollout_controls'], controls+1, limit['rollout_controls']))
    assert controls == state['attempted_controls'] == updates*limit['rollout_controls']
    assert state['substeps'] == state['attempted_substeps'] == controls*limit['substeps']//limit['controls']
    assert state['terminal_resets'] == controls//400
    cutoff, ceiling = (plan['tripwire'][key] for key in ('success_replay_rms_rad_max', 'success_replay_abs_rad_max'))
    assert all(r['drift_rms'] <= cutoff for r in rows[:-1]) and rows[-1]['drift_rms'] > cutoff
    assert all(r['drift_max'] <= ceiling for r in rows)
    trace, masks = verify_trace(record, plan, controls, rows)
    videos = verify_clips(record, trace, state['clips'])
    draws = verify_retention(plan, dict(record=str(record.relative_to(ROOT)), updates=updates))
    assert draws['weight'] == 0 and len(videos) == 1

    with np.load(ROOT/plan['anchor_data']) as data:
        anchor = data['observations'].copy(), data['teacher_targets'].copy(), read(ROOT/plan['anchor_cells'])
    with np.load(ROOT/plan['retention_data']) as data:
        retention = data['observations'].copy(), data['targets'].copy()
    cells = read(ROOT/plan['retention_cells'])
    strata = {pose: sorted({index for key, values in cells.items() if key.split('|')[0] == pose for index in values})
              for pose in sorted({key.split('|')[0] for key in cells})}
    initial, initial_drift = checkpoint_drift(record/'model-init.pt', plan, anchor, retention, strata)
    mid, mid_drift = checkpoint_drift(record/'model-25.pt', plan, anchor, retention, strata)
    stopped, stop_drift = checkpoint_drift(record/'stopped-partial.pt', plan, anchor, retention, strata)
    startup = read(record/'parent-readback.json')
    for checkpoint, metrics in ((initial_drift, dict(drift_rms=startup['parent_drift_rms_rad'],
                                                     drift_max=startup['parent_drift_max_rad'])),
                                (mid_drift, rows[24]), (stop_drift, rows[-1])):
        np.testing.assert_allclose([checkpoint['anchor_rms_rad'], checkpoint['anchor_max_rad']],
                                   [metrics['drift_rms'], metrics['drift_max']], atol=2e-5, rtol=0)
    per_update = plan['learner']['epochs']*plan['learner']['minibatches']
    adam = {}
    for role in ('actor', 'critic'):
        key = f'{role}_optimizer_state_dict'
        original = plan['inherited_adam_steps'][role]
        assert {int(item['step']) for item in initial[key]['state'].values()} == {original}
        adam[role] = []
        for point, count in ((mid, 25), (stopped, updates)):
            delta = sum(row['actor_gradient_steps'] for row in rows[:count]) if role == 'actor' else count*per_update
            assert point['completed_updates'] == count and point['control_steps'] == rows[count-1]['controls']
            assert point[f'{role}_steps_this_run'] == delta
            assert {int(item['step']) for item in point[key]['state'].values()} == {original+delta}
            for name, tensor in point[f'{role}_state_dict'].items():
                assert torch.isfinite(tensor).all()
                if not name.startswith('mlp.'):
                    exact(tensor, initial[f'{role}_state_dict'][name])
            adam[role].append(delta)
    assert not torch.cuda.is_initialized()
    figure = render_drift([startup['parent_drift_rms_rad'], *(r['drift_rms'] for r in rows)], cutoff)
    write(OUT/'stop-verification.json', dict(status='verified-guard-stop-saved-data-only',
          protocol_sha256=digest, coordinator_sha256=sha(OUT/'coordinator.json'), frozen_source_count=len(source_hashes),
          script_sha256=sha(Path(__file__)), control_record=coordinator['training_records']['control'],
          control_updates=updates, control_steps=controls, physical_substeps=state['substeps'],
          training_transitions=controls*plan['num_envs'], candidate_updates=0, evaluation_transitions=0,
          metric_sha256=sha(metric_path), cpu_check_new_physical_substeps=0, cpu_check_learning_updates=0,
          last_passing_update=updates-1, last_passing_rms_rad=rows[-2]['drift_rms'], stop_rms_rad=rows[-1]['drift_rms'],
          stop_absolute_rad=rows[-1]['drift_max'], max_absolute_rad=max(r['drift_max'] for r in rows),
          rms_limit_rad=cutoff, absolute_limit_rad=ceiling, elapsed_seconds=state['elapsed_seconds'],
          checkpoint_cpu_drift={'initial': initial_drift, 'update_25': mid_drift, 'stopped': stop_drift},
          adam_update_counts_at_25_and_stop=adam, releases=masks, videos=videos,
          replay_draws={k: v for k, v in draws.items() if k != 'sample_hashes'},
          candidate_effect_estimated=False, heldout_used=False, figure=figure))
    print('RETENTION_GUARD_VERIFIED', updates, controls*plan['num_envs'], rows[-1]['drift_rms'])


if __name__ == '__main__':
    main()
