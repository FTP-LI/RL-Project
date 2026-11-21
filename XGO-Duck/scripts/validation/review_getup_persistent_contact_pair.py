"""Review a closed pair from recorded training metrics and recovery traces."""
from pathlib import Path
import json

import numpy as np

from audit_getup_contact_duration import age_seconds
from s1_reference_inputs import ROOT, read, sha
from record_io import write

OUT = ROOT / 'records/reports/getup-persistent-contact/pair'
ASSETS = ROOT / 'docs/reports/assets'
LABELS = dict(original='官方', control='原奖励', candidate='持续接触惩罚')
COLORS = dict(original='#59636d', control='#997334', candidate='#2479a8')


def evaluate_saved(role, path):
    proof = read(path)
    assert proof['status'] == 'passed' and proof['continuous_no_physical_reset']
    record = ROOT / proof['record']
    trace = record / 'coverage-trace.npz'
    assert sha(trace) == proof['trace_sha256']
    criteria = read(record / 'protocol.json')['criteria']
    passed = np.asarray(proof['end_hold']['stable_by_environment'], dtype=bool)
    cases = read(record / 'coverage.json')['episodes']
    actual = np.asarray([row['release_pose'] for row in cases])
    assert passed.shape == actual.shape == (320,)
    assert int(passed.sum()) == proof['end_hold']['overall']['timely_holds']
    with np.load(trace) as data:
        t = data['time']
        contact = data['native_head_leg_contact']
        assert contact.shape == (550, 320)
        peak = age_seconds(contact & (t > 2 + 1e-8)[:, None])[(t > 2 + 1e-8) & (t <= 8 + 1e-8)].max(0)
        persistent = peak >= 1.2 - 1e-8
        final = t > 8 + 1e-8
        late = contact[final].any(0)
        q = data['quaternion'][final]
        vertical = 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2) / np.sum(q * q, axis=-1)
        tilt = np.rad2deg(np.arccos(np.clip(vertical, -1, 1)))
        limits = dict(height=abs(data['height'][final] - criteria['nominal_height_m']) > criteria['height_tolerance_m'],
                      tilt=tilt > criteria['tilt_max_deg'],
                      speed=np.linalg.norm(data['angular_velocity'][final], axis=-1) > criteria['angular_speed_max_rad_s'],
                      feet=~data['foot_contact'][final].all(-1),
                      joint_envelope=~data['joint_envelope_ok'][final])
        violations = {key: int((value.any(0) & ~passed).sum()) for key, value in limits.items()}
        prefix = data['qpos'][:100].copy()
        start = data['initial_qpos'].copy()
        frozen_targets = data['applied_targets'][:100].copy()
    summary = dict(role=role, successes=int(passed.sum()), failures=int((~passed).sum()),
                   persistent_in_successes=int((persistent & passed).sum()),
                   persistent_in_failures=int((persistent & ~passed).sum()),
                   final_window_contact_in_failures=int((late & ~passed).sum()),
                   final_window_violations_in_failures=violations,
                   failed_actual_supine=int(((actual == 'supine') & ~passed).sum()),
                   trace_sha256=proof['trace_sha256'], proof_sha256=sha(path))
    return summary, passed, actual, (prefix, start, frozen_targets), proof['end_hold']['overall']


def figures(training, outcomes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from figure_style import apply_figure_style
    from render_roll_angle_results import checked_save

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=.15, right=.96, bottom=.29, top=.87)
    peaks = []
    for role, rows in training.items():
        updates = [row['completed_updates'] for row in rows]
        drift = [row['raw_action_drift_rms'] for row in rows]
        peaks.append(max(drift))
        ax.plot(updates, drift, label=LABELS[role], color=COLORS[role], linewidth=1.8)
        ax.scatter(updates[-1], drift[-1], color=COLORS[role], s=24)
    ax.axhline(.05, color='#8b5b31', linestyle='--', label='停止线')
    drift_top = max(.06, max(peaks) * 1.1)
    ax.set(xlabel='完整PPO更新次数', ylabel='成功输入的输出均方根偏离（rad）',
           xlim=(0, 52), xticks=range(0, 51, 10), ylim=(0, drift_top),
           yticks=np.linspace(0, drift_top, 7))
    ax.set_title('输出偏离', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.27), ncols=3)
    result = [checked_save(fig, ASSETS / 'getup-persistent-contact-drift.png')]
    if outcomes:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=.14, right=.94, bottom=.21, top=.88)
        bottom = min(item['wilson95'][0] for item in outcomes.values()) * 100
        for x, (role, item) in enumerate(outcomes.items()):
            rate = item['rate']
            lo, hi = item['wilson95']
            ax.errorbar(x, rate * 100, yerr=100 * np.array([[rate - lo], [hi - rate]]),
                        fmt='o', color=COLORS[role], capsize=5, markersize=7)
            ax.annotate(f'{item["timely_holds"]}/{item["episodes"]}', (x, rate * 100),
                        xytext=(10, -7), textcoords='offset points', color=COLORS[role])
        ax.axhline(95, color='#8b5b31', linestyle='--', label='总体门槛')
        lower = max(0, min(82, bottom - 3))
        ax.set(xticks=range(len(outcomes)), xticklabels=[LABELS[r] for r in outcomes],
               xlim=(-.4, len(outcomes) - .4), ylim=(lower, 100),
               yticks=np.arange(np.ceil(lower / 5) * 5, 101, 5),
               ylabel='最后三秒持稳比例（%）')
        ax.set_title('起身持稳', loc='left')
        ax.legend(loc='lower left')
        result.append(checked_save(fig, ASSETS / 'getup-persistent-contact-hold.png'))
    return result


def sequence(roles):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    width, row_height = 320, 306
    canvas = Image.new('RGB', (5 * width, 76 + len(roles) * row_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, small = (ImageFont.truetype(font_path, size) for size in (24, 17))
    draw.text((10, 8), '起身关键帧', fill='#15334b', font=title)
    draw.text((10, 43), '预定仰卧相机。事件时刻与原片采样时刻分别标注。', fill='#15334b', font=small)
    provenance = []
    for row, role in enumerate(roles):
        proof = read(OUT / f'{role}-analysis.json')
        record = ROOT / proof['record']
        plan = read(record / 'protocol.json')
        capture = read(record / 'evaluation.json')
        info = next(case['video'] for case in capture['conditions'] if case['condition'] == 'supine')
        world = info['environment_index']
        assert world == plan['video_worlds']['supine']
        video = record / info['path']
        assert info['complete_clip'] and sha(video) == info['sha256']
        case = read(record / 'coverage.json')['episodes'][world]
        with np.load(record / 'coverage-trace.npz') as trace:
            ts = trace['time']
            hit = ts[(ts > 2 + 1e-8) & (ts <= 8 + 1e-8) & trace['native_head_leg_contact'][:, world]]
        events = [('释放', 2., 2.), ('首次头腿接触', float(hit[0]) if len(hit) else None, 2.),
                  ('及时持稳起点', case['hold_start_seconds'], 8.), ('动作期限', 8., 8.), ('终态', 11., 11.)]
        times = np.asarray(info['frame_times_seconds'])
        indices = [int(np.searchsorted(times, (actual if actual is not None else fallback) - 1e-8))
                   for _, actual, fallback in events]
        frames = {}
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        meta = next(reader)
        assert meta['fps'] == 25 and meta['size'] == (640, 480)
        for i, data in enumerate(reader):
            if i in indices:
                frames[i] = Image.frombytes('RGB', meta['size'], data).resize((304, 228), Image.Resampling.LANCZOS)
        assert set(indices) == frames.keys() and i + 1 == len(times) == 276
        y = 76 + row * row_height
        draw.text((10, y), f'{LABELS[role]} · 环境{world} · 首次起身' + ('通过' if case['success'] else '未通过'),
                  fill=COLORS[role], font=title)
        for col, ((label, actual, _), index) in enumerate(zip(events, indices)):
            x = col * width + 8
            event = f'{label} {actual:.2f}s' if actual is not None else f'{label}未发生'
            assert draw.textlength(event, font=small) < width - 16
            draw.text((x, y + 33), event, fill='#15334b', font=small)
            draw.text((x, y + 53), f'原片 {times[index]:.2f}s', fill='#15334b', font=small)
            canvas.paste(frames[index], (x, y + 75))
            provenance.append(dict(role=role, env_index=world, event=label, event_seconds=actual,
                                   frame_index=index, frame_seconds=float(times[index]),
                                   video=str(video.relative_to(ROOT)), video_sha256=info['sha256'],
                                   qpos_sha256=info['qpos_sha256_by_frame'][index]))
    target = ASSETS / 'getup-persistent-contact-sequence.jpg'
    canvas.save(target, quality=94)
    return dict(path=str(target.relative_to(ROOT)), sha256=sha(target), frames=provenance)


def main():
    summary = read(OUT / 'analysis.json')
    assert read(OUT / 'coordinator.json')['status'] == 'completed-review-required'
    plan = ROOT / 'configs/experiments/getup_persistent_contact_pair.json'
    assert summary['protocol_sha256'] == sha(plan) and not summary['heldout_used']
    training, traces, curves, outcomes, evaluated = {}, {}, {}, {}, {}
    for role in ('control', 'candidate'):
        path = OUT / role / 'training-verification.json'
        if not path.exists():
            continue
        proof = read(path)
        assert proof['status'] == 'passed' and proof['contact_clock_and_cost_verified']
        record = ROOT / proof['record']
        metrics = record / 'learning-metrics.jsonl'
        curves[role] = [json.loads(line) for line in metrics.read_text().splitlines()]
        state = read(record / 'state.json')
        training[role] = dict(updates=proof['rollout_updates'], transitions=proof['actual_control_transitions'],
                              drift_rms=proof['successful_state_drift_rms'],
                              drift_max=proof['successful_state_drift_max'],
                              stop_reason=proof['stop_reason'], seconds=state['elapsed_seconds'],
                              contact=proof['contact_trace'], metrics_sha256=sha(metrics),
                              clips=proof['full_action_clips'])
    for role in ('original', 'control', 'candidate'):
        path = OUT / f'{role}-analysis.json'
        if not path.exists():
            continue
        item, passed, actual, prefix, outcome = evaluate_saved(role, path)
        evaluated[role], traces[role], outcomes[role] = item, (passed, actual, prefix), outcome
        assert summary['final_hold_counts'][role] == item['successes']
    assert sum(item['transitions'] for item in training.values()) == summary['actual_learning_transitions']
    assert len(evaluated) * 176000 == summary['actual_evaluation_transitions']
    contrast = None
    if 'candidate' in traces and 'control' in traces:
        control, pose0, (q0, start0, targets0) = traces['control']
        candidate, pose1, (q1, start1, targets1) = traces['candidate']
        np.testing.assert_array_equal(start0, start1)
        np.testing.assert_array_equal(targets0, targets1)
        pose0_q, pose1_q = (q[..., 3:7].astype(np.float64) for q in (q0, q1))
        cosine = abs(np.sum(pose0_q * pose1_q, axis=-1))
        cosine /= np.linalg.norm(pose0_q, axis=-1) * np.linalg.norm(pose1_q, axis=-1)
        orientation = np.rad2deg(2 * np.arccos(np.clip(cosine, 0, 1)))
        translation = np.linalg.norm(q0[..., :3] - q1[..., :3], axis=-1)
        joints = np.rad2deg(abs(q0[..., 7:] - q1[..., 7:]))
        changed = np.flatnonzero(np.any(q0 != q1, axis=(1, 2)))
        contrast = dict(candidate_only=int((candidate & ~control).sum()),
                        control_only=int((control & ~candidate).sum()),
                        both_success=int((control & candidate).sum()),
                        both_failure=int((~control & ~candidate).sum()),
                        actual_release_pose_differences=int((pose0 != pose1).sum()),
                        first_different_control=int(changed[0] + 1) if len(changed) else None,
                        frozen_targets_and_initial_qpos_exact=True,
                        max_pre_release_translation_m=float(translation.max()),
                        max_pre_release_orientation_deg=float(orientation.max()),
                        max_pre_release_joint_difference_deg=float(joints.max()),
                        release_translation_m_max=float(translation[-1].max()),
                        release_orientation_deg_max=float(orientation[-1].max()),
                        release_joint_difference_deg_max=float(joints[-1].max()))
    report = dict(status='passed-saved-data-only', protocol_sha256=sha(plan),
                  input_summary_sha256=sha(OUT / 'analysis.json'), script_sha256=sha(Path(__file__)),
                  training=training, evaluated=evaluated, contrast=contrast,
                  new_physics_steps=0, new_learning_updates=0, heldout_used=False)
    write(OUT / 'saved-data-review.json', report)
    write(OUT / 'figure-review.json', dict(status='rendered-awaiting-visual-review',
          figures=figures(curves, outcomes), sequence=sequence(outcomes) if outcomes else None,
          data_review_sha256=sha(OUT / 'saved-data-review.json')))
    print(json.dumps({k: v for k, v in report.items() if k != 'training'}, indent=2))


if __name__ == '__main__':
    main()
