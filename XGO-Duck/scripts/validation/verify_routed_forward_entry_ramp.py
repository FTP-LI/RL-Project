"""Observe the Windows run, or score both saved CUDA suffixes independently."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'configs/experiments/continuous_routed_forward_entry_ramp.json'
OUT = ROOT / 'records/reports/continuous-routed-forward-entry-ramp'
ROLES = ('step-command', 'ramped-command')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def observe(spec):
    import verify_walk_live_page as live

    assert platform.system() == 'Windows'
    record, page = ROOT / spec['record'], ROOT / spec['page']
    state, html = (record / 'state.json').read_bytes(), page.read_bytes()
    data = json.loads(state)
    assert data['status'] == 'starting' and data['controls'] == 0 and data['run_id'] == record.name
    assert record.name.encode() in html and b'http-equiv="refresh"' in html
    ready = dict(status='starting-page-observed', observer_host=platform.system(), run_id=record.name,
        observed_at=datetime.now().astimezone().isoformat(), protocol_sha256=sha(PLAN),
        initial_state_sha256=sha256(state).hexdigest(), page_sha256=sha256(html).hexdigest())
    with (record / 'windows-observer-ready.json').open('x', encoding='utf-8') as file:
        json.dump(ready, file, indent=2)
    print('ROUTED_ENTRY_RAMP_WINDOWS_READY', record.name, flush=True)
    with patch.object(sys, 'argv', [live.__file__, str(record), str(page), '--timeout', '750']):
        live.main()


def verify(spec):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from getup_release_fork_state import digest, signature
    from record_io import write
    from standup_metrics import projected_gravity

    record = ROOT / spec['record']
    data = read(record / 'evaluation.json')
    d = spec['pair']
    n, boundary, count = d['worlds'], d['shared_controls'], d['forward_controls_per_branch']
    gate = d['diagnostic_gate']
    assert data['status'] == 'completed' and data['actor_weights_unchanged']
    assert data['protocol_sha256'] == sha(PLAN) == sha(record / 'protocol.json')
    controls, substeps = d['maximum_unique_controls'], d['maximum_physics_substeps']
    assert data['actual'] == dict(controls=controls, attempted_controls=controls, substeps=substeps,
        attempted_substeps=substeps, resets=1) and data['transitions'] == d['maximum_transitions']
    assert not any(data[key] for key in ('learning_updates', 'heldout_used', 'hardware_io'))
    assert sha(record / 'release-poses.npz') == data['release_poses_sha256']
    with np.load(record / 'release-poses.npz', allow_pickle=False) as saved:
        poses = saved['poses']
    actual = {pose: int((poses == pose).sum()) for pose in ('prone', 'supine', 'left-side', 'right-side', 'other')}
    assert len(poses) == sum(actual.values()) == n and actual == data['release_coverage']
    assert actual['right-side'] >= d['minimum_actual_right_coverage']
    assert min(actual[pose] for pose in ('prone', 'supine', 'left-side')) >= d['minimum_other_actual_pose_coverage']
    fork = read(record / 'fork.json')
    assert sha(record / 'fork.json') == data['fork_sha256']
    assert len(fork['fields']) == fork['plant']['fields'] and digest(fork['fields']) == fork['plant']['state_sha256']
    assert fork['plant']['counters'] == [boundary, boundary * 5]
    for name, key in (('fork-input.npz', 'input_sha256'), ('fork-controller.npz', 'controller_file_sha256')):
        assert sha(record / name) == fork[key]
    with np.load(record / 'fork-input.npz', allow_pickle=False) as saved:
        initial = dict(saved)
    with np.load(record / 'fork-controller.npz', allow_pickle=False) as saved:
        assert digest(signature(dict(saved))) == fork['controller_sha256']
        assert not saved['feedback_history'].any() and not saved['feedback_previous'].any()
        assert not (saved['right'] & saved['supine']).any()
        history = saved['history']
    views = read(record / 'windows-view-verification.json')
    assert views['status'] == 'passed' and views['observer_host'] == 'Windows'

    common = None
    severe, scores, videos, keyframes = {}, {}, {}, {}
    indices = (300, 312, 325, 350)
    for role in ROLES:
        folder, state = record / role, data['branches'][role]
        assert sha(folder / 'restored-fork.json') == state['restore_sha256']
        assert read(folder / 'restored-fork.json') == dict(plant=fork['plant'], controller_sha256=fork['controller_sha256'])
        assert sha(folder / 'trace.npz') == state['trace_sha256']
        assert sha(folder / 'first-motor.npz') == state['motor_sha256']
        with np.load(folder / 'first-motor.npz', allow_pickle=False) as saved:
            motor = saved['substeps']
            assert motor.shape[:2] == (5, n) and np.isfinite(motor).all()
        with np.load(folder / 'trace.npz', allow_pickle=False) as saved:
            rows = dict(saved)
        assert all(np.isfinite(value).all() for value in rows.values())
        np.testing.assert_array_equal(rows['control'], np.arange(boundary + 1, boundary + count + 1))
        np.testing.assert_array_equal(rows['sensed'][0], initial['sensed'])
        np.testing.assert_array_equal(rows['motor_joint_target'], rows['target'])
        entered = rows['mode'][0] == 2
        assert (rows['mode'] == rows['mode'][0]).all() and entered.sum() == state['forward_ready']
        assert entered.sum() >= n * d['minimum_forward_ready_fraction']
        np.testing.assert_array_equal(rows['actor_input'][0, entered, 34:48], history[2, entered])
        np.testing.assert_array_equal(rows['actor_input'][1:, entered, 34:48], rows['raw'][:-1, entered])
        assert not rows['feedback_yaw'][:50].any() and rows['feedback_yaw'][50:, entered].any()
        expected_bridge = np.arange(count) < spec['sequence']['controller']['cross_model_bridge_controls']
        np.testing.assert_array_equal(rows['bridge'][:, entered], np.broadcast_to(expected_bridge[:, None], (count, entered.sum())))
        paired_inputs = rows['mode'][0], rows['actor_input'][0, :, :48], rows['actor_input'][0, :, 49:]
        if common is None:
            common = paired_inputs
        for value, expected in zip(paired_inputs, common, strict=True):
            np.testing.assert_array_equal(value, expected)
        assert not rows['actor_input'][:, ~entered, 48:51].any()
        ramp = d['branches'][role]['ramp_controls']
        target = np.asarray(d['branches'][role]['external_forward_m_s'] *
            np.minimum(1., np.arange(1, count + 1) / max(ramp, 1)), np.float32) * np.float32(1.4)
        np.testing.assert_array_equal(rows['actor_input'][:, entered, 48], np.broadcast_to(target[:, None], (count, entered.sum())))
        assert (rows['actor_input'][:, entered, 49:] == 0).all()
        jump = float(abs(rows['target'][0, entered] - initial['previous'][entered]).max())
        assert jump <= .0400002
        qpos = rows['qpos']
        gravity = projected_gravity(qpos[..., 3:7])
        tilt = np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1., 1.)))
        fallen = (tilt > gate['severe_tilt_deg']) | (qpos[..., 2] < gate['severe_height_min_m'])
        dwell = np.lib.stride_tricks.sliding_window_view(fallen, gate['sustained_failure_controls'], axis=0).all(axis=-1)
        severe[role] = dwell.any(axis=0) & entered
        yaw, speed = abs(rows['angular_velocity'][:, entered, 2]), rows['velocity'][:, entered, 0]
        scores[role] = dict(entered=int(entered.sum()), severe_worlds=int(severe[role].sum()),
            severe_by_release={pose: int((severe[role] & (poses == pose)).sum()) for pose in actual},
            mean_yaw_rad_s=float(yaw.mean(dtype=np.float64)), second_second_forward_speed_m_s=float(speed[50:].mean(dtype=np.float64)),
            initial_maximum_applied_target_step_rad=jump)

        video = state['video']
        file = record / video['path']
        assert video['full_action'] and video['environment_index'] == 0 and sha(file) == video['sha256']
        assert video['requested_seconds'] == (boundary+count)/50 == 14
        frame_steps = list(range(0, boundary+count+1, 2))
        saved_hashes = dict(video['qpos_sha256_by_frame'])
        assert frame_steps == video['common_steps'] == list(saved_hashes)
        assert saved_hashes[boundary] == sha256(initial['qpos'][0].tobytes()).hexdigest()
        for k in frame_steps[boundary//2+1:]:
            assert saved_hashes[k] == sha256(qpos[k-boundary-1, 0].tobytes()).hexdigest()
        frames = imageio_ffmpeg.read_frames(str(file), pix_fmt='rgb24')
        info = next(frames)
        assert info['size'] == (640, 480) and info['fps'] == 25
        keyframes[role] = {}
        for index, rgb in enumerate(frames):
            if index in indices:
                keyframes[role][index] = Image.frombytes('RGB', info['size'], rgb)
        assert index + 1 == video['frames'] == len(frame_steps)
        videos[role] = dict(path=file.relative_to(ROOT).as_posix(), sha256=sha(file), seconds=14, frames=index+1,
            shared_qpos_sha256=digest({k: saved_hashes[k] for k in frame_steps if k <= boundary}))

    assert videos[ROLES[0]]['shared_qpos_sha256'] == videos[ROLES[1]]['shared_qpos_sha256']
    a, b = (scores[key] for key in ROLES)
    improvement = a['severe_worlds'] - b['severe_worlds']
    yaw_increase = b['mean_yaw_rad_s'] - a['mean_yaw_rad_s']
    speed_loss = a['second_second_forward_speed_m_s'] - b['second_second_forward_speed_m_s']
    checks = dict(reference_coverage=a['severe_worlds'] >= gate['minimum_reference_severe_worlds'],
        relative_reduction=improvement >= gate['minimum_relative_reduction_in_severe_worlds'] * a['severe_worlds'],
        absolute_reduction=improvement/n >= gate['minimum_absolute_reduction_in_severe_fraction'],
        no_yaw_increase=yaw_increase <= gate['maximum_yaw_mean_increase_rad_s'],
        second_second_speed_retained=speed_loss <= gate['maximum_second_second_forward_speed_loss_m_s'])
    rescued = severe[ROLES[0]] & ~severe[ROLES[1]]
    harmed = ~severe[ROLES[0]] & severe[ROLES[1]]
    out = OUT / 'analysis.json'
    assert not out.exists()
    image = Image.new('RGB', (1292, 570), '#f9fbfe')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 19)
    draw.text((14, 9), f'同一站立状态的前进起步 · 场景 0 · 实际释放 {poses[0]}', font=font, fill='#193147')
    for r, role in enumerate(ROLES):
        for c, k in enumerate(indices):
            x, y = 14 + c*319, 40 + r*258
            image.paste(keyframes[role][k].resize((304, 228), Image.Resampling.LANCZOS), (x, y))
            label = ('立即指令', '半秒渐升')[r]
            draw.text((x, y+229), f'{label} · {k/25:.2f} 秒', font=font, fill='#193147')
    image.save(OUT / 'entry-keyframes.jpg', quality=94)
    telemetry = read(record / 'gpu-telemetry.json')
    write(out, dict(status='passed-independent-verification', protocol_sha256=sha(PLAN), record=spec['record'],
        run_sha256=sha(record / 'evaluation.json'), branches=scores, videos=videos, keyframes_sha256=sha(OUT / 'entry-keyframes.jpg'),
        release_coverage=actual, physical_fork_fields=fork['plant']['fields'], unique_controls=controls,
        physical_substeps=substeps, resets=1, transitions=data['transitions'],
        wall_seconds=data['sampling_wall_seconds'], gpu_peak_mib=data['gpu']['peak_mib'],
        gpu_mean_percent=float(np.mean([row['gpu_percent'] for row in telemetry])),
        windows_running_snapshots=views['running_snapshots'], rescued=int(rescued.sum()), harmed=int(harmed.sum()),
        net_severe_reduction=improvement, absolute_reduction=improvement/n,
        yaw_increase_rad_s=yaw_increase, second_second_speed_loss_m_s=speed_loss, checks=checks,
        passed_diagnostic_gate=all(checks.values()), heldout_used=False, learning_updates=0))
    print('ROUTED_ENTRY_RAMP_VERIFIED', {k: v['severe_worlds'] for k, v in scores.items()}, checks, sha(out), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('observe', 'verify'))
    mode = parser.parse_args().mode
    plan = read(PLAN)
    assert plan['status'] == 'frozen-routed-forward-entry-ramp'
    (observe if mode == 'observe' else verify)(plan)
