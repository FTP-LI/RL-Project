"""One bounded GPU check of observation files, with the official fixed actor."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from getup_release_coverage import PLAN as PARENT, actor_model, checked, prepare
from evaluate_standup_reference import Tee

sys.path.insert(0, str(ROOT / 'scripts/training'))
from action_video import ActionVideo
from record_io import write
from training_live_page import publish
from training_watch import Snapshot

PLAN = ROOT / 'configs/experiments/local_observation.json'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def freeze():
    assert not PLAN.exists()
    parent = checked()
    role = deepcopy(parent['roles']['original'])
    role.update(num_envs=32, seed=3061)
    role['split']['pose_codes'] = [1, 0, 2, 3] * 8
    cfg, _, _ = prepare(role)
    assert cfg.env.scene.num_envs == 32 and cfg.env.episode_length_s == 11
    assert not torch.cuda.is_initialized()
    run_id = 'duck_getup_observation_s3061_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    sources = dict(parent['source_sha256'])
    for name in (Path(__file__), ROOT/'scripts/training/training_live_page.py', ROOT/'scripts/training/action_video.py',
                 ROOT/'configs/environments/action_recording.json', PARENT):
        sources[name.relative_to(ROOT).as_posix()] = sha(name)
    spec = dict(run_id=run_id, role=role, source_sha256=sources, episodes=2, controls_each=550,
                max_wall_seconds=240, learning_updates=0, heldout_used=False, freeze_cpu_only=True,
                purpose='运行中观察检查 · 官方固定起身策略')
    write(PLAN, spec)
    print('OBSERVATION_FROZEN', run_id, sha(PLAN), flush=True)


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    import onnxruntime as ort
    import imageio_ffmpeg

    spec = read(PLAN)
    assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
    checked()
    for name, digest in spec['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    role = spec['role']
    record = ROOT/'records/experiments'/spec['run_id']
    record.mkdir()
    write(record/'protocol.json', spec)
    (record/'README.md').write_text(
        f'# {record.name}\n\n以官方固定起身策略采集真实 GPU 轨迹，验收运行中查看。'
        '沿用共同起身物理、11 秒动作与 2 秒释放。只改观察入口及规模，32 环境、2 个完整观察窗，'
        '种子 3061，零学习与留出。结论以状态、日志和实际录像为准。\n\n'
        '[状态与曲线数据](state.json) · [冻结参数](protocol.json) · [原片](clip-step-0.mp4) · [执行日志](execution.log)\n',
        encoding='utf-8')
    maximum = spec['episodes'] * spec['controls_each']
    state = dict(run_id=record.name, status='starting', purpose=spec['purpose'], updated_at=now(),
                 controls=0, maximum_controls=maximum, transitions=0, learning_updates=0, updates=[],
                 protocol_sha256=sha(PLAN))
    write(ROOT/'records/training-current.json', dict(record=record.relative_to(ROOT).as_posix(), run_id=record.name,
                                                   view='records/training-live.html'))
    env = snapshot = video = None
    begin = time.monotonic()
    with (record/'execution.log').open('x', encoding='utf-8', buffering=1) as log, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        def report(message):
            print(message, flush=True)

        def update():
            state['updated_at'] = now()
            publish(record, state)

        try:
            report(f'OBSERVATION_START {record.name} {state["protocol_sha256"]} argv={sys.argv!r}')
            update()
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, _, _ = prepare(role)
            env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            env.seed(role['seed'])
            obs, _ = env.reset()
            actor = actor_model(role, env.device)
            session = ort.InferenceSession(str(ROOT/role['model']), providers=['CPUExecutionProvider'])
            x = obs['actor'][:8].cpu().numpy()
            expected = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in x])
            actual = actor(obs)[:8].cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            assert env.common_step_counter == env._sim_step_counter == 0 and env.step_dt == .02
            write(record/'zero-step.json', dict(status='passed', max_abs=float(np.max(np.abs(actual-expected))),
                                               device=str(env.device), physical_substeps=0))
            snapshot = Snapshot(env, record)
            video = ActionVideo(snapshot, 'getup', record)
            video.sample()
            state.update(status='running', frame=video.image['file'])
            update()
            interval = time.monotonic()
            last = 0
            zeros = torch.zeros((role['num_envs'], 14), device=env.device)
            with torch.inference_mode():
                for episode in range(spec['episodes']):
                    if episode:
                        obs, _ = env.reset()
                    for i in range(1, spec['controls_each']+1):
                        obs, reward, _, _, _ = env.step(zeros if i <= 100 else actor(obs))
                        if episode == 0:
                            video.sample()
                        step = int(env.common_step_counter)
                        if step % 50 == 0:
                            assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
                            assert time.monotonic() - begin < spec['max_wall_seconds'], 'Observation wall cap'
                            assert torch.isfinite(reward).all() and torch.isfinite(obs['actor']).all()
                            torch.cuda.synchronize()
                            end = time.monotonic()
                            rate = (step-last)*role['num_envs']/(end-interval)
                            state.update(controls=step, transitions=step*role['num_envs'])
                            if video.clips:
                                state['latest_clip'] = video.clips[-1]['file']
                            state['updates'].append(dict(control=step, transitions_per_second=rate, at=now()))
                            update()
                            report(f'OBSERVATION_PROGRESS {step}/{maximum} {rate:.1f} transitions/s')
                            last, interval = step, time.monotonic()
            assert state['controls'] == maximum and env._sim_step_counter == maximum*cfg.env.decimation
            assert len(video.clips) == 1 and video.clips[0]['full_action']
            clip = record/state['latest_clip']
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(clip))
            assert frames == video.clips[0]['frames'] == 276
            state.update(status='completed', video_sha256=sha(clip), decoded_frames=frames,
                         physical_substeps=env._sim_step_counter, wall_seconds=time.monotonic()-begin)
            report(f'OBSERVATION_COMPLETED {frames} frames {state["transitions"]} transitions')
        except BaseException as exc:
            state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
            report(state['error'])
            raise
        finally:
            try:
                if video is not None:
                    video.close()
                if snapshot is not None:
                    snapshot.close()
                if env is not None:
                    env.close()
            finally:
                update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    (freeze if parser.parse_args().freeze else run)()
