"""Bounded walking repair: sample actual gait, save at update boundaries, stop.

Change: replace constant rendering / reward-only supervision with a sampled
pose watchdog and one actual image every three minutes. Why: old runs kept
learning with inward hips and lost complete updates when the viewer closed.
This module never advances physics or selects policy actions. It does not
change rewards. Thresholds are tripwires, not proof of successful locomotion.
Recovery/roll skills require their own phase-aware watchdog, not this profile.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/experiments/training_watch.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run_name(skill, change, arm, seed, stamp=None):
    if skill not in {'walk', 'stand', 'standup', 'roll', 'switch'}:
        raise ValueError('Unknown skill')
    if arm not in {'reference', 'control', 'candidate'}:
        raise ValueError('Unknown arm')
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', change):
        raise ValueError('Use descriptive lowercase words separated by hyphens')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    stamp = stamp or datetime.now().strftime('%Y%m%dT%H%M%S')
    datetime.strptime(stamp, '%Y%m%dT%H%M%S')
    return f'duck_{skill}_{change}_{arm}_s{seed}_{stamp}'


def signed_foot_gap_mm(feet, quat):
    """Quaternion order wxyz; signed left-minus-right site distance in body y."""
    w, x, y, z = np.moveaxis(np.asarray(quat), -1, 0)
    left = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), axis=-1)
    return np.sum((feet[..., 0, :]-feet[..., 1, :])*left, axis=-1)*1000


class GaitWatch:
    def __init__(self, config, nenv, repair=None):
        self.cfg = config
        self.nenv = nenv
        self.elapsed = 0.0
        self.valid = self.inward = self.crossed = 0
        self.falls = np.zeros(nenv, dtype=bool)
        self.streak = {k: 0 for k in ('inward', 'crossed', 'falls', 'no-samples')}
        self.windows = []
        self.stop_reason = None
        self.repair = repair
        self.completed_updates = 0
        self.review = None
        if repair:
            if (not set(repair['inherited_faults']) <= {'inward', 'crossed'}
                    or not 1 <= repair['review_after_updates'] <= 50):
                raise ValueError('Repair may defer declared posture faults for at most 50 updates')

    def complete_update(self, completed):
        self.completed_updates = completed
        if self.repair and completed >= self.repair['review_after_updates']:
            fields = ('inward_fraction', 'crossed_fraction', 'fallen_environment_fraction')
            self.review = {key: dict(first_windows=float(np.mean([w[key] for w in self.windows[:2]])),
                                     last_windows=float(np.mean([w[key] for w in self.windows[-2:]])))
                           for key in fields} if self.windows else {'no_windows': True}
            self.stop_reason = self.stop_reason or 'repair-window-complete-review'

    def record_falls(self, flags):
        self.falls |= np.asarray(flags, dtype=bool)

    def sample(self, *, hips_deg, gap_mm, age_s, dt):
        hips_deg, gap_mm, age_s = map(np.asarray, (hips_deg, gap_mm, age_s))
        if (hips_deg.shape != (self.nenv, 2) or gap_mm.shape != (self.nenv,)
                or age_s.shape != (self.nenv,)):
            raise ValueError('Watch telemetry shape mismatch')
        if not all(np.isfinite(x).all() for x in (hips_deg, gap_mm, age_s)):
            self.stop_reason = 'nonfinite-gait-state'
            return None
        valid = age_s >= self.cfg['settling_seconds']
        self.valid += int(valid.sum())
        self.inward += int((valid & (hips_deg[:, 0] < -self.cfg['inward_hip_degrees'])
                            & (hips_deg[:, 1] > self.cfg['inward_hip_degrees'])).sum())
        self.crossed += int((valid & (gap_mm < self.cfg['crossed_gap_mm'])).sum())
        self.elapsed += dt
        if self.elapsed + 1e-9 < self.cfg['window_seconds']:
            return None
        sufficient = self.valid >= self.cfg['minimum_valid_samples']
        row = dict(valid_samples=self.valid, sampled_envs=self.nenv,
                   inward_fraction=self.inward/max(1, self.valid),
                   crossed_fraction=self.crossed/max(1, self.valid),
                   fallen_environment_fraction=float(self.falls.mean()))
        flags = dict(inward=sufficient and row['inward_fraction'] > self.cfg['inward_fraction_max'],
                     crossed=sufficient and row['crossed_fraction'] > self.cfg['crossed_fraction_max'],
                     falls=row['fallen_environment_fraction'] > self.cfg['fallen_environment_fraction_max'],
                     **{'no-samples': not sufficient})
        for k, flag in flags.items():
            self.streak[k] = self.streak[k]+1 if flag else 0
        row['warning'] = [k for k, v in flags.items() if v]
        row['streak'] = dict(self.streak)
        reasons = [k for k, v in self.streak.items() if v >= self.cfg['consecutive_bad_windows']]
        # A bad parent must get the declared short opportunity to change. New
        # faults, falls and invalid telemetry retain the normal stop rule.
        # At the fixed review boundary we always save and stop for analysis.
        if self.repair and self.completed_updates < self.repair['review_after_updates']:
            row['deferred_inherited_faults'] = [k for k in reasons if k in self.repair['inherited_faults']]
            reasons = [k for k in reasons if k not in self.repair['inherited_faults']]
        if reasons:
            self.stop_reason = 'sustained-' + '+'.join(reasons)
        self.windows.append(row)
        self.elapsed = 0.0
        self.valid = self.inward = self.crossed = 0
        self.falls[:] = False
        return row


class ControlledStop(Exception):
    """Complete update was saved; caller must exit, not start the next arm."""


class UnsafeState(Exception):
    """Partial or nonfinite state: retain prior checkpoint, do not save as complete."""


class UpdateBoundary:
    def __init__(self, cfg, record, clock=time.monotonic):
        self.cfg, self.record, self.clock = cfg, Path(record), clock
        self.last_time = clock()
        self.last_iteration = None
        self.last_checkpoint = None

    def complete(self, runner, iteration, reason=None):
        if runner.current_learning_iteration != iteration:
            raise ValueError('Stop hook is not after a completed optimizer update')
        due = (self.last_iteration is None or reason is not None
               or self.clock()-self.last_time >= self.cfg['checkpoint_interval_seconds']
               or iteration-self.last_iteration >= self.cfg['checkpoint_interval_updates'])
        if due:
            path = Path(runner.logger.log_dir).resolve() / f'model_{iteration}.pt'
            runner.save(str(path))
            self.last_checkpoint = str(path)
            self.last_iteration, self.last_time = iteration, self.clock()
            write(self.record/'last-checkpoint.json', dict(checkpoint=str(path), iteration=iteration,
                  common_step_counter=int(runner.env.unwrapped.common_step_counter), complete_update=True))
        if reason:
            write(self.record/'stop.json', dict(reason=reason, checkpoint=self.last_checkpoint,
                  iteration=iteration, complete_update=True, automatic_retry=False))
            raise ControlledStop(reason)


class Snapshot:
    """One reconstructed actual state; no policy inference or physics integration."""
    def __init__(self, env, record):
        import copy
        import mujoco
        self.env, self.record = env.unwrapped, Path(record)
        self.mj = mujoco
        self.model = copy.deepcopy(self.env.sim.mj_model)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = .8, 135, -15
        ids = [i for i in range(self.model.nbody)
               if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or '').endswith('trunk_base')]
        if len(ids) != 1:
            raise ValueError('Unknown root body for snapshot')
        self.root_body = ids[0]

    def render(self):
        from live_observer import VISUAL_FIELDS
        def cpu(t): return t.detach().cpu().numpy()
        sim = self.env.sim
        for name in sim.expanded_fields & VISUAL_FIELDS:
            dest = getattr(self.model, name)
            dest[:] = cpu(getattr(sim.model, name)[0]).reshape(dest.shape)
        for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'xfrc_applied'):
            dest = getattr(self.data, name)
            if dest.size:
                dest[:] = cpu(getattr(sim.data, name)[0]).reshape(dest.shape)
        self.mj.mj_forward(self.model, self.data)
        self.cam.lookat[:] = self.data.xpos[self.root_body]
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render().copy()

    def capture(self):
        from PIL import Image
        pixels = self.render()
        step = int(self.env.common_step_counter)
        name = f'frame-step-{step}.jpg'
        Image.fromarray(pixels).save(self.record/name, quality=88)
        return dict(file=name, captured_at=datetime.now(timezone.utc).isoformat(),
                    captured_unix=time.time(), common_step=step, environment_index=0,
                    source='actual-training-state', extra_physics_steps=0)

    def close(self):
        self.renderer.close()


class SparseVideo:
    """Record actual state every two 20-ms steps during a short periodic burst.

    The learner is the only owner of physics. Rendering reconstructs a copy;
    no extra rollout is generated and no policy action is changed for filming.
    """
    def __init__(self, snapshot, config, record):
        import mediapy
        import imageio_ffmpeg
        self.snapshot, self.cfg, self.record = snapshot, config, Path(record)
        self.media = mediapy
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        self.env = snapshot.env
        self.stride = round(1/(config['video_fps']*self.env.step_dt))
        if self.stride < 1 or abs(self.stride*self.env.step_dt*config['video_fps']-1) > 1e-6:
            raise ValueError('Video rate must match an integer number of actual control steps')
        self.horizon = round(config['video_seconds']/self.env.step_dt)
        self.writer = None
        self.last_started = None
        self.clips = []
        self.image = None
        self.elapsed_seconds = 0.

    def sample(self):
        from PIL import Image, ImageDraw
        start = time.monotonic()
        step = int(self.env.common_step_counter)
        if self.writer is None:
            if self.last_started is not None and start-self.last_started < self.cfg['visual_interval_seconds']:
                return
            self.last_started = start
            self.start_step = step
            self.frame_steps = []
            self.path = self.record/f'clip-step-{step}.mp4'
            self.writer = self.media.VideoWriter(str(self.path),shape=(480,640),fps=self.cfg['video_fps'],qp=24)
            self.writer.__enter__()
        if (step-self.start_step) % self.stride:
            return
        pixels = Image.fromarray(self.snapshot.render())
        draw = ImageDraw.Draw(pixels)
        draw.rectangle((0,0,640,46),fill=(17,30,41))
        draw.text((8,5),f'Actual training env 0 | step {step} | t={(step-self.start_step)*self.env.step_dt:.2f}s',fill='white')
        draw.text((8,26),self.record.name,fill='white')
        self.writer.add_image(np.asarray(pixels))
        self.frame_steps.append(step)
        if len(self.frame_steps)==1:
            name=f'frame-step-{step}.jpg';pixels.save(self.record/name,quality=88)
            self.image=dict(file=name,captured_at=datetime.now(timezone.utc).isoformat(),captured_unix=time.time(),
                common_step=step,environment_index=0,source='actual-training-state',extra_physics_steps=0)
        if step-self.start_step >= self.horizon:
            self.finish(complete=True)
        self.elapsed_seconds += time.monotonic()-start

    def finish(self, complete=False):
        if self.writer is None:
            return
        self.writer.__exit__(None,None,None)
        self.writer = None
        self.clips.append(dict(file=self.path.name,complete_clip=complete,source='actual-training-state',
            environment_index=0,frames=len(self.frame_steps),fps=self.cfg['video_fps'],
            common_steps=self.frame_steps,frame_times_seconds=[(s-self.start_step)*self.env.step_dt for s in self.frame_steps],
            captured_at=datetime.now(timezone.utc).isoformat(),captured_unix=time.time(),extra_physics_steps=0,
            resets_policy='Actual environment 0 including training resets; not a fixed evaluation'))
        write(self.record/'clips.json',self.clips)
