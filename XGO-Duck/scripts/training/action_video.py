"""Skill-aware recording for new entries; frozen old recorders stay unchanged.

Recovery and roll are transient actions: a3s excerpt can hide the fall, release
or landing. Wait for an episode boundary, then keep the complete episode.
Capture its terminal state through the recorder hook BEFORE automatic reset,
so the last frame never silently becomes the next episode's initial state.
Walking keeps short periodic clips. This class never calls env.step/reset.
"""
import time
from pathlib import Path
import torch
from training_watch import SparseVideo,read
from record_io import write

ROOT=Path(__file__).resolve().parents[2]
POLICY=ROOT/'configs/environments/action_recording.json'


def recording_plan(skill,episode_seconds,step_dt,sequence_seconds=None):
    policy=read(POLICY);profile=policy['profiles'][skill];mode=profile['mode']
    if mode=='short':seconds=profile['seconds']
    elif mode=='complete-episode':seconds=episode_seconds
    else:
        assert sequence_seconds is not None and sequence_seconds>0,'Declare the entire transition sequence duration'
        seconds=sequence_seconds
    steps=round(seconds/step_dt)
    assert steps>0 and abs(steps*step_dt-seconds)<1e-6
    return dict(video_fps=policy['fps'],video_seconds=float(seconds),
        visual_interval_seconds=policy['interval_wall_seconds'],mode=mode,skill=skill)


class ActionVideo(SparseVideo):
    def __init__(self,snapshot,skill,record,sequence_seconds=None):
        env=snapshot.env
        self.plan=recording_plan(skill,env.cfg.episode_length_s,env.step_dt,sequence_seconds)
        super().__init__(snapshot,self.plan,record)
        self.complete_action=self.plan['mode']!='short'
        self.episode_start_step=None
        self.original_pre_reset=None
        if self.complete_action:
            self.original_pre_reset=env.recorder_manager.record_pre_reset
            def before_reset(env_ids):
                if self.writer is not None and bool((env_ids==0).any()):
                    self._sample_terminal()
                self.original_pre_reset(env_ids)
            env.recorder_manager.record_pre_reset=before_reset

    def sample(self):
        age=int(self.env.episode_length_buf[0]);step=int(self.env.common_step_counter)
        if self.writer is None and self.complete_action:
            if self.last_started is not None and time.monotonic()-self.last_started<self.cfg['visual_interval_seconds']:return
            if age!=0:return  # A due interval must not start halfway through getup.
            self.episode_start_step=step
        if self.writer is not None and self.complete_action:
            assert age==step-self.episode_start_step,'Episode reset without terminal capture hook'
        super().sample()

    def _sample_terminal(self):
        step=int(self.env.common_step_counter)
        if not self.frame_steps or self.frame_steps[-1]!=step:
            stride=self.stride
            self.stride=1  # Include a real terminal frame even off the normal cadence.
            try:super().sample()
            finally:self.stride=stride
        if self.writer is not None:
            self.finish(complete=step-self.start_step>=self.horizon)

    def finish(self,complete=False):
        if self.writer is None:return
        super().finish(complete)
        clip=self.clips[-1]
        clip.update(duration_policy=self.plan['mode'],skill=self.plan['skill'],
            requested_seconds=self.cfg['video_seconds'],full_action=bool(complete and self.complete_action),
            terminal_before_reset=self.complete_action,resets_policy='One complete episode or sequence; terminal before reset' if self.complete_action else clip['resets_policy'])
        if self.episode_start_step is not None:
            clip['episode_times_seconds']=[(s-self.episode_start_step)*self.env.step_dt for s in clip['common_steps']]
        write(self.record/'clips.json',self.clips)

    def close(self):
        self.finish(complete=False)
        if self.original_pre_reset is not None:
            self.env.recorder_manager.record_pre_reset=self.original_pre_reset
            self.original_pre_reset=None
