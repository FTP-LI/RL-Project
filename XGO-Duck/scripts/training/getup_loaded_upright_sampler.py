"""Feed same-solve contacts and root pose to the verified progress clock."""

import torch
import warp as wp

from getup_loaded_upright_progress import LoadedUprightProgress
from getup_support_credit_device import SolvedFloorLoad


class GetupLoadedUprightSampler:
    def __init__(self, env, first, last_search, last, shape):
        assert (env.physics_dt,env.step_dt,env.cfg.decimation)==(.004,.02,5)
        assert env.cfg.auto_reset and env.max_episode_length==400
        assert (first,last_search,last)==(120,156,160)
        self.env,self.first,self.last_search,self.last=env,first,last_search,last
        self.solved=SolvedFloorLoad(env,single_foot_minimum=shape['foot_fraction'],both_feet_minimum=shape['both_feet_fraction'])
        self.load=wp.to_torch(self.solved.normal)
        self.clock=LoadedUprightProgress(env.num_envs,env.device,body_weight=self.solved.weight,**shape)
        self.credit=torch.zeros(env.num_envs,device=env.device)
        self.zero=self.credit.clone()
        self.before=torch.empty((env.num_envs,4),device=env.device)
        self.samples=0
        self.age=-1
        self.record=False
        self.readback=None
        self.torch_stream=torch.cuda.current_stream(env.device)
        self.read_stream=wp.stream_from_torch(self.torch_stream)
        self.original_step,self.original_physics=env.step,env.sim.step
        env.step,env.sim.step=self.step,self.physics

    def step(self,action):
        self.age=int(self.env.common_step_counter)%self.env.max_episode_length
        if self.age==0:
            self.clock.reset(self.env.episode_length_buf==0)
        self.record=self.first<=self.age<self.last
        self.samples=0
        self.credit.zero_()
        result=self.original_step(action)
        assert self.samples==(5 if self.record else 0)
        return result

    def physics(self):
        if not self.record:
            self.original_physics()
            return
        assert torch.cuda.current_stream(self.env.device)==self.torch_stream
        with wp.ScopedStream(self.read_stream,sync_enter=True,sync_exit=True):
            self.before.copy_(self.env.sim.data.qpos[:,3:7])
        self.original_physics()
        with wp.ScopedStream(self.read_stream,sync_enter=True,sync_exit=True):
            self.solved.capture()
            after=self.env.sim.data.qpos[:,3:7]
            latch=self.env._early_support_release
            enabled=latch.supine & (latch.latched & ((self.clock.elapsed>=0) | (self.age<self.last_search)))
            raw,unit,confirmed,active=self.clock.advance(self.load,self.before,after,enabled)
            self.credit.add_(unit)
            if self.readback is not None:
                self.readback(self.age,self.samples+1,self.load,self.before,after,raw,unit,confirmed,active,enabled)
        self.samples+=1

    def reward_rate(self,enabled):
        if self.record:
            assert self.samples==5
            self.solved.check()
        else:
            assert self.samples==0
        return (self.credit if enabled else self.zero)/self.env.step_dt


def loaded_upright_rate(env,*,enabled,**_settings):
    return env._loaded_upright_sampler.reward_rate(enabled)
