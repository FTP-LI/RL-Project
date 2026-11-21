"""Preserve identical causal value features in both right-side PPO arms."""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from s1_reference_inputs import read
from check_getup_right_upright_native_contract import DRAFT, TERM, install_state
from run_getup_right_upright_value import WIDTH, features


class RightUprightPPOBridge:
    def __init__(self,env,spec,enabled):
        self.env,self.enabled = env,enabled
        self.state = install_state(env,read(DRAFT),spec['release_age'],spec['alignment'])
        manager = env.reward_manager
        assert manager._term_names[-1]==TERM
        self.term = manager._term_cfgs[-1]
        assert self.term.weight==spec['reward_coefficient']==10.
        self.native = self.term.func

    def reward(self,env):
        # RewardManager skips a zero-weight term entirely. Keep the native term
        # executing in both arms and zero the control's returned contribution.
        raw = self.native(env)
        return raw if self.enabled else torch.zeros_like(raw)

    @contextmanager
    def active(self):
        with patch.object(self.term,'func',self.reward):
            yield

    def before_action(self,observation):
        age = self.env.episode_length_buf
        assert observation['actor'].shape==(self.env.num_envs,61)
        assert observation['critic'].shape==(self.env.num_envs,75)
        self.state.observe_release(observation['actor'],age)
        critic = features(self.state,age,observation['critic'])
        assert critic.shape==(self.env.num_envs,WIDTH)
        return {**observation,'critic':critic}

    def step_wrapper(self,step):
        def step_with_critic(action):
            observation,reward,terminated,timeout,info = step(action)
            return self.before_action(observation),reward,terminated,timeout,info
        return step_with_critic
