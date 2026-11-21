"""Stateless controller for the frozen static command-map confirmation.

Only the policy's command fields change. reset/map is the small interface
used by the batched evaluator, allowing later adapters to own their state
without changing the evaluator or any historical frozen experiment.
"""
import numpy as np
from command_input_mapping import transform


class Controller:
    def __init__(self,num_envs,dt):self.num_envs=num_envs;assert dt==.02
    def reset(self,env_ids):pass
    def map(self,desired,observations):
        assert desired.shape==(self.num_envs,3) and observations.shape==(self.num_envs,61)
        return transform(desired)


def create_controller(num_envs,dt):return Controller(num_envs,dt)
