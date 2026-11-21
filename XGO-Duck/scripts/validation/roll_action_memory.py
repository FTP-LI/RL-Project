"""Exact runtime-style output memory for a fixed-policy roll diagnostic.

alpha is the OLD-value coefficient: .15*old + .85*raw. All 14 outputs use
the same roll coefficient. Raw network history must remain raw, separately.
Check numeric recurrence, independent resets and nonfinite rejection in the
CPU review before freezing any physics comparison.
"""
import numpy as np


class ActionMemory:
    def __init__(self,num_envs,alpha=.15):
        assert np.isfinite(alpha) and 0<=alpha<=1
        self.alpha=alpha;self.value=np.zeros((num_envs,14),np.float32)

    def update(self,raw):
        raw=np.asarray(raw,np.float32);assert raw.shape==self.value.shape and np.isfinite(raw).all()
        self.value*=self.alpha;self.value+=raw*(1-self.alpha)
        return self.value.copy()

    def reset(self,env_ids):self.value[env_ids]=0.
