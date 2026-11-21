"""Bounded turn-only feedback; fixed policy, raw observation/history unchanged.

Both arms compute the same gyro EMA. Only designated feedback worlds add
K*(requested_yaw-filtered_gyro) to the existing 1.5x yaw input. No simulated
ground-truth velocity is used. Keep delay, noise and IMU rotation present.
Reliability checks: explicit EMA sequence, selective reset, saturation and
unchanged static/zero/linear mappings in the CPU protocol preflight.
"""
import numpy as np
from command_input_mapping import transform


class Controller:
    def __init__(self,kinds,dt=.02):
        assert set(kinds)<=set(('static','gyro-feedback'))
        self.feedback=np.asarray([k=='gyro-feedback' for k in kinds])
        self.ema=np.zeros(len(kinds),np.float32);self.ready=np.zeros(len(kinds),bool)
        self.alpha=1-np.exp(-dt/.2);self.gain=.75

    def reset(self,env_ids):
        self.ema[env_ids]=0.;self.ready[env_ids]=False

    def map(self,desired,observations):
        desired=np.asarray(desired,np.float32);observations=np.asarray(observations,np.float32)
        n=len(self.ema);assert desired.shape==(n,3) and observations.shape==(n,61)
        assert np.isfinite(desired).all() and np.isfinite(observations).all()
        gyro=observations[:,2]
        self.ema=np.where(self.ready,self.ema+self.alpha*(gyro-self.ema),gyro).astype(np.float32)
        self.ready[:]=True;out=transform(desired)
        mask=self.feedback&(desired[:,2]!=0.)
        out[mask,2]=np.clip(out[mask,2]+self.gain*(desired[mask,2]-self.ema[mask]),-1.,1.)
        return out


def create_controller(num_envs,dt):
    from s1_reference_inputs import ROOT,read
    p=read(ROOT/'configs/experiments/turn_feedback_probe.json');assert num_envs==p['parallel_worlds']
    kinds=['static']*num_envs
    for name,ids in p['groups'].items():
        for i in ids:kinds[i]=p['controllers'][name]
    return Controller(kinds,dt)
