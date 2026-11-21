"""Fixed, zero-preserving command gains for a prospective engineering check.

Why: the unchanged official policy barely turns at 0.4 rad/s but responds at
0.7, and reverse/strafe have similar weak low-input responses. Multiply only
the three policy command inputs; all metrics retain the external request.
Do not learn weights, change action history, extrapolate past measured input
bounds, or call this hardware calibration. A new protocol must verify motion.
"""
import numpy as np

GAINS=np.asarray([1.4,3.,1.5],np.float32)
LOW=np.asarray([-.2,-.24,-1.],np.float32)
HIGH=np.asarray([.4,.24,1.],np.float32)


def transform(command):
    command=np.asarray(command,np.float32)
    assert command.shape[-1]==3 and np.isfinite(command).all()
    return np.clip(command*GAINS,LOW,HIGH)
