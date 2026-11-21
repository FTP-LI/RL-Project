"""CPU checks for coordinate invariance, resets, timing and reward isolation.

Synthetic trajectories verify intended mechanics, not gait improvement.
No simulator, GPU context, model inference, training or hardware is used.
"""
from pathlib import Path
from types import SimpleNamespace as N
import hashlib
import json
import math
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from planar_velocity_reward import reward,reset_statistic,inverse_rotate,STATE,original_reward


def env(n=2):
    data=N(root_link_lin_vel_w=torch.zeros(n,3),root_link_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1))
    data.root_link_lin_vel_b=torch.zeros(n,3)
    cmd=torch.zeros(n,3)
    result=N(scene={'robot':N(data=data)},step_dt=.02,common_step_counter=0,
        command_manager=N(get_command=lambda _:cmd),command=cmd)
    return result


def measure(e,velocity,command=None,q=None):
    data=e.scene['robot'].data
    data.root_link_lin_vel_w=torch.as_tensor(velocity,dtype=torch.float32).reshape(-1,3).expand_as(data.root_link_lin_vel_w).clone()
    if q is not None:data.root_link_quat_w=torch.as_tensor(q,dtype=torch.float32).reshape(-1,4).expand_as(data.root_link_quat_w).clone()
    data.root_link_lin_vel_b=inverse_rotate(data.root_link_quat_w,data.root_link_lin_vel_w)
    if command is not None:e.command[:]=torch.tensor(command)


def call(e,candidate=True):return reward(e,.25,'twist',tau_seconds=.2,use_filtered_xy=candidate)


def main():
    cases=[]
    e=env();measure(e,[.15,0,0],[.15,0,0]);a=call(e)
    torch.testing.assert_close(a,torch.ones(2));cases.append('constant velocity starts without artificial transient')
    for k in range(100):
        e.common_step_counter=k+1
        measure(e,[.15,.03*math.sin(k),.05],[.15,0,0])
        baseline=call(e,False);original=original_reward(e,.25,'twist')
        assert torch.equal(baseline,original)
    cases.append('control output bitwise equals original installed reward')
    before=getattr(e,STATE).value.clone();updates=getattr(e,STATE).updates
    call(e);call(e)
    assert torch.equal(before,getattr(e,STATE).value) and getattr(e,STATE).updates==updates
    cases.append('repeated reward reads do not integrate extra time')
    measure(e,[0.,0,0]);reset_statistic(e,torch.tensor([0]));call(e)
    assert torch.equal(getattr(e,STATE).value[0],torch.zeros(3)) and torch.equal(getattr(e,STATE).value[1],before[1])
    cases.append('partial reset clears only reset environment, even in same step')
    reset_statistic(e,None);call(e);assert torch.equal(getattr(e,STATE).value,torch.zeros(2,3))
    cases.append('full reset has no previous-episode velocity memory')
    e=env();measure(e,[.15,0,0],[.15,0,0]);call(e)
    e.common_step_counter+=1
    measure(e,[.15,0,0],[0,-.15,0],q=[math.sqrt(.5),0,0,math.sqrt(.5)])
    torch.testing.assert_close(call(e),torch.ones(2),atol=2e-6,rtol=0)
    cases.append('world-frame memory rotates into current heading correctly')
    e=env();measure(e,[.15,0,.1],[.15,0,0]);call(e)
    e.common_step_counter+=1;measure(e,[.15,0,0]);value=call(e)
    torch.testing.assert_close(value,torch.ones(2),atol=2e-6,rtol=0)
    cases.append('vertical penalty uses raw current body velocity')
    e=env();measure(e,[.15,.1,0],[.15,0,0]);value=call(e)
    torch.testing.assert_close(value,torch.full((2,),math.exp(-.1**2/.25**2)),atol=1e-7,rtol=1e-6)
    cases.append('constant lateral drift cannot be hidden by initialization')
    e=env();raw=[];smoothed=[]
    for k in range(500):
        e.common_step_counter=k
        measure(e,[.15,.13*math.sin(2*math.pi*2*k*.02),0],[.15,0,0]);call(e)
        if k>=100:
            raw.append(float(getattr(e,STATE).last_raw_reward[0]));smoothed.append(float(getattr(e,STATE).last_filtered_reward[0]))
    raw_mean=sum(raw)/len(raw);filtered_mean=sum(smoothed)/len(smoothed)
    assert filtered_mean>raw_mean+.05
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',cases=cases,tau_seconds=.2,
        cyclic_example=dict(raw_mean_reward=raw_mean,filtered_mean_reward=filtered_mean,frequency_hz=2.,lateral_amplitude_m_s=.13),
        source_sha256=hashlib.sha256((ROOT/'scripts/training/planar_velocity_reward.py').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cuda_initialized=False,
        scope='CPU mechanical checks only; no evidence yet of better training or walking')
    folder=ROOT/'records/reports/planar-filter';folder.mkdir(exist_ok=True)
    (folder/'reward-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
