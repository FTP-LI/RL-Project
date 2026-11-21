"""CPU boundary fixtures before implementing the next policy learner."""
from pathlib import Path
import sys
import json
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_ppo_math import finite_gae,normalize_active_advantages,masked_surrogate
from s1_reference_inputs import sha
from record_io import write


def main():
    assert not torch.cuda.is_initialized()
    # A terminal middle sample severs all future values and advantages. With
    # lambda1, these are complete MC returns and a truncated tail bootstrap.
    reward=torch.tensor([[1.,10.],[2.,20.],[4.,40.]])
    value=torch.tensor([[.1,1.],[.2,2.],[.4,4.]])
    done=torch.tensor([[False,False],[True,False],[False,True]])
    adv,returns=finite_gae(reward,value,done,torch.tensor([8.,8000.]),.5,1.)
    expected=torch.tensor([[2.,30.],[2.,40.],[8.,40.]])
    torch.testing.assert_close(returns,expected,atol=1e-6,rtol=0)
    active=torch.tensor([False,False,True,True])
    advantages=torch.tensor([1000.,-2000.,1.,3.])
    normal=normalize_active_advantages(advantages,active)
    torch.testing.assert_close(normal,torch.tensor([0.,0.,-1.,1.]),atol=1e-7,rtol=0)
    log=torch.zeros(4,requires_grad=True)
    loss=masked_surrogate(log,torch.zeros(4),normal,active);loss.backward()
    torch.testing.assert_close(log.grad,torch.tensor([0.,0.,.5,-.5]),atol=0,rtol=0)
    log2=torch.tensor([100.,-100.],requires_grad=True)
    zero=masked_surrogate(log2,torch.zeros(2),torch.ones(2),torch.zeros(2,dtype=torch.bool));zero.backward()
    assert float(zero.detach())==0 and not log2.grad.any()
    # Clipped positive advantage has zero policy gradient above1.2; the frozen
    # row remains zero regardless of an extreme, irrelevant log probability.
    log3=torch.tensor([100.,.5],requires_grad=True)
    masked_surrogate(log3,torch.zeros(2),torch.ones(2),torch.tensor([False,True])).backward()
    assert not log3.grad.any()
    result=dict(status='passed',finite_episode_gae_and_timeout_separation=True,
        lambda1_matches_manual_mc_returns=True,reset_future_values_do_not_leak=True,
        freeze_rows_have_exact_zero_policy_gradient=True,active_only_advantage_normalization=True,
        fully_frozen_minibatch_loss_and_gradient_zero=True,ppo_clipping_fixture=True,
        physics_steps=0,optimizer_steps=0,cuda_initialized=False,
        learner_integration_complete=False,scope='Math fixtures only; no actor update or executable pair protocol',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/training/getup_ppo_math.py')})
    write(ROOT/'records/reports/getup-adaptation/ppo-math-preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
