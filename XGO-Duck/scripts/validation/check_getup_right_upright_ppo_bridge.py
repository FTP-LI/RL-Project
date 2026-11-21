"""Replay actual reward inputs through both PPO bridges without optimizing."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from check_getup_right_upright_native_contract import TERM
from getup_right_upright_ppo_bridge import RightUprightPPOBridge
from record_io import write


OUT = ROOT/'records/reports/getup-right-upright-clock'
CLOCK = OUT/'analysis.json'
VALUE = ROOT/'configs/experiments/getup_right_upright_value.json'
DEST = OUT/'ppo-bridge-cpu.json'


def main():
    assert not DEST.exists() and not torch.cuda.is_initialized()
    clock,spec = read(CLOCK),read(VALUE)
    assert clock['status']=='passed-actual-cuda-reward-clock' and not clock['heldout_used']
    record = ROOT/clock['record']
    source = record/'reward-trace.npz'
    assert sha(source)==clock['source_sha256'][source.relative_to(ROOT).as_posix()]
    with np.load(source) as file:
        names = ('pre_age','actor_gravity','critic_input','reward_age','z','feet','head_leg','active','raw','done')
        data = {name:torch.from_numpy(file[name]) for name in names}
    torch.set_num_threads(1)
    n = data['pre_age'].shape[1]

    def emulate_native(env):
        return env._right_upright_progress.advance(*env.native_inputs)

    def mock_env():
        term = SimpleNamespace(weight=10.,func=emulate_native)
        manager = SimpleNamespace(_term_names=[TERM],_term_cfgs=[term])
        return SimpleNamespace(num_envs=n,device='cpu',reward_manager=manager)

    a,b = mock_env(),mock_env()
    control,candidate = RightUprightPPOBridge(a,spec,False),RightUprightPPOBridge(b,spec,True)
    actor = torch.zeros(n,61)
    values = []
    with control.active(),candidate.active():
        for i in range(len(data['pre_age'])):
            a.episode_length_buf = b.episode_length_buf = data['pre_age'][i]
            actor[:,3:6] = data['actor_gravity'][i]
            observation = dict(actor=actor,critic=data['critic_input'][i])
            x,y = (bridge.before_action(observation) for bridge in (control,candidate))
            assert x['actor'] is actor and y['actor'] is actor
            torch.testing.assert_close(x['critic'],y['critic'],rtol=0,atol=0)
            fields = ('z','feet','head_leg','reward_age','active')
            a.native_inputs = b.native_inputs = tuple(data[key][i] for key in fields)
            assert not a.reward_manager._term_cfgs[-1].func(a).any()
            raw = b.reward_manager._term_cfgs[-1].func(b)
            torch.testing.assert_close(raw,data['raw'][i],rtol=0,atol=6e-7)
            values.append(raw)
            torch.testing.assert_close(control.state.trigger_age,candidate.state.trigger_age,rtol=0,atol=0)
            done = torch.nonzero(data['done'][i]).flatten()
            if i+1==250:
                done = torch.cat((done,torch.tensor(clock['selective_reset_ids'])))
            for bridge in (control,candidate):
                bridge.state.reset(done)
    assert a.reward_manager._term_cfgs[-1].func is emulate_native and b.reward_manager._term_cfgs[-1].func is emulate_native
    signal = torch.stack(values)
    assert not torch.cuda.is_initialized()
    write(DEST,dict(status='passed-matched-cpu-ppo-bridge',physical_substeps=0,learning_updates=0,heldout_used=False,
        sampled_input_rows=signal.numel(),nonzero_candidate_samples=int(torch.count_nonzero(signal)),
        control_contribution_exactly_zero=True,critic_features_bitwise_equal=True,actor_observation_preserved=True,
        maximum_candidate_readback_error=float((signal-data['raw']).abs().max()),
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (CLOCK,VALUE,source,Path(__file__),
                     ROOT/'scripts/validation/getup_right_upright_ppo_bridge.py')}))
    print('RIGHT_UPRIGHT_PPO_BRIDGE_CPU',signal.numel(),int(torch.count_nonzero(signal)),flush=True)


if __name__=='__main__':
    main()
