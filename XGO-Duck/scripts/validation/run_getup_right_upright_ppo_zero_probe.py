"""Check matched zero steps with the isolated reward probe context corrected."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

import run_getup_right_upright_ppo_zero as original
import run_getup_right_upright_ppo_zero_readback as readback
from getup_right_upright_progress import RightUprightProgress
from record_io import write
from run_local_getup_ppo import check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_right_upright_ppo_zero_probe.json'
OUT = original.OUT/'probe'
STAGE = 'S3-right-side-upright-ppo-zero-probe'
SCRIPT = Path(__file__).resolve()
native = original.native


def reward_probe(spec,env,features,bridge):
    """Use an isolated state and the actual registered term without stepping physics."""
    live = bridge.state
    probe = RightUprightProgress(env.num_envs,env.device,spec['release_age'],spec['alignment'],
                                 spec['entry'],spec['horizon'])
    actor = features['actor'].clone()
    actor[:3,3:6] = actor.new_tensor([[0.,-1.,0.],[0.,-1.,0.],[0.,1.,0.]])
    ages = torch.zeros_like(env.episode_length_buf)
    ages[:3] = spec['release_age']
    tilt = actor.new_full((env.num_envs,),30.)
    feet = torch.zeros((env.num_envs,2),device=env.device,dtype=torch.bool)
    feet[:3] = True
    heads = feet[:,0].clone()
    with (
        patch.object(env,'_right_upright_progress',probe),
        patch.object(env,'episode_length_buf',ages),
        patch.object(native,'inputs',lambda _: (-(tilt*(math.pi/180.)).cos(),feet,heads)),
    ):
        probe.observe_release(actor,ages)
        assert probe.right[:3].tolist() == [True,True,False]
        for _ in range(len(probe.tilts)):
            ages[:3] += 1
            assert not bridge.term.func(env).any()
        assert probe.trigger_age[:3].tolist() == [int(ages[0]),int(ages[0]),-1]
        ages[:3] += 1
        tilt[:3] = tilt.new_tensor([24.,36.,24.])
        raw = bridge.term.func(env)
        expected = [.2,-.2,0.] if bridge.enabled else [0.,0.,0.]
        torch.testing.assert_close(raw[:3],raw.new_tensor(expected),rtol=0,atol=1e-6 if bridge.enabled else 0)
        assert not raw[3:].any()
        native.reset_reward(env,torch.tensor([0,2],device=env.device))
        assert probe.release_known[:3].tolist() == [False,True,False]
        assert not probe.supported[:,[0,2]].any() and probe.trigger_age[[0,2]].tolist() == [-1,-1]
    assert env._right_upright_progress is live and not live.release_known.any()
    return dict(raw=raw[:3].tolist(), weighted=(raw[:3]*bridge.term.weight).tolist(),
                clock=original.signature(vars(probe)), reset_ids=[0,2])


def freeze():
    check_project()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == readback.STAGE and ROOT/stage['current_protocol'] == readback.PLAN
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    spec = deepcopy(read(readback.PLAN))
    assert all(sha(ROOT/path) == digest for path,digest in spec['sources'].items())
    log = readback.OUT/'control.log'
    failed = readback.OUT/'analysis.json'
    startup = readback.OUT/'control/startup.json'
    assert read(failed)['status'] == 'stopped-review-required' and read(failed)['protocol_sha256'] == sha(readback.PLAN)
    assert "bad operand type for unary +: '_patch'" in log.read_text() and read(startup)['physical_steps'] == 0
    assert read(failed)['runs']['control']['exit_code'] == 1 and not (readback.OUT/'control/result.json').exists()
    paths = (SCRIPT,readback.PLAN,readback.OUT/'preflight.json',readback.OUT/'pair-exit.txt',startup,failed,log)
    spec['sources'].update({original.relative(path):sha(path) for path in paths})
    spec['stage'] = STAGE
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='frozen-probe-context-only',protocol_sha256=sha(PLAN),
          original_protocol_sha256=sha(readback.PLAN),physical_substeps=0,learning_updates=0))
    print('RIGHT_UPRIGHT_PPO_PROBE_FROZEN',sha(PLAN),flush=True)


if __name__ == '__main__':
    command, = sys.argv[1:]
    if command == 'freeze':
        freeze()
    else:
        with patch.multiple(original,PLAN=PLAN,OUT=OUT,STAGE=STAGE,__file__=str(SCRIPT),reward_probe=reward_probe):
            spec = original.checked()
            if command == 'pair':
                original.pair(spec)
            else:
                assert command in original.ROLES
                original.arm(spec,command)
