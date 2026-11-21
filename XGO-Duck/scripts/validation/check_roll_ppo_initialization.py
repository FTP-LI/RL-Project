"""Run a full learner reset and resource readback, with physical forward steps disabled."""
from pathlib import Path
import argparse
from dataclasses import asdict
from unittest.mock import patch
import sys

import run_roll_ppo_capacity as trial

ROOT=trial.ROOT
DRAFT=trial.DRAFT
REPORT=ROOT/'records/reports/roll-friction-ppo-capacity-bounded/learner-reset'


def run(n):
    p=trial.read(trial.P)
    state=p['staged_training_request']
    cap=p['goal_execution']['roll_ppo_capacity']
    trial.require(trial.allowed(p,trial.STAGE) and not state['simulation_running'] and not state['learning_running']
                  and cap['status']=='review-required' and cap['next_trial']=='phase-1-pass-1-e3584',
                  'Learner reset-only diagnosis is not the current phase')
    frozen=ROOT/cap['protocol']
    plan=trial.read(frozen)
    trial.require(trial.sha(frozen)==cap['sha256'] and n in (3584,4096),'Frozen zero step inputs changed')
    for base,name in ((ROOT,'source_sha256'),(Path('/'),'installed_sha256')):
        trial.require(all(trial.sha(base/f)==h for f,h in plan[name].items()),'Frozen config source changed')
    from roll_friction_ppo import (ort, torch, prepare, MemoryWatch, ManagerBasedRlEnv, configure_torch_backends,
        RslRlVecEnvWrapper, Simulation, MjlabOnPolicyRunner, initial_setup, resources)
    assert not torch.cuda.is_initialized()
    arm=dict(run_id=f'roll-ppo-initialization-only-e{n}',environments=n,updates=20,friction_role='official-distribution')
    report=REPORT/f'e{n}'
    report.mkdir(parents=True)
    recipe=plan['recipe']
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/recipe['official_onnx']),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map
    config=prepare(plan,arm,meta)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch=MemoryWatch(recipe['hard_gpu_fraction'])

    def no_step(*_, **__):
        raise RuntimeError('Zero-step initialization is prohibited from executing dynamics')

    raw=None
    try:
        with watch.active(),patch.object(ManagerBasedRlEnv,'step',no_step),patch.object(Simulation,'step',no_step):
            raw=ManagerBasedRlEnv(cfg=config.env,device='cuda:0',render_mode=None)
            wrapped=RslRlVecEnvWrapper(raw,clip_actions=config.agent.clip_actions)
            wrapped.seed(recipe['seed'])
            wrapped.reset()
            runner=MjlabOnPolicyRunner(wrapped,asdict(config.agent),None,'cuda:0')
            with patch.object(runner.alg,'update',no_step):
                zero=initial_setup(raw,wrapped,runner,torch.load(ROOT/recipe['parent_checkpoint'],map_location='cpu',weights_only=True),
                                   session,meta,arm,report)
            watch.sample()
            watch.check()
        assert raw.common_step_counter==raw._sim_step_counter==0 and zero['matched_actor_critic_and_adam']
        result=dict(status='passed-actual-learner-reset-only',preflight_sha256=trial.sha(frozen),environments=n,
                    physical_substeps=0,learning_updates=0,hardware_io=False,
                    zero_step_sha256=trial.sha(report/'zero-step.json'),resource=resources(watch))
        trial.put(report/'result.json',result)
        print('PPO_ACTUAL_LEARNER_ZERO',n,result['resource']['peak_used_mib'],flush=True)
    except BaseException as exc:
        trial.put(report/'failure.json',dict(reason=f'{type(exc).__name__}: {exc}',environments=n,
                   controls=None if raw is None else raw.common_step_counter,
                   physics=None if raw is None else raw._sim_step_counter,learning_updates=0))
        raise
    finally:
        if raw is not None:
            raw.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('environments',type=int,choices=(3584,4096))
    run(parser.parse_args().environments)
