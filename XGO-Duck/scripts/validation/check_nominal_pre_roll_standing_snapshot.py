"""Qualify the GPU restore and controller inventory needed for the standing pair."""

import fcntl
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from check_continuous_forward_fork_state import controller_snapshot, restore_controller as restore_roll_latch
from getup_release_fork_state import clone, digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
from local_getup_scale import gpu_sample
from nominal_skill_composition_controller import Composition
from recover_nominal_skill_composition_zero import isolated_components
from record_io import write
from routed_forward_entry_ramp import capture_controller, restore_controller
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent


ROOT, read, rel, sha = parent.ROOT,parent.read,parent.rel,parent.sha
PLAN = ROOT/'configs/experiments/nominal_skill_composition_pre_roll_stand_draft.json'
OUT = ROOT/'records/reports/nominal-skill-composition/pre-roll-stand-snapshot-preflight.json'


class StandingFork:
    def __init__(self, env, composition, actor, maximum_warp_bytes):
        self.control,self.actor=composition,actor
        self.plant=SceneReleaseSnapshot(env,composition.commands,maximum_warp_bytes)
        self.fields=self.capture()
        self.expected=digest(signature(self.fields))

    def capture(self):
        ctl=self.control
        return dict(commands=capture_controller(ctl.commands,self.actor),roll_latch=controller_snapshot(ctl.roll_route),
            roll_policy={key:clone(getattr(ctl.roll,key)) for key in ('history','previous','previous_features')},
            eligibility={key:getattr(ctl,key).copy() for key in ('roll_entry','roll_ready','stand_reads')},
            turn_map={name:getattr(ctl.commands.turn_map,name).copy() for name in ('ema','ready')})

    def restore(self):
        physical=self.plant.restore()
        ctl=self.control
        restore_controller(ctl.commands,self.actor,self.fields['commands'])
        restore_roll_latch(ctl.roll_route,self.fields['roll_latch'])
        for key in ('history','previous'):
            getattr(ctl.roll,key).copy_(self.fields['roll_policy'][key])
        ctl.roll.previous_features=clone(self.fields['roll_policy']['previous_features'])
        for key,value in self.fields['eligibility'].items():
            np.copyto(getattr(ctl,key),value)
        for key,value in self.fields['turn_map'].items():
            np.copyto(getattr(ctl.commands.turn_map,key),value)
        assert digest(signature(self.capture()))==self.expected
        return dict(physical=physical,controller_sha256=self.expected)


def main():
    project=read(parent.PROJECT)
    state=project['goal_execution'][parent.KEY]
    task=project['staged_training_request']
    stage=parent.STAGE+'-pre-roll-pair-preparation'
    assert parent.allowed(project,stage) and not task['simulation_running'] and not OUT.exists()
    assert task['current_protocol']==rel(PLAN) and sha(PLAN)==state['next_paired_design_sha256']
    design=read(PLAN)
    sources={rel(PLAN):sha(PLAN)}
    for name in ('parent_protocol','parent_analysis','handoff_review'):
        path=ROOT/design[name]
        assert sha(path)==design[name+'_sha256']
        sources[rel(path)]=sha(path)
    plan=read(ROOT/design['parent_protocol'])
    parent.base.shared.assert_sources(plan)
    for file in (Path(__file__).resolve(),Path(__file__).with_name('getup_release_scene_state.py'),
                 Path(__file__).with_name('getup_release_fork_state.py'),
                 Path(__file__).with_name('routed_forward_entry_ramp.py'),
                 Path(__file__).with_name('check_continuous_forward_fork_state.py')):
        sources[rel(file)]=sha(file)
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(), \
         patch.object(feedback,'components',isolated_components):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=parent.cuda_env(plan)
        try:
            obs,_=env.reset(seed=design['mechanism_development_seed'])
            # Initialize the release latches on copies of real reset inputs.
            # The live environment clock stays zero throughout this fixture.
            actors=parent.load_actors(plan,env.device,lambda:plan['composition']['release_control'])
            actor=actors['getup']
            actor(parent.base.shared.td('actor',obs['actor'].clone()))
            ctl=Composition(plan,actors,env.device)
            before=gpu_sample()
            limit=plan['composition']['observer']['gpu_memory_fraction_limit']
            available=int((before['total_mib']*limit-before['used_mib'])*1024**2)
            fork=StandingFork(env,ctl,actor,available)
            initial=fork.plant.verify()
            assert initial['counters']==[0,0]
            saved_names=set(fork.plant.expected)
            assert {'warp.qpos','env.obs_buf','controller.history','rng'}<=saved_names
            assert any(name.startswith('obs-delay.actor.') for name in saved_names)
            assert any(name.startswith('actuator.0.delay.') for name in saved_names)
            env.sim.data.qpos[0,0].add_(.03125)
            ctl.commands.history[0,0,0].add_(.125)
            ctl.roll.history[0,0].add_(.25)
            ctl.roll_route.dwell[0]=7
            ctl.stand_reads[0]=4
            ctl.commands.turn_map.ema[0]=.3
            actor.calls+=2
            torch.rand(1,device=env.device)
            difference={key for key,value in fork.plant.live_signatures().items() if value!=fork.plant.expected[key]}
            assert {'warp.qpos','controller.history','rng'}<=difference
            assert digest(signature(fork.capture()))!=fork.expected
            restored=fork.restore()
            after=gpu_sample()
            assert restored['physical']==initial and after['used_mib']/after['total_mib']<=limit
        finally:
            env.close()
    write(OUT,dict(status='verified-actual-cuda-zero-restore',source_sha256=sources,fixture_control=0,
        planned_matched_prefix_control=design['common_snapshot_control'],physical=initial,
        controller_sha256=restored['controller_sha256'],physical_fields_changed_in_fixture=sorted(difference),
        restored_release_latches=True,restored_roll_and_gyro_memory=True,before_gpu=before,after_gpu=after,
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False))
    print('NOMINAL_PRE_ROLL_SNAPSHOT_ZERO_VERIFIED',sha(OUT),initial['fields'],initial['warp_bytes'],flush=True)


if __name__=='__main__':
    main()
