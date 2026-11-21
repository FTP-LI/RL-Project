"""Exercise complete CPU snapshot restoration for the prospective roll entry."""
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from roll_entry_snapshot import RollEntrySnapshot
from run_continuous_roll_stand_walk import configuration
from s1_reference_inputs import ROOT, read, sha
from record_io import write


PLAN = ROOT/'configs/experiments/continuous_roll_entry_isolation_draft.json'
OUT = ROOT/'records/reports/continuous-roll-stand-walk'


def main():
    from mjlab.envs import ManagerBasedRlEnv

    draft = read(PLAN)
    source = ROOT/draft['source']
    parent = ROOT/draft['parent_protocol']
    evidence, p = read(source), read(parent)
    assert evidence['protocol_sha256']==sha(parent) and not evidence['heldout_used']
    assert evidence['roll']['continuous']['failures']['head-penetration-exceeded']==draft['worlds']==128
    assert draft['status']=='draft-not-executable' and not draft['hardware_io'] and not draft['heldout_used']
    expected = draft['shared_stand_controls']+len(draft['branches'])*draft['maximum_roll_controls_per_branch']
    assert expected==draft['maximum_controls_all_branches']
    assert expected*draft['worlds']==draft['maximum_control_transitions'] and expected*5==draft['maximum_physics_substeps']
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=configuration(p),device='cpu',render_mode=None)
    try:
        env.seed(draft['development_seed'])
        env.reset()
        memory = SimpleNamespace(previous=torch.zeros(env.num_envs,14),history=torch.zeros(env.num_envs,14),
                                 previous_features=np.zeros((env.num_envs,2),dtype=np.float32))
        fork = RollEntrySnapshot(env,memory,268435456)
        start = fork.verify()
        assert start['counters']==[0,0]
        env.sim.data.qvel.add_(.125)
        env.action_manager.get_term('joint_pos')._processed_actions.fill_(.2)
        memory.previous.fill_(.01)
        memory.history.fill_(.03)
        memory.previous_features.fill(.01)
        env._roulade_last_update_step += 77
        random.random()
        np.random.random()
        torch.rand(8)
        assert fork.live_signatures()!=fork.expected
        restored = fork.restore()
        assert restored==start and fork.verify()==start
        labels = sorted(fork.expected)
        assert {'controller.previous','controller.history','controller.previous_features','rng','env._sim_step_counter',
                'env.obs_buf','env._roulade_last_update_step','env._ground_support_cache'}<=set(labels)
        assert not any(label.startswith('obs-delay.') for label in labels)
        assert any('.delay.ring.' in label for label in labels)
        assert any(label.startswith('sensor.feet_ground_contact.') for label in labels)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    output = OUT/'roll-entry-fork-cpu-preflight.json'
    assert not output.exists()
    write(output,dict(status='passed-cpu-zero-step-fork',source_sha256={path.relative_to(ROOT).as_posix():sha(path)
        for path in (PLAN,source,parent,Path(__file__),ROOT/'scripts/validation/roll_entry_snapshot.py',
                     ROOT/'scripts/validation/getup_release_scene_state.py',
                     ROOT/'scripts/validation/getup_release_fork_state.py')},
        verified=start,field_names=labels,mutated='velocity, processor target, actor history and CPU RNG',
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Actual CPU reset-state rollback. Standing-to-roll k=50 and CUDA rollback remain to verify.'))
    print('ROLL_ENTRY_FORK_CPU',start,flush=True)


if __name__=='__main__':
    main()
