"""Shared zero-step and recording contracts for the anchor studies.

Protect both RNG streams around the inherited standing-model import too.
Its CPU-only import fork can otherwise reseed CUDA after the physical reset.
Keep all inherited HOME/history/slew checks, then verify physical/processor
state and full scene arrays. Recording hashes the actual source qpos at each
render call; the renderer uses only a copy and never advances physics.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from getup_interface_value import startup as original_startup
from common_scene_contract import model_arrays,compare_arrays
from training_watch import Snapshot
from record_io import write
from s1_reference_inputs import read


def startup(env,obs,actor,plan,record,cpu_scene):
    cpu=lambda x:x.detach().cpu().numpy().copy()
    term=env.action_manager.get_term('joint_pos')
    names=('_raw_actions','_processed_actions','previous_target','previous_valid',
           'standing_mask','requested_target','projected_destination')
    saved={k:cpu(getattr(term,k)) for k in names}
    qpos=cpu(env.sim.data.qpos);qvel=cpu(env.sim.data.qvel)
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(env.device).clone()
    with torch.random.fork_rng(devices=[0]):
        original_startup(env,obs,actor,plan,record)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(env.device))
    for name,value in saved.items():np.testing.assert_array_equal(cpu(getattr(term,name)),value)
    np.testing.assert_array_equal(qpos,cpu(env.sim.data.qpos));np.testing.assert_array_equal(qvel,cpu(env.sim.data.qvel))
    with np.load(cpu_scene) as f:compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
    assert env.common_step_counter==0 and obs['actor'].shape==(env.num_envs,61) and obs['critic'].shape==(env.num_envs,75)
    proof=read(record/'startup.json');proof.update(complete_import_cpu_cuda_rng_restored=True,
        actual_processor_memory_exact=True,new_frozen_cpu_scene_exact=True,physical_steps=0)
    write(record/'startup.json',proof)


class RecordingSnapshot(Snapshot):
    def __init__(self,env,record):
        super().__init__(env,record);self.recorded_states=[]

    def render(self):
        qpos=self.env.sim.data.qpos[0].detach().cpu().numpy().copy()
        pixels=super().render()
        np.testing.assert_array_equal(self.data.qpos.astype(qpos.dtype),qpos)
        self.recorded_states.append(dict(common_step=int(self.env.common_step_counter),
            qpos_sha256=hashlib.sha256(qpos.tobytes()).hexdigest()))
        return pixels

    def close(self):
        write(self.record/'recording-states.json',self.recorded_states)
        super().close()
