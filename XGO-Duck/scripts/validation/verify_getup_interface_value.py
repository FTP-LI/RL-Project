"""Independent CPU value and target audit for the fresh interface dataset.

Reuse the frozen Monte Carlo/split/normalizer/optimizer checker with explicit
new paths, then independently reconstruct all400x1024x14 physical requests.
No simulation, fitting, tolerance changes or reinterpretation of gait scores.
"""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_interface_value import PLAN,OUT,checked
from s1_reference_inputs import read,sha
from record_io import write


def main():
    import verify_getup_value_bootstrap as legacy
    p=checked();assert not torch.cuda.is_initialized()
    legacy.PLAN,legacy.OUT,legacy.checked=PLAN,OUT,checked
    legacy.main()
    record=Path(read(OUT/'progress.json')['path']);proof=read(OUT/'analysis.json')
    with np.load(record/'value-data.npz') as f:
        raw=f['actions'].copy();requests=f['requested_targets'].copy();applied=f['applied_targets'].copy()
    home=np.asarray(p['model_home'],np.float32);physical=np.asarray(p['simulation_home'],np.float32)
    limits=np.asarray(p['joint_limits'],np.float32)
    desired=raw+home;desired[:100]=physical
    np.testing.assert_array_equal(requests,desired)
    expected=desired.copy()
    for step in range(250,400):
        destination=np.clip(desired[step],limits[:,0],limits[:,1])
        expected[step]=expected[step-1]+np.clip(destination-expected[step-1],-.04,.04)
    np.testing.assert_allclose(applied,expected,atol=1e-6,rtol=0)
    original=read(ROOT/'configs/experiments/getup_value_bootstrap.json')
    assert p['initial_std']==.1 and original['initial_std']==.05
    assert p['actor_noise']=={'base_ang_vel':.06,'projected_gravity':.03,'joint_pos':.01,'joint_vel':.25}
    startup=read(record/'startup.json');assert startup['new_cpu_scene_exact'] and startup['stochastic_probe_rng_restored']
    proof.update(common_interface_reconstructed=True,all_requested_targets_checked=int(requests.size),
        all_processed_targets_checked=int(applied.size),maximum_target_reconstruction_error=float(abs(applied-expected).max()),
        common_noise=p['actor_noise'],fixed_exploration_std=.1,
        new_critic_for_new_recipe=True,original_value_probe_unchanged=True,
        wrapper_sha256=sha(Path(__file__)),source_count=len(p['source_sha256']))
    write(OUT/'analysis.json',proof)
    print('INTERFACE_VALUE_VERIFIED',json.dumps(proof),flush=True)


if __name__=='__main__':main()
