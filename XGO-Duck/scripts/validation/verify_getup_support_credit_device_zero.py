"""Compile and check the CUDA weight-transfer reader without advancing physics."""

import fcntl
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
import warp as wp

from check_nominal_low_friction_support_credit import (
    KEY, PREVIOUS_KEY, PROJECT, RESULT, ROOT, STAGE, credit, digest, load, permitted, write,
)
from getup_support_credit_device import (
    CHANNELS, SolvedFloorLoad, evaluate_weight_transfer, reduce_solved_floor_load, vec5,
)
from roll_contact_clock_capture import ContactClock, reduce_contacts
import run_nominal_low_friction_support_effort as experiment


def fixture(device, config):
    """Include head contacts in either order, a one-foot case, and a masked constraint."""
    kinds = np.array([-1,1,2,3,0], dtype=np.int32)
    pairs = np.array([[0,1],[1,0],[0,2],[3,0],[0,4],[2,3],[0,3],[0,2],[0,1],[0,3]], dtype=np.int32)
    worlds = np.array([0,0,0,0,0,0,1,1,2,2], dtype=np.int32)
    forces = np.array([.75,.25,3.,3.,.1,1.,2.,0.,0.,0.], dtype=np.float32)
    used, capacity, size = 8,len(pairs),3
    count = wp.array([used],dtype=int,device=device)
    int_array = lambda values: wp.array(values,dtype=int,device=device)
    addresses = np.arange(capacity,dtype=np.int32).reshape(-1,1)
    addresses[7,0] = -1
    summed = wp.zeros((size,len(CHANNELS)),dtype=float,device=device)
    invalid = wp.zeros(1,dtype=int,device=device)
    reward = wp.empty(size,dtype=float,device=device)
    cube = wp.array(np.tile(forces,(size,1)),dtype=float,device=device)
    wp.launch(reduce_solved_floor_load,dim=capacity,device=device,inputs=[
        int(mujoco.mjtCone.mjCONE_ELLIPTIC),capacity,0,size,count,
        wp.array(pairs,dtype=wp.vec2i,device=device),int_array(worlds),int_array(kinds),
        wp.zeros(capacity,dtype=wp.mat33,device=device),wp.zeros(capacity,dtype=vec5,device=device),
        int_array(np.ones(capacity,np.int32)),int_array(addresses),cube,capacity,summed,invalid,
    ])
    assert int(invalid.numpy()[0])==0
    _, expected = reduce_contacts(pairs[:used],worlds[:used],forces[:used],0,kinds,size)
    np.testing.assert_allclose(summed.numpy(),expected,atol=2e-6,rtol=0.)
    weight = 8.0
    wp.launch(evaluate_weight_transfer,dim=size,device=device,inputs=[
        summed,weight,config['per_foot_load_fraction'],config['total_foot_load_fraction'],reward,
    ])
    expected_credit = credit(expected[:,1:],weight,config)
    np.testing.assert_allclose(reward.numpy(),expected_credit,atol=2e-7,rtol=0.)
    return dict(valid_contact_slots=used,fixture_worlds=size,expected_force_n=expected.tolist(),
                expected_credit=expected_credit.tolist(),actual_credit=reward.numpy().tolist())


def main():
    project = load(PROJECT)
    permitted(project,STAGE)
    goal, task = project['goal_execution'], project['staged_training_request']
    archived = goal[KEY]
    result = load(RESULT,archived['result_sha256'])
    assert archived['status']=='reviewed' and result['gpu_readback_preflight_admitted']
    spec = load(ROOT/archived['protocol'],archived['protocol_sha256'])
    original = goal[PREVIOUS_KEY]
    assert original['status']=='reviewed'
    original_protocol = ROOT/original['protocol']
    session = load(original_protocol,original['protocol_sha256'])
    prior = ROOT/original['zero']
    receipt = load(prior,original['zero_sha256'])
    reset = prior.with_name('reset.npz')
    assert digest(reset)==receipt['reset_sha256']
    output = RESULT.with_name('support-credit-device-zero.json')
    assert not output.exists() and KEY+'_device_zero' not in goal
    device_code = ROOT/'scripts/training/getup_support_credit_device.py'
    sources = {item.relative_to(ROOT).as_posix():digest(item) for item in (
        Path(__file__).resolve(),device_code,original_protocol,prior,reset,RESULT,ROOT/archived['protocol'],
    )}
    parent = experiment.parent
    with patch.object(parent,'config',experiment.config):
        env = parent.cuda_env(session)
    try:
        obs,_ = env.reset(seed=session['environment']['seed'])
        qpos,qvel,ctrl = (experiment.cpu(getattr(env.sim.data,key)) for key in ('qpos','qvel','ctrl'))
        rng = torch.get_rng_state(), torch.cuda.get_rng_state(env.device)
        with np.load(reset,allow_pickle=False) as target:
            for key,actual in dict(qpos=qpos,qvel=qvel,ctrl=ctrl,actor=experiment.cpu(obs['actor'])).items():
                np.testing.assert_array_equal(actual,target[key],err_msg=key)
        gpu = SolvedFloorLoad(env,single_foot_minimum=spec['per_foot_load_fraction'],
                              both_feet_minimum=spec['total_foot_load_fraction'])
        clock = ContactClock(env,dict(capture_seconds=[0.,.004],maximum_batched_substeps=1,
                                     load_threshold_nominal_weight_fraction=spec['per_foot_load_fraction']))
        assert gpu.weight==clock.weight and gpu.capacity==clock.slots and gpu.floor==clock.floor
        points, expected = clock.read_contacts()
        gpu.capture()
        gpu.check()
        normals = gpu.normal.numpy()
        np.testing.assert_allclose(normals,expected,atol=2e-6,rtol=0.)
        np.testing.assert_allclose(experiment.cpu(gpu.torch_credit()),credit(expected[:,1:],gpu.weight,spec),
                                   atol=2e-7,rtol=0.)
        independent = fixture(env.sim.wp_device,spec)
        for key,before in (('qpos',qpos),('qvel',qvel),('ctrl',ctrl)):
            np.testing.assert_array_equal(experiment.cpu(getattr(env.sim.data,key)),before,err_msg=key)
        assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state(env.device))
        assert env.common_step_counter==env._sim_step_counter==0
        write(output,dict(status='verified-cuda-zero-and-nonzero-fixture',sources=sources,worlds=env.num_envs,
            actual_initial_contact_points=int(points[:,0].sum()),baseline_contact_capacity=gpu.capacity,
            actual_initial_max_force_error_n=float(np.max(np.abs(normals-expected))),fixture=independent,
            state_and_rng_bitwise_unchanged=True,physical_steps=0,learning_updates=0,hardware_io=False,
            speed_and_active_contacts_verified=False))
    finally:
        env.close()
    current = load(PROJECT)
    permitted(current,STAGE)
    goal,task = current['goal_execution'],current['staged_training_request']
    assert goal[KEY]['result_sha256']==archived['result_sha256'] and KEY+'_device_zero' not in goal
    rel = output.relative_to(ROOT).as_posix()
    goal[KEY+'_device_zero'] = dict(status='verified-cuda-zero-and-fixture',evidence=rel,evidence_sha256=digest(output))
    info = dict(current_evidence=rel,current_protocol_execution=STAGE+'-cuda-zero-verified')
    task.update(info,next_gate='Compare GPU and CPU readback on active solved contacts and measure real rollout cost before the learning pair.')
    current.update(info)
    write(PROJECT,current)
    print('GETUP_SUPPORT_CREDIT_DEVICE_ZERO',digest(output),flush=True)


if __name__=='__main__':
    lock_file = Path.home()/'XGO-Duck/training.lock'
    with lock_file.open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
