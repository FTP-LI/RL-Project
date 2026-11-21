"""Compare CPU and CUDA reductions of the same active, saved-pose contact buffers."""

import fcntl
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from check_nominal_low_friction_support_credit import (
    KEY, PREVIOUS_KEY, PROJECT, RESULT, ROOT, STAGE, credit, digest, load, permitted, write,
)
from getup_support_credit_device import CHANNELS, SolvedFloorLoad
from roll_contact_clock_capture import ContactClock
import run_nominal_low_friction_support_effort as experiment


CONTROLS = (124, 128, 132, 142, 150)
CALLS_PER_BURST = 32
BURSTS = 4


def benchmark(device, legacy, gpu):
    def timed(read):
        wp.synchronize_device(device)
        start = perf_counter_ns()
        for _ in range(CALLS_PER_BURST):
            read()
        wp.synchronize_device(device)
        return (perf_counter_ns()-start)/1_000_000/CALLS_PER_BURST

    # Alternate the two orders to reduce the influence of warm-up and GPU clocks.
    cpu_time, gpu_time = [], []
    for index in range(BURSTS):
        if index%2==0:
            cpu_time.append(timed(legacy.read_contacts))
            gpu_time.append(timed(gpu.capture))
        else:
            gpu_time.append(timed(gpu.capture))
            cpu_time.append(timed(legacy.read_contacts))
    gpu.check()
    return dict(cpu_median_ms=median(cpu_time),gpu_median_ms=median(gpu_time),cpu_bursts_ms=cpu_time,gpu_bursts_ms=gpu_time)


def main():
    project = load(PROJECT)
    permitted(project,STAGE)
    goal = project['goal_execution']
    zero = goal[KEY+'_device_zero']
    previous, archived = goal[PREVIOUS_KEY], goal[KEY]
    proof = load(ROOT/zero['evidence'],zero['evidence_sha256'])
    assert proof['status']=='verified-cuda-zero-and-nonzero-fixture'
    for path,value in proof['sources'].items():
        assert digest(ROOT/path)==value,path
    protocol = ROOT/previous['protocol']
    plan = load(protocol,previous['protocol_sha256'])
    score_path = ROOT/previous['result']
    score = load(score_path,previous['result_sha256'])
    spec = load(ROOT/archived['protocol'],archived['protocol_sha256'])
    collection_path = ROOT/previous['collection']
    folder = collection_path.parent
    raw = load(collection_path,previous['collection_sha256'])
    manifest = load(folder/raw['support_effort']['path'],raw['support_effort']['sha256'])
    by_window, = (row for row in manifest['windows'] if row['window']=='release')
    early_path = folder/by_window['file']
    assert digest(early_path)==by_window['sha256'] and manifest['contact_channels']==list(CHANNELS)
    device_zero, output = ROOT/zero['evidence'], RESULT.with_name('support-credit-device-load.json')
    assert not output.exists() and KEY+'_device_load' not in goal
    sources = {path.relative_to(ROOT).as_posix():digest(path) for path in (
        Path(__file__).resolve(),device_zero,protocol,score_path,collection_path,early_path,
    )}
    samples = {}
    for item in raw['chunks']:
        here = [q for q in CONTROLS if item['first']<=q<=item['last']]
        if not here:
            continue
        path = folder/item['file']
        assert digest(path)==item['sha256']
        sources[path.relative_to(ROOT).as_posix()] = item['sha256']
        with np.load(path,allow_pickle=False) as data:
            positions, velocities = data['qpos'],data['qvel']
            for control in here:
                i = control-item['first']
                samples[control] = positions[i],velocities[i]
    assert sorted(samples)==list(CONTROLS)
    with np.load(early_path,allow_pickle=False) as source:
        desired = np.array(CONTROLS)*5+1
        index = np.searchsorted(source['substep'],desired)
        np.testing.assert_array_equal(source['substep'][index],desired)
        motor = source['requested_torque_nm'][index]
    with patch.object(experiment.parent,'config',experiment.config):
        env = experiment.parent.cuda_env(plan)
    results = []
    try:
        env.reset(seed=plan['environment']['seed'])
        gpu = SolvedFloorLoad(env,single_foot_minimum=spec['per_foot_load_fraction'],
                              both_feet_minimum=spec['total_foot_load_fraction'])
        legacy = ContactClock(env,dict(capture_seconds=[0.,.004],maximum_batched_substeps=1,
                                      load_threshold_nominal_weight_fraction=spec['per_foot_load_fraction']))
        actuator, = env.scene['robot'].actuators
        assert list(actuator._target_names)==manifest['joint_order']
        ids = actuator._global_ctrl_ids
        sim = env.sim
        for j,control in enumerate(CONTROLS):
            qpos,qvel = samples[control]
            sim.data.qpos.copy_(torch.as_tensor(qpos,device=env.device))
            sim.data.qvel.copy_(torch.as_tensor(qvel,device=env.device))
            sim.data.ctrl[:,ids] = torch.as_tensor(motor[j],device=env.device)
            # Recompute contacts at the saved pose. This is a stationary
            # comparison of the two readers, without integration or learning.
            sim.forward()
            points, expected = legacy.read_contacts()
            gpu.capture()
            gpu.check()
            actual = gpu.normal.numpy()
            target_credit = credit(expected[:,1:],gpu.weight,spec)
            actual_credit = experiment.cpu(gpu.torch_credit())
            np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=0.)
            np.testing.assert_allclose(actual_credit,target_credit,atol=1e-5,rtol=0.)
            np.testing.assert_array_equal(experiment.cpu(sim.data.qpos),qpos)
            assert env.common_step_counter==env._sim_step_counter==0
            results.append(dict(control=control,worlds=env.num_envs,resolved_floor_contacts=int(points[:,0].sum()),
                loaded_head_worlds=int((expected[:,1]>.02*gpu.weight).sum()),
                loaded_foot_worlds=int((expected[:,2:].max(axis=-1)>.02*gpu.weight).sum()),
                maximum_force_error_n=float(np.max(np.abs(actual-expected))),
                maximum_credit_error=float(np.max(np.abs(actual_credit-target_credit))),
                timing=benchmark(env.sim.wp_device,legacy,gpu)))
        assert all(row['resolved_floor_contacts']>0 and row['loaded_foot_worlds']>0 for row in results), results
        assert any(row['loaded_head_worlds']>0 for row in results), results
    finally:
        env.close()
    write(output,dict(status='verified-active-contact-readers',method='identical reconstructed CUDA forward buffer for both readers',
        worlds=score['original_getup']['overall']['episodes'],foot_sliding_friction=score['foot_sliding_friction'],
        controls=results,bursts_per_reader=BURSTS,calls_per_burst=CALLS_PER_BURST,physical_steps=0,learning_updates=0,
        hardware_io=False,integrated_training_throughput_measured=False,sources=sources))
    current=load(PROJECT)
    permitted(current,STAGE)
    path = output.relative_to(ROOT).as_posix()
    current['goal_execution'][KEY+'_device_load']=dict(status='verified',evidence=path,evidence_sha256=digest(output))
    info = dict(current_evidence=path,current_protocol_execution=STAGE+'-active-contact-verified')
    current['staged_training_request'].update(info,next_gate='Integrate the GPU reward with the held 4 ms motor clock and benchmark the actual sampler before an equal-sample PPO comparison.')
    current.update(info)
    write(PROJECT,current)
    print('GETUP_SUPPORT_CREDIT_DEVICE_ACTIVE',digest(output),[(row['control'],row['resolved_floor_contacts'],
          round(row['timing']['cpu_median_ms'],5),round(row['timing']['gpu_median_ms'],5)) for row in results],flush=True)


if __name__=='__main__':
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
