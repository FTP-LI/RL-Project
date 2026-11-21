"""Check the active CUDA solver clock on identical archived motor inputs."""

from pathlib import Path
import fcntl

from mjlab.scene import Scene
from mjlab.sim import Simulation
import numpy as np
import torch
import warp as wp

from record_io import write
from replay_nominal_roll_sagittal_coupling import read_sources
from review_nominal_skill_composition_handoff import ROOT, read, sha
from run_nominal_skill_composition import config


KEY = 'nominal_roll_timestep_cuda_clock'
SAME_FIELDS = ('qpos', 'qvel', 'qacc_warmstart', 'act', 'ctrl', 'time')


def measured(sim, value, macro_seconds, start, rows):
    step = sim.wp_model.opt.timestep
    graph = sim.step_graph
    with wp.ScopedDevice(sim.wp_device):
        step.fill_(value)
    expected = float(np.float32(value))
    np.testing.assert_array_equal(step.numpy(), np.full(step.shape, expected, dtype=np.float32))
    sim.reset()
    sim.data.qpos.copy_(start[0])
    sim.data.qvel.copy_(start[1])
    sim.data.ctrl.zero_()
    initial = {k: getattr(sim.data, k).clone() for k in SAME_FIELDS}
    assert not initial['time'].any()
    repeats = round(macro_seconds / value)
    elapsed, intervals, count = [], [], 0
    for phase, command in enumerate(rows, 1):
        begin = sim.data.time.clone()
        sim.data.ctrl.copy_(command)
        for part in range(1, repeats + 1):
            sim.step()
            count += 1
            torch.testing.assert_close(sim.data.ctrl.clone(), command, atol=0, rtol=0)
            clock = sim.data.time.clone()
            torch.testing.assert_close(clock - begin, torch.full_like(clock, part * value), atol=2e-8, rtol=0)
            elapsed.append(float(clock[0].item()))
        torch.testing.assert_close(sim.data.time.clone(), torch.full_like(begin, phase * macro_seconds), atol=2e-8, rtol=0)
        intervals.append(float(sim.data.time[0].item()))
    assert graph is sim.step_graph and sim.use_cuda_graph and sim.step_graph is not None
    assert sim.cfg.mujoco.timestep == sim.mj_model.opt.timestep == macro_seconds
    assert bool(torch.isfinite(sim.data.qpos.clone()).all() and torch.isfinite(sim.data.qvel.clone()).all())
    return initial, dict(warp_step_s=expected, held_motor_updates=len(rows), executed_solver_steps=count,
        solver_clock_first_world_s=elapsed, macro_clock_first_world_s=intervals, input_unchanged_within_every_interval=True)


def main():
    project_file = ROOT / 'project.json'
    project = read(project_file)
    goal = project['goal_execution']
    state, task = goal[KEY], project['staged_training_request']
    plan = read(ROOT / state['protocol'], state['protocol_sha256'])
    assert task['current_stage'] == plan['stage'] and state['status'] == 'frozen'
    assert goal['status'] == 'active' and not goal['pause_requested_by_user']
    assert not (task['simulation_running'] or task['learning_running'] or task['pause_requested'] or task['deployment_authorized_now'])
    previous = plan['previous']
    cpu = read(ROOT / previous['protocol'], previous['protocol_sha256'])
    final = read(ROOT / previous['evidence'], previous['evidence_sha256'])
    check = read(ROOT / previous['paired_readback'], previous['paired_readback_sha256'])
    assert final['original_4ms_replay_calibrated'] and check['timestep_result_sha256'] == previous['evidence_sha256']
    link = cpu['archived_events']
    source = read(ROOT / link['protocol'], link['protocol_sha256'])
    plant_spec, cohorts, marker, clock = read_sources(source)
    n = plan['worlds_per_source_cohort']
    events = cohorts[0][:n] + cohorts[1][:n]
    worlds = np.array([entry['world'] for entry in events])
    time = np.array([entry['start'] for entry in events]) - clock['substep'][0] - 1
    assert marker['sha256'] == final['raw_clock_sha256']
    plant_cfg = config(plant_spec)
    plant = Scene(plant_cfg.scene, device='cpu').compile()
    assert torch.cuda.is_available()
    target = 'cuda:0'
    sim = Simulation(num_envs=len(events), cfg=plant_cfg.sim, model=plant, device=target)
    base = tuple(torch.as_tensor(clock[key][time, worlds], device=target) for key in ('qpos_before', 'qvel_before'))
    values = []
    for k in range(plan['control_intervals']):
        motor = np.zeros((len(events), plant.nu), dtype=np.float32)
        motor[:, marker['actuator_layout']['motor_ids']] = clock['motor_ctrl'][time + k, worlds]
        values.append(torch.as_tensor(motor, device=target))
    samples = [measured(sim, duration, plan['fixed_motor_period_s'], base, values) for duration in plan['solver_step_s']]
    for field in SAME_FIELDS:
        torch.testing.assert_close(samples[0][0][field], samples[1][0][field], atol=0, rtol=0)
    operations = sum(row['executed_solver_steps'] for _, row in samples)
    assert operations == plan['maximum_inner_warp_steps'] and operations * len(events) == plan['maximum_scene_inner_steps']
    initial = plan['initial_attempt']
    assert sha(ROOT / initial['log']) == initial['log_sha256']
    assert sha(ROOT / initial['preserved_source']) == initial['preserved_source_sha256']
    assert operations + initial['batched_steps'] == plan['maximum_cumulative_warp_steps']
    assert operations * len(events) + initial['world_steps'] == plan['maximum_cumulative_scene_steps']
    evidence = ROOT / plan['evidence']
    assert not evidence.exists()
    write(evidence, dict(protocol_sha256=state['protocol_sha256'], script_sha256=sha(Path(__file__)),
        device=torch.cuda.get_device_name(0), scene_count=len(events), full_reset_readbacks_equal=list(SAME_FIELDS),
        starting_scenes=worlds.tolist(), prior_clock_sha256=marker['sha256'],
        solver_passes=operations, physical_scene_steps=operations * len(events),
        initially_stopped_scene_steps=initial['world_steps'], cumulative_scene_steps=plan['maximum_cumulative_scene_steps'],
        modes=[row for _, row in samples], policy_updates=0, hardware_io=False,
        closed_loop_tested=False))
    state.update(status='reviewed', evidence=plan['evidence'], evidence_sha256=sha(evidence))
    task.update(current_protocol_execution='roll-timestep-cuda-clock-qualified', current_evidence=state['evidence'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=state['evidence'])
    write(project_file, project)
    print('NOMINAL_ROLL_TIMESTEP_CUDA_CLOCK', sha(evidence), operations, [row['macro_clock_first_world_s'] for _, row in samples], flush=True)


if __name__ == '__main__':
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
