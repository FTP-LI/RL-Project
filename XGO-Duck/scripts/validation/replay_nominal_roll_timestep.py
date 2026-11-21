"""Calibrate a short CPU numerical replay before changing the timestep."""

from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np

from record_io import write
from replay_nominal_roll_sagittal_coupling import COHORTS, read_sources
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_contact_clock_capture import contact_layout, reduce_contacts
from run_nominal_skill_composition import config


KEY = 'nominal_roll_timestep_replay'


def replay_plant(plant, step, commands, qpos, qvel, load_layout, repeats):
    plant.opt.timestep = step
    env = mujoco.MjData(plant)
    floor, kind, _ = load_layout
    force, final = [], []
    wrench = np.zeros(6)
    for motor, pose, velocity in zip(commands, qpos, qvel, strict=True):
        mujoco.mj_resetData(plant, env)
        env.qpos[:] = pose
        env.qvel[:] = velocity
        rows = []
        for value in np.repeat(motor, repeats, axis=0):
            env.ctrl[:] = value
            mujoco.mj_step(plant, env)
            geom = np.asarray(env.contact.geom).copy()
            load = np.empty(env.ncon)
            for i in range(env.ncon):
                mujoco.mj_contactForce(plant, env, i, wrench)
                load[i] = wrench[0]
            _, measured = reduce_contacts(geom, np.zeros(env.ncon, dtype=np.int32), load, floor, kind, 1)
            rows.append(measured[0])
        assert abs(env.time - len(motor) * repeats * step) < 1e-12
        assert np.isfinite(env.qpos).all() and np.isfinite(env.qvel).all()
        final.append(env.qpos.copy())
        force.append(rows)
    return dict(force=np.asarray(force), final_qpos=np.asarray(final))


def cohort_calibration(actual, expected, pose, threshold):
    original = expected[..., 0] > threshold
    current = actual['force'][..., 0] > threshold
    end = np.linalg.norm((actual['final_qpos'] - pose)[:, :3], axis=1) * 1000
    agree = (current == original).all(axis=1)
    return dict(events=len(end), exact_three_solve_load_patterns=int(agree.sum()),
        pattern_reproduced_fraction_each_event=float(agree.mean()),
        individual_4ms_load_flags_matched=int((current == original).sum()),
        median_endpoint_xyz_error_mm=float(np.median(end)), p90_endpoint_xyz_error_mm=float(np.quantile(end, .9)),
        original_loaded_at_center=int(original[:, 1].sum()), replay_loaded_at_center=int(current[:, 1].sum()),
        original_loaded_on_last_solve=int(original[:, -1].sum()), replay_loaded_on_last_solve=int(current[:, -1].sum()))


def main():
    registry = ROOT / 'project.json'
    project = read(registry)
    state, task = project['goal_execution'][KEY], project['staged_training_request']
    study = read(ROOT / state['protocol'], state['protocol_sha256'])
    assert state['status'] == 'frozen' and study['stage'] == task['current_stage']
    assert not (task['simulation_running'] or task['learning_running'] or task['pause_requested'] or task['deployment_authorized_now'])
    head = study['head_compensation_review']
    assert not read(ROOT / head['file'], head['sha256'])['admitted_neck_signs']
    prior = study['archived_events']
    source = read(ROOT / prior['protocol'], prior['protocol_sha256'])
    prior_analysis = read(ROOT / prior['evidence'], prior['evidence_sha256'])
    plant_spec, groups, marker, clock = read_sources(source)
    assert prior_analysis['source_contact_clock_sha256'] == marker['sha256']
    event = groups[0] + groups[1]
    worlds = np.array([row['world'] for row in event])
    index = np.array([row['start'] for row in event]) - clock['substep'][0]
    steps = study['original_frames_per_event']
    taken = index[:, None] - 1 + np.arange(steps)
    assert steps == 3 and taken.min() >= 0 and taken.max() + 1 < len(clock['substep'])
    np.testing.assert_array_equal(clock['substep'][taken[:, 1]], [row['start'] for row in event])
    start, end = taken[:, 0], taken[:, -1] + 1
    pos, vel = (clock[key][start, worlds] for key in ('qpos_before', 'qvel_before'))
    observed = clock['normal_force_n'][taken, worlds[:, None]]
    following = clock['qpos_before'][end, worlds]
    cfg = config(plant_spec)
    plant = Scene(cfg.scene, device='cpu').compile()
    cfg.sim.mujoco.apply(plant)
    assert plant.opt.timestep == study['archive_period_s']
    assert plant.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    layout = contact_layout(plant)
    threshold = marker['force_threshold_n']
    assert abs(layout[2] - marker['nominal_weight_n']) < 1e-8
    assert (observed[:, 0, 1] > threshold).all() and (observed[:, 1, 0] <= threshold).all()
    ids = marker['actuator_layout']['motor_ids']
    assert sorted(ids) == list(range(plant.nu))
    commands = np.zeros((len(event), steps, plant.nu))
    commands[..., ids] = clock['motor_ctrl'][taken, worlds[:, None]]
    buckets = dict(zip(COHORTS, (slice(0, len(groups[0])), slice(len(groups[0]), None)), strict=True))
    previous = replay_plant(plant, study['archive_period_s'], commands, pos, vel, layout, 1)
    cohorts = {key: cohort_calibration({field: rows[pick] for field, rows in previous.items()},
        observed[pick], following[pick], threshold) for key, pick in buckets.items()}
    condition = study['ground_load_calibration']
    calibrated = all(row['pattern_reproduced_fraction_each_event'] >= condition['minimum_full_pattern_reproduction_fraction_each_cohort']
        and row['median_endpoint_xyz_error_mm'] <= condition['maximum_median_endpoint_xyz_displacement_mm_each_cohort']
        for row in cohorts.values())
    refined = {}
    by_period = dict(baseline_force=previous['force'], baseline_final_qpos=previous['final_qpos'])
    advanced = len(event) * steps
    if calibrated:
        repeats = round(study['archive_period_s'] / study['refined_period_s'])
        assert repeats == 2
        new = replay_plant(plant, study['refined_period_s'], commands, pos, vel, layout, repeats)
        advanced += len(event) * steps * repeats
        for label, pick in buckets.items():
            before = previous['force'][pick, :, 0] <= threshold
            after = new['force'][pick, :, 0] <= threshold
            delta = after.sum(axis=1) * study['refined_period_s'] - before.sum(axis=1) * study['archive_period_s']
            refined[label] = dict(episodes=len(delta), median_paired_unloaded_time_delta_ms=float(np.median(delta) * 1000))
        by_period.update(refined_force=new['force'], refined_final_qpos=new['final_qpos'])
    assert advanced <= study['max_cpu_physics_steps']
    path = ROOT / study['evidence']
    data = path.with_suffix('.npz')
    assert not (path.exists() or data.exists())
    np.savez_compressed(data, original_force=observed, original_final_qpos=following, scene=worlds,
        first_original_substep=clock['substep'][start], **by_period)
    write(path, dict(protocol_sha256=state['protocol_sha256'], script_sha256=sha(Path(__file__)),
        raw_clock_sha256=marker['sha256'], calibration=cohorts, original_4ms_replay_calibrated=bool(calibrated),
        refined_2ms_candidate_measured=bool(calibrated), refined=refined, archived_4ms_step_s=study['archive_period_s'],
        estimated_input_scope='Replay of three recorded actuator signals on same exposed kinematics; missing full CUDA solver state',
        archive_arrays=data.relative_to(ROOT).as_posix(), archive_arrays_sha256=sha(data),
        new_cpu_physics_steps=advanced, learning_updates=0, reserved_seed_used=False, hardware_io=False))
    state.update(status='reviewed', evidence=study['evidence'], evidence_sha256=sha(path), calibrated=bool(calibrated))
    task.update(current_protocol_execution='roll-timestep-replay-reviewed', current_evidence=state['evidence'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=state['evidence'])
    write(registry, project)
    print('NOMINAL_ROLL_TIMESTEP_REPLAY', sha(path), cohorts, 'calibration', bool(calibrated), 'steps', advanced, flush=True)


if __name__ == '__main__':
    main()
