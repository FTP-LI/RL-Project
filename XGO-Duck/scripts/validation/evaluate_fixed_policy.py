"""Fixed-command ONNX reference evaluation; no training, optimizer or hardware.

Uses the archived play task with documented overrides, model-metadata home and
scale, and raw policy actions (no UNO Q runtime filters). This tests the policy
in the training simulator, not parity with the real deployment control loop.
Only the first episode of each parallel environment counts. Videos show env 0
until its first terminal event; failed/reset episodes cannot inflate survival.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import mediapy as media
import numpy as np
import onnxruntime as ort
import torch
import mujoco
from PIL import Image, ImageDraw
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from evaluation_metrics import FirstEpisodeMetrics

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL = ROOT/'configs/experiments/formal_velocity_tracking.json'
DIAGNOSTICS = ROOT/'configs/experiments/reference_adaptation_diagnostics.json'
AMPLITUDE = ROOT/'configs/experiments/reference_command_amplitude.json'
MECHANISM = ROOT/'configs/experiments/gait_mechanism_diagnostic.json'
STRAFE = ROOT/'configs/experiments/walking_strafe_supplement.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fixed_twist(cfg, command):
    cfg.heading_command = False
    cfg.ranges.heading = None
    for name in ('rel_standing_envs', 'rel_heading_envs', 'rel_world_envs',
                 'rel_forward_envs', 'rel_turn_in_place_envs', 'rel_backward_envs', 'init_velocity_prob'):
        setattr(cfg, name, 0.0)
    cfg.ranges.lin_vel_x = (command[0], command[0])
    cfg.ranges.lin_vel_y = (command[1], command[1])
    cfg.ranges.ang_vel_z = (command[2], command[2])
    cfg.resampling_time_range = (1000.0, 1000.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='2 environments, 2 s each; not formal acceptance')
    parser.add_argument('--onnx', type=Path, default=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_walk.onnx')
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--diagnostic-variant', choices=('reference_repeat', 'runtime_home', 'runtime_filter', 'actuator_kp6'))
    variant.add_argument('--command-amplitude', action='store_true', help='Frozen command magnitude diagnostic; never replaces formal low-speed acceptance')
    variant.add_argument('--heldout-plan', type=Path, help='Frozen final-model selection; consumes one held-out attempt')
    variant.add_argument('--mechanism-variant', choices=('reference', 'foot_contact', 'hip_roll_guard'),
                         help='Frozen short fixed-weight adduction diagnostic; never formal acceptance')
    variant.add_argument('--walking-supplement', action='store_true',
                         help='Frozen left/right strafe supplement; reuse compatible saved five-command traces')
    args = parser.parse_args()
    if args.heldout_plan and args.preflight:
        parser.error('Held-out data cannot be used for a preflight')
    protocol_path = STRAFE if args.walking_supplement else (MECHANISM if args.mechanism_variant else (AMPLITUDE if args.command_amplitude else PROTOCOL))
    protocol = json.loads(protocol_path.read_text())
    if args.walking_supplement:
        assert not args.preflight, 'Use the frozen 32-episode supplement'
        assert protocol['status'] == 'frozen-before-simulation'
        assert digest(PROTOCOL) == protocol['formal_protocol_sha256']
        formal = json.loads(PROTOCOL.read_text())
        assert protocol['task'] == formal['task']
        for field in ('episodes_per_condition', 'episode_seconds', 'development_seed', 'criteria_per_condition'):
            assert protocol['evaluation'][field] == formal['evaluation'][field]
        assert set(protocol['evaluation']['commands']) == {'strafe_left', 'strafe_right'}
    if args.mechanism_variant:
        assert not args.preflight, 'Use the explicit frozen short diagnostic budget'
        assert protocol['status']=='frozen-before-simulation'
        assert digest(args.onnx)==protocol['model_sha256']
        assert digest(PROTOCOL)==protocol['formal_protocol_sha256']
        for name,expected in protocol['training_source_sha256'].items():
            assert digest(WORK/name)==expected,name
    if args.command_amplitude:
        # Change commands only: test whether archived keyboard magnitudes elicit
        # walking. Larger-command success cannot erase the original low-speed failure.
        formal = json.loads(PROTOCOL.read_text())
        assert digest(PROTOCOL) == protocol['formal_protocol_sha256']
        assert digest(args.onnx) == protocol['model_sha256']
        assert protocol['task'] == formal['task']
        for field in ('episodes_per_condition', 'episode_seconds', 'development_seed', 'criteria_per_condition'):
            assert protocol['evaluation'][field] == formal['evaluation'][field]
        for name, expected in protocol['source_basis_sha256'].items():
            assert digest(ROOT/name) == expected, name
    evaluation = protocol['evaluation']
    seed = evaluation['heldout_seed' if args.heldout_plan else 'development_seed']
    n = 2 if args.preflight else evaluation['episodes_per_condition']
    seconds = 2 if args.preflight else evaluation['episode_seconds']
    mode = 'heldout' if args.heldout_plan else ('preflight' if args.preflight else 'reference')
    if args.command_amplitude:
        mode = 'amplitude_preflight' if args.preflight else 'amplitude_diagnostic'
    if args.mechanism_variant:
        mode='mechanism_'+args.mechanism_variant
    if args.walking_supplement:
        mode='strafe_supplement'
    diagnostic = None
    if args.diagnostic_variant:
        plan = json.loads(DIAGNOSTICS.read_text())
        assert digest(PROTOCOL) == plan['formal_protocol_sha256']
        assert digest(args.onnx) == plan['model_sha256']
        assert (seed, evaluation['episodes_per_condition'], evaluation['episode_seconds']) == (plan['seed'], plan['num_envs'], plan['episode_seconds'])
        diagnostic = plan['variants'][args.diagnostic_variant]
        mode = ('diagnostic_preflight_' if args.preflight else 'diagnostic_') + args.diagnostic_variant
    is_official = args.onnx.resolve() == (ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_walk.onnx').resolve()
    label = 'official_walk' if is_official else 'custom_onnx'
    run_id = f'{label}_{mode}_{datetime.now().strftime("%Y%m%dT%H%M%S")}_s{seed}'
    output = ROOT/'records/evaluations'/run_id
    selection = None
    if args.heldout_plan:
        # Only the random seed changes. Freeze provenance and claim the single
        # attempt before creating an environment; never resample after peeking.
        from heldout_evaluation import claim_attempt, check_environment
        selection = claim_attempt(ROOT, WORK, args.heldout_plan, args.onnx, run_id, output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'videos').mkdir()
    model = args.onnx.resolve()
    report = dict(run_id=run_id, status='running', phase=mode, pid=os.getpid(),
                  started_at=datetime.now(timezone.utc).isoformat(), seed=seed,
                  num_envs=n, episode_seconds=seconds, conditions=[],
                  training_started=False, hardware_verified=False,
                  scope='ONNX reference in training simulation; model home; no runtime action/IMU filters',
                  model=str(model), model_sha256=digest(model), protocol_sha256=digest(protocol_path),
                  linear_xy_error_definition='time_mean_planar_l2_then_episode_mean',
                  packages={name: importlib.metadata.version(name) for name in
                            ('mjlab', 'torch', 'mujoco', 'mujoco-warp', 'onnxruntime')})
    save_json(output/'evaluation.json', report)
    save_json(ROOT/'records/evaluations/current-evaluation.json', dict(run_id=run_id, pid=os.getpid(), path=str(output)))
    shutil.copy2(protocol_path, output/'protocol.json')
    if selection is not None:
        report['evaluation_seed_role'] = 'heldout'
        report['heldout_selection_sha256'] = digest(args.heldout_plan)
        report['scope'] = 'Held-out final-policy simulation; selected before seed 2001; no hardware acceptance'
        shutil.copy2(args.heldout_plan, output/'heldout-selection.json')
        shutil.copy2(Path(__file__).with_name('heldout_evaluation.py'), output/'heldout_evaluation.py')
        shutil.copy2(Path(__file__).with_name('prepare_policy_export.py'), output/'prepare_policy_export.py')
    if args.command_amplitude:
        report['command_amplitude_diagnostic'] = True
        report['scope'] = 'Fixed-weight command magnitude diagnostic; not replacement formal or hardware acceptance'
        report['formal_protocol_sha256'] = protocol['formal_protocol_sha256']
    if args.mechanism_variant:
        report['mechanism_variant']=args.mechanism_variant
        report['scope']='Short fixed-weight adduction mechanism diagnostic; no learning, formal acceptance or hardware claim'
    if args.walking_supplement:
        report['walking_supplement'] = True
        report['scope'] = 'Two additional fixed commands; seven-command screening is recomputed separately from traces, not legacy pass flags'
    if diagnostic is not None:
        shutil.copy2(DIAGNOSTICS, output/DIAGNOSTICS.name)
        report['diagnostic_plan_sha256'] = digest(DIAGNOSTICS)
        report['diagnostic_variant'] = args.diagnostic_variant
        report['diagnostic_settings'] = diagnostic
        report['scope'] = 'Fixed-weight single-factor simulation diagnostic; not held-out or hardware acceptance'
    for name in ('evaluate_fixed_policy.py', 'evaluation_metrics.py'):
        shutil.copy2(Path(__file__).with_name(name), output/name)
    env = None
    try:
        assert digest(WORK/'uv.lock') == '0ef769135d86e63c03663b2df570c0217217b541ab234460a879dff8fb260fa9'
        source_files = ['uv.lock', 'src/mjlab_microduck/tasks/__init__.py',
                        'src/mjlab_microduck/tasks/mdp.py',
                        'src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py',
                        'src/mjlab_microduck/robot/xgoduck_constants.py',
                        'src/mjlab_microduck/robot/xgoduck/robot_walk.xml']
        report['source_sha256'] = {name:digest(WORK/name) for name in source_files}
        if args.walking_supplement:
            assert report['source_sha256'] == protocol['training_source_sha256']
        if selection is not None:
            assert report['source_sha256'] == selection['training_source_sha256']
            assert report['packages'] == selection['packages']
        if args.command_amplitude:
            assert report['source_sha256'] == protocol['training_source_sha256']
        report['evaluator_sha256'] = digest(Path(__file__))
        report['metric_code_sha256'] = digest(Path(__file__).with_name('evaluation_metrics.py'))
        configure_torch_backends()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        assert inp.shape == [1, 61] and out.shape == [1, 14]
        metadata = session.get_modelmeta().custom_metadata_map
        names = metadata['joint_names'].split(',')
        home = np.array(metadata['default_joint_pos'].split(','), dtype=np.float32)
        scale = float(metadata['action_scale'])
        assert home.shape == (14,) and np.isfinite(home).all() and math.isfinite(scale) and scale > 0
        report['model_metadata'] = metadata
        model_home = home.copy()
        alpha = np.zeros(14, dtype=np.float32)
        if diagnostic is not None:
            # Only the selected factor changes. The runtime-home diagnostic
            # changes both relative observations and target reconstruction,
            # while leaving the physical reset pose and dynamics unchanged.
            runtime_dir = ROOT/'official_resources/xgoduck_runtime_arduino/python'
            sys.path.insert(0, str(runtime_dir))
            import rl_core
            assert Path(rl_core.__file__).resolve() == runtime_dir/'rl_core.py'
            report['runtime_core_sha256'] = digest(runtime_dir/'rl_core.py')
            if diagnostic['home'] == 'runtime':
                home = np.deg2rad(rl_core.HOME_DEG).astype(np.float32)
            if diagnostic['action_filter'] == 'runtime':
                # Preserve the archived EMA exactly, including ID 20 in [5:10].
                # Observation history still contains RAW network outputs.
                alpha[:] = rl_core.ACTION_ALPHA
                alpha[5:10] = rl_core.ACTION_ALPHA_HEAD
        cfg = load_env_cfg(protocol['task'], play=True)
        cfg.seed = seed
        cfg.scene.num_envs = n
        cfg.auto_reset = False
        cfg.episode_length_s = seconds
        cfg.curriculum = {}
        if diagnostic is not None:
            actuators = cfg.scene.entities['robot'].articulation.actuators
            assert len(actuators) == 1 and actuators[0].kp_fw == 5.0
            # A source-informed sensitivity test, not a real firmware P/D model.
            actuators[0].kp_fw = diagnostic['kp_fw']
        cfg.events.pop('push_robot', None)
        fixed_twist(cfg.commands['twist'], [0, 0, 0])
        for name in ('head_pose', 'body_pose'):
            cfg.commands[name].ranges = ((0.0, 0.0),)*len(cfg.commands[name].ranges)
            cfg.commands[name].zero_command_prob = 1.0
            cfg.commands[name].resampling_time_range = (1000.0, 1000.0)
        cfg.viewer.width, cfg.viewer.height = 960, 720
        cfg.viewer.distance, cfg.viewer.elevation = 1.2, -20
        cfg.viewer.max_extra_envs = 0
        if args.mechanism_variant=='foot_contact':
            # One factor: allow the two feet to collide; retain floor contact,
            # geometry, friction and actuator settings. No reward is changed.
            robot_cfg=cfg.scene.entities['robot']
            assert len(robot_cfg.collisions)==1
            collision=robot_cfg.collisions[0]
            robot_cfg.collisions=(replace(collision,
                contype={r'^(left|right)_foot_collision$':3,**collision.contype},
                conaffinity={r'^(left|right)_foot_collision$':2,**collision.conaffinity}),)
        dump_yaml(output/'env.yaml', asdict(cfg))
        if selection is not None:
            check_environment(ROOT, selection, output/'env.yaml')
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array')
        robot = env.scene['robot']
        joint_ids, physical_joint_names = robot.find_joints(names, preserve_order=True)
        assert physical_joint_names == names
        action_term = env.action_manager.get_term('joint_pos')
        assert list(action_term._target_names) == names
        sim_home = action_term._offset.detach().cpu().numpy()
        if sim_home.ndim == 1:
            sim_home = np.broadcast_to(sim_home, (n, 14))
        sim_scale = np.asarray(action_term._scale if not torch.is_tensor(action_term._scale)
                               else action_term._scale.detach().cpu().numpy())
        assert np.isfinite(sim_scale).all() and (sim_scale != 0).all()
        actor_cfg = cfg.observations['actor']
        assert list(actor_cfg.terms) == ['base_ang_vel', 'projected_gravity', 'joint_pos',
                                       'joint_vel', 'actions', 'command', 'head_command', 'body_command']
        assert not actor_cfg.history_length
        assert all(t.scale is None and t.clip is None for t in actor_cfg.terms.values())
        feet, foot_names = robot.find_sites(['left_foot', 'right_foot'], preserve_order=True)
        assert foot_names == ['left_foot', 'right_foot']
        if args.mechanism_variant:
            model=env.sim.mj_model
            gids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+side+'_foot_collision') for side in ('left','right')]
            assert min(gids)>=0
            report['foot_collision_masks']=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),
                contype=int(model.geom_contype[g]),conaffinity=int(model.geom_conaffinity[g])) for g in gids]
            expected_mask=(3,2) if args.mechanism_variant=='foot_contact' else (1,0)
            assert all((v['contype'],v['conaffinity'])==expected_mask for v in report['foot_collision_masks'])
            report['hip_roll_target_guard_deg']=protocol['hip_roll_target_guard_deg'] if args.mechanism_variant=='hip_roll_guard' else None
        report.update(dict(step_dt=env.step_dt, physics_dt=env.physics_dt,
                           actor_corruption=actor_cfg.enable_corruption,
                           simulation_home=sim_home.tolist(), simulation_action_scale=sim_scale.tolist(),
                           policy_home=home.tolist(), model_home=model_home.tolist(),
                           action_filter_alpha=alpha.tolist(),
                           foot_sites=foot_names,
                           remaining_events={name:dict(mode=value.mode, function=value.func.__name__) for name,value in cfg.events.items()},
                           adapter='obs joint_pos += sim_home-policy_home; previous raw model action; applied=alpha*filtered_history+(1-alpha)*raw; sim_action=(policy_home+scale*applied-sim_home)/sim_scale',
                           frame_policy='25 fps env 0 only through first outcome; timestamp list saved; no reset footage'))
        horizon = round(seconds/env.step_dt)
        assert abs(horizon*env.step_dt-seconds) < 1e-6
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        print(f'EVALUATION {run_id}: {n} episodes x {seconds}s x {len(evaluation["commands"])} commands; no training', flush=True)
        for condition, command_list in evaluation['commands'].items():
            wall_start = time.monotonic()
            command = np.asarray(command_list, dtype=np.float32)
            fixed_twist(env.command_manager.get_term('twist').cfg, command_list)
            obs, _ = env.reset(seed=seed)
            previous = np.zeros((n, 14), dtype=np.float32)
            filtered = np.zeros_like(previous)
            initial_xy = robot.data.root_link_pos_w[:, :2].detach().cpu().numpy().copy()
            initial_joint_positions = robot.data.joint_pos[:, joint_ids].detach().cpu().numpy().copy()
            stats = FirstEpisodeMetrics(initial_xy, horizon, env.step_dt)
            trace = {name: [] for name in ('active', 'xy', 'velocity', 'yaw_rate', 'observations', 'actions', 'applied_actions', 'requested_joint_targets', 'joint_positions', 'joint_velocities', 'contact', 'foot_speed', 'terminated', 'truncated', 'fell_over', 'nan_state', 'out_of_bounds')}
            # Preserve posture evidence for ordinary final-checkpoint evaluation
            # as well as fixed-policy interventions. Logging does not modify
            # actions, physics, sampling horizon or acceptance thresholds.
            trace.update({name:[] for name in ('policy_joint_targets','foot_positions','root_quaternion','self_collision')})
            video_path = output/'videos'/f'{condition}_env0.mp4'
            frames_at = []
            with media.VideoWriter(str(video_path), shape=(720,960), fps=25, qp=22) as video:
                def frame(step, outcome):
                    pixels = env.render()
                    image = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 960, 56), fill=(16, 25, 34))
                    draw.text((12, 8), f'{condition} | env 0 | seed {seed} | t={step*env.step_dt:.2f}s | {outcome}', fill='white')
                    draw.text((12, 30), f'command vx={command[0]:.2f} vy={command[1]:.2f} yaw={command[2]:.2f} | {mode} | ONNX inference, no training', fill='white')
                    video.add_image(np.asarray(image))
                    frames_at.append(step*env.step_dt)
                    if step == 0 or outcome != 'running':
                        image.save(output/'videos'/f'{condition}_{"start" if step == 0 else "end"}.jpg')
                frame(0, 'running')
                # Persistent manager buffers are reused by reset between conditions.
                # no_grad avoids allocating inference-only tensors inside the loop.
                with torch.no_grad():
                    for step in range(1, horizon+1):
                        actor = obs['actor'].detach().cpu().numpy().copy()
                        assert actor.shape == (n,61)
                        np.testing.assert_allclose(actor[:,48:51], np.broadcast_to(command,(n,3)), atol=1e-6, rtol=0)
                        np.testing.assert_allclose(actor[:,51:61], 0, atol=1e-6, rtol=0)
                        actor[:,6:20] += sim_home-home
                        actor[:,34:48] = previous
                        assert np.isfinite(actor[stats.active]).all(), 'Nonfinite live observation'
                        actions = np.zeros((n,14), dtype=np.float32)
                        for i in np.flatnonzero(stats.active):
                            actions[i] = session.run([out.name], {inp.name:actor[i:i+1]})[0][0]
                        assert np.isfinite(actions).all(), 'Nonfinite ONNX action'
                        filtered = alpha*filtered + (1-alpha)*actions
                        targets = home+scale*filtered
                        policy_targets=targets.copy()
                        if args.mechanism_variant=='hip_roll_guard':
                            # One factor: bound requested hip-roll targets to
                            # +/-10 deg to test posture sensitivity. This is a
                            # diagnostic guard, not a calibrated hardware limit
                            # or proof that an unmodified policy can walk well.
                            roll=[names.index(side+'_hip_roll') for side in ('left','right')]
                            bound=np.deg2rad(protocol['hip_roll_target_guard_deg'])
                            targets[:,roll]=np.clip(targets[:,roll],-bound,bound)
                        sim_action = (targets-sim_home)/sim_scale
                        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(sim_action, device=env.device, dtype=torch.float32))
                        def cpu(value):
                            return value.detach().cpu().numpy().copy()
                        xy = cpu(robot.data.root_link_pos_w[:, :2])
                        velocity = cpu(robot.data.root_link_lin_vel_b[:, :2])
                        yaw = cpu(robot.data.root_link_ang_vel_b[:, 2])
                        joint_positions = cpu(robot.data.joint_pos[:, joint_ids])
                        joint_velocities = cpu(robot.data.joint_vel[:, joint_ids])
                        foot_speed = np.linalg.norm(cpu(robot.data.site_lin_vel_w[:, feet, :2]), axis=2)
                        contact = cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2)
                        term, timeout = cpu(terminated), cpu(truncated)
                        fell = cpu(env.termination_manager.get_term('fell_over'))
                        nan = cpu(env.termination_manager.get_term('nan_state'))
                        bounds = cpu(env.termination_manager.get_term('out_of_terrain_bounds'))
                        assert all(np.isfinite(v[stats.active]).all() for v in (xy, velocity, yaw, foot_speed, joint_positions, joint_velocities)), 'Nonfinite simulator measurement'
                        active = stats.update(step, xy, velocity, yaw, command, actions-previous,
                                              foot_speed, contact, term, timeout, fell, nan, bounds)
                        for key, value in dict(active=active, xy=xy, velocity=velocity, yaw_rate=yaw,
                                               observations=actor, actions=actions, contact=contact, foot_speed=foot_speed,
                                               applied_actions=filtered.copy(), requested_joint_targets=targets,
                                               joint_positions=joint_positions, joint_velocities=joint_velocities,
                                               terminated=term, truncated=timeout, fell_over=fell, nan_state=nan, out_of_bounds=bounds).items():
                            # previous aliases actions below; terminal reset clears
                            # previous in-place. Keep the recorded action independent
                            # so an episode's final applied action is not lost.
                            trace[key].append(value.copy() if key == 'actions' else value)
                        for key,value in dict(policy_joint_targets=policy_targets,
                            foot_positions=cpu(robot.data.site_pos_w[:,feet]),
                            root_quaternion=cpu(robot.data.root_link_quat_w),
                            self_collision=cpu(env.scene['self_collision'].data.found)).items():
                            trace[key].append(value)
                        if active[0] and (step%2 == 0 or not stats.active[0]):
                            frame(step, 'running' if stats.active[0] else stats.reason[0])
                        previous = actions
                        if step%100 == 0 or not stats.active.any() or step == horizon:
                            print(f'{condition}: {step}/{horizon} steps | first episodes active {stats.active.sum()}/{n} | {time.monotonic()-wall_start:.1f}s wall', flush=True)
                        if not stats.active.any():
                            break
                        reset_ids = torch.nonzero(terminated | truncated, as_tuple=False).squeeze(-1)
                        if len(reset_ids):
                            # Manager requires reset of terminal envs. Their subsequent
                            # trajectories remain excluded by FirstEpisodeMetrics.active.
                            obs, _ = env.reset(env_ids=reset_ids)
                            previous[reset_ids.cpu().numpy()] = 0
                            filtered[reset_ids.cpu().numpy()] = 0
            result = stats.result()
            result.update(condition=condition, command=command_list, seed=seed,
                          finite_observations_and_actions=True, wall_seconds=time.monotonic()-wall_start,
                          video=dict(path=str(video_path.relative_to(output)), sha256=digest(video_path),
                                     frame_times_seconds=frames_at, env_index=0, outcome=stats.reason[0]))
            limits = evaluation['criteria_per_condition']
            result['meets_numeric_criteria'] = bool(
                result['full_episode_survival'] >= limits['full_episode_survival_min']
                and result['linear_xy_mae_m_s'] <= limits['linear_xy_mae_max_m_s']
                and result['yaw_mae_rad_s'] <= limits['yaw_mae_max_rad_s']
                and (condition != 'stop' or result['final_displacement_m_median'] <= limits['stop_drift_max_m_median']))
            result['formal_acceptance_eligible'] = not args.preflight and diagnostic is None and not args.command_amplitude and not args.mechanism_variant and not args.walking_supplement
            np.savez_compressed(output/f'{condition}-trace.npz', initial_xy=initial_xy,
                                initial_joint_positions=initial_joint_positions,
                                **{name:np.stack(values) for name,values in trace.items()})
            save_json(output/f'{condition}.json', result)
            report['conditions'].append(result)
            save_json(output/'evaluation.json', report)
            print(f'{condition}: survival={result["full_episode_survival"]:.3f}, falls={result["fall_count_excluding_timeouts"]}, xy MAE={result["linear_xy_mae_m_s"]:.4f}, yaw MAE={result["yaw_mae_rad_s"]:.4f}', flush=True)
        report['status'] = 'completed'
        report['meets_all_numeric_criteria'] = all(v['meets_numeric_criteria'] for v in report['conditions'])
        report['meets_all_reference_criteria'] = not args.preflight and diagnostic is None and not args.command_amplitude and not args.mechanism_variant and not args.walking_supplement and report['meets_all_numeric_criteria']
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_json(output/'evaluation.json', report)
        print(f'EVALUATION {report["status"]}: {output}', flush=True)
    raise SystemExit(0 if report['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
