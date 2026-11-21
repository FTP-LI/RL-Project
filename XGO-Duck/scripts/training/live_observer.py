"""Read-only native display of environment 0 in the actual training process.

What/why: show robot motion continuously and collect the reported leg-adduction
symptom. Copy one GPU state to an isolated CPU render model at a configured wall
time cap (default 4 Hz). Never call env.step/reset/policy, never copy mouse perturbations back, and
never draw all 4096 environments. Renderer cost is measured, not assumed free.
The trainer remains responsible for every rollout and optimizer update.

After the completed repair pilot, record and display the head-pose request
beside measured actuator angles. This distinguishes commanded turning from
tracking error without changing commands, rewards, targets or the simulator.
Persist display-close timing even when final cleanup updates the status.
"""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import time
import threading

import mujoco
import mujoco.viewer
import numpy as np


VISUAL_FIELDS = frozenset(('qpos0', 'geom_rgba', 'geom_size', 'geom_pos',
    'geom_quat', 'mat_rgba', 'site_pos', 'site_quat', 'body_pos', 'body_quat',
    'body_ipos', 'body_inertia', 'body_iquat', 'body_mass', 'cam_pos', 'cam_quat',
    'cam_fovy', 'cam_intrinsic', 'light_pos', 'light_dir'))


def cpu(value):
    return value.detach().cpu().numpy().copy()


def open_passive(model, data):
    before = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    return handle, [thread for thread in threading.enumerate() if thread not in before]


def close_passive(handle, threads):
    handle.close()
    # Handle.close requests exit; wait for the renderer to release its context
    # before Python's GLFW atexit handler or the copied model is destroyed.
    deadline = time.perf_counter()+5
    for thread in threads:
        thread.join(timeout=max(0,deadline-time.perf_counter()))
    assert not any(thread.is_alive() for thread in threads), 'Native viewer cleanup did not finish'


class LiveObserver:
    def __init__(self, env, record, max_fps=4.0):
        self.env = env.unwrapped
        self.record = Path(record)
        self.period = 1.0 / max_fps
        self.next_frame = 0.0
        self.frames = 0
        self.total_seconds = 0.0
        self.handle = None
        self.viewer_threads = []
        self.closed_reported = False
        self.model = copy.deepcopy(self.env.sim.mj_model)
        self.data = mujoco.MjData(self.model)
        assert self.model is not self.env.sim.mj_model
        assert not np.shares_memory(self.model.qpos0, self.env.sim.mj_model.qpos0)
        self.robot = self.env.scene['robot']
        self.feet, names = self.robot.find_sites(['left_foot', 'right_foot'], preserve_order=True)
        assert names == ['left_foot', 'right_foot']
        self.joint_names = list(self.robot.joint_names)
        self.head_names = ['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']
        self.head_indices = [self.joint_names.index(name) for name in self.head_names]
        self.root_body = self.name_id(mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
        self.joint_ids = [self.name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.log = (self.record/'live-view.jsonl').open('a', buffering=1)
        self.metadata = dict(status='starting', started_at=datetime.now(timezone.utc).isoformat(),
            display='Actual training environment 0; native MuJoCo window',
            environment_index=0, max_wall_fps=max_fps, joint_names=self.joint_names,
            head_joint_names=self.head_names,
            head_angle_note='Actuator-joint angles; error excludes any passive backlash-link deflection',
            telemetry_stops_when_display_closes=True,
            joint_ranges_rad=self.model.jnt_range[self.joint_ids].tolist(),
            foot_sites=names, input_from_training_only=True, extra_environment_steps=0,
            compiled_geometries=[dict(id=i,
                name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i),
                body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[i])),
                contype=int(self.model.geom_contype[i]),conaffinity=int(self.model.geom_conaffinity[i]))
                for i in range(self.model.ngeom)],
            explicit_contact_pairs=[dict(geom1=int(self.model.pair_geom1[i]),geom2=int(self.model.pair_geom2[i]))
                for i in range(self.model.npair)],
            mouse_perturbations_applied=False, isolated_cpu_render_model=True,
            contact_note='Ground contact flags from GPU sensors; CPU reconstructed contacts are diagnostic only',
            stable_gait_verified=False)
        self.save_status()

    def name_id(self, kind, suffix):
        count = {mujoco.mjtObj.mjOBJ_BODY:self.model.nbody,
                 mujoco.mjtObj.mjOBJ_JOINT:self.model.njnt}[kind]
        matches = [i for i in range(count)
                   if (mujoco.mj_id2name(self.model, kind, i) or '').rsplit('/',1)[-1] == suffix]
        assert len(matches) == 1, (kind, suffix, matches)
        return matches[0]

    def save_status(self):
        self.metadata.update(frames=self.frames, total_observer_seconds=self.total_seconds)
        target = self.record/'live-view-status.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+'\n')
        temporary.replace(target)

    def setup(self):
        self.copy_state()
        self.handle, self.viewer_threads = open_passive(self.model, self.data)
        self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.handle.cam.trackbodyid = self.root_body
        self.handle.cam.distance = 0.8
        self.handle.cam.azimuth = 135
        self.handle.cam.elevation = -15
        # No input callback is registered and sync_viewer_to_env is never used.
        self.metadata.update(status='display-open', opened_at=datetime.now(timezone.utc).isoformat())
        self.save_status()
        print(f'LIVE_VIEW_OPEN: actual training env 0; max {1/self.period:g} Hz wall time; no extra simulation steps', flush=True)

    def copy_state(self):
        sim = self.env.sim
        for name in sim.expanded_fields & VISUAL_FIELDS:
            source = cpu(getattr(sim.model, name)[0])
            destination = getattr(self.model, name)
            destination[:] = source.reshape(destination.shape)
        for name in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'xfrc_applied'):
            destination = getattr(self.data, name)
            if destination.size:
                destination[:] = cpu(getattr(sim.data, name)[0]).reshape(destination.shape)
        self.data.time = self.env.common_step_counter * self.env.step_dt
        # Forward kinematics/contact reconstruction on the isolated CPU copy;
        # this does not integrate physics or modify the live GPU environment.
        mujoco.mj_forward(self.model, self.data)

    def observe(self):
        now = time.perf_counter()
        if now < self.next_frame:
            return
        self.next_frame = now + self.period
        if not self.handle or not self.handle.is_running():
            if not self.closed_reported:
                self.metadata.update(status='display-closed-training-continues',
                    display_closed_at=datetime.now(timezone.utc).isoformat(),
                    display_closed_common_step=int(self.env.common_step_counter),
                    display_close_observation='Viewer handle was not running; initiating cause is unknown')
                self.save_status()
                print('LIVE_VIEW_CLOSED: training continues; display closure recorded', flush=True)
                self.closed_reported = True
            return
        self.copy_state()
        feet = cpu(self.robot.data.site_pos_w[0, self.feet])
        command = cpu(self.env.command_manager.get_command('twist')[0])
        joints = cpu(self.robot.data.joint_pos[0])
        head_command = cpu(self.env.command_manager.get_command('head_pose')[0])
        head_home = cpu(self.robot.data.default_joint_pos[0, self.head_indices])
        head_speed = cpu(self.robot.data.joint_vel[0, self.head_indices])
        assert head_command.shape == head_home.shape == head_speed.shape == (4,)
        head_actual = joints[self.head_indices]
        head_target = head_home + head_command
        ground = cpu(self.env.scene['feet_ground_contact'].data.found[0]).reshape(2,-1).any(axis=1)
        self_contact = bool(cpu(self.env.scene['self_collision'].data.found[0]).any())
        rotation = self.data.xmat[self.root_body].reshape(3,3)
        lateral = float((feet[0]-feet[1]) @ rotation[:,1])
        contacts = []
        for contact in self.data.contact[:self.data.ncon]:
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                     for g in contact.geom]
            contacts.append(dict(geoms=names, distance_m=float(contact.dist)))
        row = dict(at=datetime.now(timezone.utc).isoformat(),
            common_step_counter=int(self.env.common_step_counter),
            command=command.tolist(), joint_positions_rad=joints.tolist(),
            head_pose_command_rad=head_command.tolist(),
            head_pose_target_rad=head_target.tolist(),
            head_joint_positions_rad=head_actual.tolist(),
            head_joint_velocities_rad_s=head_speed.tolist(),
            head_pose_error_actuator_rad=(head_actual-head_target).tolist(),
            foot_site_positions_world_m=feet.tolist(),
            foot_site_distance_xy_m=float(np.linalg.norm(feet[0,:2]-feet[1,:2])),
            signed_foot_lateral_gap_m=lateral, ground_contact=ground.tolist(),
            gpu_self_collision_detected=self_contact,
            cpu_reconstructed_contacts=contacts)
        with self.handle.lock():
            # Discard mouse forces even in the isolated rendering copy.
            self.data.xfrc_applied[:] = 0
        if self.handle.is_running():
            target_text = ' / '.join(f'{v:.1f}' for v in np.rad2deg(head_target))
            actual_text = ' / '.join(f'{v:.1f}' for v in np.rad2deg(head_actual))
            self.handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                'LIVE TRAINING ENV 0\nControl step\nCommand vx / vy / yaw\nFoot lateral gap\nGround L / R\nHead target deg N / P / Y / R\nHead actual deg N / P / Y / R',
                f'Actual state, max {1/self.period:g} Hz\n{row["common_step_counter"]}\n'
                f'{command[0]:.2f} / {command[1]:.2f} / {command[2]:.2f}\n'
                f'{lateral*1000:.1f} mm\n{bool(ground[0])} / {bool(ground[1])}\n'
                f'{target_text}\n{actual_text}'))
            self.handle.sync()
        cost = time.perf_counter()-now
        self.frames += 1
        self.total_seconds += cost
        row['observer_wall_seconds'] = cost
        self.log.write(json.dumps(row, allow_nan=False)+'\n')
        if self.frames == 1 or self.frames % 40 == 0:
            self.metadata.update(status='displaying-live-state', last_sample=row)
            self.save_status()
            print(f'LIVE_VIEW: frame={self.frames} step={row["common_step_counter"]} '
                  f'foot_gap={lateral:.4f}m command={command.tolist()} observer_total={self.total_seconds:.3f}s', flush=True)

    def close(self):
        if self.handle:
            close_passive(self.handle, self.viewer_threads)
        self.log.close()
        self.metadata.update(status='observer-finished', finished_at=datetime.now(timezone.utc).isoformat())
        self.save_status()
