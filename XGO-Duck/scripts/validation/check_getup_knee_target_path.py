"""Check one proposed knee pulse through the actual CPU actuator path.

Process the same archived raw action twice at a synthetic episode age, with
the same pinned scene and restored controller and actuator state. Calling
the target and motor writers does not integrate, evaluate motion, or learn.
"""
import numpy as np
import torch

import getup_shared_release as shared
import getup_knee_support as support
from common_scene_contract import compare_arrays, model_arrays
from getup_release_scene_state import SceneReleaseSnapshot
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT / 'configs/experiments/getup_knee_support_draft.json'
OUT = ROOT / 'records/reports/getup-knee-support'
CONTROL_HISTORY = ('_action', '_prev_action', '_prev_prev_action')


def main():
    from mjlab.envs import ManagerBasedRlEnv

    torch.set_num_threads(4)
    draft = read(PLAN)
    proof = read(OUT / 'preflight.json')
    assert proof['admitted_for_next_preflight'] and proof['protocol_sha256'] == sha(PLAN)
    assert proof['component_sha256'] == sha(ROOT / 'scripts/validation/getup_knee_support.py')
    settings = draft['mechanism']
    parent = ROOT / draft['matched_state_basis']
    spec = read(parent)
    shared.assert_sources(spec)
    p = spec['environment']
    source = proof['studies']['1050']['roles']['original']
    source_path = ROOT / source['path']
    assert sha(source_path) == source['sha256']
    source_world = next(i for i, step in enumerate(source['trigger_control_by_environment']) if step > 0)
    age = source['trigger_control_by_environment'][source_world]
    with np.load(source_path) as trace:
        saved_raw = trace['raw_policy_actions'][age, source_world]
    cfg, display, meta = shared.prepare(p)
    assert not display['native_live_window']
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cpu', render_mode=None)
    try:
        env.reset(seed=p['seed'])

        def no_physics(*args, **kwargs):
            raise AssertionError('A target-path preflight must not integrate')

        env.sim.step = no_physics
        env.step = no_physics
        with np.load(shared.PARENT / 'cpu-scene-fields.npz') as saved:
            compare_arrays(model_arrays(env.sim.mj_model), {name: saved[name] for name in saved.files})
        manager = env.action_manager
        term = manager.get_term('joint_pos')
        robot = env.scene['robot']
        names = meta['joint_names'].split(',')
        robot_ids, model_names = robot.find_joints(names, preserve_order=True)
        assert list(model_names) == names
        knee_ids = [names.index(name) for name in settings['joint_order']]
        limits = torch.tensor(p['joint_limits'])[knee_ids]
        pulse = support.KneeSupportPulse(p['num_envs'], env.device, settings)
        pulse.trigger[:2] = age
        env.episode_length_buf.fill_(age)
        raw = torch.as_tensor(saved_raw).expand(p['num_envs'], -1).clone()
        left = knee_ids[0]
        raw[1, left] = limits[0, 1] + .01 - p['model_home'][left]
        baseline_raw = raw.clone()
        memory = shared.make_memory(p, meta, env.device)
        snapshot = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
        initial = snapshot.verify()
        physical = [value.clone() for value in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.time)]

        def path(enabled):
            manager.process_action(raw)
            request = term.requested_target.clone()
            support.apply_after_processing(term, pulse, age + 1, knee_ids, limits, enabled)
            torch.testing.assert_close(raw, baseline_raw, atol=0, rtol=0)
            torch.testing.assert_close(manager._action, baseline_raw, atol=0, rtol=0)
            controls = []
            for _ in range(env.cfg.decimation):
                manager.apply_action()
                torch.testing.assert_close(robot.data.joint_pos_target[:, robot_ids], term._processed_actions, atol=0, rtol=0)
                env.scene.write_data_to_sim()
                controls.append(env.sim.data.ctrl.clone())
            for before, after in zip(physical, (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.time), strict=True):
                torch.testing.assert_close(before, after.clone(), atol=0, rtol=0)
            torch.testing.assert_close(term.previous_target, term._processed_actions, atol=0, rtol=0)
            return request, term._processed_actions.clone(), [getattr(manager, key).clone() for key in CONTROL_HISTORY], torch.stack(controls)

        with torch.no_grad():
            reference = path(False)
            assert snapshot.restore() == initial
            candidate = path(True)
            assert snapshot.restore() == initial
        assert initial['counters'] == [0, 0] and not torch.cuda.is_initialized()
        torch.testing.assert_close(reference[0], candidate[0], atol=0, rtol=0)
        torch.testing.assert_close(reference[2], candidate[2], atol=0, rtol=0)
        expected = torch.zeros_like(reference[1])
        expected[0, knee_ids] = torch.tensor(settings['signed_offset']) * pulse.step_offset
        torch.testing.assert_close(candidate[1] - reference[1], expected, atol=1e-7, rtol=0)
        motor_difference = (candidate[3] - reference[3]).abs()
        assert torch.isfinite(motor_difference).all()
        assert not motor_difference[:, 1:].any() and motor_difference[:, 0].any()
        writes = [i + 1 for i in range(env.cfg.decimation) if motor_difference[i, 0].any()]
        result = dict(status='passed-cpu-zero-step', draft_sha256=sha(PLAN), source_preflight_sha256=sha(OUT / 'preflight.json'),
            parent_protocol_sha256=sha(parent), archived_raw_source=source['path'], archived_raw_sha256=source['sha256'],
            archived_world=source_world, synthetic_age_controls=age, native_snapshot=initial,
            raw_policy_and_manager_history_exact=True, native_dispatched_knee_delta_rad=expected[0, knee_ids].tolist(),
            other_world_and_joint_targets_exact=True, preexisting_knee_excess_preserved=True,
            motor_writes_with_difference=writes, max_motor_ctrl_difference=float(motor_difference.max()),
            actual_physics_steps=0, actual_env_steps=0, learning_updates=0, hardware_io=False,
            scope='CPU target and actuator buffer writes with no integration; GPU and behavior remain untested.')
        write(OUT / 'target-path.json', result)
        print('KNEE_TARGET_PATH_OK', initial['fields'], writes, result['max_motor_ctrl_difference'])
    finally:
        env.close()


if __name__ == '__main__':
    main()
