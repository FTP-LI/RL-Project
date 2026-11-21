"""Check native right-turn inputs and startup reward without physics steps."""

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_parent import DRAFT, OUT
from common_scene_contract import compare_arrays, model_arrays
from common_walk_startup import make_actor
from common_walk_targets import prepare
from evaluate_fixed_policy import fixed_twist
from gyro_turn_controller import Controller
from record_io import write
from walk_backward_ppo import actor_mean
from walk_right_startup import RightTurnRollout, startup_yaw_deficit


TERM = "right_startup_yaw_deficit"
CPU = OUT / "native-cpu.json"


def source_plan():
    draft = read(DRAFT)
    proof = read(OUT / "parent-readback.json")
    assert proof["status"] == "passed-right-turn-parent-and-retention"
    assert sha(DRAFT) == proof["draft_sha256"]
    assert sha(OUT / "right-turn-retention.npz") == proof["bank_sha256"]
    assert sha(ROOT / draft["actor_checkpoint"]) == proof["checkpoint_sha256"]
    assert sha(ROOT / draft["actor_reference"]) == proof["onnx_sha256"]
    plan_path = ROOT / draft["fixed_routing_reference"]
    result_path, expected = draft["evidence"]["matched_dynamic"]
    assert sha(ROOT / result_path) == expected == proof["archive_result_sha256"]
    assert sha(plan_path) == read(ROOT / result_path)["protocol_sha256"]
    return draft, read(plan_path)


def configuration(draft, parent, worlds, seed, weight):
    from mjlab.managers import RewardTermCfg

    p = deepcopy(parent["cases"]["1141"]["environment"])
    p.update(seed=seed, parallel_worlds=worlds, episodes_per_command=worlds,
             groups={"turn-right": list(range(worlds))})
    external = p["commands"]["turn-right"]
    settings = draft["reward_proposal"]
    assert external == [0., 0., settings["target_external_yaw_rad_s"]]
    cfg = prepare(p)
    cfg.auto_reset = True
    cfg.is_finite_horizon = True
    fixed_twist(cfg.commands["twist"], external)
    assert TERM not in cfg.rewards and cfg.episode_length_s == 20. and cfg.decimation == 5
    cfg.rewards[TERM] = RewardTermCfg(func=startup_yaw_deficit, weight=weight,
        params={"startup_controls": settings["last_startup_control_one_based"]})
    return cfg, p


def load_actor(draft, p, device):
    model = make_actor(p, device)
    checkpoint = torch.load(ROOT / draft["actor_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    return model


def first_inputs(env, draft, parent, p, observations, actor):
    record = ROOT / parent["cases"]["1141"]["record"]
    scene = record / "scene-static-fields.npz"
    with np.load(scene, allow_pickle=False) as reference:
        compare_arrays(model_arrays(env.sim.mj_model), {key: reference[key] for key in reference.files})
    pipe = RightTurnRollout(env, p["joint_names"], p["model_home"])
    command = env.command_manager.get_command("twist")
    expected = command.new_tensor(p["commands"]["turn-right"]).expand_as(command)
    torch.testing.assert_close(command, expected, atol=0, rtol=0)
    torch.testing.assert_close(observations["actor"][:, 48:51], expected, atol=0, rtol=0)
    actor_obs, critic = pipe.inputs(observations)
    assert actor_obs.shape == (env.num_envs, 61) and critic.shape == (env.num_envs, 77)
    assert not actor_obs[:, 34:48].any() and (critic[:, -1] == 1.).all()
    first = Controller(["gyro-feedback"] * env.num_envs).map(expected.cpu().numpy(), actor_obs.cpu().numpy())
    np.testing.assert_allclose(actor_obs[:, 48:51].cpu().numpy(), first, atol=1.2e-7, rtol=0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / draft["actor_reference"]), options, providers=["CPUExecutionProvider"])
    probes = actor_obs[:32].cpu().numpy()
    exported = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in probes])
    with torch.no_grad():
        recovered = actor_mean(actor, actor_obs)[:len(probes)].cpu().numpy()
    np.testing.assert_allclose(recovered, exported, atol=2e-6, rtol=0)
    assert env.common_step_counter == env._sim_step_counter == 0
    return pipe, float(np.abs(recovered-exported).max()), sha(scene)


def replay_mapping(env, pipe, draft, parent, observations):
    score = read(ROOT / draft["evidence"]["matched_dynamic"][0])
    maximum = 0.
    rows = 0
    for seed, entry in score["records"].items():
        record = ROOT / entry["path"]
        receipt = record / "verification.json"
        assert sha(receipt) == entry["verification_sha256"]
        proof = read(receipt)
        ids = parent["cases"][seed]["environment"]["groups"]["turn-right"]
        command = env.command_manager.get_command("twist").numpy().copy()
        controller = Controller(["gyro-feedback"] * len(ids))
        for name, expected in (("shared-prefix", proof["shared_prefix_sha256"]),
                               ("reference-a", proof["roles"]["reference-a"]["trace_sha256"])):
            trace = record / f"trace-{name}.npz"
            assert sha(trace) == expected
            with np.load(trace, allow_pickle=False) as saved:
                for archived, active in zip(saved["observations"][:, ids], saved["active"][:, ids], strict=True):
                    fixture = dict(actor=torch.from_numpy(archived), critic=observations["critic"])
                    env.episode_length_buf.fill_(rows % 1000)
                    reference = controller.map(command, archived)
                    mapped, _ = pipe.inputs(fixture)
                    actual = mapped[:, 48:51].numpy()[active]
                    np.testing.assert_allclose(reference[active], archived[active, 48:51], atol=2e-7, rtol=0)
                    np.testing.assert_allclose(actual, reference[active], atol=2e-7, rtol=0)
                    maximum = max(maximum, float(np.abs(actual-reference[active]).max(initial=0.)))
                    rows += 1
    assert rows == 2000
    return dict(replayed_batched_controls=rows, maximum_command_error_rad_s=maximum)


def check_reset_and_reward(env, pipe, obs, actor):
    from check_walk_backward_native_rollout import check_value_targets

    with torch.no_grad():
        actor_obs, _ = pipe.inputs(obs)
        actions = actor_mean(actor, actor_obs)
    env.action_manager.process_action(actions)
    reset_ids = torch.tensor([0, 3], dtype=torch.int64)
    kept = torch.ones(env.num_envs, dtype=torch.bool)
    kept[reset_ids] = False
    env.episode_length_buf.fill_(50)
    saved_ema = pipe.gyro_ema.clone()
    next_obs, _ = env.reset(env_ids=reset_ids)
    next_actor, next_critic = pipe.inputs(next_obs)
    torch.testing.assert_close(next_actor[kept, 34:48], actions[kept], atol=0, rtol=0)
    assert not next_actor[reset_ids, 34:48].any() and (next_critic[reset_ids, -1] == 1.).all()
    gyro = next_obs["actor"][:, 2]
    torch.testing.assert_close(pipe.gyro_ema[reset_ids], gyro[reset_ids], atol=0, rtol=0)
    torch.testing.assert_close(pipe.gyro_ema[kept], (saved_ema + pipe.alpha*(gyro-saved_ema))[kept], atol=0, rtol=0)
    torch.testing.assert_close(next_critic[kept, -1], torch.full_like(next_critic[kept, -1], .95))
    env.episode_length_buf[:4] = torch.tensor([0, 1, 200, 201])
    reward = env.reward_manager.compute(env.step_dt)
    term = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
    yaw = env.scene["robot"].data.root_link_ang_vel_b[:4, 2].numpy()
    expected = np.where([False, True, True, False], -np.clip((.4+yaw)/.4, 0, 1), 0.)
    np.testing.assert_allclose(term[:4].numpy(), expected, atol=2e-7, rtol=0)
    torch.testing.assert_close(reward, env.reward_manager._step_reward.sum(dim=1)*env.step_dt, atol=2e-7, rtol=2e-6)
    return dict(partial_reset_ids=reset_ids.tolist(), reward_ages=[0, 1, 200, 201],
                boundary_unscaled_reward=term[:4].tolist(), value=check_value_targets())


def reward_sign_fixture():
    command = torch.tensor([[0., 0., -.4]] * 4 + [[0., 0., .4], [.15, 0., -.4]])
    yaw = torch.tensor([.2, -.2, -.4, -.8, 0., 0.])
    robot = SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.stack((yaw*0, yaw*0, yaw), dim=1)))
    env = SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.ones(6),
                          command_manager=SimpleNamespace(get_command=lambda name: command))
    expected = torch.tensor([-1., -.5, 0., 0., 0., 0.])
    torch.testing.assert_close(startup_yaw_deficit(env, 200), expected, atol=0, rtol=0)
    return expected.tolist()


def main():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardManager

    assert not CPU.exists()
    torch.set_num_threads(4)
    draft, parent = source_plan()
    cfg, p = configuration(draft, parent, 32, 1146, weight=1.)
    order = ("self.episode_length_buf += 1", "self.reward_manager.compute(", "record_pre_reset(",
             "self._reset_idx(reset_env_ids)", "self.obs_buf = self.observation_manager.compute(")
    native = inspect.getsource(ManagerBasedRlEnv.step)
    indices = [native.index(name) for name in order]
    assert indices == sorted(set(indices))
    env = ManagerBasedRlEnv(cfg, device="cpu", render_mode=None)
    try:
        obs, _ = env.reset(seed=p["seed"])
        actor = load_actor(draft, p, "cpu")
        pipe, error, scene_hash = first_inputs(env, draft, parent, p, obs, actor)
        mapping = replay_mapping(env, pipe, draft, parent, obs)
        obs, _ = env.reset(seed=p["seed"])
        checks = check_reset_and_reward(env, pipe, obs, actor)
        assert env.common_step_counter == env._sim_step_counter == 0
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    libraries = [Path(inspect.getsourcefile(cls)) for cls in (ManagerBasedRlEnv, RewardManager)]
    sources = (Path(__file__), DRAFT, OUT / "parent-readback.json", ROOT / draft["fixed_routing_reference"],
               ROOT / "scripts/training/walk_right_startup.py", ROOT / "scripts/validation/gyro_turn_controller.py")
    result = dict(status="passed-native-right-turn-cpu-zero-step", physical_substeps=0, learning_updates=0,
        actor_onnx_max_error=error, common_scene_sha256=scene_hash, source_order=list(order),
        native_derived_yaw_lag_substeps=1, sign_fixture=reward_sign_fixture(), **mapping, **checks,
        source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in sources},
        library_source_sha256={str(path): sha(path) for path in libraries})
    write(CPU, result)
    print("RIGHT_STARTUP_NATIVE_CPU", result, flush=True)


if __name__ == "__main__":
    main()
