"""Check the native rollout interface before admitting any new simulation."""

from copy import deepcopy
from dataclasses import asdict
import inspect
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from common_scene_contract import compare_arrays, model_arrays
from common_walk_startup import make_actor, td
from common_walk_targets import prepare
from command_input_mapping import transform
from evaluate_fixed_policy import fixed_twist
from getup_ppo_math import finite_gae
from record_io import write
from walk_backward_native_rollout import NativeBackwardRollout

CONFIG = ROOT / "configs/experiments/common_walk_backward_online_focus_draft.json"
PARENT = ROOT / "configs/experiments/common_walk_backward_stall_capture_capacity.json"
CAPTURE = ROOT / "records/reports/common-walk-backward-stall-capture/capacity"
OUT = ROOT / "records/reports/common-walk-backward-online-focus"
SCENE = CAPTURE / "cpu-scene-fields.npz"
PROOF = OUT / "native-cpu.json"


def inputs():
    prior = read(OUT / "readback.json")
    capture = read(CAPTURE / "analysis.json")
    assert prior["status"] == "passed-real-trace-weight-readback" and prior["plan_sha256"] == sha(CONFIG)
    assert capture["protocol_sha256"] == sha(PARENT) and not capture["capture_admitted"]
    assert prior["source_receipts"]["capture_result"][1] == sha(CAPTURE / "analysis.json")
    assert prior["weight_source_sha256"] == sha(ROOT / "scripts/training/walk_backward_online_focus.py")
    plan = read(PARENT)
    assert plan["reference_sha256"][SCENE.relative_to(ROOT).as_posix()] == sha(SCENE)
    return plan, read(CONFIG)


def configuration(parent, count, seed):
    plan = deepcopy(parent["environment"])
    plan.update(seed=seed, parallel_worlds=count, episodes_per_command=count, groups={"backward": list(range(count))})
    external = plan["commands"]["backward"]
    assert external == [-0.1, 0.0, 0.0]
    internal = transform(external).tolist()
    np.testing.assert_allclose(internal, [-0.14, 0.0, 0.0], atol=1e-8, rtol=0)
    cfg = prepare(plan)
    cfg.auto_reset = True
    cfg.is_finite_horizon = True
    fixed_twist(cfg.commands["twist"], external)
    assert cfg.decimation == 5 and cfg.episode_length_s == 20.0
    return cfg, plan, internal


def check_inputs(env, plan, policy_command, focus, obs):
    with np.load(SCENE, allow_pickle=False) as fields:
        compare_arrays(model_arrays(env.sim.mj_model), {name: fields[name] for name in fields.files})
    pipe = NativeBackwardRollout(env, plan["joint_names"], plan["model_home"], policy_command, focus)
    actor_obs, critic_obs = pipe.inputs(obs)
    n = env.num_envs
    external = actor_obs.new_tensor(plan["commands"]["backward"]).expand(n, -1)
    torch.testing.assert_close(env.command_manager.get_command("twist"), external, atol=0, rtol=0)
    torch.testing.assert_close(obs["actor"][:, 48:51], external, atol=0, rtol=0)
    torch.testing.assert_close(actor_obs[:, 48:51], pipe.policy_command.expand(n, -1), atol=0, rtol=0)
    torch.testing.assert_close(actor_obs[:, 34:48], torch.zeros_like(actor_obs[:, 34:48]), atol=0, rtol=0)
    torch.testing.assert_close(critic_obs[:, :-1], obs["critic"], atol=0, rtol=0)
    torch.testing.assert_close(critic_obs[:, -1], torch.ones_like(critic_obs[:, -1]), atol=0, rtol=0)
    assert torch.isfinite(actor_obs).all() and torch.isfinite(critic_obs).all()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ROOT / plan["model"]
    assert sha(model) == plan["model_sha256"]
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    expected = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in actor_obs[:32].cpu().numpy()])
    actor = make_actor(plan, env.device)
    with torch.no_grad():
        result = actor(td(actor_obs))
    error = float(np.max(np.abs(result[:32].cpu().numpy() - expected)))
    np.testing.assert_allclose(result[:32].cpu().numpy(), expected, atol=2e-5, rtol=2e-5)
    return pipe, actor, result, {"actor_width": actor_obs.shape[1], "critic_width": critic_obs.shape[1],
                                  "maximum_official_action_error": error, "checked_rows": len(expected)}


def check_reward(env):
    # mjlab stores each weighted term before dt scaling in _step_reward.
    reward = env.reward_manager.compute(dt=env.step_dt)
    scale = env.step_dt if env.reward_manager._scale_by_dt else 1.0
    restored = env.reward_manager._step_reward.sum(dim=1) * scale
    torch.testing.assert_close(reward, restored, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(reward).all() and "track_linear_velocity" in env.reward_manager.active_terms
    return {"terms": list(env.reward_manager.active_terms), "scale": scale,
            "maximum_decomposition_error": (reward - restored).abs().max().item()}


def check_value_targets():
    rewards = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    values = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    dones = torch.tensor([[False, False], [True, False], [False, True]])
    final = torch.tensor([7., 101.])
    _, returns = finite_gae(rewards, values, dones, final, gamma=0.99, lam=0.95)
    # Done samples target their own native rewards. Only a live last sample
    # bootstraps, even if a reset observation exists for a finished row.
    torch.testing.assert_close(returns[dones], rewards[dones])
    torch.testing.assert_close(returns[-1, 0], rewards[-1, 0] + 0.99 * final[0])
    expected_first = rewards[0, 0] + 0.99 * (0.05 * values[1, 0] + 0.95 * rewards[1, 0])
    torch.testing.assert_close(returns[0, 0], expected_first)
    return {"done_returns_equal_reward": True, "live_window_bootstraps": True,
            "post_reset_value_excluded": True}


def main():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PROOF.exists() and not torch.cuda.is_initialized()
    parent, focus = inputs()
    cfg, plan, internal = configuration(parent, count=16, seed=1132)
    baseline = prepare(plan)
    expected = asdict(baseline)
    expected.update(auto_reset=True, is_finite_horizon=True)
    fixed_twist(baseline.commands["twist"], plan["commands"]["backward"])
    expected["commands"] = asdict(baseline)["commands"]
    assert asdict(cfg) == expected
    assert cfg.rewards["track_linear_velocity"].params["command_name"] == "twist"
    ordered = ("self.episode_length_buf += 1", "self.termination_manager.compute()",
               "self.reward_manager.compute(", "self._reset_idx(reset_env_ids)", "self.sim.forward()",
               "self.obs_buf = self.observation_manager.compute(")
    source = inspect.getsource(ManagerBasedRlEnv.step)
    positions = [source.index(item) for item in ordered]
    assert positions == sorted(set(positions))
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    try:
        obs, _ = env.reset(seed=plan["seed"])
        pipe, _, actions, interface = check_inputs(env, plan, internal, focus, obs)
        env.action_manager.process_action(actions)
        actor_obs, _ = pipe.inputs(obs)
        torch.testing.assert_close(actor_obs[:, 34:48], actions, atol=0, rtol=0)
        reset_ids = torch.tensor([0, 3], dtype=torch.int64)
        kept = torch.ones(env.num_envs, dtype=torch.bool)
        kept[reset_ids] = False
        env.episode_length_buf.fill_(50)
        obs, _ = env.reset(env_ids=reset_ids)
        actor_obs, critic_obs = pipe.inputs(obs)
        torch.testing.assert_close(actor_obs[kept, 34:48], actions[kept], atol=0, rtol=0)
        assert not actor_obs[reset_ids, 34:48].any() and not env.episode_length_buf[reset_ids].any()
        torch.testing.assert_close(critic_obs[reset_ids, -1], torch.ones(len(reset_ids)))
        torch.testing.assert_close(critic_obs[kept, -1], torch.full((int(kept.sum()),), 0.95))
        assert env.common_step_counter == env._sim_step_counter == 0
        reward = check_reward(env)
        value = check_value_targets()
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    files = (Path(__file__), PARENT, CONFIG, SCENE, OUT / "readback.json",
             ROOT / "scripts/training/walk_backward_native_rollout.py",
             ROOT / "scripts/training/getup_ppo_math.py",
             ROOT / "scripts/training/walk_backward_online_focus.py")
    library = Path(inspect.getsourcefile(ManagerBasedRlEnv))
    result = dict(status="passed-native-backward-cpu-zero-step", physical_substeps=0, learning_updates=0,
                  heldout_used=False, hardware_io=False, worlds=16, seed=plan["seed"],
                  external_command=plan["commands"]["backward"], policy_command=internal,
                  finite_episode_seconds=20, decimation=5, partial_reset_ids=reset_ids.tolist(),
                  native_step_order=list(ordered), interface=interface, reward=reward, value=value,
                  library_source_sha256={str(library): sha(library)},
                  source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in files})
    write(PROOF, result)
    print("NATIVE_BACKWARD_CPU_ZERO", interface, reward, value, flush=True)


if __name__ == "__main__":
    main()
