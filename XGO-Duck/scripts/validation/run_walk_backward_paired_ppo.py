"""Train a bounded native PPO pair from the verified shared value checkpoint."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import hashlib
from pathlib import Path
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from action_video import ActionVideo
from check_walk_backward_native_rollout import check_inputs, configuration, inputs
from getup_anchor_runtime import RecordingSnapshot
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from run_walk_backward_ppo_headroom import model_digest
from walk_backward_export import export_walking_actor
from walk_backward_ppo import BackwardRolloutBatch, action_and_log_prob, actor_mean, losses
from walk_backward_value import SharedBackwardValue
import training_live_page as live


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_paired_ppo.json"
VALUE = PLAN.with_stem("common_walk_backward_shared_value")
EXPORT = ROOT / "records/reports/common-walk-backward-online-focus/export-interface-verification.json"
EVAL_REFERENCE = ROOT / "configs/experiments/common_walk_actuator_yaw_feedback_heldout.json"
ARMS = ("control", "candidate")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def publish(folder, page, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", page):
        live.publish(folder, state)


def freeze():
    check_project()
    assert not PLAN.exists()
    value = read(VALUE)
    prior = ROOT / value["record"]
    receipt = read(prior / "verification.json")
    assert receipt["status"] == "passed-independent-native-backward-value-readback" and receipt["admitted"]
    assert receipt["analysis_sha256"] == sha(prior / "analysis.json")
    assert receipt["checkpoint_sha256"] == sha(prior / "shared-value.pt")
    assert read(EXPORT)["status"] == "passed-walking-export-readback"
    parent, focus = inputs()
    model = ROOT / parent["environment"]["model"]
    actor_sha = parent["environment"]["model_sha256"]
    assert actor_sha == sha(model) == value["official_actor_sha256"] == read(EXPORT)["official_sha256"]
    bank = (ROOT / focus["retention"][0]).with_name("anchors.npz")
    assert sha(bank) == read(EXPORT)["anchor_sha256"]
    checksums = dict(value["source_sha256"])
    files = (ENTRY, ENTRY.with_name("verify_walk_backward_paired_ppo.py"), VALUE,
             prior / "verification.json", prior / "analysis.json", prior / "shared-value.pt",
             model, bank, EXPORT, EVAL_REFERENCE, ROOT / "scripts/training/walk_backward_export.py")
    checksums.update({rel(path): sha(path) for path in files})
    seed = 1138
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_root = ROOT / "records/experiments"
    assert not any(f"_s{seed}_" in path.name for path in run_root.iterdir())
    roles = {}
    for arm in ARMS:
        folder = run_root / f"duck_walk_backward-focus-ppo_{arm}_s{seed}_{stamp}"
        page = ROOT / f"records/walk-backward-paired-ppo-{arm}-live.html"
        assert not folder.exists() and not page.exists()
        roles[arm] = dict(record=rel(folder), page=rel(page), focused=arm == "candidate")
    n, horizon, updates = value["worlds"], 24, 42
    spec = dict(status="frozen-native-backward-ppo-pair", seed=seed, worlds=n, horizon=horizon,
                updates_per_arm=updates, controls_per_arm=horizon*updates, transitions_per_arm=n*horizon*updates,
                physical_substeps_per_arm=5*horizon*updates, arms=roles, shared_value=rel(prior / "shared-value.pt"),
                official_actor=rel(model), official_actor_sha256=actor_sha, anchor_bank=rel(bank),
                epochs=5, minibatches=4, initial_std=value["initial_std"], gamma=value["gamma"], lam=0.95,
                clip=0.2, anchor_coefficient=0.02, actor_lr=0.00002, critic_lr=0.0001,
                target_kl=0.02, severe_kl_stop=0.1, gradient_norm_max=1.0,
                checkpoints=[14, 28, 42], memory_cap_fraction=value["memory_cap_fraction"], deadline_seconds=480,
                source_sha256=checksums, library_source_sha256=value["library_source_sha256"],
                evaluation_reference=rel(EVAL_REFERENCE), heldout_used=False, hardware_io=False,
                extension=False, exact_mid_episode_resume=False)
    for path, digest in checksums.items():
        assert sha(ROOT / path) == digest, path
    write(PLAN, spec)
    for arm, info in roles.items():
        folder, page = ROOT / info["record"], ROOT / info["page"]
        folder.mkdir()
        write(folder / "protocol.json", spec)
        (folder / "README.md").write_text(
            f"# 后退配对 PPO · {'候选' if info['focused'] else '对照'}\n\n"
            f"[冻结方案](../../../{rel(PLAN)}) · [运行观察](../../{page.name})\n",
            encoding="utf-8",
        )
        publish(folder, page, dict(status="starting", run_id=folder.name, purpose="等待同起点 PPO",
                                   controls=0, maximum_controls=spec["controls_per_arm"],
                                   transitions=0, learning_updates=0, updates=[]))
    print("BACKWARD_PAIR_FROZEN", sha(PLAN), roles, flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-backward-ppo-pair"
    assert not any(spec[key] for key in ("extension", "heldout_used", "hardware_io", "exact_mid_episode_resume"))
    for path, digest in spec["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in spec["library_source_sha256"].items():
        assert sha(Path(path)) == digest, path
    return spec


def batch_update(actor, critic, storage, tail, anchors, spec, rng, optimizers, focused, watch):
    batch = storage.prepare(tail, spec["gamma"], spec["lam"])
    assert all(value.isfinite().all() for value in batch.values())
    actor_opt, critic_opt = optimizers
    actor_open = True
    steps = 0
    rows = []
    count = len(batch["done"])
    assert count % spec["minibatches"] == 0
    for _ in range(spec["epochs"]):
        ids = torch.randperm(count, generator=rng, device=tail.device)
        for block in ids.chunk(spec["minibatches"]):
            watch.check()
            policy, value, kl, anchor = losses(actor, critic, batch, block, anchors, spec, focused)
            scalars = [float(t.detach()) for t in (policy, value, kl, anchor)]
            assert np.isfinite(scalars).all() and scalars[2] < spec["severe_kl_stop"], "PPO numeric or KL gate"
            actor_open &= scalars[2] <= spec["target_kl"]
            critic_opt.zero_grad(set_to_none=True)
            value.backward()
            assert torch.nn.utils.clip_grad_norm_(critic.parameters(), spec["gradient_norm_max"]).isfinite()
            critic_opt.step()
            if actor_open:
                actor_opt.zero_grad(set_to_none=True)
                policy.backward()
                assert torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), spec["gradient_norm_max"]).isfinite()
                actor_opt.step()
                steps += 1
            rows.append(scalars)
    return dict(actor_steps=steps, critic_steps=len(rows), max_kl=max(row[2] for row in rows),
                policy_loss=float(np.mean([row[0] for row in rows])),
                value_loss=float(np.mean([row[1] for row in rows])),
                anchor_mse=float(np.mean([row[3] for row in rows])),
                stalled_samples=int((batch["weight"] > 1).sum()), native_episode_ends=int(batch["done"].sum()),
                mean_reward=float(batch["reward"].mean()))


def initial_signature(actor, critic, env, shared_value):
    digest = hashlib.sha256()
    for tensor in (env.sim.data.qpos, env.sim.data.qvel, env.episode_length_buf, env.action_manager.action):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return dict(physical_sha256=digest.hexdigest(), actor_sha256=model_digest(actor),
                critic_sha256=model_digest(critic), shared_value_sha256=sha(shared_value), physical_substeps=0)


def save_checkpoint(path, actor, critic, optimizers, rngs, state, spec, arm):
    actor_opt, critic_opt = optimizers
    candidate = path.with_suffix(".partial")
    torch.save(dict(actor_state_dict=actor.state_dict(), critic_state_dict=critic.state_dict(),
                    actor_optimizer_state_dict=actor_opt.state_dict(), critic_optimizer_state_dict=critic_opt.state_dict(),
                    action_rng=rngs[0].get_state(), batch_rng=rngs[1].get_state(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    arm=arm, completed_updates=state["learning_updates"], controls=state["controls"],
                    actor_steps=state["actor_steps"], critic_steps=state["critic_steps"], protocol_sha256=sha(PLAN),
                    shared_value_sha256=spec["source_sha256"][spec["shared_value"]], recovery_boundary="native reset"),
               candidate)
    candidate.replace(path)


def run(spec, arm):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    check_project()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    info = spec["arms"][arm]
    folder, page = ROOT / info["record"], ROOT / info["page"]
    state = read(folder / "state.json")
    assert state["status"] == "starting"
    watch = MemoryWatch(spec["memory_cap_fraction"])
    parent, focus = inputs()
    n, horizon = spec["worlds"], spec["horizon"]
    cfg, plan, internal = configuration(parent, n, spec["seed"])
    shared = ROOT / spec["shared_value"]
    deadline = monotonic()+spec["deadline_seconds"]
    env = camera = video = None
    history = []
    state.update(actor_steps=0, critic_steps=0)
    try:
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            obs, _ = env.reset(seed=spec["seed"])
            pipe, actor, _, startup = check_inputs(env, plan, internal, focus, obs)
            assert env.common_step_counter == env._sim_step_counter == 0
            critic = SharedBackwardValue(read(VALUE)["seed"], env.device)
            saved = torch.load(shared, map_location="cpu", weights_only=True)
            assert saved["admitted"] and not saved["actor_updates"]
            critic.load_state_dict(saved["critic_state_dict"], strict=True)
            signature = initial_signature(actor, critic, env, shared)
            assert signature["actor_sha256"] == saved["actor_sha256"]
            if arm == "candidate":
                other = ROOT / spec["arms"]["control"]["record"]
                assert read(other / "state.json")["status"] == "completed"
                assert signature == read(other / "startup.json")["signature"], "Paired start differs"
            write(folder / "startup.json", dict(**startup, signature=signature))
            frozen = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()
                      if not name.startswith("mlp.")}
            for parameter in actor.mlp.parameters():
                parameter.requires_grad_(True)
            actor_opt = torch.optim.Adam(actor.mlp.parameters(), lr=spec["actor_lr"])
            critic_opt = torch.optim.Adam(critic.parameters(), lr=spec["critic_lr"])
            critic_opt.load_state_dict(saved["critic_optimizer_state_dict"])
            for group in critic_opt.param_groups:
                group["lr"] = spec["critic_lr"]
            optimizers = (actor_opt, critic_opt)
            rngs = tuple(torch.Generator(device=env.device).manual_seed(spec["seed"]+offset)
                         for offset in (10000, 20000))
            with np.load(ROOT / spec["anchor_bank"], allow_pickle=False) as bank:
                anchors = tuple(torch.as_tensor(bank[key], device=env.device) for key in ("observations", "teacher_action"))
            storage = BackwardRolloutBatch(horizon, n, env.device)
            watch.sample()
            camera = RecordingSnapshot(env, folder)
            camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.1, 90, -20
            video = ActionVideo(camera, "walk", folder)
            video.sample()
            state.update(status="running", frame=video.image["file"])
            publish(folder, page, state)
            probe_start = pipe.inputs(obs)[0][:128].cpu().numpy()
            for update in range(1, spec["updates_per_arm"]+1):
                tick = monotonic()
                for index in range(horizon):
                    watch.check()
                    assert monotonic() < deadline, "PPO deadline"
                    with torch.no_grad():
                        actor_obs, critic_obs = pipe.inputs(obs)
                        mean = actor_mean(actor, actor_obs)
                        action, log_prob = action_and_log_prob(mean, spec["initial_std"], rngs[0])
                        value = critic(critic_obs).squeeze(-1)
                        step = pipe.advance(action, focused=True)
                        obs = step.observations
                        storage.put(index, actor_obs, critic_obs, mean, action, log_prob, value, step)
                    if video.writer is not None or (
                        monotonic()-video.last_started >= video.cfg["visual_interval_seconds"]
                        and int(env.episode_length_buf[0]) == 0
                    ):
                        video.sample()
                    if video.clips and video.clips[-1]["complete_clip"]:
                        state["latest_clip"] = video.clips[-1]["file"]
                torch.cuda.synchronize()
                elapsed = monotonic()-tick
                with torch.no_grad():
                    tail = critic(pipe.inputs(obs)[1]).squeeze(-1)
                row = batch_update(actor, critic, storage, tail, anchors, spec, rngs[1], optimizers, info["focused"], watch)
                row.update(update=update, sampling_seconds=elapsed, optimizer_seconds=monotonic()-tick-elapsed)
                history.append(row)
                state.update(controls=env.common_step_counter, transitions=n*env.common_step_counter, learning_updates=update)
                state["actor_steps"] += row["actor_steps"]
                state["critic_steps"] += row["critic_steps"]
                state["updates"].append(dict(transitions_per_second=n*horizon/elapsed))
                state["purpose"] = f"后退 PPO · {arm} · 回报 {row['mean_reward']:.3f} · KL {row['max_kl']:.4f}"
                write(folder / "ppo-metrics.json", history)
                if update in spec["checkpoints"]:
                    save_checkpoint(folder / f"checkpoint-{update}.pt", actor, critic, optimizers, rngs, state, spec, arm)
                publish(folder, page, state)
                print("BACKWARD_PAIRED_PPO", arm, row, flush=True)
            watch.sample()
            watch.check()
            assert env.common_step_counter == spec["controls_per_arm"] and env._sim_step_counter == spec["physical_substeps_per_arm"]
            assert all(torch.equal(actor.state_dict()[name], tensor) for name, tensor in frozen.items())
            assert state["actor_steps"] > 0 and model_digest(actor) != signature["actor_sha256"]
            probes = np.concatenate((anchors[0].cpu().numpy(), probe_start, pipe.inputs(obs)[0][:128].cpu().numpy()))
            probe_file = folder / "onnx-probes.npz"
            np.savez_compressed(probe_file, observations=probes)
            export = export_walking_actor(actor, ROOT / spec["official_actor"], spec["official_actor_sha256"],
                                          folder / "policy.onnx", probes)
            final = folder / f"checkpoint-{spec['updates_per_arm']}.pt"
            result = dict(status="completed-native-backward-paired-ppo-arm", arm=arm, protocol_sha256=sha(PLAN),
                          signature=signature, actor_sha256=model_digest(actor), critic_sha256=model_digest(critic),
                          checkpoint_sha256=sha(final), export=export, export_probe_sha256=sha(probe_file),
                          memory=watch.summary(),
                          controls=state["controls"], transitions=state["transitions"], learning_updates=state["learning_updates"],
                          actor_steps=state["actor_steps"], critic_steps=state["critic_steps"],
                          stalled_samples=sum(row["stalled_samples"] for row in history))
            write(folder / "analysis.json", result)
            state.update(status="completed")
            print("BACKWARD_PAIRED_ARM_COMPLETE", result, flush=True)
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        if video is not None:
            video.close()
        if camera is not None:
            camera.close()
        if env is not None:
            state.update(controls=env.common_step_counter, transitions=n*env.common_step_counter)
            env.close()
        write(folder / "gpu-telemetry.json", watch.rows)
        publish(folder, page, state)


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(checked(), args.arm)


if __name__ == "__main__":
    main()
