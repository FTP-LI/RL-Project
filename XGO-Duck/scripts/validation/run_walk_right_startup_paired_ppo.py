"""Run equal-sample PPO on the admitted right-turn actor and shared value start."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import monotonic
from typing import NamedTuple
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from action_video import ActionVideo
from check_walk_right_startup_native import OUT, TERM, configuration, first_inputs, load_actor, source_plan
from getup_anchor_runtime import RecordingSnapshot
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
import run_walk_backward_paired_ppo as common_ppo
from run_walk_right_startup_probe import publish, require_observer
from run_walk_right_startup_shared_value import PLAN as VALUE, checked as checked_value
from walk_backward_export import export_walking_actor
from walk_backward_ppo import BackwardRolloutBatch, action_and_log_prob, actor_mean, losses
from walk_backward_value import SharedBackwardValue


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_startup_paired_ppo.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
ARMS = ("control", "candidate")


class Transition(NamedTuple):
    reward: torch.Tensor
    actor_weight: torch.Tensor
    terminated: torch.Tensor
    timed_out: torch.Tensor


def relative(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    check_project()
    assert not PLAN.exists()
    draft = read(DRAFT)
    assert draft["status"] == "draft-right-turn-paired-ppo"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "actor_updates_authorized",
                                    "critic_updates_authorized", "heldout_used", "hardware_io"))
    value = checked_value()
    verify_path = ROOT / draft["shared_value_verification"]
    proof = read(verify_path)
    assert proof["status"] == "passed-independent-native-right-turn-value-readback" and proof["admitted"]
    assert proof["protocol_sha256"] == sha(VALUE) and value["worlds"] == draft["worlds"]
    record = ROOT / value["record"]
    checkpoint = record / "shared-value.pt"
    assert proof["receipt_sha256"] == sha(record / "verification.json")
    assert proof["checkpoint_sha256"] == sha(checkpoint)
    assert read(ROOT / draft["reward_signal"])["new_reward_coefficient"] == draft["reward_coefficient"]
    _, routing = source_plan()
    official = ROOT / routing["models"]["official"][0]
    assert sha(official) == routing["models"]["official"][1]
    bank, bank_receipt = (OUT / name for name in ("right-turn-retention.npz", "parent-readback.json"))
    assert sha(bank) == read(bank_receipt)["bank_sha256"]
    paths = (ENTRY, DRAFT, VALUE, verify_path, record / "verification.json", checkpoint, bank, bank_receipt,
             ROOT / draft["reward_signal"], official, Path(common_ppo.__file__),
             ROOT / "scripts/training/walk_backward_export.py",
             ENTRY.with_name("observe_walk_right_startup_paired_ppo.py"),
             ENTRY.with_name("verify_walk_right_startup_paired_ppo.py"))
    hashes = {**value["source_sha256"], **{relative(path): sha(path) for path in paths}}
    seed = draft["seed"]
    assert all(not any(f"_s{seed}_" in path.name for path in (ROOT / "records" / part).iterdir())
               for part in ("experiments", "evaluations", "benchmarks"))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    arms = {arm: dict(record=f"records/experiments/duck_walk_right-startup-ppo_{arm}_s{seed}_{stamp}",
                      page=f"records/walk-right-startup-ppo-{arm}-live.html",
                      weight=draft["reward_coefficient"] if arm == "candidate" else 0.) for arm in ARMS}
    controls = draft["horizon"]*draft["updates_per_arm"]
    assert controls == 336 and draft["checkpoints"][-1] == draft["updates_per_arm"]
    spec = dict(draft, **draft["optimizer"])
    spec.update(status="frozen-native-right-turn-paired-ppo", arms=arms,
        source_sha256=hashes, library_source_sha256=value["library_source_sha256"], shared_value=relative(checkpoint),
        shared_value_seed=value["seed"], anchor_bank=relative(bank), official_actor=relative(official),
        official_actor_sha256=sha(official), controls_per_arm=controls, physical_substeps_per_arm=controls*5,
        transitions_per_arm=controls*draft["worlds"], actor_updates_authorized=draft["updates_per_arm"]*2,
        critic_updates_authorized=draft["updates_per_arm"]*2, physical_substeps_authorized=controls*10,
        deadline_seconds_per_arm=240)
    write(PLAN, spec)
    for arm, item in arms.items():
        record, page = (ROOT / item[key] for key in ("record", "page"))
        record.mkdir()
        write(record / "protocol.json", spec)
        (record / "README.md").write_text(
            "# 右转启动 PPO\n\n固定同一策略与价值起点，按原指令只调整右转启动。\n\n"
            f"[冻结方案](../../../{relative(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        publish(record, page, dict(status="starting", run_id=record.name, purpose=f"右转启动 PPO · {arm}",
                controls=0, transitions=0, maximum_controls=controls, learning_updates=0, updates=[]))
    print("RIGHT_STARTUP_PPO_FROZEN", sha(PLAN), arms, flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-right-turn-paired-ppo"
    assert not spec["heldout_used"] and not spec["hardware_io"]
    for path, checksum in spec["source_sha256"].items():
        assert sha(ROOT / path) == checksum, path
    for path, checksum in spec["library_source_sha256"].items():
        assert sha(Path(path)) == checksum, path
    return spec


def gradient_gate(actor, critic, storage, tail, anchors, spec, watch, base_mib, base_reserved):
    batch = storage.prepare(tail, spec["gamma"], spec["lam"])
    assert all(tensor.isfinite().all() for tensor in batch.values())
    done = batch["done"]
    torch.testing.assert_close(batch["returns"][done], batch["reward"][done], atol=3e-7, rtol=2e-6)
    before = [common_ppo.model_digest(model) for model in (actor, critic)]
    # Reserve the actor's two lazy Adam moment tensors. The critic already owns both.
    moments = [torch.zeros_like(parameter) for parameter in actor.mlp.parameters() for _ in range(2)]
    indices = torch.arange(len(done)//spec["minibatches"], device=tail.device)
    started = monotonic()
    policy, value, kl, anchor = losses(actor, critic, batch, indices, anchors, spec, focused=False)
    value.backward()
    policy.backward()
    actor_norm = torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), spec["gradient_norm_max"])
    critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), spec["gradient_norm_max"])
    scalars = [float(x.detach()) for x in (policy, value, kl, anchor, actor_norm, critic_norm)]
    assert np.isfinite(scalars).all() and min(scalars[-2:]) > 0. and scalars[2] < 2e-7
    watch.sample()
    watch.check()
    estimated = base_mib + (torch.cuda.max_memory_reserved()-base_reserved)/2**20
    total = watch.rows[-1]["total_mib"]
    measured = watch.summary()["peak_mib"]
    margin = total*spec["memory_cap_fraction"] - max(measured, estimated)
    assert margin >= spec["minimum_reserved_headroom_mib"], "PPO first-batch memory headroom"
    assert before == [common_ppo.model_digest(model) for model in (actor, critic)]
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    return dict(status="passed-real-first-batch-gradient", gradient_minibatch=len(indices), measured_peak_mib=measured,
        estimated_peak_mib=estimated, margin_mib=margin, policy_loss=scalars[0], value_loss=scalars[1], kl=scalars[2],
        anchor_mse=scalars[3], actor_grad_norm=scalars[4], critic_grad_norm=scalars[5],
        optimizer_steps=0, actor_and_critic_unchanged=True, gradient_seconds=monotonic()-started,
        reserved_actor_moment_bytes=sum(x.numel()*x.element_size() for x in moments))


def run(spec, arm):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    item = spec["arms"][arm]
    folder, page = (ROOT / item[key] for key in ("record", "page"))
    state = read(folder / "state.json")
    assert state["status"] == "starting" and not (folder / "execution-exit.txt").exists()
    watch = MemoryWatch(spec["memory_cap_fraction"])
    n, horizon = spec["worlds"], spec["horizon"]
    env = camera = video = None
    code = 1
    state.update(actor_steps=0, critic_steps=0)
    try:
        with patch("run_walk_right_startup_probe.PLAN", PLAN):
            require_observer(folder, page, state, arm)
        check_project()
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        draft, parent = source_plan()
        cfg, p = configuration(draft, parent, n, spec["seed"], weight=item["weight"])
        deadline = monotonic()+spec["deadline_seconds_per_arm"]
        with watch.active():
            env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode=None)
            obs, _ = env.reset(seed=spec["seed"])
            actor = load_actor(draft, p, env.device)
            pipe, parity, scene = first_inputs(env, draft, parent, p, obs, actor)
            shared = ROOT / spec["shared_value"]
            saved = torch.load(shared, map_location="cpu", weights_only=True)
            assert saved["admitted"] and saved["actor_updates"] == 0
            critic = SharedBackwardValue(spec["shared_value_seed"], env.device)
            critic.load_state_dict(saved["critic_state_dict"], strict=True)
            signature = common_ppo.initial_signature(actor, critic, env, shared)
            assert signature["actor_sha256"] == saved["actor_sha256"]
            if arm == "candidate":
                control = ROOT / spec["arms"]["control"]["record"]
                assert (control / "execution-exit.txt").read_text().strip() == "0"
                assert signature == read(control / "startup.json")["signature"], "Paired start differs"
            write(folder / "startup.json", dict(signature=signature, scene_sha256=scene, actor_onnx_error=parity,
                  reward_coefficient=item["weight"]))
            frozen = {name: x.clone() for name, x in actor.state_dict().items() if not name.startswith("mlp.")}
            actor.mlp.requires_grad_(True)
            actor_opt = torch.optim.Adam(actor.mlp.parameters(), lr=spec["actor_lr"])
            critic_opt = torch.optim.Adam(critic.parameters(), lr=spec["critic_lr"])
            critic_opt.load_state_dict(saved["critic_optimizer_state_dict"])
            for group in critic_opt.param_groups:
                group["lr"] = spec["critic_lr"]
            optimizers = (actor_opt, critic_opt)
            rngs = tuple(torch.Generator(device=env.device).manual_seed(spec["seed"]+offset)
                         for offset in (10000, 20000))
            with np.load(ROOT / spec["anchor_bank"], allow_pickle=False) as bank:
                anchors = tuple(torch.as_tensor(bank[name], device=env.device) for name in ("observations", "actions"))
            assert anchors[0].shape == (1024, 61) and anchors[1].shape == (1024, 14)
            storage = BackwardRolloutBatch(horizon, n, env.device)
            weights = torch.ones(n, device=env.device)
            watch.sample()
            base_mib, base_reserved = watch.rows[-1]["used_mib"], torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            rm = env.reward_manager
            costs = torch.empty((horizon, n), device=env.device)
            assert rm.active_terms[-1] == TERM and rm._scale_by_dt
            original = rm.compute

            def capture(dt):
                reward = original(dt)
                costs[index].copy_(rm._step_reward[:, -1]*dt)
                return reward

            rm.compute = capture
            history, archived_targets = [], []
            camera = RecordingSnapshot(env, folder)
            camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.05, 75, -20
            video = ActionVideo(camera, "walk", folder)
            video.sample()
            poses = [env.sim.data.qpos[0].clone()]
            state.update(status="running", frame=video.image["file"])
            publish(folder, page, state)
            for update in range(1, spec["updates_per_arm"]+1):
                tick = monotonic()
                for index in range(horizon):
                    watch.check()
                    assert monotonic() < deadline, "PPO wall limit"
                    with torch.no_grad():
                        actor_obs, critic_obs = pipe.inputs(obs)
                        mean = actor_mean(actor, actor_obs)
                        action, log_prob = action_and_log_prob(mean, spec["initial_std"], rngs[0])
                        value = critic(critic_obs).squeeze(-1)
                        obs, reward, term, timeout, _ = env.step(action)
                        storage.put(index, actor_obs, critic_obs, mean, action, log_prob, value,
                                    Transition(reward, weights, term, timeout))
                    if video.writer is not None:
                        poses.append(env.sim.data.qpos[0].clone())
                        video.sample()
                    if video.clips and video.clips[-1]["complete_clip"]:
                        state["latest_clip"] = video.clips[-1]["file"]
                torch.cuda.synchronize()
                elapsed = monotonic()-tick
                assert not env.termination_manager.get_term("nan_state").any()
                assert int(env.sim.wp_data.nacon.numpy()[0]) < env.sim.wp_data.contact.dist.shape[0]
                with torch.no_grad():
                    tail = critic(pipe.critic_inputs(obs)).squeeze(-1)
                if update == 1:
                    gate = gradient_gate(actor, critic, storage, tail, anchors, spec, watch, base_mib, base_reserved)
                    write(folder / "first-gradient-check.json", gate)
                if update in (1, spec["updates_per_arm"]):
                    batch = storage.prepare(tail, spec["gamma"], spec["lam"])
                    targets = {key: batch[key].reshape(storage.shape).cpu().numpy()
                               for key in ("reward", "value", "done", "returns")}
                    archived_targets.append(dict(**targets, tail=tail.cpu().numpy(), cost=costs.cpu().numpy()))
                start = monotonic()
                row = common_ppo.batch_update(actor, critic, storage, tail, anchors, spec, rngs[1], optimizers,
                                              False, watch)
                row.update(update=update, sampling_seconds=elapsed, optimizer_seconds=monotonic()-start,
                           penalized_transitions=int((costs < 0).sum()), mean_added_cost=float(costs.mean()))
                history.append(row)
                state.update(controls=env.common_step_counter, transitions=env.common_step_counter*n, learning_updates=update)
                state["actor_steps"] += row["actor_steps"]
                state["critic_steps"] += row["critic_steps"]
                state["updates"].append(dict(control=state["controls"], transitions_per_second=n*horizon/elapsed))
                state["purpose"] = f"右转 PPO · {arm} · 回报 {row['mean_reward']:.3f} · KL {row['max_kl']:.4f}"
                write(folder / "ppo-metrics.json", history)
                if update in spec["checkpoints"]:
                    with patch.object(common_ppo, "PLAN", PLAN):
                        common_ppo.save_checkpoint(folder / f"checkpoint-{update}.pt", actor, critic,
                                                   optimizers, rngs, state, spec, arm)
                publish(folder, page, state)
                print("RIGHT_STARTUP_PPO", arm, row, flush=True)
            assert env.common_step_counter == spec["controls_per_arm"]
            assert env._sim_step_counter == spec["physical_substeps_per_arm"]
            assert all(torch.equal(actor.state_dict()[name], tensor) for name, tensor in frozen.items())
            actor_final = common_ppo.model_digest(actor)
            assert state["actor_steps"] > 0 and actor_final != signature["actor_sha256"]
            probe_inputs = np.concatenate((anchors[0].cpu().numpy(), actor_obs[:128].cpu().numpy()))
            probes = folder / "onnx-probes.npz"
            np.savez_compressed(probes, observations=probe_inputs)
            export = export_walking_actor(actor, ROOT / spec["official_actor"], spec["official_actor_sha256"],
                                          folder / "policy.onnx", probe_inputs)
            trace = folder / "value-targets.npz"
            np.savez_compressed(trace, **{key: np.stack([row[key] for row in archived_targets])
                                         for key in archived_targets[0]}, video_qpos=torch.stack(poses).cpu().numpy())
            assert video.clips[-1]["complete_clip"] and video.clips[-1]["frames"] == 101
            watch.sample()
            watch.check()
            final = folder / f"checkpoint-{spec['updates_per_arm']}.pt"
            write(folder / "analysis.json", dict(status="completed-native-right-turn-ppo-arm", arm=arm,
                signature=signature, actor_sha256=actor_final, critic_sha256=common_ppo.model_digest(critic),
                checkpoint_sha256=sha(final), protocol_sha256=sha(PLAN), export=export, probes_sha256=sha(probes),
                trace_sha256=sha(trace), video_sha256=sha(folder / state["latest_clip"]), memory=watch.summary(),
                physical_substeps=env._sim_step_counter, **{k: state[k] for k in ("controls", "transitions",
                    "learning_updates", "actor_steps", "critic_steps")}))
            state["status"] = "completed"
            code = 0
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
        (folder / "execution-exit.txt").write_text(f"{code}\n", encoding="utf-8")


def run_pair(spec):
    ledger = OUT / "paired-ppo-execution.json"
    assert not ledger.exists()
    results = []
    with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm, item in spec["arms"].items():
            check_project()
            record = ROOT / item["record"]
            command = [sys.executable, "-B", "-u", str(ENTRY), "--arm", arm]
            (record / "command.txt").write_text("MUJOCO_GL=egl "+shlex.join(command)+"\n", encoding="utf-8")
            print("RIGHT_STARTUP_PPO_START", arm, flush=True)
            with (record / "execution.log").open("xb") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, MUJOCO_GL="egl"), timeout=350, check=False)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    code = 124
            if not (record / "execution-exit.txt").exists():
                (record / "execution-exit.txt").write_text(f"{code}\n", encoding="utf-8")
                state = read(record / "state.json")
                state.update(status="stopped-review-required", reason=f"Worker exit {code}")
                publish(record, ROOT/item["page"], state)
            results.append(dict(arm=arm, returncode=code, record=item["record"]))
            write(ledger, results)
            print("RIGHT_STARTUP_PPO_EXIT", arm, code, flush=True)
            if code:
                raise SystemExit(code)


def main():
    parser = ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--freeze", action="store_true")
    command.add_argument("--run", action="store_true")
    command.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run_pair(checked())
    else:
        run(checked(), args.arm)


if __name__ == "__main__":
    main()
