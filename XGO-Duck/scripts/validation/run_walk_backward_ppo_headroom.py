"""Measure real native PPO gradients and memory with the parent weights fixed."""

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
from check_walk_backward_native_rollout import OUT, check_inputs, configuration, inputs
from getup_anchor_runtime import RecordingSnapshot
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from walk_backward_ppo import BackwardRolloutBatch, action_and_log_prob, actor_mean, losses, make_value_model
import training_live_page as live


ENTRY = Path(__file__).resolve()
SCALE = ROOT / "configs/experiments/common_walk_backward_native_scale_refinement.json"
PLAN = ROOT / "configs/experiments/common_walk_backward_ppo_headroom.json"
PAGE = ROOT / "records/walk-backward-ppo-headroom-live.html"
SEED = 1136
TRACE_KEYS = ("reward", "value", "done", "weight", "returns")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def publish(folder, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", PAGE):
        live.publish(folder, state)


def freeze():
    assert not PLAN.exists() and not PAGE.exists()
    check_project()
    parent, focus = inputs()
    scale = read(SCALE)
    source = ROOT / scale["record"]
    verification = read(source / "verification.json")
    summary = read(source / "summary.json")
    assert verification["status"] == "passed-independent-scale-verification"
    assert verification["summary_sha256"] == sha(source / "summary.json")
    assert summary["protocol_sha256"] == sha(SCALE)
    n = verification["selected_worlds"]
    assert n == summary["selected"]["worlds"] and n in scale["sizes"]
    anchor_receipt = ROOT / focus["retention"][0]
    assert sha(anchor_receipt) == focus["retention"][1]
    anchor_path = anchor_receipt.with_name("anchors.npz")
    assert sha(anchor_path) == read(anchor_receipt)["bank_sha256"]
    reference = ROOT / "configs/experiments/common_walk_backward_native_probe.json"
    frozen_probe = read(reference)
    paths = (ENTRY, ENTRY.with_name("verify_walk_backward_ppo_headroom.py"), SCALE, source / "verification.json",
             source / "summary.json", anchor_receipt, anchor_path, reference,
             ROOT / "scripts/training/walk_backward_ppo.py")
    checksums = {**frozen_probe["source_sha256"], **{rel(path): sha(path) for path in paths}}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT / "records/evaluations" / f"duck_walk_ppo-headroom_reference_s{SEED}_{stamp}"
    assert not any(f"_s{SEED}_" in path.name for path in folder.parent.iterdir())
    spec = dict(status="frozen-native-backward-ppo-headroom", seed=SEED, worlds=n, record=rel(folder),
                source_sha256=checksums, library_source_sha256=frozen_probe["library_source_sha256"],
                policy_model_sha256=parent["environment"]["model_sha256"], anchors=rel(anchor_path),
                warmup_controls=96, horizon=24, rollout_batches=5, minibatches=4,
                initial_std=0.05, gamma=0.99, lam=0.95, clip=0.2, anchor_coefficient=0.02,
                timeout_probe_world=1, timeout_probe_control=112, maximum_controls=216,
                maximum_batched_substeps=1080, maximum_transitions=n*216, deadline_seconds=300,
                memory_cap_fraction=scale["memory_cap_fraction"], minimum_reserved_headroom_mib=256,
                optimizer_reserve="Two same-device tensors per trainable parameter for Adam moments; no optimizer is invoked.",
                learning_updates=0, heldout_used=False, hardware_io=False, extension=False)
    for path, digest in checksums.items():
        assert sha(ROOT / path) == digest, path
    folder.mkdir()
    write(PLAN, spec)
    write(folder / "protocol.json", spec)
    (folder / "README.md").write_text(
        "# 后退 PPO 显存核验\n\n实际采样、回报和梯度计算，官方权重固定。\n\n"
        f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{PAGE.name})\n",
        encoding="utf-8",
    )
    publish(folder, dict(status="starting", run_id=folder.name, purpose="后退 PPO 前向与反向 · 官方权重固定",
                         controls=0, maximum_controls=216, transitions=0, learning_updates=0, updates=[]))
    print("BACKWARD_PPO_HEADROOM_FROZEN", rel(folder), n, sha(PLAN), flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-backward-ppo-headroom"
    assert not any(spec[key] for key in ("heldout_used", "hardware_io", "learning_updates", "extension"))
    for path, digest in spec["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in spec["library_source_sha256"].items():
        assert sha(Path(path)) == digest, path
    return spec


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def backward_passes(actor, critic, storage, tail, anchors, plan, rng, watch):
    batch = storage.prepare(tail, plan["gamma"], plan["lam"])
    assert all(tensor.isfinite().all() for tensor in batch.values())
    done = batch["done"]
    torch.testing.assert_close(batch["returns"][done], batch["reward"][done], atol=2e-7, rtol=2e-6)
    indices = torch.randperm(len(done), device=done.device, generator=rng)
    assert len(done) % plan["minibatches"] == 0
    timings, results = [], []
    for block_id, block in enumerate(indices.chunk(plan["minibatches"])):
        watch.check()
        started = monotonic()
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        policy, value, kl, anchor = losses(actor, critic, batch, block, anchors, plan, focused=True)
        if block_id == 0:
            with torch.no_grad():
                plain = losses(actor, critic, batch, block, anchors, plan, focused=False)
            for measured, reference in zip((value, kl, anchor), plain[1:], strict=True):
                torch.testing.assert_close(measured, reference)
            policy_difference = float((policy.detach() - plain[0]).abs())
        value.backward()
        policy.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
        value_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        result = [float(t.detach()) for t in (policy, value, kl, anchor, actor_norm, value_norm)]
        assert np.isfinite(result).all() and min(result[-2:]) > 0 and result[2] < 2e-7
        results.append(result)
        timings.append(monotonic()-started)
    columns = ("policy_loss", "value_loss", "kl", "anchor_mse", "actor_grad_norm", "critic_grad_norm")
    metrics = dict(zip(columns, np.mean(results, axis=0).tolist(), strict=True))
    metrics.update(gradient_seconds=sum(timings), weighted_transitions=int((batch["weight"] > 1).sum()),
                   completed_episodes=int(done.sum()), tail_values=tail.numel(),
                   paired_actor_loss_difference=policy_difference)
    trace = {name: batch[name].reshape(storage.shape).cpu().numpy().copy() for name in TRACE_KEYS}
    trace["tail"] = tail.cpu().numpy().copy()
    return metrics, trace


def run(spec):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    folder = ROOT / spec["record"]
    state = read(folder / "state.json")
    assert state["status"] == "starting"
    n, horizon = spec["worlds"], spec["horizon"]
    total = spec["warmup_controls"] + horizon * spec["rollout_batches"]
    assert total == spec["maximum_controls"] and total * n == spec["maximum_transitions"]
    check_project()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    parent, focus = inputs()
    cfg, plan, internal = configuration(parent, n, spec["seed"])
    watch = MemoryWatch(spec["memory_cap_fraction"])
    env = None
    camera = video = None
    metrics, traces = [], []
    begin = monotonic()
    try:
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            obs, _ = env.reset(seed=spec["seed"])
            pipe, actor, _, interface = check_inputs(env, plan, internal, focus, obs)
            for parameter in actor.mlp.parameters():
                parameter.requires_grad_(True)
            critic = make_value_model(spec["seed"], env.device)
            initial = [model_digest(model) for model in (actor, critic)]
            assert env.common_step_counter == env._sim_step_counter == 0
            watch.sample()
            startup_mib = watch.rows[-1]["used_mib"]
            base_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            storage = BackwardRolloutBatch(horizon, n, env.device)
            # Adam is lazy. Reserve both moment tensors while keeping the
            # optimizer and model parameters untouched in this admission run.
            moments = [torch.zeros_like(p) for model in (actor.mlp, critic) for p in model.parameters()
                       for _ in range(2)]
            moment_bytes = sum(item.numel() * item.element_size() for item in moments)
            action_rng = torch.Generator(device=env.device).manual_seed(spec["seed"] + 10000)
            batch_rng = torch.Generator(device=env.device).manual_seed(spec["seed"] + 20000)
            with np.load(ROOT / spec["anchors"], allow_pickle=False) as bank:
                anchors = tuple(torch.as_tensor(bank[k], device=env.device) for k in ("observations", "teacher_action"))
            assert anchors[0].shape == (3072, 61) and anchors[1].shape == (3072, 14)
            watch.sample()
            write(folder / "startup.json", dict(interface=interface, worlds=n, gpu_mib=startup_mib,
                                                adam_moment_reserve_bytes=moment_bytes, physical_substeps=0))
            camera = RecordingSnapshot(env, folder)
            camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.1, 90, -20
            video = ActionVideo(camera, "walk", folder)
            video.sample()
            state.update(status="running", frame=video.image["file"])
            publish(folder, state)
            last_tick = monotonic()
            for control in range(1, total + 1):
                watch.check()
                assert monotonic()-begin <= spec["deadline_seconds"]
                with torch.no_grad():
                    forced = control == spec["timeout_probe_control"]
                    row = spec["timeout_probe_world"]
                    if forced:
                        env.episode_length_buf[row] = env.max_episode_length - 1
                    actor_obs, critic_obs = pipe.inputs(obs)
                    mean = actor_mean(actor, actor_obs)
                    action, log_prob = action_and_log_prob(mean, spec["initial_std"], action_rng)
                    value = critic(critic_obs).squeeze(-1)
                    step = pipe.advance(action, focused=True)
                    obs = step.observations
                    if forced:
                        assert step.timed_out[row] and not env.episode_length_buf[row]
                    if control > spec["warmup_controls"]:
                        index = (control - spec["warmup_controls"] - 1) % horizon
                        storage.put(index, actor_obs, critic_obs, mean, action, log_prob, value, step)
                if video.writer is not None:
                    video.sample()
                if video.clips:
                    state["latest_clip"] = video.clips[-1]["file"]
                state.update(controls=control, transitions=control*n)
                if control % horizon == 0:
                    torch.cuda.synchronize()
                    now = monotonic()
                    state["updates"].append(dict(control=control, transitions_per_second=horizon*n/(now-last_tick)))
                    if control > spec["warmup_controls"]:
                        with torch.no_grad():
                            tail = critic(pipe.inputs(obs)[1]).squeeze(-1)
                        row_metrics, trace = backward_passes(actor, critic, storage, tail, anchors, spec, batch_rng, watch)
                        row_metrics["control"] = control
                        metrics.append(row_metrics)
                        traces.append(trace)
                        write(folder / "gradient-metrics.json", metrics)
                        watch.sample()
                    state["diagnostic_batches"] = len(metrics)
                    state["purpose"] = (
                        f"后退 PPO · 已完成 {len(metrics)}/{spec['rollout_batches']} 批梯度核验 · 官方权重固定"
                    )
                    publish(folder, state)
                    print("BACKWARD_PPO_HEADROOM", control, total, len(metrics), flush=True)
                    last_tick = monotonic()
            assert env.common_step_counter == total and env._sim_step_counter == spec["maximum_batched_substeps"]
            assert video.clips[-1]["complete_clip"] and video.clips[-1]["frames"] == 101
            assert sum(row["weighted_transitions"] for row in metrics) > 0
            assert max(row["paired_actor_loss_difference"] for row in metrics) > 0
            final = [model_digest(model) for model in (actor, critic)]
            assert initial == final
            assert len(moments) > 0
            watch.sample()
            watch.check()
            peak_reserved = torch.cuda.max_memory_reserved()
            estimated_peak = startup_mib + (peak_reserved - base_reserved) / 2**20
        trace_path = folder / "native-ppo-targets.npz"
        np.savez_compressed(trace_path, **{key: np.stack([row[key] for row in traces]) for key in (*TRACE_KEYS, "tail")})
        cap_mib = watch.rows[-1]["total_mib"] * spec["memory_cap_fraction"]
        observed_peak = watch.summary()["peak_mib"]
        margin = cap_mib - max(observed_peak, estimated_peak)
        result = dict(status="passed-native-ppo-gradient-readback", admitted=margin >= spec["minimum_reserved_headroom_mib"],
                      protocol_sha256=sha(PLAN), actual_transitions=total*n,
                      batched_physical_substeps=spec["maximum_batched_substeps"], actor_weights_exact=True,
                      critic_weights_exact=True, before_model_digests=initial, after_model_digests=final,
                      torch_reserved_peak_bytes=peak_reserved,
                      estimated_device_peak_mib=estimated_peak, memory=watch.summary(), memory_cap_mib=cap_mib,
                      memory_margin_mib=margin, gradient_minibatches=len(metrics)*spec["minibatches"],
                      trace_sha256=sha(trace_path), video_sha256=sha(folder / state["latest_clip"]), learning_updates=0)
        write(folder / "analysis.json", result)
        state.update(status="completed", admitted=result["admitted"])
        print("BACKWARD_PPO_HEADROOM_RESULT", result, flush=True)
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        if video is not None:
            video.close()
        if camera is not None:
            camera.close()
        if env is not None:
            state.update(controls=env.common_step_counter, transitions=env.common_step_counter*n)
            env.close()
        write(folder / "gpu-telemetry.json", watch.rows)
        publish(folder, state)


def main():
    parser = ArgumentParser(description=__doc__)
    args = parser.add_mutually_exclusive_group(required=True)
    args.add_argument("--freeze", action="store_true")
    args.add_argument("--run", action="store_true")
    if parser.parse_args().freeze:
        freeze()
    else:
        spec = checked()
        with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(spec)


if __name__ == "__main__":
    main()
