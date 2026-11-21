"""Freeze and measure the first real CUDA rollout of the native backward path."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
from time import monotonic, process_time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from s1_reference_inputs import ROOT, read, sha
from check_walk_backward_native_rollout import (
    OUT, PROOF, check_inputs, configuration, inputs, td,
)
from action_video import ActionVideo
from getup_anchor_runtime import RecordingSnapshot
from getup_ppo_math import finite_gae
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
import training_live_page as live

ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_native_probe.json"
PAGE = ROOT / "records/walk-backward-native-probe-live.html"
SEED = 1133
FIELDS = ("forward_velocity", "completed_age", "reward", "value", "terminated", "timed_out", "actor_weight")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    from common_walk_forward_phase_screen import check_sources

    assert not PLAN.exists() and not PAGE.exists()
    check_project()
    parent, _ = inputs()
    check_sources(parent)
    cpu = read(PROOF)
    assert cpu["status"] == "passed-native-backward-cpu-zero-step" and cpu["physical_substeps"] == 0
    assert not any(f"_s{SEED}_" in folder.name for folder in (ROOT / "records/evaluations").iterdir())
    hashes = dict(cpu["source_sha256"])
    witness = ROOT / read(OUT.parent / "common-walk-backward-stall-capture/capacity/analysis.json")["record"]
    witness = witness / "windows-view-verification.json"
    assert read(witness)["status"] == "passed"
    files = (ENTRY, ENTRY.with_suffix(".sh"), PROOF, witness, ROOT / "scripts/training/action_video.py",
             ROOT / "scripts/training/getup_anchor_runtime.py", ROOT / "scripts/training/training_live_page.py",
             ROOT / "scripts/validation/run_local_getup_ppo.py")
    hashes.update({relative(path): sha(path) for path in files})
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT / "records/evaluations" / f"duck_walk_native-rollout-probe_reference_s{SEED}_{stamp}"
    spec = dict(status="frozen-native-backward-probe", seed=SEED, record=relative(folder), page=relative(PAGE),
                source_sha256=hashes, library_source_sha256=cpu["library_source_sha256"], worlds=1024,
                warmup_controls=64, timed_controls=320, maximum_batched_substeps=1920,
                maximum_transitions=393216, deadline_seconds=240, memory_cap_fraction=0.85,
                minimum_tagged_worlds=16, timeout_probe_world=1, timeout_probe_control=24,
                timeout_probe_scope="Set only this world's episode age to maximum minus one before native step.",
                policy_model_sha256=parent["environment"]["model_sha256"],
                value_network="Fresh frozen 77-512-256-128-1 ELU for timing and boundary verification.",
                source_witness=relative(witness), learning_updates=0, heldout_used=False, hardware_io=False,
                retry=False, extension=False)
    for path, digest in hashes.items():
        assert sha(ROOT / path) == digest, path
    folder.mkdir()
    write(PLAN, spec)
    write(folder / "protocol.json", spec)
    (folder / "README.md").write_text(
        "# 原生后退采样\n\n固定官方权重，首次检查实际 CUDA 原生重置和回报边界，"
        "用同一轮记录包含录像和写盘开销的净吞吐。\n\n"
        f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{PAGE.name})\n",
        encoding="utf-8",
    )
    state = dict(status="starting", run_id=folder.name, purpose="固定官方后退策略 · CUDA 原生采样",
                 controls=0, maximum_controls=384, transitions=0, learning_updates=0, updates=[],
                 updated_at=datetime.now(timezone.utc).isoformat())
    with patch.object(live, "PAGE", PAGE):
        live.publish(folder, state)
    print("NATIVE_BACKWARD_PROBE_FROZEN", folder.name, sha(PLAN), flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-backward-probe"
    assert not any(spec[key] for key in ("heldout_used", "hardware_io", "learning_updates", "retry", "extension"))
    for path, digest in spec["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in spec["library_source_sha256"].items():
        assert sha(Path(path)) == digest, path
    return spec


def replay_weights(trace, focus, minimum):
    age, weight, speed = (trace[key] for key in ("completed_age", "actor_weight", "forward_velocity"))
    ended = trace["terminated"] | trace["timed_out"]
    block = focus["block_controls"]
    start, stop = block * 2, focus["focus_stop_control"]
    alive_stall = np.zeros(age.shape[1], dtype=bool)
    ever_tagged = alive_stall.copy()
    for t in range(len(age)):
        weighted = alive_stall & (age[t] > start) & (age[t] <= stop)
        np.testing.assert_array_equal(weight[t], np.where(weighted, focus["focused_actor_loss_multiplier"], 1))
        alive_stall &= ~(ended[t] | (age[t] == 1))
        if t + 1 >= start:
            tail = slice(t + 1 - start, t + 1)
            on_clock = (age[tail] == np.arange(1, start + 1)[:, None]).all(axis=0)
            speed_windows = speed[tail].reshape(2, block, -1).mean(axis=1, dtype=np.float64)
            static = (np.abs(speed_windows) <= focus["maximum_absolute_block_speed_m_s"]).all(axis=0)
            alive_stall |= on_clock & static & ~ended[tail].any(axis=0)
            ever_tagged |= alive_stall
    assert ever_tagged.sum() >= minimum
    return dict(tagged_worlds=int(ever_tagged.sum()), weighted_transitions=int((weight > 1).sum()),
                native_terminations=int(trace["terminated"].sum()), native_timeouts=int(trace["timed_out"].sum()),
                independently_reconstructed_weights=True)


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    spec = checked()
    folder = ROOT / spec["record"]
    state = read(folder / "state.json")
    assert state["status"] == "starting"
    check_project()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    parent, focus = inputs()
    n = spec["worlds"]
    cfg, plan, internal = configuration(parent, n, spec["seed"])
    total = spec["warmup_controls"] + spec["timed_controls"]
    assert n * total == spec["maximum_transitions"] and total * cfg.decimation == spec["maximum_batched_substeps"]
    watch = MemoryWatch(spec["memory_cap_fraction"])
    begin = monotonic()
    env = None
    video = None

    def publish():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with patch.object(live, "PAGE", PAGE):
            live.publish(folder, state)

    try:
        with watch.active(), torch.no_grad():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            obs, _ = env.reset(seed=spec["seed"])
            pipe, actor, _, interface = check_inputs(env, plan, internal, focus, obs)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(spec["seed"])
                critic = nn.Sequential(nn.Linear(77, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                       nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(env.device).eval()
            watch.sample()
            assert env.common_step_counter == env._sim_step_counter == 0
            write(folder / "startup.json", dict(**interface, worlds=n, physical_substeps=0,
                                                initial_gpu_memory_mib=watch.rows[-1]["used_mib"]))
            camera = RecordingSnapshot(env, folder)
            camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.1, 90, -20
            try:
                video = ActionVideo(camera, "walk", folder)
                video.sample()
                state.update(status="running", frame=video.image["file"])
                publish()
                trace = {key: [] for key in FIELDS}
                timing = dict(write_seconds=0.0, measured_log_seconds=0.0)
                chunk_started = monotonic()
                for control in range(1, total + 1):
                    watch.check()
                    assert monotonic() - begin <= spec["deadline_seconds"], "Probe wall time exhausted"
                    row = spec["timeout_probe_world"]
                    forced = control == spec["timeout_probe_control"]
                    if forced:
                        env.episode_length_buf[row] = env.max_episode_length - 1
                    actor_obs, critic_obs = pipe.inputs(obs)
                    action = actor(td(actor_obs))
                    value = critic(critic_obs).squeeze(-1)
                    sample = pipe.advance(action, focused=True)
                    obs = sample.observations
                    for key in FIELDS:
                        trace[key].append((value if key == "value" else getattr(sample, key)).clone())
                    if forced:
                        actor_next, critic_next = pipe.inputs(obs)
                        assert sample.timed_out[row] and not sample.terminated[row]
                        assert sample.completed_age[row] == env.max_episode_length and not env.episode_length_buf[row]
                        assert not actor_next[row, 34:48].any() and critic_next[row, -1] == 1
                        assert sample.actor_weight[row] == 1
                        native = env.reward_manager._step_reward.sum(dim=1) * env.step_dt
                        torch.testing.assert_close(sample.reward, native, atol=2e-7, rtol=2e-6)
                        boundary = dict(world=row, control=control, last_age=int(sample.completed_age[row]),
                                        next_age=int(env.episode_length_buf[row]), last_reward=float(sample.reward[row]),
                                        next_history_zero=True, native_reward_reconstruction_error=float((sample.reward-native).abs().max()))
                    if video.writer is not None:
                        video.sample()
                    if video.clips and video.clips[-1]["complete_clip"]:
                        state["latest_clip"] = video.clips[-1]["file"]
                    state.update(controls=control, transitions=n * control)
                    if control % 32 == 0:
                        torch.cuda.synchronize()
                        now = monotonic()
                        state["updates"].append(dict(control=control, transitions_per_second=32*n/(now-chunk_started),
                                                     measured=control > spec["warmup_controls"]))
                        chunk_started = now
                        tick = monotonic()
                        publish()
                        print("NATIVE_BACKWARD_CUDA", control, "/", total, "worlds", n, flush=True)
                        timing["write_seconds"] += monotonic()-tick
                        if control > spec["warmup_controls"]:
                            timing["measured_log_seconds"] += monotonic()-tick
                    if control == spec["warmup_controls"]:
                        measured_started, cpu_started = monotonic(), process_time()
                        render_started = video.elapsed_seconds
                        chunk_started = measured_started
                torch.cuda.synchronize()
                timing.update(wall_seconds=monotonic()-measured_started, process_cpu_seconds=process_time()-cpu_started,
                              measured_render_seconds=video.elapsed_seconds-render_started)
                assert env.common_step_counter == total and env._sim_step_counter == spec["maximum_batched_substeps"]
                assert video.clips[-1]["complete_clip"] and video.clips[-1]["frames"] == 101
                _, critic_obs = pipe.inputs(obs)
                last = critic(critic_obs).squeeze(-1)
                stacked = {key: torch.stack(values) for key, values in trace.items()}
                dones = stacked["terminated"] | stacked["timed_out"]
                _, returns = finite_gae(stacked["reward"], stacked["value"], dones, last)
                assert torch.isfinite(returns).all() and torch.isfinite(stacked["reward"]).all()
                torch.testing.assert_close(returns[dones], stacked["reward"][dones], atol=2e-7, rtol=2e-6)
                boundary["return_target"] = float(returns[boundary["control"]-1, row])
                arrays = {key: value.cpu().numpy() for key, value in stacked.items()}
                arrays.update(value_target=returns.cpu().numpy(), value_tail=last.cpu().numpy())
            finally:
                if video is not None:
                    video.close()
                camera.close()
            watch.sample()
            watch.check()
        trace_path = folder / "native-rollout.npz"
        tick = monotonic()
        np.savez_compressed(trace_path, **arrays)
        timing["trace_archive_seconds"] = monotonic()-tick
        replay = replay_weights(arrays, focus, spec["minimum_tagged_worlds"])
        clip_path = folder / state["latest_clip"]
        result = dict(status="passed-native-cuda-rollout", protocol_sha256=sha(PLAN), **replay, interface=interface,
                      boundary=boundary, timing=timing, actual_transitions=n*total, batched_physical_substeps=5*total,
                      measured_transitions=n*spec["timed_controls"], net_transitions_per_second=n*spec["timed_controls"]/timing["wall_seconds"],
                      memory=watch.summary(), trace_sha256=sha(trace_path), video_sha256=sha(clip_path), learning_updates=0)
        write(folder / "analysis.json", result)
        state.update(status="completed")
        print("NATIVE_BACKWARD_CUDA_PASS", result, flush=True)
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        if env is not None:
            state["controls"] = int(env.common_step_counter)
            state["transitions"] = n * state["controls"]
            env.close()
        write(folder / "gpu-telemetry.json", watch.rows)
        publish()


def main():
    parser = ArgumentParser(description=__doc__)
    args = parser.add_mutually_exclusive_group(required=True)
    args.add_argument("--freeze", action="store_true")
    args.add_argument("--sample", action="store_true")
    if parser.parse_args().freeze:
        freeze()
    else:
        sample()


if __name__ == "__main__":
    main()
