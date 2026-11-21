"""Fit one right-turn value network from matched native first-episode rewards."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import gc
from pathlib import Path
from time import monotonic, process_time, sleep
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from action_video import ActionVideo
from check_walk_right_startup_native import TERM, configuration, first_inputs, load_actor, source_plan
from getup_anchor_runtime import RecordingSnapshot
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from run_walk_backward_ppo_headroom import model_digest
import run_walk_backward_shared_value as shared_fit
from run_walk_right_startup_probe import PLAN as PROBE, checked as checked_probe, require_observer
import training_live_page as live
from walk_backward_ppo import actor_mean
from walk_backward_value import episode_returns, world_split


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_startup_shared_value.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
PAGE = ROOT / "records/walk-right-startup-shared-value-live.html"
OUT = ROOT / "records/reports/common-walk-right-startup-ppo"
SEED = 1147


def relative(path):
    return path.relative_to(ROOT).as_posix()


def publish(folder, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", PAGE):
        live.publish(folder, state)
    metrics = folder / "value-metrics.json"
    if metrics.exists():
        rows = read(metrics)
        errors = [row["validation"]["normalized_mse"] for row in rows]
        peak = max(.25, *errors)
        points = " ".join(f"{35+5.8*row['update']:.2f},{135-106*error/peak:.2f}"
                          for row, error in zip(rows, errors, strict=True))
        gate = 135-106*.25/peak
        panel = f"""<section id="value-fit" data-update="{rows[-1]['update']}">
<h2>共享价值验证曲线</h2>
<svg viewBox="0 0 640 160" role="img" aria-label="验证归一化均方误差">
<path d="M35 24V135H615" fill="none" stroke="#a4b6c4"/>
<path d="M35 {gate:.2f}H615" stroke="#ae6a23" stroke-dasharray="5 4"/>
<polyline points="{points}" fill="none" stroke="#087e8b" stroke-width="2"/>
<text x="35" y="17">归一化 MSE · 虚线 0.25</text></svg>
<p>更新 {rows[-1]['update']} · MSE {errors[-1]:.6f} · EV {rows[-1]['validation']['explained_variance']:.6f}</p>
</section>"""
        temp = PAGE.with_suffix(".value.tmp")
        html = PAGE.read_text(encoding="utf-8").replace("</html>", panel+"</html>")
        deadline = monotonic()+1.
        while True:
            try:
                temp.write_text(html, encoding="utf-8")
                temp.replace(PAGE)
                break
            except PermissionError:
                if monotonic() >= deadline:
                    raise
                sleep(.05)


def targets(data, spec):
    native, candidate, done = (data[key] for key in ("native_reward", "candidate_reward", "done"))
    midpoint = (native + candidate)*.5
    returns, valid = episode_returns(midpoint, done, spec["gamma"])
    terminal = done.argmax(axis=0)
    windows = data["prefix_yaw"][50:150].reshape(2, 50, -1).mean(axis=1, dtype=np.float64)
    label = (terminal >= 200) & (np.abs(windows).max(axis=0) <= spec["slow_yaw_rad_s"])
    train, validation = world_split(label, spec["validation_fraction"], spec["split_seed"])
    split = dict(train_worlds=train.tolist(), validation_worlds=validation.tolist(),
                 labelled_worlds=int(label.sum()), labelled_validation_worlds=int(label[validation].sum()))
    return returns, valid, label, split


def target_fixture():
    native = np.zeros((205, 3), dtype=np.float32)
    candidate = native.copy()
    done = np.zeros_like(native, dtype=bool)
    done[[1, 199, 204], np.arange(3)] = True
    native[0:2, 0] = [2., 4.]
    candidate[0:2, 0] = [0., 2.]
    native[199, 1], candidate[199, 1] = 4., 2.
    native[204, 2], candidate[204, 2] = 3., 3.
    prefix = np.zeros((200, 3), dtype=np.float32)
    spec = dict(gamma=.99, slow_yaw_rad_s=.05, validation_fraction=.25, split_seed=SEED)
    reward, valid, label, split = targets(
        dict(native_reward=native, candidate_reward=candidate, done=done, prefix_yaw=prefix), spec)
    np.testing.assert_allclose(reward[:3, 0], [3.97, 3., 0.], atol=3e-7, rtol=0)
    np.testing.assert_array_equal(valid.sum(axis=0), [2, 200, 205])
    np.testing.assert_array_equal(label, [False, False, True])
    assert set(split["train_worlds"]).isdisjoint(split["validation_worlds"])
    return dict(terminal_rewards=reward[done].tolist(), completed_episode_lengths=valid.sum(axis=0).tolist(),
                post_terminal_targets_zero=bool(not reward[~valid].any()))


def freeze():
    check_project()
    assert not PLAN.exists() and not PAGE.exists()
    draft = read(DRAFT)
    assert draft["status"] == "draft-right-turn-shared-value"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "critic_updates_authorized",
                                    "actor_updates_authorized", "heldout_used", "hardware_io"))
    checksums = dict(draft[k] for k in ("source", "reward_scale_evidence"))
    for name, checksum in checksums.items():
        assert sha(ROOT / name) == checksum, name
    probe = checked_probe()
    probe_result, signal = (read(ROOT / draft[k][0]) for k in ("source", "reward_scale_evidence"))
    assert probe_result["status"] == "passed-independent-native-right-turn-probe"
    assert probe_result["protocol_sha256"] == sha(PROBE)
    assert signal["value_draft_admitted"] and signal["source_result_sha256"] == draft["source"][1]
    assert signal["new_reward_coefficient"] == draft["reward_coefficient"] == .25
    assert probe_result["worlds"] == probe["worlds"] == draft["worlds"] == 3072
    assert all(result["peak_mib"] < .85*16384 for result in probe_result["cases"].values())
    source_plan()
    for path in (DRAFT, ENTRY, PROBE, ENTRY.with_name("observe_walk_right_startup_shared_value.py"),
                 ENTRY.with_name("verify_walk_right_startup_shared_value.py"), Path(shared_fit.__file__),
                 ROOT / "scripts/training/walk_backward_value.py", ROOT / "scripts/training/record_io.py"):
        checksums[relative(path)] = sha(path)
    for name, checksum in probe["source_sha256"].items():
        assert name not in checksums or checksums[name] == checksum
        checksums[name] = checksum
    assert all(not any(f"_s{SEED}_" in path.name for path in (ROOT / "records" / part).iterdir())
               for part in ("experiments", "evaluations", "benchmarks"))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT / "records/experiments" / f"duck_walk_right-startup-shared-value_reference_s{SEED}_{stamp}"
    controls, n = draft["first_episode_maximum_controls"], draft["worlds"]
    fitting = draft["value_learning_plan"]
    spec = dict(draft, status="frozen-native-right-turn-shared-value", seed=SEED, record=relative(folder),
        page=relative(PAGE), source_sha256=checksums, library_source_sha256=probe["library_source_sha256"],
        maximum_transitions=n*controls, physical_substeps_authorized=controls*5, critic_updates_authorized=fitting["updates"],
        maximum_value_updates=fitting["updates"], value_minibatch_size=fitting["minibatch"],
        value_learning_rate=fitting["learning_rate"], validation=draft["validation_gate"], gamma=.99,
        memory_cap_fraction=probe["memory_cap_fraction"], deadline_seconds=450, slow_yaw_rad_s=.05,
        input_scale_floor=.001, gradient_norm_max=1., report_interval_updates=10,
        split_seed=SEED+20000, shuffle_seed=SEED+30000, target_fixture=target_fixture(),
        sampling_policy="Deterministic nominal outputs from the fixed learned right-turn parent; no exploration noise.")
    assert controls == 1000 and fitting["updates"] == 100
    folder.mkdir()
    write(PLAN, spec)
    write(folder / "protocol.json", spec)
    (folder / "README.md").write_text(
        "# 右转共享价值拟合\n\n用固定策略采集原生完整首回合，以两组奖励均值拟合共同价值起点。\n\n"
        f"[冻结方案](../../../{relative(PLAN)}) · [运行观察](../../{PAGE.name})\n", encoding="utf-8")
    publish(folder, dict(status="starting", run_id=folder.name, purpose="右转共享价值 · 等待首回合采样",
                         controls=0, maximum_controls=controls, transitions=0, learning_updates=0, updates=[]))
    print("RIGHT_STARTUP_VALUE_FROZEN", relative(folder), sha(PLAN), flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-right-turn-shared-value"
    assert not any(spec[k] for k in ("actor_updates_authorized", "heldout_used", "hardware_io"))
    for name, checksum in spec["source_sha256"].items():
        assert sha(ROOT / name) == checksum, name
    for name, checksum in spec["library_source_sha256"].items():
        assert sha(Path(name)) == checksum, name
    return spec


@torch.no_grad()
def collect(spec, folder, state, watch, deadline):
    from mjlab.envs import ManagerBasedRlEnv

    draft, parent = source_plan()
    n, horizon, stride = (spec[k] for k in ("worlds", "first_episode_maximum_controls", "critic_observation_stride"))
    cfg, p = configuration(draft, parent, n, spec["seed"], weight=spec["reward_coefficient"])
    env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode=None)
    camera = video = None
    try:
        obs, _ = env.reset(seed=spec["seed"])
        actor = load_actor(draft, p, env.device)
        pipe, parity, scene = first_inputs(env, draft, parent, p, obs, actor)
        initial = model_digest(actor)
        assert env.common_step_counter == env._sim_step_counter == 0 and horizon == env.max_episode_length
        watch.sample()
        write(folder / "startup.json", dict(physical_substeps=0, actor_onnx_max_error=parity,
              scene_static_sha256=scene, actor_sha256=initial, initial_gpu_memory_mib=watch.rows[-1]["used_mib"]))
        data = {k: torch.empty((horizon, n), device=env.device) for k in ("native_reward", "candidate_reward", "yaw")}
        data.update(done=torch.empty((horizon, n), dtype=torch.bool, device=env.device),
            timeout=torch.empty((horizon, n), dtype=torch.bool, device=env.device),
            critic_obs=torch.empty((horizon//stride, n, 77), device=env.device))
        robot, rm = env.scene["robot"], env.reward_manager
        assert rm.active_terms[-1] == TERM and rm._scale_by_dt
        original = rm.compute
        captured = {}

        def capture(dt):
            reward = original(dt)
            # Original native terms stay intact. Read both reward streams before reset.
            captured["native_reward"] = rm._step_reward[:, :-1].sum(dim=1)*dt
            captured["yaw"] = robot.data.root_link_ang_vel_b[:, 2].clone()
            return reward

        rm.compute = capture
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        camera = RecordingSnapshot(env, folder)
        camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.05, 75, -20
        video = ActionVideo(camera, "walk", folder)
        video.sample()
        poses = [env.sim.data.qpos[0].clone()]
        state.update(status="running", purpose="右转共享价值 · 完整首回合采样", frame=video.image["file"])
        publish(folder, state)
        begin = tick = monotonic()
        cpu_start = process_time()
        publish_seconds = 0.
        contact_peak = 0
        for t in range(horizon):
            watch.check()
            assert monotonic() < deadline, "Right-turn value sampling deadline"
            actor_obs, critic_obs = pipe.inputs(obs)
            if t % stride == 0:
                data["critic_obs"][t//stride].copy_(torch.where(alive[:, None], critic_obs, 0.))
            action = actor_mean(actor, actor_obs)
            obs, reward, terminated, timed_out, _ = env.step(action)
            done = alive & (terminated | timed_out)
            data["done"][t].copy_(done)
            data["timeout"][t].copy_(alive & timed_out)
            for key, sample in (("native_reward", captured["native_reward"]), ("candidate_reward", reward),
                                ("yaw", captured["yaw"])):
                data[key][t].copy_(torch.where(alive, sample, 0.))
            alive &= ~done
            if video.writer is not None:
                poses.append(env.sim.data.qpos[0].clone())
                video.sample()
            if video.clips and video.clips[-1]["complete_clip"]:
                state["latest_clip"] = video.clips[-1]["file"]
            state.update(controls=t+1, transitions=n*(t+1))
            if (t+1) % 25 == 0:
                torch.cuda.synchronize()
                assert not env.termination_manager.get_term("nan_state").any()
                contacts = int(env.sim.wp_data.nacon.numpy()[0])
                assert 0 <= contacts < env.sim.wp_data.contact.dist.shape[0]
                contact_peak = max(contacts, contact_peak)
                state["updates"].append(dict(control=t+1, transitions_per_second=n*25/(monotonic()-tick)))
                tick = monotonic()
                publish(folder, state)
                publish_seconds += monotonic()-tick
                if (t+1) % 100 == 0:
                    print("RIGHT_STARTUP_VALUE_SAMPLE", t+1, horizon, int(alive.sum()), flush=True)
                tick = monotonic()
        assert env.common_step_counter == horizon and env._sim_step_counter == spec["physical_substeps_authorized"]
        assert not alive.any() and video.clips[-1]["frames"] == 101 and video.clips[-1]["complete_clip"]
        assert initial == model_digest(actor)
        timing = dict(sampling_wall_seconds=monotonic()-begin, sampling_cpu_seconds=process_time()-cpu_start,
                      publish_seconds=publish_seconds, render_seconds=video.elapsed_seconds)
        result = {k: array.cpu().numpy() for k, array in data.items()}
        result.update(prefix_yaw=result["yaw"][:200].copy(), video_qpos=torch.stack(poses).cpu().numpy())
        return result, initial, timing, contact_peak
    finally:
        if video is not None:
            video.close()
        if camera is not None:
            camera.close()
        state.update(controls=env.common_step_counter, transitions=env.common_step_counter*n)
        env.close()


def run(spec):
    from mjlab.utils.torch import configure_torch_backends

    check_project()
    folder = ROOT / spec["record"]
    state = read(folder / "state.json")
    assert state["status"] == "starting" and not (folder / "execution-exit.txt").exists()
    watch = MemoryWatch(spec["memory_cap_fraction"])
    code = 1
    try:
        with patch("run_walk_right_startup_probe.PLAN", PLAN):
            require_observer(folder, PAGE, state, "shared-value")
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        deadline = monotonic() + spec["deadline_seconds"]
        with watch.active():
            data, actor_hash, timing, contacts = collect(spec, folder, state, watch, deadline)
            gc.collect()
            torch.cuda.empty_cache()
            state["purpose"] = "右转共享价值 · 保存首回合并复算目标"
            publish(folder, state)
            archive = folder / "first-episodes.npz"
            returns, valid, data["label"], split = targets(data, spec)
            start = monotonic()
            np.savez_compressed(archive, **data)
            timing["trace_archive_seconds"] = monotonic()-start
            write(folder / "split.json", split)
            assert split["labelled_worlds"] >= spec["min_delayed_worlds"], "Too few delayed starts"
            assert split["labelled_validation_worlds"] >= spec["min_delayed_validation_worlds"], "Too few validation delayed starts"
            start = monotonic()
            # Reuse the already verified fitter without rewriting its frozen historical source.
            with patch.multiple(shared_fit, publish=publish, PLAN=PLAN):
                score = shared_fit.fit(spec, folder, state, watch, deadline, data, returns, valid, split, actor_hash)
            timing["value_fit_wall_seconds"] = monotonic()-start
            watch.sample()
            watch.check()
        score.update(status="completed-native-right-turn-shared-value", protocol_sha256=sha(PLAN),
            first_episodes_sha256=sha(archive), actual_transitions=state["transitions"], valid_first_transitions=int(valid.sum()),
            physical_substeps=spec["physical_substeps_authorized"], native_first_episodes=spec["worlds"], timing=timing,
            clip_sha256=sha(folder / state["latest_clip"]), split_sha256=sha(folder / "split.json"),
            contact_peak_sampled=contacts, memory=watch.summary())
        write(folder / "analysis.json", score)
        state.update(status="completed", admitted=score["admitted"])
        code = 0
        print("RIGHT_STARTUP_VALUE_COMPLETE", score, flush=True)
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        write(folder / "gpu-telemetry.json", watch.rows)
        publish(folder, state)
        (folder / "execution-exit.txt").write_text(f"{code}\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(checked())


if __name__ == "__main__":
    main()
