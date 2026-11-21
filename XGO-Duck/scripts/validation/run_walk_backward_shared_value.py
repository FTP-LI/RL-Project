"""Collect complete native first episodes and fit one shared backward critic."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import gc
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
from walk_backward_ppo import action_and_log_prob, actor_mean
from walk_backward_value import SharedBackwardValue, episode_returns, world_split
import training_live_page as live


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_shared_value.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
HEADROOM = PLAN.with_stem("common_walk_backward_ppo_headroom")
PAGE = ROOT / "records/walk-backward-shared-value-live.html"


def publish(folder, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", PAGE):
        live.publish(folder, state)


def freeze():
    check_project()
    assert not PLAN.exists() and not PAGE.exists()
    spec = read(DRAFT)
    assert spec["status"] == "draft-native-backward-value-no-execution"
    assert not any(spec[key] for key in ("physical_substeps_authorized", "critic_updates_authorized",
                                        "actor_updates_authorized", "heldout_used", "hardware_io"))
    checksums = dict(spec[key] for key in ("headroom_verification", "headroom_analysis"))
    verification = read(ROOT / spec["headroom_verification"][0])
    proof = read(ROOT / spec["headroom_analysis"][0])
    headroom = read(HEADROOM)
    assert verification["status"] == "passed-independent-native-ppo-headroom" and verification["admitted"]
    assert verification["analysis_sha256"] == spec["headroom_analysis"][1]
    assert proof["protocol_sha256"] == sha(HEADROOM)
    assert headroom["policy_model_sha256"] == spec["official_actor_sha256"]
    assert verification["worlds"] == spec["worlds"] and headroom["initial_std"] == spec["initial_std"]
    checksums.update(headroom["source_sha256"])
    for path in (DRAFT, HEADROOM, ENTRY, ENTRY.with_name("verify_walk_backward_shared_value.py"),
                 ROOT / "scripts/training/walk_backward_value.py"):
        checksums[path.relative_to(ROOT).as_posix()] = sha(path)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT / "records/experiments" / f"duck_walk_shared-value_reference_s{spec['seed']}_{stamp}"
    assert not any(f"_s{spec['seed']}_" in path.name for path in folder.parent.iterdir())
    spec.update(status="frozen-native-backward-shared-value", record=folder.relative_to(ROOT).as_posix(),
                source_sha256=checksums, library_source_sha256=headroom["library_source_sha256"],
                physical_substeps_authorized=spec["maximum_batched_substeps"],
                critic_updates_authorized=spec["maximum_value_updates"], deadline_seconds=480,
                input_scale_floor=0.001, gradient_norm_max=1.0, report_interval_updates=10,
                split_seed=spec["seed"]+20000, shuffle_seed=spec["seed"]+30000)
    assert spec["worlds"] * spec["first_episode_maximum_controls"] == spec["maximum_transitions"]
    for path, digest in checksums.items():
        assert sha(ROOT / path) == digest, path
    folder.mkdir()
    write(PLAN, spec)
    write(folder / "protocol.json", spec)
    (folder / "README.md").write_text(
        "# 后退共享价值拟合\n\n[冻结方案](../../../configs/experiments/common_walk_backward_shared_value.json)"
        " · [运行观察](../../walk-backward-shared-value-live.html)\n",
        encoding="utf-8",
    )
    publish(folder, dict(status="starting", run_id=folder.name, purpose="后退共享价值 · 等待固定策略采样",
                         controls=0, maximum_controls=spec["first_episode_maximum_controls"],
                         transitions=0, learning_updates=0, updates=[]))
    print("BACKWARD_SHARED_VALUE_FROZEN", spec["record"], sha(PLAN), flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-backward-shared-value"
    assert not any(spec[key] for key in ("actor_updates_authorized", "heldout_used", "hardware_io"))
    for path, digest in spec["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in spec["library_source_sha256"].items():
        assert sha(Path(path)) == digest, path
    return spec


@torch.no_grad()
def collect(spec, folder, state, watch, deadline):
    from mjlab.envs import ManagerBasedRlEnv

    parent, focus = inputs()
    n, horizon, stride = (spec[key] for key in ("worlds", "first_episode_maximum_controls", "critic_observation_stride"))
    label_control = 2 * focus["block_controls"]
    cfg, plan, internal = configuration(parent, n, spec["seed"])
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    camera = video = None
    try:
        obs, _ = env.reset(seed=spec["seed"])
        pipe, actor, _, interface = check_inputs(env, plan, internal, focus, obs)
        initial = model_digest(actor)
        assert env.common_step_counter == env._sim_step_counter == 0 and horizon == env.max_episode_length
        watch.sample()
        data = dict(reward=torch.empty((horizon, n), device=env.device),
                    done=torch.empty((horizon, n), dtype=torch.bool, device=env.device),
                    prefix_velocity=torch.empty((label_control, n), device=env.device),
                    critic_obs=torch.empty((horizon//stride, n, 77), device=env.device))
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        rng = torch.Generator(device=env.device).manual_seed(spec["seed"]+10000)
        write(folder / "startup.json", dict(**interface, actor_sha256=initial, physical_substeps=0))
        camera = RecordingSnapshot(env, folder)
        camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.1, 90, -20
        video = ActionVideo(camera, "walk", folder)
        video.sample()
        state.update(status="running", purpose="后退共享价值 · 固定策略采样", frame=video.image["file"])
        publish(folder, state)
        tick = monotonic()
        for t in range(horizon):
            watch.check()
            assert monotonic() < deadline
            actor_obs, critic_obs = pipe.inputs(obs)
            if t % stride == 0:
                data["critic_obs"][t//stride].copy_(critic_obs)
            action, _ = action_and_log_prob(actor_mean(actor, actor_obs), spec["initial_std"], rng)
            step = pipe.advance(action, focused=True)
            obs = step.observations
            ended = alive & (step.terminated | step.timed_out)
            data["reward"][t].copy_(torch.where(alive, step.reward, 0))
            data["done"][t].copy_(ended)
            alive &= ~ended
            if t < label_control:
                data["prefix_velocity"][t].copy_(step.forward_velocity)
            if t + 1 == label_control:
                data["label"] = alive & pipe.focus.tagged
            if video.writer is not None:
                video.sample()
            if video.clips and video.clips[-1]["complete_clip"]:
                state["latest_clip"] = video.clips[-1]["file"]
            state.update(controls=t+1, transitions=n*(t+1))
            if (t + 1) % 25 == 0:
                torch.cuda.synchronize()
                state["updates"].append(dict(transitions_per_second=n*25/(monotonic()-tick)))
                publish(folder, state)
                if (t + 1) % 100 == 0:
                    print("BACKWARD_VALUE_SAMPLE", t+1, horizon, int(alive.sum()), flush=True)
                tick = monotonic()
        assert env.common_step_counter == horizon and env._sim_step_counter == spec["maximum_batched_substeps"]
        assert not alive.any() and video.clips[-1]["frames"] == 101
        assert initial == model_digest(actor)
        return {key: tensor.cpu().numpy() for key, tensor in data.items()}, initial
    finally:
        if video is not None:
            video.close()
        if camera is not None:
            camera.close()
        state.update(controls=env.common_step_counter, transitions=n*env.common_step_counter)
        env.close()


def targets(data, spec):
    reward, done = data["reward"], data["done"]
    returns, valid = episode_returns(reward, done, spec["gamma"])
    _, focus = inputs()
    block = focus["block_controls"]
    speed = data["prefix_velocity"].reshape(2, block, -1).mean(axis=1, dtype=np.float64)
    label = valid[2*block] & (np.abs(speed).max(axis=0) <= focus["maximum_absolute_block_speed_m_s"])
    np.testing.assert_array_equal(data["label"], label)
    train, validation = world_split(label, spec["validation_fraction"], spec["split_seed"])
    assert len(set(train) & set(validation)) == 0 and len(train)+len(validation) == len(label)
    split = dict(train_worlds=train.tolist(), validation_worlds=validation.tolist(), labelled_worlds=int(label.sum()),
                 labelled_validation_worlds=int(label[validation].sum()))
    return returns, valid, split


def fit(spec, folder, state, watch, deadline, data, returns, valid, split, actor_hash):
    stride = spec["critic_observation_stride"]
    label = data["label"]
    x = data["critic_obs"]
    y, kept = returns[::stride], valid[::stride]
    assert x.shape == (*y.shape, 77)

    def tensors(ids):
        use = kept[:, ids]
        return (torch.as_tensor(x[:, ids][use], device="cuda:0"), torch.as_tensor(y[:, ids][use], device="cuda:0"),
                torch.as_tensor(np.broadcast_to(label[ids], use.shape)[use], device="cuda:0"))

    train_x, train_y, _ = tensors(split["train_worlds"])
    val_x, val_y, val_label = tensors(split["validation_worlds"])
    model = SharedBackwardValue(spec["seed"], "cuda:0")
    model.input_center.copy_(train_x.mean(dim=0))
    model.input_scale.copy_(train_x.std(dim=0, unbiased=False).clamp_min(spec["input_scale_floor"]))
    model.reward_scale.copy_(train_y.square().mean().sqrt())
    assert model.reward_scale > 0 and torch.isfinite(train_x).all() and torch.isfinite(val_x).all()
    train_y /= model.reward_scale
    val_y /= model.reward_scale
    initial_critic = model_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["value_learning_rate"])
    rng = torch.Generator(device="cuda:0").manual_seed(spec["shuffle_seed"])
    rows = []

    @torch.no_grad()
    def evaluate(update):
        pred = torch.cat([model.normalized(batch).squeeze(-1) for batch in val_x.split(4096)])
        error = (pred-val_y).double()

        def metrics(e, y):
            return dict(normalized_mse=float(e.square().mean()),
                        explained_variance=float(1-e.var(unbiased=False)/y.double().var(unbiased=False)),
                        bias=float(e.mean()), samples=len(y))

        row = dict(update=update, validation=metrics(error, val_y),
                   stalled=metrics(error[val_label], val_y[val_label]),
                   other=metrics(error[~val_label], val_y[~val_label]))
        rows.append(row)
        write(folder / "value-metrics.json", rows)
        state["purpose"] = (f"共享价值拟合 · {update}/{spec['maximum_value_updates']} · "
                            f"验证 EV {row['validation']['explained_variance']:.3f} · MSE {row['validation']['normalized_mse']:.3f}")
        publish(folder, state)
        print("BACKWARD_VALUE_FIT", row, flush=True)

    watch.sample()
    evaluate(0)
    for k in range(1, spec["critic_updates_authorized"]+1):
        watch.check()
        assert monotonic() < deadline
        idx = torch.randint(len(train_y), (spec["value_minibatch_size"],), device=train_y.device, generator=rng)
        loss = (model.normalized(train_x[idx]).squeeze(-1)-train_y[idx]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_norm_max"])
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        state["learning_updates"] = k
        if k % spec["report_interval_updates"] == 0:
            evaluate(k)
    assert rows[-1]["update"] == state["learning_updates"]
    score, gate = rows[-1]["validation"], spec["validation"]
    admitted = score["explained_variance"] >= gate["explained_variance_min"] and score["normalized_mse"] <= gate["normalized_mse_max"]
    checkpoint = folder / "shared-value.pt"
    final_critic = model_digest(model)
    assert final_critic != initial_critic
    torch.save(dict(critic_state_dict=model.state_dict(), critic_optimizer_state_dict=optimizer.state_dict(),
                    critic_updates=state["learning_updates"], actor_updates=0, actor_sha256=actor_hash,
                    protocol_sha256=sha(PLAN), split_sha256=sha(folder / "split.json"), admitted=admitted),
               checkpoint)
    return dict(admitted=admitted, checkpoint_sha256=sha(checkpoint), initial_critic_sha256=initial_critic,
                final_critic_sha256=final_critic, actor_before_sha256=actor_hash, actor_after_sha256=actor_hash,
                critic_updates=state["learning_updates"], actor_updates=0, result=rows[-1])


def run(spec):
    from mjlab.utils.torch import configure_torch_backends

    check_project()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    folder = ROOT / spec["record"]
    state = read(folder / "state.json")
    assert state["status"] == "starting"
    watch = MemoryWatch(spec["memory_cap_fraction"])
    deadline = monotonic() + spec["deadline_seconds"]
    try:
        with watch.active():
            data, actor = collect(spec, folder, state, watch, deadline)
            gc.collect()
            torch.cuda.empty_cache()
            state["purpose"] = "后退共享价值 · 保存首回合并核对回报"
            publish(folder, state)
            archive = folder / "first-episodes.npz"
            returns, valid, split = targets(data, spec)
            np.savez_compressed(archive, **data)
            write(folder / "split.json", split)
            assert split["labelled_worlds"] >= spec["minimum_labelled_first_episodes"], "Stall coverage below minimum"
            assert split["labelled_validation_worlds"] >= spec["minimum_labelled_validation_worlds"], "Validation stall coverage below minimum"
            result = fit(spec, folder, state, watch, deadline, data, returns, valid, split, actor)
            watch.sample()
            watch.check()
        result.update(status="completed-native-backward-shared-value", protocol_sha256=sha(PLAN),
                      first_episodes_sha256=sha(archive), actual_transitions=state["transitions"],
                      clip_sha256=sha(folder / state["latest_clip"]), split_sha256=sha(folder / "split.json"),
                      memory=watch.summary(), native_first_episodes=spec["worlds"])
        write(folder / "analysis.json", result)
        state.update(status="completed", admitted=result["admitted"])
        print("BACKWARD_SHARED_VALUE_COMPLETE", result, flush=True)
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        write(folder / "gpu-telemetry.json", watch.rows)
        publish(folder, state)


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
