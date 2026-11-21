"""Admit a new observed getup recipe using real CUDA initialization only.

The trainer is imported only after this run's local view has been observed.
Initialization reads back actor, critic, Adam and physical RNG state. This
entry never calls env.step or optimizer.update; a failure closes this run.
It reuses the tested supervisor and keeps closed training sources unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import observed_getup_entry as entry

ROOT = entry.ROOT
ENTRY = Path(__file__).resolve()


def command(path: Path, role: str) -> list[str]:
    return [sys.executable, "-B", str(ENTRY), "--mode", "worker",
            "--protocol", str(path), "--arm", role]


def checked(path: Path, role: str) -> tuple[dict, dict]:
    if not sys.platform.startswith("linux") or not path.is_relative_to(ROOT / "configs/experiments"):
        raise ValueError("Use a project protocol and the existing WSL interpreter")
    spec = entry.read(path)
    if (spec.get("status") != "frozen-observed-cuda-zero" or spec.get("learning_authorized") is not False
            or spec.get("automatic_retry") is not False or spec.get("physical_steps") != 0):
        raise ValueError("Only a new zero-step protocol is allowed")
    draft_path = entry.inside(spec["training_draft"])
    if entry.sha(draft_path) != spec["training_draft_sha256"]:
        raise ValueError("Training draft changed")
    draft = entry.read(draft_path)
    arm = draft["arms"][role]
    zero_id = spec["zero_run_ids"][role]
    if not entry.supervisor._authorized(path, entry.sha(path), spec["stage"], zero_id,
                                        command(path, role), before_start=True):
        raise ValueError("Project is paused or this exact zero protocol is not admitted")
    sources = draft["source_sha256"]
    parent = entry.read(entry.inside(draft["parent_protocol"]))["training"]["candidate"]
    required = entry.REQUIRED_SOURCES | set(parent["source_sha256"]) | {
        draft["parent_protocol"], ENTRY.relative_to(ROOT).as_posix()}
    if not required.issubset(sources) or sources != spec["source_sha256"]:
        raise ValueError("Incomplete zero/training source manifest")
    for name, expected in sources.items():
        if entry.sha(entry.inside(name)) != expected:
            raise ValueError("Source changed: " + name)
    if any(sources[name] != digest for name, digest in parent["source_sha256"].items()):
        raise ValueError("Inherited source differs")
    linux_root = (Path.home() / "XGO-Duck").resolve()
    for field, base in (("training_source_sha256", linux_root / "workspace/xgoduck_rl"),
                        ("library_source_sha256", Path("/"))):
        for name, expected in parent[field].items():
            source = (base / name).resolve()
            if not source.is_relative_to(linux_root) or entry.sha(source) != expected:
                raise ValueError("WSL dependency differs: " + name)
    if (arm["num_envs"] <= 0 or arm["num_envs"] % 4
            or spec["training_settings_sha256"][role] != entry.settings_digest(draft, role)):
        raise ValueError("Zero-step settings differ from the declared training")
    record = entry.inside(draft["zero_step_record"]) / role
    report = entry.inside(spec["coordinator"]) / role
    if (not report.is_relative_to(ROOT / "records/reports") or not record.is_relative_to(ROOT / "records")
            or record.exists() or (report / "entry-result.json").exists()
            or (ROOT / "records/experiments" / (zero_id + "-native-state.json")).exists()):
        raise ValueError("Run already exists or paths are invalid")
    entry.supervisor._seconds(tuple(spec["supervision"].values()))
    if spec["supervision"]["startup_seconds"] <= spec["supervision"]["viewing_seconds"]:
        raise ValueError("Startup must include viewing time")
    return spec, draft


def zero_worker(spec: dict, draft: dict, role: str, path: Path) -> dict:
    run_id = spec["zero_run_ids"][role]
    protocol_sha = entry.sha(path)
    report = entry.inside(spec["coordinator"]) / role
    record = entry.inside(draft["zero_step_record"]) / role
    report.mkdir(parents=True, exist_ok=True)

    def allowed():
        return entry.supervisor._authorized(path, protocol_sha, spec["stage"], run_id,
                                            command(path, role), before_start=False)

    observer = entry.NativePPOObservation(run_id, report, allowed,
        environments=draft["arms"][role]["num_envs"], planned_updates=draft["arms"][role]["training_updates"],
        stall_seconds=spec["supervision"]["stall_seconds"], purpose="实际 CUDA 零步检查；本次不采样、不学习")
    result = {"status": "not-started", "run_id": run_id, "physical_substeps": 0,
              "learning_updates": 0, "protocol_sha256": protocol_sha, "automatic_retry": False}
    try:
        observer.start()
        print("WAITING_FOR_CURRENT_VIEW", observer.view, flush=True)
        deadline = time.monotonic() + spec["supervision"]["viewing_seconds"]
        while not entry.viewing_confirmed(report / "viewing-confirmation.json", observer, protocol_sha):
            observer.require_active()
            if time.monotonic() >= deadline:
                raise TimeoutError("Current zero-step view was not confirmed")
            time.sleep(.2)
        observer.require_active()
        # This import boundary prevents hidden GPU work while waiting for view.
        from functools import partial
        from unittest.mock import patch
        import torch
        from mjlab.utils.torch import configure_torch_backends
        import run_local_getup_ppo as worker

        p = worker.arm_plan(draft, role)
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        with observer.lock:
            observer.state.update(status="gpu-zero-step", purpose="CUDA 场景与保存点读回；物理步 0，学习更新 0")
        optimizer = partial(worker.AdaptiveGetupPPO, adaptive_settings=draft["adaptive_settings"],
            reward_scale=p["reward_scale"], replay_limits=draft["limits_rad"],
            step_scales=draft["step_scales"], check_active=observer.require_active)

        def inspect(plan, output, scene):
            observer.require_active()
            output.mkdir(parents=True)
            env, obs, learner, _ = worker.rollout.initialize(plan, output, scene)
            try:
                if (not obs["actor"].is_cuda or not next(learner.actor.parameters()).is_cuda
                        or env.common_step_counter != 0 or env._sim_step_counter != 0):
                    raise ValueError("Expected actual CUDA with zero physical steps")
                torch.cuda.synchronize()
                observer.require_active()
                return worker.gpu_sample()
            finally:
                env.close()

        started = time.monotonic()
        with patch.object(worker.retention, "RecoveryRetentionPPO", optimizer):
            telemetry = worker.retention.run_with_retention(inspect, p, record, worker.SCENE)
        proof = entry.read(record / "parent-readback.json")
        if not (proof["matched_networks_and_adam"] and proof["physical_rng_unchanged"]
                and proof["physical_substeps"] == 0):
            raise ValueError("Parent readback failed")
        fraction = telemetry["used_mib"] / telemetry["total_mib"]
        result.update(status="passed" if fraction <= draft["memory_fraction_limit"] else "memory-ceiling",
            envs=p["num_envs"], telemetry=telemetry, observed_gpu_memory_fraction=fraction,
            seconds=time.monotonic()-started, parent=proof, training_run_id=p["run_id"],
            training_settings_sha256=entry.settings_digest(draft, role),
            training_source_sha256=draft["source_sha256"])
        worker.write(record / "result.json", result)
    except BaseException as exc:
        result.update(status="stopped-review-required", reason=type(exc).__name__ + ": " + str(exc))
    finally:
        try:
            observer.close(result)
        except BaseException as exc:
            result.update(status="stopped-review-required", cleanup_error=repr(exc))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (report / "entry-result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("check", "supervise", "worker"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    try:
        path = args.protocol.resolve()
        spec, draft = checked(path, args.arm)
        if args.mode == "check":
            print(json.dumps({"status": "admitted-read-only", "cuda_started": False,
                              "physical_substeps": 0, "learning_updates": 0}))
            return 0
        if args.mode == "worker":
            expected = command(path, args.arm)
            expected[expected.index("worker")] = "supervise"
            actual = (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if actual != expected:
                raise ValueError("Worker requires this entry's supervisor")
            result = zero_worker(spec, draft, args.arm, path)
        else:
            report = entry.inside(spec["coordinator"]) / args.arm
            report.mkdir(parents=True, exist_ok=True)
            run_id = spec["zero_run_ids"][args.arm]
            timing = {k: v for k, v in spec["supervision"].items() if k != "viewing_seconds"}
            result = entry.supervisor.run_supervised(command(path, args.arm), cwd=ROOT,
                log_path=report / "execution.log", receipt_path=report / "supervisor-result.json",
                source=ROOT / "records/experiments" / (run_id + "-native-state.json"), run_id=run_id,
                protocol=path, protocol_sha256=entry.sha(path), stage=spec["stage"], **timing)
            if result["status"] == "child-exited" and result["exit_code"] == 0:
                result = entry.read(report / "entry-result.json")
            view = entry.finalize_view(report, run_id, result)
            if view["status"] != "saved" and result["status"] == "passed":
                result["status"] = "observation-review-required"
        print(json.dumps({k: v for k, v in result.items() if k != "training_source_sha256"}, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 3
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
