"""Collector adapter for the S17-style simulator; no simulator is started here.

``make_guarded_collector`` is designed to replace only the historical
``collect`` function. The caller must independently validate source hashes,
authorize a frozen plan, inject the requested physical factors, and perform a
real zero-step check. This file does not modify the historical entry point.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import monotonic

import numpy as np

from guarded_roll_loop import run_guarded_loop
from roll_stand_walk_sequence import Sequence, score_forward


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_guarded_collector(base, may_continue):
    """Return a collector with the same call signature as ``base.collect``."""
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        robot = env.scene["robot"]
        joints, order = robot.find_joints(plan["walking"]["joint_names"], preserve_order=True)
        assert order == plan["walking"]["joint_names"]
        feet, sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        assert list(sites) == ["left_foot", "right_foot"]
        head = robot.find_bodies("jaw_soft")[0][0]
        tolerance = np.deg2rad(plan["roll"]["actual_joint_limit_tolerance_deg"])
        limit_np = base.cpu(limits)
        alive = np.ones(env.num_envs, dtype=bool)
        sequence = Sequence(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        snapshot = base.Snapshot(env, folder)
        video = base.ActionVideo(snapshot, "switch", folder, plan["maximum_controls"] * .02)
        states, controls, artifact_errors = [], [], []
        head_over = np.zeros(env.num_envs, dtype=bool)
        joint_over = np.zeros(env.num_envs, dtype=bool)
        nonfinite_samples = 0
        started = monotonic()
        view = {"status": "running", "run_id": plan["run_id"], "purpose": "有界连续仿真",
                "controls": 0, "maximum_controls": plan["maximum_controls"], "transitions": 0,
                "learning_updates": 0, "updates": []}
        log = folder / "execution.log"

        def publish(step, status="running", frame=None, reason=None):
            view.update(status=status, controls=step, transitions=step * plan["worlds"],
                        updated_at=base.datetime.now(base.TZ).isoformat())
            if frame:
                view["frame"] = frame
            if reason:
                view["stop_reason"] = reason
            if video.clips and video.clips[0].get("complete_clip"):
                view["latest_clip"] = video.clips[0]["file"]
            with base.patch.object(base.live, "PAGE", base.PAGE):
                base.live.publish(folder, view)

        def capture():
            return base.capture(env, endpoint, alive, joints, head, feet, limit_np, tolerance)

        def advance(step, current_sequence):
            nonlocal obs, alive
            observed = obs["actor"].clone()
            target, saved = policy.next(step, observed, current_sequence)
            obs, _, term, timeout, _ = env.step(target - sim_home)
            received = env.action_manager.get_term("joint_pos")._processed_actions
            base.torch.testing.assert_close(received, target, rtol=0, atol=1e-6)
            alive &= ~base.cpu(term | timeout)
            controls.append(dict(saved, observed=base.cpu(observed), terminated=base.cpu(term),
                                 truncated=base.cpu(timeout), actual_motor_target=base.cpu(received)))
            endpoint.capture()
            assert not endpoint.first_mismatches, endpoint.first_mismatches
            return capture()

        def record(step, frame, phase_valid):
            nonlocal nonfinite_samples
            active = np.asarray(frame["active"], dtype=bool)
            head_over[:] |= active & (np.asarray(frame["head_penetration_m"]) > plan["roll"]["criteria"]["head_penetration_max_m"])
            joint_over[:] |= active & ~np.asarray(frame["joint_envelope_ok"], dtype=bool)
            nonfinite_samples += sum(int(np.count_nonzero(~np.isfinite(np.asarray(value))))
                                     for value in frame.values() if np.asarray(value).dtype.kind in "fc")
            states.append(dict(frame, phase_next=sequence.phase.copy(),
                               guard_phase_valid=np.asarray(phase_valid, dtype=bool)))

        def observe(step, frame, decision):
            if step == 0:
                video.sample()
            elif video.writer is not None:
                if alive[0]:
                    video.sample()
                else:
                    video._sample_terminal()
            if step and step % 25 == 0 and monotonic() - started >= plan["maximum_sampling_wall_seconds"]:
                raise TimeoutError("guarded sampling wall budget exceeded")
            if step == 0 or step % 50 == 0 or decision["stop"]:
                elapsed = max(monotonic() - started, 1e-9)
                view["updates"].append({"control": step, "transitions_per_second": step * plan["worlds"] / elapsed})
                text = f"GUARDED_ROLL {step}/{plan['maximum_controls']} stop={decision['reason']}"
                print(text, flush=True)
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")
                image = snapshot.capture()
                publish(step, frame=image["file"])

        result = {"status": "stopped", "reason": "initialization-error", "last_recorded_control": -1}
        try:
            result = run_guarded_loop(plan, sequence, capture(), advance, record, observe, may_continue)
        except (Exception, KeyboardInterrupt) as exc:
            result = {"status": "stopped", "reason": "collector-error", "last_recorded_control": len(states) - 1,
                      "actual_physical_steps": "unknown-read-caller-counter", "error_type": type(exc).__name__}
        finally:
            for label, action in (
                ("video-close", video.close),
                ("trace-save", lambda: np.savez_compressed(folder / "trace.npz", **{key: np.asarray([row[key] for row in states]) for key in states[0]}) if states else None),
                ("commands-save", lambda: np.savez_compressed(folder / "commands.npz", **{key: np.asarray([row[key] for row in controls]) for key in controls[0]}) if controls else None),
                ("endpoint-save", lambda: endpoint.save(plan["maximum_controls"] + 1)),
                ("snapshot-close", snapshot.close),
            ):
                try:
                    action()
                except (Exception, KeyboardInterrupt) as exc:
                    artifact_errors.append({"artifact": label, "error_type": type(exc).__name__})

        recorded = result["last_recorded_control"]
        state_count_valid = len(states) == len(controls) + 1 and recorded == len(controls)
        summary = {
            "trace_valid": bool(state_count_valid and all(bool(row["guard_phase_valid"]) for row in states)),
            "nonfinite_samples": int(nonfinite_samples), "head_over_limit_worlds": int(head_over.sum()),
            "joint_over_limit_worlds": int(joint_over.sum()),
        }
        completed = (result["status"] == "completed" and recorded == plan["maximum_controls"] and
                     state_count_valid and not artifact_errors and summary["trace_valid"])
        if completed:
            try:
                arrays = {key: np.asarray([row[key] for row in states]) for key in states[0]}
                score = score_forward(arrays, sequence.forward_at, plan)
                base.write(folder / "raw-score.json", {"roll_switch_at": sequence.stand_at.tolist(),
                           "forward_at": sequence.forward_at.tolist(), "forward": score})
            except (Exception, KeyboardInterrupt) as exc:
                artifact_errors.append({"artifact": "raw-score", "error_type": type(exc).__name__})
                completed = False
        reason = (result["reason"] or "artifact-or-score-error") if not completed else None
        manifest = {
            "status": "collected-review-required" if completed else "guarded-stop-review-required",
            "stop_reason": reason, "guard_result": result, "last_recorded_control": recorded,
            "recorded_control_transitions": len(controls) * plan["worlds"],
            "actual_control_steps": getattr(env, "common_step_counter", None),
            "actual_physics_substeps": getattr(env, "_sim_step_counter", None),
            "saved_states": len(states), "saved_controls": len(controls),
            "safety_summary": summary, "artifact_errors": artifact_errors,
            "trace_sha256": sha(folder / "trace.npz") if (folder / "trace.npz").is_file() else None,
            "command_sha256": sha(folder / "commands.npz") if (folder / "commands.npz").is_file() else None,
            "learning_updates": 0, "hardware_io": False, "video": video.clips,
        }
        base.write(folder / "guarded-evaluation.json", manifest)
        if not completed:
            base.write(folder / "guarded-stop.json", manifest)
        try:
            publish(max(recorded, 0), status=manifest["status"], reason=reason)
        except (Exception, KeyboardInterrupt):
            # The stop and artifact manifest have already been persisted.
            pass
        return manifest

    return collect
