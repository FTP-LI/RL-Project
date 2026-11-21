"""Independent 128-episode roll evaluation for the airborne-support pilot.

The official, matched control and candidate ONNX files are evaluated under the
same original roll physics and seed.  The candidate reward is not used during
evaluation; only the policy file changes.  Full traces and an env-0 video are
kept for every role so the promotion gate can be audited later.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys

import onnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import evaluate_roll_reference as reference
from record_io import write
from roll_top_contact import adapt as adapt_model_home


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metadata(model):
    return {item.key: item.value for item in onnx.load(model).metadata_props}


def evaluate_role(base_plan, role, model_path, seed, stamp, base_config):
    model_path = Path(model_path)
    model_meta = metadata(model_path)
    role_plan = dict(base_plan)
    role_plan.update(
        status="frozen-before-simulation",
        role=role,
        seed=seed,
        episodes=128,
        model=str(model_path.relative_to(ROOT)),
        model_sha256=sha(model_path),
        learning_updates=0,
        hardware_io=False,
    )
    role_plan["source_sha256"] = {
        "scripts/validation/evaluate_roll_reference.py": sha(ROOT / "scripts/validation/evaluate_roll_reference.py"),
        "scripts/validation/evaluate_roll_airborne_support.py": sha(Path(__file__)),
        "scripts/validation/roll_metrics.py": sha(ROOT / "scripts/validation/roll_metrics.py"),
        "scripts/validation/evaluate_fixed_policy.py": sha(ROOT / "scripts/validation/evaluate_fixed_policy.py"),
        "scripts/training/record_io.py": sha(ROOT / "scripts/training/record_io.py"),
    }
    run_id = f"duck_roll_airborne-support_eval_{role}_s{seed}_{stamp}"
    folder = ROOT / "records" / "evaluations" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "videos").mkdir()
    write(folder / "protocol.json", role_plan)
    state = dict(
        status="running",
        run_id=run_id,
        role=role,
        path=str(folder),
        started_at=datetime.now(timezone.utc).isoformat(),
        model=str(model_path),
        model_sha256=role_plan["model_sha256"],
        seed=seed,
        episodes=128,
        learning_updates=0,
        hardware_io=False,
    )
    reference.MODEL = model_path
    def adapted_config(plan):
        cfg = base_config(plan)
        return adapt_model_home(cfg, model_meta, False)

    reference.config = adapted_config
    try:
        reference.evaluate(role_plan, folder, state)
        state["status"] = "completed"
        state["ended_at"] = datetime.now(timezone.utc).isoformat()
        write(folder / "evaluation.json", state)
        return state
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), ended_at=datetime.now(timezone.utc).isoformat())
        write(folder / "evaluation.json", state)
        raise


def main():
    assert os.isatty(1), "Use one visible terminal for the evaluation run"
    base = read(ROOT / "configs/experiments/roll_reference.json")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    roles = {
        "official": ROOT / "official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx",
        "control": ROOT / "artifacts/duck_roll_airborne-support_control_s1210_20261114T004321/policy.onnx",
        "candidate": ROOT / "artifacts/duck_roll_airborne-support_candidate_s1210_20261114T004321/policy.onnx",
    }
    states = {}
    base_config = reference.config
    for role, model in roles.items():
        states[role] = evaluate_role(base, role, model, 1211, stamp, base_config)
        result = states[role]["conditions"][0]
        print(f"ROLL_AIRBORNE_SUPPORT_EVAL {role}: {result['successes']}/{result['attempts']}", flush=True)

    control = states["control"]["conditions"][0]
    candidate = states["candidate"]["conditions"][0]
    control_rows = control["episodes"]
    candidate_rows = candidate["episodes"]
    new_penetration = any(
        row["head_penetration_max_m"] > 0.005 for row in candidate_rows
    ) and not any(row["head_penetration_max_m"] > 0.005 for row in control_rows)
    new_joint_limits = any(row["joint_violation_samples"] > 0 for row in candidate_rows) and not any(
        row["joint_violation_samples"] > 0 for row in control_rows
    )
    gate = dict(
        status="passed" if (
            candidate["successes"] >= 116
            and candidate["successes"] >= control["successes"] + 4
            and not new_penetration
            and not new_joint_limits
        ) else "failed",
        candidate_successes=candidate["successes"],
        control_successes=control["successes"],
        minimum_successes=116,
        minimum_additional_successes=4,
        new_head_penetration=new_penetration,
        new_joint_limit_failures=new_joint_limits,
        continue_learning=False,
        continue_router=False,
    )
    result = dict(
        status="completed-independent-evaluation",
        protocol="configs/experiments/roll_airborne_support_evaluation.json",
        seed=1211,
        episodes=128,
        roles={
            role: {
                "run_id": states[role]["run_id"],
                "successes": states[role]["conditions"][0]["successes"],
                "evaluation": f"records/evaluations/{states[role]['run_id']}/evaluation.json",
                "trace": f"records/evaluations/{states[role]['run_id']}/forward-roll-trace.npz",
                "video": f"records/evaluations/{states[role]['run_id']}/videos/forward-roll_env0.mp4",
            }
            for role in states
        },
        gate=gate,
        physical_steps=sum(450 * 128 for _ in states),
        learning_updates=0,
        hardware_io=False,
    )
    out = ROOT / "records/reports/roll-airborne-support"
    write(out / "evaluation.json", result)
    write(out / "evaluation-gate.json", gate)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
