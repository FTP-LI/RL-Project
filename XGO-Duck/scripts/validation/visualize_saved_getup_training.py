"""Visualize two archived PPO runs without starting training or simulation.

The runs have equal transition budgets but different update boundaries.  Plot
the actual saved per-update values against cumulative transitions, and keep the
historical evaluation separate from the training curves.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from figure_style import apply_figure_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/historical-getup-training-comparison"
SOURCE = ROOT / "records/reports/getup-local-ppo-20261110T011523"
PLAN = ROOT / "configs/experiments/local_getup_ppo.json"
EVAL_PLAN = ROOT / "configs/experiments/local_getup_ppo_evaluation.json"
FIELDS = ("environment_reward_mean", "value_loss", "drift_rms", "net_transitions_per_second")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_evidence() -> tuple[list[dict], dict, dict]:
    plan, training, evaluation = read(PLAN), read(SOURCE / "training-verification.json"), read(SOURCE / "evaluation-analysis.json")
    if training["protocol_sha256"] != sha(PLAN) or evaluation["protocol_sha256"] != sha(EVAL_PLAN):
        raise ValueError("Historical protocol differs from the saved verification")
    evidence = []
    for role in ("control", "throughput"):
        spec, check = plan["arms"][role], training["arms"][role]
        folder = ROOT / check["record"]
        observation, receipt = read(folder / "observation.json"), read(folder / "online-result.json")
        log = folder / "learning-metrics.jsonl"
        if sha(folder / "observation.json") != receipt["observation_sha256"] or sha(log) != receipt["samples_sha256"]:
            raise ValueError("Historical observation or raw learning log changed")
        raw = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        rows = observation["rows"]
        if (observation["run_id"] != spec["run_id"] or observation["environments"] != spec["num_envs"]
                or observation["completed_updates"] != spec["training_updates"]
                or len(rows) != len(raw) or len(rows) != spec["training_updates"]
                or [row["update"] for row in rows] != list(range(1, len(rows) + 1))
                or any(any(row.get(key) != value for key, value in original.items())
                       for row, original in zip(rows, raw, strict=True))):
            raise ValueError("Saved update rows do not match the frozen run and raw log")
        if any(not math.isfinite(float(row[key])) for row in rows for key in FIELDS):
            raise ValueError("Nonfinite historical metric")
        transitions = [row["controls"] * spec["num_envs"] for row in rows]
        if (transitions != sorted(set(transitions)) or transitions[-1] != receipt["actual_transitions"]
                or transitions[-1] != plan["limits"][role]["transitions"]):
            raise ValueError("Saved transition counts are incomplete")
        video = ROOT / check["videos"][0]["path"]
        if sha(video) != check["videos"][0]["sha256"] or not check["videos"][0]["full_action"]:
            raise ValueError("Full historical training video changed or is missing")
        telemetry_path = folder / "gpu-telemetry.json"
        telemetry = read(telemetry_path)
        samples = telemetry["samples"]
        if (telemetry["failure"] is not None or not samples
                or any(not 0 <= float(sample["gpu_percent"]) <= 100 for sample in samples)
                or max(float(sample["used_mib"]) for sample in samples) != check["peak_mib"]):
            raise ValueError("Recorded whole-device GPU telemetry is incomplete")
        evidence.append({"role": role, "environments": spec["num_envs"], "run_id": spec["run_id"],
                         "record": relative(folder), "observation_sha256": sha(folder / "observation.json"),
                         "log_sha256": sha(log), "video": relative(video), "video_sha256": sha(video),
                         "updates": len(rows), "transitions": transitions[-1], "rows": rows,
                         "peak_mib": check["peak_mib"], "wall_seconds": check["wall_seconds"],
                         "total_net_transitions_per_second": check["mean_net_transitions_per_second"],
                         "gpu_telemetry_sha256": sha(telemetry_path), "gpu_samples": len(samples),
                         "gpu_utilization_percent_median": statistics.median(float(row["gpu_percent"]) for row in samples),
                         "gpu_utilization_scope": "Whole-device periodic samples including startup and other processes; not a process-specific measurement."})
    if evidence[0]["transitions"] != evidence[1]["transitions"]:
        raise ValueError("Historical arms have unequal budgets")
    if training["status"] != "passed-online-training-integrity" or evaluation["status"] != "passed-independent-verification":
        raise ValueError("Saved verification is incomplete")
    return evidence, plan, evaluation


def check_labels(fig) -> None:
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()
    for axis in fig.axes:
        for scale, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            low, high = sorted(limits)
            for tick in scale.get_major_ticks() + scale.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden.update((tick.label1, tick.label2))
    labels = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text)
              if t.get_visible() and t.get_text().strip() and t not in hidden]
    frames = [(spine, spine.get_window_extent(renderer)) for axis in fig.axes
              for spine in axis.spines.values() if spine.get_visible()]
    ticks = {axis: set(axis.get_xticklabels(which="both") + axis.get_yticklabels(which="both")) for axis in fig.axes}
    hits = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(labels)
            for b, bb in labels[i + 1:] if ba.overlaps(bb)]
    hits += [(t.get_text(), "axis") for t, box in labels for spine, edge in frames
             if box.overlaps(edge) and t not in ticks[spine.axes]]
    outside = [t.get_text() for t, box in labels if not fig.bbox.contains(box.x0, box.y0)
               or not fig.bbox.contains(box.x1, box.y1)]
    if hits or outside:
        raise ValueError(f"Chart text collision or overflow: {hits}, {outside}")


def build() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from PIL import Image

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font = None
    apply_figure_style(font=font, sizes=(11, 10, 9))
    import matplotlib.pyplot as plt

    evidence, plan, evaluation = load_evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / name).exists() for name in ("training-curves.png", "training-metrics.csv", "comparison.json")):
        raise FileExistsError("Keep the prior comparison; output already exists")
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 10.0), sharex=True, layout="constrained")
    titles = ("环境回报", "价值损失", "成功样本动作偏移", "运行中累计吞吐")
    labels = ("平均回报", "损失", "RMS / rad", "转移 / 秒")
    colors, markers = ("#005a9c", "#b65419"), ("o", "s")
    for arm, color, marker in zip(evidence, colors, markers, strict=True):
        rows = arm["rows"]
        x = [row["controls"] * arm["environments"] / 1_000_000 for row in rows]
        for axis, field in zip(axes, FIELDS, strict=True):
            axis.plot(x, [row[field] for row in rows], color=color, marker=marker, markersize=2.1,
                      linewidth=1.1, label=f"{arm['environments']:,} 个环境")
    for axis, title, label in zip(axes, titles, labels, strict=True):
        axis.set_title(title, loc="left")
        axis.set_ylabel(label)
        axis.margins(x=0.035, y=0.18)
    axes[2].axhline(plan["limits_rad"][0], linestyle="--", color="#666666", linewidth=.9)
    axes[2].text(.02, plan["limits_rad"][0] + .0007, "RMS 上限 0.05 rad",
                 transform=axes[2].get_yaxis_transform(), color="#555555", fontsize=9, va="bottom")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("累计训练转移（百万条）")
    fig.suptitle("历史起身训练指标", fontsize=12)
    check_labels(fig)
    chart = OUT / "training-curves.png"
    # The project bundles only the style helper.  Save without tight cropping
    # so axes' display bounds can be mapped directly to temporary QA crops.
    matplotlib.rcParams["savefig.bbox"] = None
    fig.savefig(chart, dpi=300)
    renderer = fig.canvas.get_renderer()
    canvas_height = fig.bbox.height
    scale = 300 / fig.dpi
    boxes = {}
    for index, axis in enumerate(axes):
        box = axis.get_tightbbox(renderer)
        boxes[str(index)] = (max(0, int(box.x0 * scale) - 12),
                             max(0, int((canvas_height - box.y1) * scale) - 12),
                             int(box.x1 * scale) + 12, int((canvas_height - box.y0) * scale) + 12)
    with Image.open(chart) as image:
        for key, box in boxes.items():
            image.crop(box).save(OUT / f"qa-panel-{key}.png")
    plt.close(fig)

    table = OUT / "training-metrics.csv"
    columns = ("role", "environments", "update", "transitions", *FIELDS, "policy_loss", "elapsed")
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for arm in evidence:
            for row in arm["rows"]:
                writer.writerow({"role": arm["role"], "environments": arm["environments"], "update": row["update"],
                                 "transitions": row["controls"] * arm["environments"],
                                 **{field: row[field] for field in FIELDS}, "policy_loss": row["policy_loss"], "elapsed": row["elapsed"]})
    selected = {}
    for role in ("fixed-weighted", "n1536", "n3072"):
        result = evaluation["arms"][role]
        selected[role] = {"combined_successes": result["combined_successes"],
                          "all_motion_gates_passed": result["all_motion_gates_passed"],
                          "actual_supine": result["timing"]["actual"]["supine"],
                          "actual_right_side": result["timing"]["actual"]["right-side"]}
    summary = {"status": "historical-data-only", "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "Two saved training runs and their previously verified evaluation; no new training, physics or hardware.",
               "protocol": relative(PLAN), "protocol_sha256": sha(PLAN),
               "training_verification": relative(SOURCE / "training-verification.json"),
               "training_verification_sha256": sha(SOURCE / "training-verification.json"),
               "evaluation": relative(SOURCE / "evaluation-analysis.json"),
               "evaluation_sha256": sha(SOURCE / "evaluation-analysis.json"), "script_sha256": sha(Path(__file__)),
               "arms": [{key: value for key, value in arm.items() if key != "rows"} for arm in evidence],
               "historical_evaluation": selected, "chart_sha256": sha(chart), "csv_sha256": sha(table),
               "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
               "plot_text_overlap_check": "passed", "visual_inspection": "pending"}
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "arms": summary["arms"], "chart_sha256": summary["chart_sha256"],
                      "panel_crops": sorted(boxes), "physical_steps": 0, "learning_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("This script has no run or training options")
    build()
