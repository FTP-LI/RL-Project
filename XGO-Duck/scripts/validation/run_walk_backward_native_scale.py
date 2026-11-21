"""Time the same native backward path in two opposing CUDA scale passes."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_local_getup_ppo import check_project
import run_walk_backward_native_probe as probe
import training_live_page as live

SOURCE = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_native_scale.json"
PAGE = ROOT / "records/walk-backward-native-scale-live.html"
SIZES = (1024, 2048, 4096, 6144)
SEED = 1134


def rel(path):
    return path.relative_to(ROOT).as_posix()


def start_state(path, worlds):
    return dict(status="starting", run_id=f"{path.parent.name}_{path.name}", purpose=f"原生后退采样 · {worlds:,} 并行环境",
                controls=0, maximum_controls=384, transitions=0, learning_updates=0, updates=[],
                updated_at=datetime.now(timezone.utc).isoformat())


def freeze():
    assert not PLAN.exists() and not PAGE.exists()
    check_project()
    prior = probe.checked()
    reference = ROOT / prior["record"]
    witness = reference / "windows-view-verification.json"
    verification = reference / "verification.json"
    proof = read(verification)
    assert proof["status"] == "passed-completed-native-probe-readback"
    assert proof["protocol_sha256"] == sha(probe.PLAN)
    assert proof["observer_sha256"] == sha(witness) and read(witness)["status"] == "passed"
    assert proof["analysis_sha256"] == sha(reference / "analysis.json")
    assert prior["seed"] != SEED and prior["warmup_controls"] + prior["timed_controls"] == 384
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    record = ROOT / "records/benchmarks" / f"duck_walk_native-scale_reference_s{SEED}_{stamp}"
    record.mkdir()
    trials = []
    for pass_number, order in enumerate((SIZES, SIZES[::-1]), 1):
        for size in order:
            folder = record / f"pass-{pass_number}-n{size}"
            folder.mkdir()
            subplan = dict(prior, record=rel(folder), page=rel(PAGE), worlds=size, seed=SEED,
                           maximum_transitions=size * 384)
            local = folder / "protocol.json"
            write(local, subplan)
            write(folder / "state.json", start_state(folder, size))
            (folder / "README.md").write_text(
                f"# {size:,} 环境 · 第 {pass_number} 遍\n\n固定官方策略，64 步热身和 320 步计时。\n\n"
                "[本批说明](../README.md) · [参数](protocol.json)\n", encoding="utf-8",
            )
            trials.append(dict(index=len(trials), pass_number=pass_number, worlds=size, protocol=rel(local),
                               protocol_sha256=sha(local)))
    spec = dict(status="frozen-native-backward-scale", record=rel(record), page=rel(PAGE), seed=SEED, sizes=SIZES,
                warmup_controls=prior["warmup_controls"], timed_controls=prior["timed_controls"], repeats=2,
                memory_cap_fraction=prior["memory_cap_fraction"], minimum_paired_gain=0.05, trials=trials,
                maximum_transitions=sum(row["worlds"] * 384 for row in trials), maximum_batched_substeps=1920*len(trials),
                source_sha256={rel(p): sha(p) for p in (SOURCE, probe.PLAN, verification, witness, reference / "analysis.json")},
                learning_updates=0, hardware_io=False, heldout_used=False,
                scope="Same official actor, native reward, reference value network, storage and recording. Sequential GPU trials.")
    write(PLAN, spec)
    write(record / "protocol.json", spec)
    (record / "README.md").write_text(
        "# 原生后退并行规模\n\n两遍以相反顺序运行同一采样和录像链路。选择更大档位时，两遍净吞吐都需提高至少 5%。"
        "若显存触及冻结上限，停止本档并跳过其同档复测和更大档。\n\n"
        f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{PAGE.name})\n",
        encoding="utf-8",
    )
    first = (ROOT / trials[0]["protocol"]).parent
    with patch.object(live, "PAGE", PAGE):
        live.publish(first, start_state(first, trials[0]["worlds"]))
    print("NATIVE_BACKWARD_SCALE_FROZEN", rel(record), spec["maximum_transitions"], sha(PLAN), flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-backward-scale" and not spec["learning_updates"]
    for path, digest in spec["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for row in spec["trials"]:
        assert sha(ROOT / row["protocol"]) == row["protocol_sha256"]
    return spec


def trial(index):
    spec = checked()
    row = spec["trials"][index]
    path = ROOT / row["protocol"]
    with patch.multiple(probe, PLAN=path, PAGE=PAGE):
        probe.sample()


def summarize(spec, entries):
    good = {(row["worlds"], row["pass_number"]): row for row in entries if row["status"] == "passed"}
    eligible = []
    selected = None
    for size in spec["sizes"]:
        pair = [good.get((size, repeat)) for repeat in (1, 2)]
        if any(row is None for row in pair):
            continue
        rates = [row["net_transitions_per_second"] for row in pair]
        gain = [new / old - 1 for new, old in zip(rates, selected["rates"])] if selected else None
        admitted = gain is None or min(gain) >= spec["minimum_paired_gain"]
        row = dict(worlds=size, rates=rates, relative_to_current_selection=gain, selected_at_comparison=admitted)
        eligible.append(row)
        if admitted:
            selected = row
    return dict(status="passed-scale-comparison" if selected else "review-required",
                protocol_sha256=sha(PLAN), selected=selected, comparisons=eligible, trials=entries,
                measured_transitions=sum(row.get("measured_transitions", 0) for row in entries),
                actual_transitions=sum(row.get("actual_transitions", 0) for row in entries),
                learning_updates=0, heldout_used=False)


def run():
    spec = checked()
    record = ROOT / spec["record"]
    assert not (record / "progress.json").exists()
    lock_path = Path.home() / "XGO-Duck/training.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries, cap_size = [], None
        for row in spec["trials"]:
            check_project()
            folder = (ROOT / row["protocol"]).parent
            entry = dict(row, record=rel(folder))
            if cap_size is not None and row["worlds"] >= cap_size:
                entry.update(status="skipped-by-earlier-capacity-stop")
            else:
                command = [sys.executable, "-B", "-u", str(SOURCE), "--trial", str(row["index"])]
                (folder / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
                print("NATIVE_SCALE_TRIAL", row["pass_number"], row["worlds"], flush=True)
                try:
                    with (folder / "execution.log").open("xb") as log:
                        finished = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                                  env=dict(os.environ, MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1"),
                                                  timeout=320, check=False)
                    exit_code, timeout_reason = finished.returncode, ""
                except subprocess.TimeoutExpired:
                    exit_code, timeout_reason = 124, "Trial reached its process wall-time cap"
                (folder / "execution-exit.txt").write_text(f"{exit_code}\n", encoding="ascii")
                state = read(folder / "state.json")
                entry["actual_transitions"] = state["transitions"]
                if exit_code == 0:
                    result = read(folder / "analysis.json")
                    gpu = read(folder / "gpu-telemetry.json")
                    assert result["protocol_sha256"] == row["protocol_sha256"]
                    assert state["status"] == "completed" and result["learning_updates"] == 0
                    assert result["actual_transitions"] == row["worlds"] * 384
                    for key in ("net_transitions_per_second", "measured_transitions", "timing", "memory"):
                        entry[key] = result[key]
                    entry.update(status="passed", analysis_sha256=sha(folder / "analysis.json"),
                                 median_sampled_gpu_utilization=statistics.median(item["gpu_percent"] for item in gpu),
                                 process_cpu_core_equivalent=result["timing"]["process_cpu_seconds"]/result["timing"]["wall_seconds"])
                else:
                    reason = timeout_reason or state.get("reason", f"Child exit {exit_code}")
                    capacity = "Observed GPU memory ceiling" in reason or "out of memory" in reason.lower()
                    entry.update(status="capacity-stop" if capacity else "stopped-review-required", reason=reason)
                    if timeout_reason:
                        state.update(status="stopped-review-required", reason=reason,
                                     updated_at=datetime.now(timezone.utc).isoformat())
                        with patch.object(live, "PAGE", PAGE):
                            live.publish(folder, state)
                    if capacity:
                        cap_size = row["worlds"]
            entries.append(entry)
            write(record / "progress.json", dict(status="running", trials=entries, total_trials=len(spec["trials"])))
            print("NATIVE_SCALE_RESULT", row["pass_number"], row["worlds"], entry["status"],
                  entry.get("net_transitions_per_second"), flush=True)
            if entry["status"] == "stopped-review-required":
                raise RuntimeError(f"Native scale trial needs review: {folder}")
        result = summarize(spec, entries)
        write(record / "summary.json", result)
        write(record / "progress.json", dict(status="completed", trials=entries, total_trials=len(spec["trials"])))
        print("NATIVE_SCALE_SELECTED", result["selected"], flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    args = parser.add_mutually_exclusive_group(required=True)
    args.add_argument("--freeze", action="store_true")
    args.add_argument("--run", action="store_true")
    args.add_argument("--trial", type=int)
    opts = parser.parse_args()
    if opts.freeze:
        freeze()
    elif opts.run:
        run()
    else:
        trial(opts.trial)


if __name__ == "__main__":
    main()
