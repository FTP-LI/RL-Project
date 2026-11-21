"""Render a run's own saved data as local VS Code files, without a server.

This is an observation adapter, not a training launcher or a stop controller.
The caller must independently freeze its protocol and stop if this process
dies, its heartbeat is stale, or its health is not fresh. Historical mode is
explicitly labelled and must never be used as evidence of a running job.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RECORDS = (ROOT / "records").resolve()
sys.path.insert(0, str(ROOT / "scripts/validation"))

METRICS = (
    ("environment_reward_mean", "环境回报", "平均回报"),
    ("value_loss", "价值损失", "损失"),
    ("net_transitions_per_second", "采样速度", "转移 / 秒"),
    ("transitions_per_second", "采样速度", "转移 / 秒"),
)
ACTIVE = {"starting", "running", "training", "gpu-zero-step"}


def checked_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(RECORDS):
        raise ValueError("path must be inside project records")
    return path


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def metrics(state: dict) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows = state.get("rows", state.get("updates", []))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid metric rows")
    available = []
    for key, label, unit in METRICS:
        if key == "transitions_per_second" and any(x[0] == "net_transitions_per_second" for x in available):
            continue
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in values):
                raise ValueError(f"invalid metric: {key}")
            available.append((key, label, unit))
    return rows, available


def media_path(source: Path, value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("invalid media reference")
    path = (source.parent / value).resolve()
    if not path.is_relative_to(source.parent.resolve()) or not path.is_file():
        raise ValueError("media reference missing or outside the run folder")
    return path


def plot(output: Path, rows: list[dict], fields: list[tuple[str, str, str]], historical: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.text import Text
    from figure_style import apply_figure_style

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    font = None
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
    apply_figure_style(font=font, sizes=(10, 9, 8))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fields), 1, figsize=(7, 2.3 * len(fields)), squeeze=False, layout="constrained")
    fig.suptitle("历史运行指标" if historical else "运行指标")
    for axis, (key, label, unit) in zip(axes[:, 0], fields):
        points = [(index + 1, float(row[key])) for index, row in enumerate(rows) if row.get(key) is not None]
        axis.plot([point[0] for point in points], [point[1] for point in points],
                  color="#086b80", marker="o", markersize=2.5, linewidth=1.2)
        axis.set_title(label, loc="left")
        axis.set_ylabel(unit)
        axis.margins(x=0.04, y=0.12)
    axes[-1, 0].set_xlabel("已记录指标序号")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden_ticks = set()
    for axis in axes[:, 0]:
        for scale, bounds in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            low, high = sorted(bounds)
            for tick in scale.get_major_ticks() + scale.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden_ticks.update((tick.label1, tick.label2))
    labels = [(item, item.get_window_extent(renderer)) for item in fig.findobj(Text)
              if item.get_visible() and item.get_text().strip() and item not in hidden_ticks]
    frames = [(spine, spine.get_window_extent(renderer)) for axis in axes[:, 0]
              for spine in axis.spines.values() if spine.get_visible()]
    own_ticks = {axis: set(axis.get_xticklabels(which="both") + axis.get_yticklabels(which="both"))
                 for axis in axes[:, 0]}
    collisions = [(a.get_text(), b.get_text()) for index, (a, a_box) in enumerate(labels)
                  for b, b_box in labels[index + 1:] if a_box.overlaps(b_box)]
    collisions.extend((item.get_text(), "axis") for item, box in labels for spine, edge in frames
                      if box.overlaps(edge) and item not in own_ticks[spine.axes])
    outside = [item.get_text() for item, box in labels
               if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)]
    if collisions or outside:
        plt.close(fig)
        raise ValueError(f"chart label collision or overflow: {collisions}, {outside}")
    data = io.BytesIO()
    fig.savefig(data, format="png", dpi=140)
    plt.close(fig)
    atomic_write(output / "native-curve.png", data.getvalue())


def publish(source: Path, output: Path, run_id: str, mode: str, max_age: float,
            last_chart: float, chart_interval: float) -> float:
    raw = source.read_bytes()
    state = json.loads(raw)
    if not isinstance(state, dict) or state.get("run_id") != run_id:
        raise ValueError("run identifier mismatch")
    updated = timestamp(state["updated_at"])
    now = datetime.now(timezone.utc)
    age = (now - updated).total_seconds()
    status = state.get("status")
    if not isinstance(status, str):
        raise ValueError("run status missing")
    if mode == "live" and (age < -5 or age > max_age):
        raise ValueError("source timestamp stale or in the future")
    if mode == "live" and status not in ACTIVE:
        raise ValueError("source is not in an active state")
    rows, fields = metrics(state)
    frame, clip = media_path(source, state.get("frame")), media_path(source, state.get("latest_clip"))
    frame_sha = None
    if frame is not None:
        if frame.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("unsupported frame format")
        frame_name = "native-latest" + frame.suffix.lower()
        frame_bytes = frame.read_bytes()
        expected_header = b"\x89PNG\r\n\x1a\n" if frame.suffix.lower() == ".png" else b"\xff\xd8\xff"
        if not frame_bytes.startswith(expected_header):
            raise ValueError("frame bytes do not match extension")
        atomic_write(output / frame_name, frame_bytes)
        frame_sha = digest(frame_bytes)
    else:
        frame_name = None
    # The output directory is created for this observer invocation. Remove
    # only its own prior frame names so an absent/new-format frame cannot look
    # like current footage to someone who left the image tab open.
    for suffix in (".jpg", ".jpeg", ".png"):
        previous = output / ("native-latest" + suffix)
        if previous.name != frame_name and previous.is_file():
            previous.unlink()

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=["sample_index", *[field[0] for field in fields]])
    writer.writeheader()
    for index, row in enumerate(rows, 1):
        writer.writerow({"sample_index": index, **{key: row.get(key) for key, _, _ in fields}})
    atomic_write(output / "native-metrics.csv", buf.getvalue().encode("utf-8"))
    current = time.monotonic()
    if fields and (last_chart == 0 or current - last_chart >= chart_interval):
        plot(output, rows, fields, mode == "historical")
        chart_record = {"source_sha256": digest(raw), "source_updated_at": updated.isoformat()}
        atomic_write(output / "native-curve-source.json", (json.dumps(chart_record, ensure_ascii=False) + "\n").encode("utf-8"))
        last_chart = current

    chart_source = json.loads((output / "native-curve-source.json").read_text(encoding="utf-8")) if (output / "native-curve-source.json").is_file() else None

    label = "历史记录，不是当前运行" if mode == "historical" else "运行数据：以来源时间和健康状态为准"
    latest = [f"{name}: {next((row[key] for row in reversed(rows) if row.get(key) is not None), '无')}"
              for key, name, _ in fields]
    source_rel = source.relative_to(ROOT).as_posix()
    lines = ["XGO-Duck 本机运行观察", label, f"运行编号: {run_id}", f"来源状态: {status}",
             f"运行用途: {state.get('purpose', '来源未提供')}",
             f"来源更新时间 UTC: {updated.isoformat()}", f"观察时间 UTC: {now.isoformat()}",
             f"来源年龄秒: {age:.1f}", f"来源: {source_rel}", f"来源 SHA-256: {digest(raw)}",
             f"控制步: {state.get('controls', '未提供')}",
             f"学习更新: {state.get('completed_updates', state.get('learning_updates', '未提供'))}",
             f"指标样本数: {len(rows)}", *latest,
             f"最新帧: {frame_name or '等待来源提供'}", f"最新录像: {clip.relative_to(ROOT).as_posix() if clip else '等待来源提供'}",
             f"曲线: native-curve.png；数据更新时间: {chart_source['source_updated_at']}" if chart_source else "曲线: 等待真实指标",
             "注意：观察器未接入训练控制器；它自身不能停止训练。"]
    atomic_write(output / "native-status.txt", ("\n".join(lines) + "\n").encode("utf-8"))
    health = {"kind": "native-run-observer", "mode": mode, "health": "historical" if mode == "historical" else "fresh",
              "observer_pid": os.getpid(),
              "run_id": run_id, "source": source_rel, "source_sha256": digest(raw),
              "source_updated_at": updated.isoformat(), "heartbeat_at": now.isoformat(),
              "source_age_seconds": age, "source_status": status, "metric_rows": len(rows),
              "frame_source_sha256": frame_sha, "chart_source": chart_source,
              "clip_source": clip.relative_to(ROOT).as_posix() if clip else None,
              "protocol_verified": False, "controller_stop_integrated": False}
    atomic_write(output / "native-health.json", (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return last_chart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=("historical", "live"))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--max-age", type=float, default=60)
    parser.add_argument("--chart-interval", type=float, default=30)
    args = parser.parse_args()
    source, output = checked_path(args.source), checked_path(args.output)
    if args.interval <= 0 or args.max_age <= 0 or args.chart_interval <= 0:
        raise ValueError("intervals must be positive")
    if source.parent == output or output == RECORDS:
        raise ValueError("source or output is not valid")
    output.mkdir(parents=True, exist_ok=False)
    last_chart = 0.0
    try:
        if not source.is_file():
            raise ValueError("source file is missing")
        while True:
            last_chart = publish(source, output, args.run_id, args.mode, args.max_age, last_chart, args.chart_interval)
            if not args.follow:
                break
            time.sleep(args.interval)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        error = {"kind": "native-run-observer", "mode": args.mode, "health": "error", "run_id": args.run_id,
                 "observer_pid": os.getpid(), "error_type": type(exc).__name__, "error": str(exc),
                 "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                 "protocol_verified": False, "controller_stop_integrated": False}
        atomic_write(output / "native-status.txt", ("观察记录出错，不能视为运行正常。\n" + str(exc) + "\n").encode("utf-8"))
        atomic_write(output / "native-health.json", (json.dumps(error, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(error, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "snapshot-written", "mode": args.mode, "run_id": args.run_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
