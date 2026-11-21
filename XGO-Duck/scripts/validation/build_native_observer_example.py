"""Make a clearly historical VS Code example from one archived PPO observation.

This only reads saved metrics and media. It does not start a simulator, train a
policy, or serve a page. A future run needs its own separate observer admission.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from figure_style import apply_figure_style

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "records/training-archive/experiments/duck_getup_early-support-ppo-native_control_s1085_20261110T034608"
OUTPUT = ROOT / "records/reports/native-observer-preflight"
FIELDS = (
    ("environment_reward_mean", "环境回报", "平均回报"),
    ("value_loss", "价值损失", "损失"),
    ("net_transitions_per_second", "采样速度", "转移 / 秒"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.text import Text

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font = None
    apply_figure_style(font=font, sizes=(10, 9, 8))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    observation = SOURCE / "observation.json"
    saved = json.loads(observation.read_text(encoding="utf-8"))
    rows = saved["rows"]
    updates = [row["update"] for row in rows]
    if updates != list(range(1, saved["completed_updates"] + 1)):
        raise ValueError("Historical updates are missing or out of order")
    series = {}
    for field, _, _ in FIELDS:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite historical metric: {field}")
        series[field] = values

    clip = SOURCE / saved["latest_clip"]
    if not clip.is_file():
        raise FileNotFoundError("Recorded historical clip is missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), layout="constrained")
    figure.suptitle("历史训练指标", ha="center")
    for axis, (field, title, ylabel) in zip(axes, FIELDS):
        axis.plot(updates, series[field], color="#086b80", linewidth=1.4, marker="o", markersize=3)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        axis.margins(x=0.04, y=0.12)
    axes[-1].set_xlabel("更新次数")
    chart = OUTPUT / "historical-metrics.png"
    figure.savefig(chart, dpi=180)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    hidden_ticks = set()
    for axis in axes:
        for scale, bounds in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            low, high = sorted(bounds)
            for tick in scale.get_major_ticks() + scale.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden_ticks.update((tick.label1, tick.label2))
    labels = [(item, item.get_window_extent(renderer)) for item in figure.findobj(Text)
              if item.get_visible() and item.get_text().strip() and item not in hidden_ticks]
    frames = [(spine, spine.get_window_extent(renderer)) for axis in axes
              for spine in axis.spines.values() if spine.get_visible()]
    own_ticks = {axis: set(axis.get_xticklabels(which="both") + axis.get_yticklabels(which="both"))
                 for axis in axes}
    collisions = [(a.get_text(), b.get_text()) for index, (a, box_a) in enumerate(labels)
                  for b, box_b in labels[index + 1:] if box_a.overlaps(box_b)]
    collisions.extend((label.get_text(), "axis") for label, box in labels for spine, edge in frames
                      if box.overlaps(edge) and label not in own_ticks[spine.axes])
    outside = [item.get_text() for item, box in labels
               if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(box.x1, box.y1)]
    if collisions or outside:
        raise ValueError(f"Figure text collision or overflow: {collisions}, {outside}")
    plt.close(figure)

    record = {
        "scope": "historical-only-not-live", "run_id": saved["run_id"],
        "source": observation.relative_to(ROOT).as_posix(), "source_sha256": sha(observation),
        "video": clip.relative_to(ROOT).as_posix(), "video_sha256": sha(clip),
        "script_sha256": sha(Path(__file__)), "chart_sha256": sha(chart),
        "recorded_at": saved["updated_at"], "completed_updates": saved["completed_updates"],
        "metric_fields": list(series), "point_count_per_metric": len(updates),
        "physics_steps_this_example": 0, "learning_updates_this_example": 0,
    }
    (OUTPUT / "historical-example.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# 历史训练观察示例

这是已归档训练记录，**当前没有训练或仿真**。训练记录时间为 {saved['updated_at']}，共完成 {saved['completed_updates']} 次更新，使用 {saved['environments']} 个环境。训练曲线来自该运行保存的逐次更新数值，不能单独证明动作质量或模型改进。

![历史训练指标](historical-metrics.png)

[原始指标](../../training-archive/experiments/{SOURCE.name}/observation.json) · [训练录像](../../training-archive/experiments/{SOURCE.name}/{clip.name}) · [数据来源与哈希](historical-example.json) · [观察预检](README.md)

图中“环境回报”是记录中的环境步平均回报，“价值损失”是价值网络损失，“采样速度”是该训练记录中的净转移数/秒。此例只有一个训练组，不能判断修改是否带来收益。录像为训练时保存的实际画面，并非当前直播。

本图由项目内的 `scripts/validation/build_native_observer_example.py` 读取原始记录生成。运行需要已安装 Matplotlib 的项目 Python 环境；脚本不执行物理步或学习更新。新训练要有自己的实时数据、录像和停止记录，不能使用本示例充当准入凭据。
"""
    (OUTPUT / "historical-example.md").write_text(report, encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
