"""Extract annotated phase keyframes from an existing router video.

The source video is decoded only; no simulation step is added. Labels identify
the declared route phase and control step so the montage remains traceable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT = Path("/mnt/c/Windows/Fonts/msyh.ttc")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/reports/assets/router-full-sequence-s1218-keyframes.jpg")
    args = parser.parse_args()
    video = args.video.resolve()
    # These times cover every declared phase without fabricating a pose.
    anchors = [(0.0, "站立开始", 0), (0.5, "站立稳定", 25), (1.0, "进入滚翻", 50),
               (3.0, "滚翻中段", 150), (7.0, "滚翻后段", 350), (10.0, "落地站立", 500),
               (10.5, "接力保持", 525), (11.0, "进入前进", 550), (16.0, "前进早段", 800),
               (26.0, "前进后段", 1300), (31.0, "序列结束", 1550)]
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24", input_params=["-threads", "1"],
                                        output_params=["-threads", "1"])
    meta = next(reader)
    fps, (width, height) = float(meta["fps"]), meta["size"]
    wanted = {max(0, int(round(seconds * fps))): (seconds, label, control) for seconds, label, control in anchors}
    frames = {}
    for index, pixels in enumerate(reader):
        if index in wanted:
            frames[index] = Image.frombytes("RGB", (width, height), pixels)
    reader.close()
    assert len(frames) == len(wanted), (len(frames), len(wanted), sorted(set(wanted) - set(frames)))
    font = ImageFont.truetype(str(FONT), size=22)
    small = ImageFont.truetype(str(FONT), size=18)
    cols, cell_w, cell_h = 3, 480, 330
    rows = (len(anchors) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "#101923")
    draw = ImageDraw.Draw(canvas)
    manifest = []
    for pos, (seconds, label, control) in enumerate(anchors):
        index = max(0, int(round(seconds * fps)))
        frame = frames[index].copy()
        frame.thumbnail((cell_w - 16, cell_h - 66), Image.Resampling.LANCZOS)
        x, y = (pos % cols) * cell_w, (pos // cols) * cell_h
        canvas.paste(frame, (x + (cell_w - frame.width) // 2, y + 34))
        draw.text((x + 10, y + 7), f"{label} · 控制步 {control}", font=font, fill="#f6d365")
        draw.text((x + 10, y + cell_h - 25), f"录像时间 {seconds:.1f} s · 帧 {index}", font=small, fill="#d3e1ee")
        manifest.append({"label": label, "control": control, "time_seconds": seconds, "frame_index": index})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=94, subsampling=0)
    out_manifest = args.out.with_suffix(".json")
    out_manifest.write_text(json.dumps({"status": "passed", "source": str(video.relative_to(ROOT)),
                                        "source_sha256": sha(video), "output": str(args.out.relative_to(ROOT)),
                                        "output_sha256": sha(args.out), "fps": fps, "anchors": manifest},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.out.relative_to(ROOT)),
                      "manifest": str(out_manifest.relative_to(ROOT)), "anchors": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
