"""Make a labelled viewing copy; preserve every frame of the original video."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RUN = "duck_roll-native_nominal_s1241_20261115T230228"
SOURCE = ROOT / "records/evaluations" / RUN / "clip-step-0.mp4"
OUT = ROOT / "records/reports" / RUN


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target, receipt = OUT / "evaluation-view.mp4", OUT / "evaluation-view.json"
    if target.exists() or receipt.exists(): raise FileExistsError("Do not replace an existing viewing copy")
    analysis = json.loads((OUT / "independent-analysis.json").read_text(encoding="utf-8"))
    assert sha(SOURCE) == analysis["clips"][0]["sha256"]
    reader = imageio_ffmpeg.read_frames(str(SOURCE), pix_fmt="rgb24")
    meta = next(reader)
    assert meta["size"] == (640, 480) and meta["fps"] == 25
    writer = imageio_ffmpeg.write_frames(str(target), meta["size"], pix_fmt_in="rgb24", pix_fmt_out="yuv420p", fps=25, quality=8)
    writer.send(None)
    title = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 19)
    detail = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 14)
    count = 0
    try:
        for count, frame in enumerate(reader, 1):
            image = Image.frombytes("RGB", meta["size"], frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 640, 45), fill=(17, 30, 41))
            draw.text((8, 2), "官方策略仿真评估 · 非训练", font=title, fill="white")
            draw.text((9, 26), f"环境 0  |  控制步 {(count-1)*2}  |  时间 {(count-1)*.04:.2f} 秒", font=detail, fill="white")
            writer.send(image.tobytes())
    finally:
        writer.close()
    assert count == analysis["clips"][0]["frames"] == 776
    check = imageio_ffmpeg.read_frames(str(target), pix_fmt="rgb24")
    actual = next(check)
    decoded = sum(1 for _ in check)
    assert actual["size"] == meta["size"] and actual["fps"] == meta["fps"] and decoded == count
    result = {"status": "labelled-view-copy-verified", "checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": RUN, "original": SOURCE.relative_to(ROOT).as_posix(), "original_sha256": sha(SOURCE),
              "copy": target.relative_to(ROOT).as_posix(), "copy_sha256": sha(target), "frames": decoded,
              "fps": 25, "seconds_encoded": decoded / 25, "physics_steps": 0,
              "scope": "Re-encoded viewing copy with a corrected top label; original frames and source video remain available."}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
