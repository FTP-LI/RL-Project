"""Select video frames at measured roll angles, without running a simulation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roll_metrics import rotation

ROOT = Path(__file__).resolve().parents[2]
RUN = "duck_roll-native_nominal_s1241_20261115T230228"
SOURCE = ROOT / "records/evaluations" / RUN
OUT = ROOT / "records/reports" / RUN


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    image_path, receipt_path = OUT / "keyframes.jpg", OUT / "keyframes.json"
    if image_path.exists() or receipt_path.exists(): raise FileExistsError("Keep the existing keyframe record")
    result = json.loads((OUT / "independent-analysis.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed-baseline-independently-verified" and result["controls"] == 1550
    trace = np.load(SOURCE / "trace.npz", allow_pickle=False)
    phases = trace["phase_next"][:, 0]
    matrices = rotation(trace["quaternion"][50:217, 0, :])
    lateral = np.maximum(np.linalg.norm(matrices[:, :2, 1], axis=-1), 1e-6)
    forward = (matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 1, 0] * matrices[:, 0, 1]) / lateral
    angle = np.degrees(np.unwrap(np.arctan2(-matrices[:, 2, 0], forward)))
    angle -= angle[0]
    even_steps = np.arange(50, 217, 2)
    chosen = []
    for target in (90, 180, 270, 360):
        step = int(even_steps[np.abs(angle[even_steps - 50] - target).argmin()])
        measured = float(angle[step - 50])
        chosen.append((step, 1, f"旋转约 {measured:.0f}°", measured))
    steps = [(0, 0, "起始站立", None), (50, 1, "切入滚翻", None), *chosen,
             (266, 3, "开始前进", None), (1550, 3, "路线终点", None)]
    assert len({step for step, _, _, _ in steps}) == len(steps)
    video = SOURCE / "clip-step-0.mp4"
    assert sha(video) == result["clips"][0]["sha256"]
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    header = next(reader)
    assert header["fps"] == 25 and header["size"] == (640, 480)
    wanted = {step // 2 for step, _, _, _ in steps}
    images = {}
    for index, frame in enumerate(reader):
        if index in wanted:
            # The recorder's historical top banner says "training" even
            # for evaluation. Crop that banner; retain the full original video.
            images[index] = Image.frombytes("RGB", header["size"], frame).crop((0, 46, 640, 480))
        if index >= max(wanted): break
    assert set(images) == wanted
    font = "/mnt/c/Windows/Fonts/msyh.ttc"
    title, subtitle, label = (ImageFont.truetype(font, size) for size in (34, 20, 24))
    canvas = Image.new("RGB", (1680, 840), "#f3f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), "官方标称路线 · 环境 0", font=title, fill="#193447")
    draw.text((30, 65), "按机身旋转角度选择滚翻关键帧；图片来自本次实际评估录像", font=subtitle, fill="#486273")
    frames = []
    for index, (step, expected, name, measured) in enumerate(steps):
        assert int(phases[step]) == expected
        x, y = 30 + (index % 4) * 415, 100 + (index // 4) * 350
        canvas.paste(images[step // 2].resize((400, 300), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 3, y + 305), f"{step * .02:05.2f} 秒  ·  {name}", font=label, fill="#193447")
        frames.append({"step": step, "seconds": step * .02, "video_frame": step // 2, "phase_code": expected,
                       "label": name, "measured_roll_degrees": measured})
    draw.text((30, 812), "控制周期 0.02 秒；截图裁去录像旧标题条，完整过程保留在原片。", font=subtitle, fill="#486273")
    canvas.save(image_path, quality=91, subsampling=0)
    receipt = {"status": "event-keyframes-from-original-video", "checked_at": datetime.now(timezone.utc).isoformat(),
               "run_id": RUN, "environment_index": 0, "frames": frames, "image_sha256": sha(image_path),
               "source_trace_sha256": sha(SOURCE / "trace.npz"), "source_video_sha256": sha(video),
               "crop_pixels_top": 46, "scope": "Selected video frames at recorded roll angles; no new simulation."}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "frames": len(frames), "image_sha256": receipt["image_sha256"]}, ensure_ascii=False))


if __name__ == "__main__": main()
