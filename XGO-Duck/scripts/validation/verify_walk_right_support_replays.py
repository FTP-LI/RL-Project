"""Read back all comparative right-turn video frames against saved physical state."""

from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT / "records/reports/common-walk-right-support-yaw-training/seven-command"
PLAN = ROOT / "configs/experiments/common_walk_right_support_seven_command.json"


def main():
    spec, analysis = read(PLAN), read(OUT / "analysis.json")
    assert analysis["protocol_sha256"] == sha(PLAN)
    samples = {}
    for seed in spec["development_seeds"]:
        path = OUT / f"right-startup-s{seed}-replay.json"
        data = read(path)
        world = data["selected_world"]
        assert data["seed"] == seed and data["protocol_sha256"] == sha(PLAN) and not data["physical_substeps"]
        rows = data["frames"]
        assert [row["control"] for row in rows] == list(range(0, 1001, 2))
        assert all(abs(row["seconds"]-row["control"]*.02) < 1e-8 for row in rows)
        for arm in spec["arms"]:
            receipt = analysis["records"][f"{seed}-{arm}"]
            folder = ROOT / receipt["path"]
            proof_path = folder / "verification.json"
            assert sha(proof_path) == receipt["verification_sha256"] == data["arms"][arm]["verification_sha256"]
            proof = read(proof_path)
            initial = folder / "physical-initial.npz"
            assert sha(initial) == read(folder / "evaluation.json")["initial_state_sha256"]
            with np.load(initial, allow_pickle=False) as saved:
                poses = [saved["qpos"][world:world+1]]
            for name, expected in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
                file = folder / f"trace-{name}.npz"
                assert sha(file) == expected
                with np.load(file, allow_pickle=False) as saved:
                    poses.append(saved["qpos"][saved["active"][:, world], world])
            poses = np.concatenate(poses)
            assert len(poses) == 1001
            for row in rows:
                frame = row["sources"][arm]
                assert frame["source_control"] == row["control"]
                assert sha256(poses[row["control"]].tobytes()).hexdigest() == frame["qpos_sha256"]
        video, card = (ROOT / data[key] for key in ("video", "keyframes"))
        assert sha(video) == data["video_sha256"] and sha(card) == data["keyframes_sha256"]
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
        assert count == 501 and abs(seconds-count/25) <= .05
        with Image.open(card) as image:
            assert image.size == (1152, 288*len(data["keyframe_controls"]))
            image.verify()
        samples[str(seed)] = dict(world=world, frames=count, seconds=seconds, verified_poses=count*len(spec["arms"]),
            video=data["video"], video_sha256=sha(video), keyframes=data["keyframes"], keyframes_sha256=sha(card),
            replay_proof_sha256=sha(path))
    write(OUT / "replay-verification.json", dict(status="passed-independent-right-turn-replay-readback",
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(PLAN), frames_verified=samples, physical_substeps=0))
    print("RIGHT_SUPPORT_REPLAYS_VERIFIED", {seed: item["frames"] for seed, item in samples.items()}, flush=True)


if __name__ == "__main__":
    main()
