"""Inspect the stopped S20 arm's saved frames and MP4 container without decoding."""
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "records/evaluations/duck_roll-domain-parameter_high-mid_s1240_20261114T113334"
OUT = ROOT / "records/reports/roll-domain-parameter-random-s20/partial-media-audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boxes(path):
    size = path.stat().st_size
    offset = 0
    result = []
    with path.open("rb") as handle:
        while offset + 8 <= size:
            handle.seek(offset)
            length, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if length == 1:
                length = struct.unpack(">Q", handle.read(8))[0]
                header = 16
            if length == 0:
                length = size - offset
            valid = length >= header and offset + length <= size
            result.append({"offset": offset, "size": length, "type": kind.decode("ascii", "replace"), "complete": valid})
            if not valid or not length:
                break
            offset += length
    return result


def main():
    video = RECORD / "clip-step-0.mp4"
    entries = boxes(video)
    frames = sorted(RECORD.glob("frame-step-*.jpg"), key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    result = {
        "status": "container-inspected-not-decoded",
        "partial_arm": "high-mid",
        "video": {
            "path": str(video.relative_to(ROOT)).replace("\\", "/"),
            "bytes": video.stat().st_size,
            "sha256": sha(video),
            "boxes": entries,
            "has_complete_moov": any(item["type"] == "moov" and item["complete"] for item in entries),
        },
        "frames": {
            "count": len(frames),
            "first": frames[0].name if frames else None,
            "last": frames[-1].name if frames else None,
            "last_sha256": sha(frames[-1]) if frames else None,
        },
        "complete_trace_npz_present": (RECORD / "trace.npz").is_file(),
        "resumable_physics_checkpoint_present": False,
        "scope": "MP4 container structure only; no playback or content verification",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
