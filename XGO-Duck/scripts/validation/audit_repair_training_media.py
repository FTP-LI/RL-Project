"""Decode saved repair-pilot videos and record their exact provenance on CPU.

Use a completed copied clip while learning continues, or every saved clip after
learning. Partial clips are identified explicitly. No simulation, optimizer,
window control, or training restart is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    record = ROOT/'records/experiments'/args.run_id
    assert record.parent == ROOT/'records/experiments'
    request = json.loads((record/'training-request.json').read_text())
    assert request['phase'] == 'repair-pilot'
    protocol = json.loads((record/'protocol.json').read_text())
    progress = json.loads((record/'progress.json').read_text())
    log_dir = Path(progress['log_dir'])
    if not log_dir.is_absolute():
        log_dir = Path.home()/'XGO-Duck/workspace/xgoduck_rl'/log_dir
    output = record/'media-check'
    output.mkdir(exist_ok=True)
    checks = []
    for video in sorted((record/'videos').glob('*.mp4')):
        original = log_dir/'videos/train'/video.name
        digest = sha(video)
        assert digest == sha(original), video.name
        start = int(video.stem.rsplit('-', 1)[-1])
        assert start % protocol['video_interval_control_steps'] == 0
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(reader)
        first = last = None
        count = 0
        for frame in reader:
            if first is None:
                first = frame
            last = frame
            count += 1
        assert count > 0 and count <= protocol['video_length_control_steps']
        assert metadata['fps'] == 50 and tuple(metadata['size']) == (960, 720)
        assert digest == sha(video) == sha(original), 'Video changed during decode'
        complete = count == protocol['video_length_control_steps']
        if not complete:
            assert (record/'training-result.json').exists(), 'Copy of unfinished active clip'
        for label, frame in [('start', first), ('end', last)]:
            Image.frombytes('RGB', tuple(metadata['size']), frame).save(output/f'{video.stem}-{label}.jpg', quality=90)
        checks.append(dict(file=str(video.relative_to(ROOT)), sha256=digest,
            original_file=str(original), decoded_frames=count, fps=50,
            duration_seconds=count/50, size=metadata['size'], complete_clip=complete,
            relative_control_step=start, display='Saved training environment 0; not live'))
    assert checks, 'No copied clips yet'
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        run_id=args.run_id, arm=request['arm'], updates_at_check=progress['new_updates'],
        clips=checks, training_finished=(record/'training-result.json').exists(),
        final_budget_verified=False, stable_gait_verified=False,
        source_sha256=sha(Path(__file__)), scope='Copied training media and full decode only')
    (output/'check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
