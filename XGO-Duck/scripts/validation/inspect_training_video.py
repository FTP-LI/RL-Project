"""Decode a recorded video, save a representative frame and media provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import mediapy as media
import imageio_ffmpeg
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('record_dir',type=Path)
p.add_argument('--live',action='store_true',help='Inspect already closed/copied clips while training continues; never claim run completion')
args=p.parse_args()
record=args.record_dir.resolve()
# A running experiment has a request but no final result. This explicit option
# only decodes saved media; it does not launch training or alter its process.
request_file='training-request.json' if args.live else 'training-result.json'
request=json.loads((record/request_file).read_text())
entries=[]
# Training step numbers, not lexicographic names, define the recording order.
# This only changes the media index; decoded video bytes are preserved.
video_key=lambda path: (int(path.stem.rsplit('-',1)[-1]) if path.stem.rsplit('-',1)[-1].isdigit() else -1,path.name)
for path in sorted((record/'videos').glob('*.mp4'),key=video_key):
    frames=media.read_video(path)
    assert len(frames)>0 and frames.shape[-1]==3
    fps=float(frames.metadata.fps)
    frame_index=min(len(frames)-1,len(frames)//2)
    still=path.with_suffix('.jpg')
    media.write_image(still,frames[frame_index])
    entries.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'frames':len(frames),'fps':fps,'duration_seconds':len(frames)/fps,
        'height':frames.shape[1],'width':frames.shape[2],
        'frame_preview':still.name,'preview_frame_index':frame_index,
        'run_id':request['run_id'],'seed':request['seed'],
        'source_sha256':request['task_source_sha256'],
        'captured_while_training_active':args.live,
        'scope':'Actual training rollout of env 0; policy changes during clip; stochastic actions, randomized commands and resets; not a checkpoint evaluation'})
(record/'media-manifest.json').write_text(json.dumps(entries,indent=2))
print(json.dumps(entries,indent=2))
