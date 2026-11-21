"""CPU decode and source-state verification for the full8s replay."""
from pathlib import Path
import hashlib
import json
import numpy as np
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    out=ROOT/'records/reports/action-video';m=read(out/'replay.json');source=ROOT/m['source_record']
    assert sha(source/'value-data.npz')==m['source_dataset_sha256']
    assert sha(source/'live-startup.npz')==m['initial_state_sha256']
    with np.load(source/'value-data.npz') as f:states=f['qpos'].copy()
    with np.load(source/'live-startup.npz') as f:initial=f['qpos'].copy()
    rows=[]
    for row in m['videos']:
        index=row['environment_index'];path=ROOT/row['path'];assert sha(path)==row['sha256']
        assert row['common_steps']==list(range(0,401,2))
        expected=[initial[index] if k==0 else states[k-1,index] for k in row['common_steps']]
        assert [hashlib.sha256(x.tobytes()).hexdigest() for x in expected]==row['qpos_sha256_by_frame']
        reader=imageio_ffmpeg.read_frames(str(path));meta=next(reader);frames=sum(1 for _ in reader)
        assert frames==row['frames']==201 and meta['fps']==25.
        assert abs(meta['duration']-8.04)<.05
        assert row['episode_times_seconds'][0]==0 and row['episode_times_seconds'][-1]==8.
        rows.append(dict(path=row['path'],frames=frames,fps=meta['fps'],decoded_duration=meta['duration'],
            source_state_hashes_all_exact=True,complete_episode_covered=True,sha256=sha(path)))
    result=dict(status='passed',videos=rows,source_replay_manifest_sha256=sha(out/'replay.json'),
        frames_interpolated=0,new_physics_steps=0,new_learning_updates=0,
        scope='Full8s saved-state replay,201frames including both endpoints; separate from original camera recordings',
        script_sha256=sha(Path(__file__)))
    write(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
