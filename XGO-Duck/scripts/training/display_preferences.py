"""Apply the user's recorded-video display preference to future training entries.

Only recording fields in TrainConfig change. The returned settings explicitly
disable native live windows; the future runner must skip LiveObserver when
native_live_window is false. Do not fake an open observer or call old entries
that unconditionally create one. Completed/interrupted sources stay archived.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FILE=ROOT/'configs/environments/training_display.json'


def prepare_display(cfg, *, sparse_snapshots=False):
    settings=json.loads(FILE.read_text())
    assert settings['mode']=='recorded-clips-and-progress'
    assert settings['native_live_window'] is False
    assert settings['visible_terminal_required'] is True
    assert settings['automatic_window_resize_or_reposition'] is False
    assert settings['apply_to']=='future-training-entries'
    assert (settings['video_interval_control_steps'],settings['video_length_control_steps'],
            settings['recording_playback_fps'])==(1200,200,50)
    # New guarded entry owns sparse wall-time snapshots. Historical entries
    # retain their frozen video behavior and original configuration hashes.
    if sparse_snapshots:
        prepared=replace(cfg, video=False)
        return prepared, {**settings, 'mode':'wall-time-snapshots',
            'video_enabled':False, 'snapshot_interval_wall_seconds':180}, [
            dict(field='video',before=cfg.video,after=False)]
    prepared=replace(cfg,video=settings['video_enabled'],
        video_interval=settings['video_interval_control_steps'],
        video_length=settings['video_length_control_steps'])
    before,after=asdict(cfg),asdict(prepared)
    changed=[]
    for key in ('video','video_interval','video_length'):
        if before[key]!=after[key]:changed.append(dict(field=key,before=before[key],after=after[key]))
        after[key]=before[key]
    assert after==before
    return prepared,settings,changed


def main():
    from profile_components import prepare
    import torch
    cfg,_,_,_=prepare('future_recorded_display_check')
    prepared,settings,changes=prepare_display(cfg)
    assert prepared.env is cfg.env and prepared.agent is cfg.agent
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=hashlib.sha256(FILE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes=changes,native_live_window=settings['native_live_window'],
        env_and_agent_unchanged=True,training_started=False,graphics_context_created=False,
        running_pilot_changed=False,future_runner_integration_required=True,
        scope='Real TrainConfig preparation for future entries; no training or window action')
    out=ROOT/'records/reports/training-display-preference-check.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def save_model_metadata(env, record):
    """Read joint names and limits for later trace checks without any viewer.

    Uses the already compiled training model; creates no CPU simulation/data,
    performs no forward/step and does not access a policy or random generator.
    """
    import mujoco
    env=env.unwrapped
    names=list(env.scene['robot'].joint_names)
    model=env.sim.mj_model
    ids=[]
    for name in names:
        matches=[i for i in range(model.njnt)
                 if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) or '').rsplit('/',1)[-1]==name]
        assert len(matches)==1,(name,matches)
        ids.append(matches[0])
    metadata=dict(joint_names=names,joint_ranges_rad=model.jnt_range[ids].tolist(),
        source='Actual compiled training model',native_live_window_created=False,extra_environment_steps=0)
    (Path(record)/'training-model-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
