"""Explain saved turn errors before proposing another bounded controller.

No environment steps. Use every first-episode sample, not a selected video.
Check the unscaled, noisy pre-action gyro against the preceding post-action
yaw rate; distinguish bias from fast oscillation without changing old gates.
"""
from pathlib import Path
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    progress=read(ROOT/'records/reports/command-mapping/probe-progress.json')
    folder=Path(progress['path']);report=read(folder/'evaluation.json');rows=[]
    assert report['status']=='failed'
    for item in report['conditions']:
        name=item['condition']
        if not name.startswith('turn-'):continue
        path=folder/f'{name}-trace.npz';assert sha(path)==item['trace_sha256']
        with np.load(path) as t:
            yaw=t['yaw_rate'];active=t['active'];gyro=t['observations'][:,:,2]
            assert active.all() and yaw.shape==(600,16)
            target=item['command'][2];error=yaw-target
            means=yaw[50:].reshape(11,50,16).mean(axis=1)
            previous_error=gyro[1:]-yaw[:-1]
            rows.append(dict(condition=name,target=target,
                full_signed_speed=float(np.sign(target)*yaw.mean()),
                full_yaw_mae=float(abs(error).mean()),
                post1s_yaw_mae=float(abs(error[50:]).mean()),
                post1s_bias=float(error[50:].mean()),
                post1s_yaw_std=float(np.mean(np.std(yaw[50:],axis=0))),
                one_second_yaw_mae=float(abs(means-target).mean()),
                pre_action_gyro_previous_state_mae=float(abs(previous_error).mean()),
                pre_action_gyro_previous_state_max=float(abs(previous_error).max()),
                trace_sha256=sha(path)))
    assert len(rows)==2
    result=dict(status='diagnostic-only',source_status=report['status'],rows=rows,
        source_report_sha256=sha(folder/'evaluation.json'),script_sha256=sha(Path(__file__)),
        learning_updates=0,environment_steps=0,legacy_verdicts_unchanged=True,
        proposal='Compare fixed mapping against bounded feedback using only the existing noisy gyro; preserve raw yaw criteria and fixed budget')
    out=ROOT/'records/reports/turn-feedback';out.mkdir(exist_ok=True)
    write(out/'prior-yaw-analysis.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
