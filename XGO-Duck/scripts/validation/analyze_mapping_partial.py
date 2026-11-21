"""Review completed cases without overwriting the failed mapping attempt.

The frozen pointwise standing-trajectory tolerance failed. This separate
diagnostic checks the same-input policy identity and reports already completed
motion only. It cannot promote the failed protocol or fabricate its two
unexecuted lateral conditions. No additional physical samples are generated.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from planar_block_metrics import block_statistic
from command_input_mapping import transform

OUT=ROOT/'records/reports/command-mapping'


def main():
    state=read(OUT/'probe-progress.json');assert state['status']=='failed'
    folder=Path(state['path']);plan=read(folder/'protocol.json')
    assert sha(folder/'protocol.json')==sha(ROOT/'configs/experiments/command_mapping_probe.json')
    control=Path(plan['control_evaluation']);criteria=plan['development_criteria'];rows=[]
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    names=state['model_metadata']['joint_names'].split(',');hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    no_op_max=0.;replays=0
    for item in state['conditions']:
        name=item['condition'];command=np.asarray(item['command']);path=folder/f'{name}-trace.npz'
        assert sha(path)==item['trace_sha256']
        with np.load(path) as value:t={k:value[k].copy() for k in value.files}
        with np.load(control/f'{plan["control_conditions"][name]}-trace.npz') as ref:
            keep=np.r_[0:48,51:61]
            first_equal=bool(np.array_equal(t['observations'][0][:,keep],ref['observations'][0][:,keep]))
        active=t['active'];assert active.shape==(600,16) and active.all()
        if name=='stand':
            for step in range(0,600,40):
                for actor in t['observations'][step]:
                    converted=actor.copy();converted[48:51]=transform(converted[48:51])
                    assert np.array_equal(actor,converted)
                    a=session.run(None,{'obs':actor[None]})[0];b=session.run(None,{'obs':converted[None]})[0]
                    no_op_max=max(no_op_max,float(abs(a-b).max()));replays+=1
        axis=int(np.argmax(abs(command)));sign=float(np.sign(command[axis]))
        values=t['yaw_rate'] if axis==2 else t['velocity'][:,:,axis]
        speed=float(values.mean()*sign)
        blocks=block_statistic(t['velocity'],active,command)
        angles=np.rad2deg(t['joint_positions'][50:,:,hips]);inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
        shared=(item['full_episode_survival']>=criteria['survival_fraction_min'] and
            item['yaw_mae_rad_s']<=criteria['yaw_error_max_rad_s'] and
            speed>=criteria['minimum_signed_command_axis_speed'].get(name,0.) and
            (name!='stand' or item['final_displacement_m_median']<=criteria['stand_drift_median_max_m']))
        rows.append(dict(condition=name,external_command=command.tolist(),policy_input=transform(command).tolist(),
            first_noncommand_observation_exact=first_equal,survivors=16,signed_speed_full_episode=speed,
            instantaneous_error=item['linear_xy_mae_m_s'],block_error=float(blocks['mean_error_per_episode'].mean()),
            yaw_error=item['yaw_mae_rad_s'],inward_fraction=float(inward.mean()),
            legacy_numeric_pass=bool(shared and item['linear_xy_mae_m_s']<=criteria['linear_xy_l2_error_max'][name]),
            predeclared_block_numeric_pass=bool(shared and blocks['mean_error_per_episode'].mean()<=criteria['linear_xy_l2_error_max'][name] and inward.mean()<=.01),
            trace_sha256=sha(path)))
    assert no_op_max==0.
    result=dict(status='diagnostic-only-original-attempt-remains-failed',checked_at=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=sha(folder/'protocol.json'),original_failed_progress_sha256=sha(OUT/'probe-progress.json'),
        script_sha256=sha(Path(__file__)),completed_cases=len(rows),episodes=16*len(rows),rows=rows,
        zero_input_same_observation_inference=dict(inputs=replays,max_abs_error=no_op_max),
        missing_conditions=[k for k in plan['commands'] if k not in [x['condition'] for x in rows]],
        cause_boundary='First physical-step divergence despite identical first observation/action; unrecorded solver/model state not fully excluded',
        scope='Retrospective partial engineering review only; no full protocol pass, stage acceptance or hardware claim')
    write(OUT/'partial-behavior-summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
