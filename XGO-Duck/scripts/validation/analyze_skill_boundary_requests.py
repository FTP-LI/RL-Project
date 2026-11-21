"""CPU switch-request audit on saved stable policy observations.

At an identical recorded pre-action state, express q in the destination HOME,
zero its raw-action history as the archived runtime does, and infer once.
Compare the first new request to the preceding raw physical request. These
are hypothetical requests, not executed transitions or actuator measurements.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/skill-interfaces'


def main():
    audit=read(OUT/'analysis.json');assert audit['status']=='passed'
    models={x['role']:x for x in audit['models']};sessions={}
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for role in ('walk-fixed-reference','getup','roll'):
        row=models[role];path=ROOT/row['path'];assert sha(path)==row['sha256']
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
        sessions[role]=session
    getup=Path(read(ROOT/'records/reports/standup-reference/progress.json')['path'])
    roll=Path(read(ROOT/'records/reports/roll-reference/progress.json')['path'])
    walk=Path(read(ROOT/'records/reports/official-actor-bootstrap/screen-analysis.json')['rows']['official']['stop']['record'])
    if not walk.is_absolute():walk=ROOT/walk
    cases=[(getup,pose,'getup','walk-fixed-reference') for pose in ('prone','supine','left-side','right-side')]
    cases += [(roll,'forward-roll','roll','walk-fixed-reference'),(walk,'stop','walk-fixed-reference','roll')]
    records=[];packets={}
    for number,(folder,condition,source_role,target_role) in enumerate(cases):
        report=read(folder/'evaluation.json');check=read(folder/'verification.json')
        assert check['status']=='passed' and check['report_sha256']==sha(folder/'evaluation.json')
        assert report['model_sha256']==models[source_role]['sha256'],source_role
        item=next(x for x in report['conditions'] if x['condition']==condition)
        path=folder/f'{condition}-trace.npz'
        if 'trace_sha256' in item:expected_hash=item['trace_sha256']
        else:expected_hash=next(x['trace_sha256'] for x in check['checks'] if x['condition']==condition)
        assert sha(path)==expected_hash
        with np.load(path) as trace:
            observation=trace['observations'][-1].copy()
            active=trace['active'][-1].astype(bool)
        assert observation.shape==(32,61) and np.isfinite(observation[active]).all()
        gravity=observation[:,3:6]
        tilt=np.rad2deg(np.arccos(np.clip(-gravity[:,2]/np.linalg.norm(gravity,axis=1),-1,1)))
        eligible=active&(tilt<=15)
        original=observation[eligible].copy();assert len(original)>0
        source_home=np.deg2rad(models[source_role]['model_home_deg']).astype(np.float32)
        target_home=np.deg2rad(models[target_role]['model_home_deg']).astype(np.float32)
        np.testing.assert_allclose(models[source_role]['action_scale'],1.)
        np.testing.assert_allclose(models[target_role]['action_scale'],1.)
        old_target=source_home+original[:,34:48]
        converted=original.copy();converted[:,6:20]+=source_home-target_home
        converted[:,34:48]=0.;converted[:,48:61]=0.
        np.testing.assert_allclose(original[:,6:20]+source_home,converted[:,6:20]+target_home,atol=1e-7,rtol=0)
        session=sessions[target_role]
        new_raw=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in converted])
        new_target=target_home+new_raw;assert np.isfinite(new_target).all()
        jump=np.rad2deg(abs(new_target-old_target));current_q=original[:,6:20]+source_home
        names=models[source_role]['joint_names'];assert names==models[target_role]['joint_names']
        head=[i for i,x in enumerate(names) if x.startswith(('head_','neck_'))]
        leg=[i for i in range(14) if i not in head]
        key=f'case{number}'
        for name,value in dict(original=original,converted=converted,old_target=old_target,new_raw=new_raw,
            new_target=new_target,current_q=current_q,environment_index=np.flatnonzero(eligible)).items():packets[key+'_'+name]=value
        records.append(dict(source=source_role,destination=target_role,condition=condition,
            trace=str(path.relative_to(ROOT)),trace_sha256=sha(path),report_sha256=sha(folder/'evaluation.json'),
            recorded_frame='last recorded pre-action observation',episodes_total=32,eligible_stable_states=int(eligible.sum()),
            excluded_unstable_or_inactive=int((~eligible).sum()),
            peak_target_jump_deg=dict(median=float(np.median(jump.max(axis=1))),p95=float(np.quantile(jump.max(axis=1),.95)),maximum=float(jump.max())),
            leg_maximum_jump_deg=float(jump[:,leg].max()),head_maximum_jump_deg=float(jump[:,head].max()),
            per_joint_maximum_jump_deg={name:float(value) for name,value in zip(names,jump.max(axis=0))},
            maximum_new_request_from_measured_position_deg=float(np.rad2deg(abs(new_target-current_q)).max())))
    packet=OUT/'boundary-request-inputs.npz';np.savez_compressed(packet,**packets)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),cases=records,
        script_sha256=sha(Path(__file__)),source_audit_sha256=sha(OUT/'analysis.json'),
        inputs=str(packet.relative_to(ROOT)),inputs_sha256=sha(packet),models={r:models[r]['sha256'] for r in sessions},
        home_conversion_preserves_measured_q=True,history_reset='zero destination raw output history',
        command='destination twist/head/body zero',external_filter=False,
        simulated_transitions=0,learning_updates=0,hardware_io=False,
        scope='CPU hypothetical first-request audit, no executed transition, applied actuator jump, stability or hardware claim')
    write(OUT/'boundary-requests.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
