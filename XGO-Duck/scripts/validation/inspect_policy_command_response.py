"""CPU command sensitivity at identical saved standing states.

Reuse the current fixed official model, not the older archive used for the
historical amplitude trial. Change only the three twist inputs, retain raw
action history and all measured state inputs. This cannot measure resulting
motion; it screens whether a bounded physical response probe is worthwhile.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def commands():
    result={'stand':[0.,0.,0.]}
    for label,axis,values in (
        ('forward',0,(.15,.25,.4)),('backward',0,(-.1,-.15,-.2)),
        ('turn-left',2,(.4,.7,1.)),('turn-right',2,(-.4,-.7,-1.)),
        ('strafe-left',1,(.08,.16,.24)),('strafe-right',1,(-.08,-.16,-.24))):
        for i,value in enumerate(values):
            command=[0.,0.,0.];command[axis]=value;result[f'{label}-{i}']=command
    return result


def main():
    out=ROOT/'records/reports/command-response';out.mkdir(exist_ok=True)
    audit=read(ROOT/'records/reports/skill-interfaces/analysis.json')
    model=next(x for x in audit['models'] if x['role']=='walk-fixed-reference')
    path=ROOT/model['path'];assert sha(path)==model['sha256']
    record=ROOT/read(ROOT/'records/reports/official-actor-bootstrap/screen-analysis.json')['rows']['official']['stop']['record']
    checked=read(record/'verification.json');assert checked['status']=='passed'
    assert checked['report_sha256']==sha(record/'evaluation.json')
    trace_path=record/'stop-trace.npz'
    assert sha(trace_path)==next(x['trace_sha256'] for x in checked['checks'] if x['condition']=='stop')
    with np.load(trace_path) as trace:observation=trace['observations'][-1].copy();active=trace['active'][-1]
    assert observation.shape==(32,61) and active.all() and np.isfinite(observation).all()
    np.testing.assert_allclose(observation[:,48:61],0.,atol=1e-7,rtol=0)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    names=model['joint_names'];legs=[i for i,x in enumerate(names) if not x.startswith(('head_','neck_'))]
    raw=[];rows=[];cmds=commands();baseline=None
    for label,command in cmds.items():
        assert -.4<=command[0]<=.4 and abs(command[1])<=.3 and abs(command[2])<=1.
        actor=observation.copy();actor[:,48:51]=command
        assert np.array_equal(actor[:,:48],observation[:,:48]) and np.array_equal(actor[:,51:],observation[:,51:])
        actions=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in actor]);assert np.isfinite(actions).all()
        if baseline is None:baseline=actions.copy()
        delta=np.rad2deg(actions-baseline)
        rows.append(dict(condition=label,command=command,states=32,
            leg_output_delta_rms_deg=float(np.sqrt(np.mean(delta[:,legs]**2))),
            maximum_output_delta_deg=float(abs(delta).max()),
            per_joint_mean_delta_deg={k:float(v) for k,v in zip(names,delta.mean(axis=0))}))
        raw.append(actions)
    packet=out/'cpu-command-inputs.npz'
    np.savez_compressed(packet,standing_inputs=observation,commands=np.asarray(list(cmds.values()),np.float32),raw_outputs=np.stack(raw))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),conditions=rows,
        model=model,trace=str(trace_path.relative_to(ROOT)),trace_sha256=sha(trace_path),
        inputs=str(packet.relative_to(ROOT)),inputs_sha256=sha(packet),script_sha256=sha(Path(__file__)),
        model_calls=32*len(cmds),only_twist_inputs_changed=True,history_preserved=True,
        simulation_steps=0,learning_updates=0,hardware_io=False,
        scope='Static first-output sensitivity; no directional motion, calibration, stability or deployment claim')
    write(out/'cpu-response.json',result)
    print(json.dumps({k:result[k] for k in ('status','model_calls','scope')},indent=2))
    for row in rows:print(row['condition'],row['command'],'leg delta RMS degrees',round(row['leg_output_delta_rms_deg'],4))


if __name__=='__main__':main()
