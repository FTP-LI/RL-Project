"""Independently reconstruct original roll outcomes and output filtering.

Retain original contact/rotation/landing criteria. Validate raw action history,
exact EMA recurrence, selected physical targets, model replay, configuration,
source hashes and complete real video before comparing motion outcomes.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from verify_roll_reference import independent
from roll_action_memory import ActionMemory

OUT=ROOT/'records/reports/roll-action-memory'


def verify(arm):
    state=read(OUT/arm/'progress.json');assert state['status']=='completed'
    folder=Path(state['path']);plan=read(folder/'protocol.json');report=read(folder/'evaluation.json')
    pre=read(OUT/'preflight.json')
    assert plan['arm']==arm and report['protocol_sha256']==sha(folder/'protocol.json')==pre['arm_protocol_sha256'][arm]
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==pre['config']
    assert sha(ROOT/plan['model'])==plan['model_sha256']==report['model_sha256']
    for source,digest in plan['source_sha256'].items():assert sha(folder/Path(source).name)==digest
    assert sha(folder/'expanded-model-initial.npz')==report['expanded_model_sha256']
    item=report['conditions'][0];path=folder/'forward-roll-trace.npz';assert sha(path)==item['trace_sha256']
    with np.load(path) as saved:t={key:saved[key].copy() for key in saved.files}
    assert t['raw_policy_actions'].shape==(451,32,14) and t['active'].shape==(451,32)
    assert np.isfinite(t['observations'][t['active']]).all()
    np.testing.assert_allclose(t['observations'][:,:,48:][t['active']],0,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['observations'][1:,:,34:48][t['active'][1:]],t['raw_policy_actions'][:-1][t['active'][1:]],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(t['filtered_policy_actions'][0],0.)
    memory=ActionMemory(32)
    for step in range(1,451):
        filtered=memory.update(t['raw_policy_actions'][step])
        np.testing.assert_allclose(t['filtered_policy_actions'][step],filtered,atol=1e-7,rtol=0)
        memory.reset(np.flatnonzero(t['terminated'][step]|t['truncated'][step]))
    expected=t['raw_policy_actions'] if arm=='control' else t['filtered_policy_actions']
    np.testing.assert_array_equal(t['applied_policy_actions'],expected)
    np.testing.assert_allclose(t['applied_targets'][1:],np.asarray(report['policy_home'])+expected[1:],atol=1e-6,rtol=0)
    rows=independent(t,plan['criteria'],np.asarray(report['joint_limits']),np.deg2rad(plan['actual_joint_limit_tolerance_deg']))
    for current,original in zip(rows,item['episodes']):
        for key,value in current.items():
            if value is None or isinstance(value,(bool,int)):assert value==original[key]
            else:np.testing.assert_allclose(value,original[key],atol=2e-4,rtol=0)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    replayed=0;error=0.
    for step in range(1,451,30):
        for index in np.flatnonzero(t['active'][step]):
            raw=session.run(None,{'obs':t['observations'][step,index:index+1]})[0][0]
            np.testing.assert_allclose(raw,t['raw_policy_actions'][step,index],atol=2e-5,rtol=2e-5)
            replayed+=1;error=max(error,float(abs(raw-t['raw_policy_actions'][step,index]).max()))
    video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
    assert meta['fps']==25 and meta['size']==(960,720) and frames==len(item['video']['frame_times_seconds'])
    n=len(rows);k=sum(x['success'] for x in rows);assert n==32 and item['successes']==k
    assert item['numeric_gate_passed']==(k/n>=plan['success_fraction_min'])
    z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),arm=arm,successes=k,episodes=n,
        numeric_gate_passed=k/n>=plan['success_fraction_min'],wilson95=[max(0.,center-half),min(1.,center+half)],
        rows=rows,trace_sha256=sha(path),report_sha256=sha(folder/'evaluation.json'),protocol_sha256=sha(folder/'protocol.json'),
        video_sha256=sha(video),decoded_frames=frames,replayed_inputs=replayed,max_inference_error=error,script_sha256=sha(Path(__file__)),
        learning_updates=0,scope=plan['scope'])
    write(folder/'verification.json',result);write(OUT/arm/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result
