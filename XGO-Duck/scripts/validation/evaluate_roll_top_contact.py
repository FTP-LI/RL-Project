"""One visible fixed-policy roll screen with explicit post-export binding.

The model hash is supplied only after checkpoint/export verification. Physics,
seed, episode budget and criteria were frozen before learning. No optimization
or source reward change occurs in this evaluation.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import shutil
import sys
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from collect_roll_policy import evaluate
from verify_roll_reference import independent

OUT=ROOT/'records/reports/roll-top-contact'


def verify(folder):
    folder=Path(folder);p=read(folder/'protocol.json');r=read(folder/'evaluation.json');role=p['role']
    assert r['status']=='completed' and sha(folder/'protocol.json')==r['protocol_sha256']
    assert p['frozen_master_sha256']==sha(ROOT/'configs/experiments/roll_top_contact_pair.json')
    assert sha(ROOT/p['model'])==p['model_sha256']==r['model_sha256']
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==read(OUT/'preflight.json')['evaluation_config']
    for name,digest in p['source_sha256'].items():assert sha(folder/Path(name).name)==digest
    assert sha(folder/'expanded-model-initial.npz')==r['expanded_model_sha256']
    item=r['conditions'][0];path=folder/'forward-roll-trace.npz';assert sha(path)==item['trace_sha256']
    with np.load(path) as value:t={k:value[k].copy() for k in value.files}
    np.testing.assert_allclose(t['observations'][1:,:,34:48][t['active'][1:]],t['raw_policy_actions'][:-1][t['active'][1:]],atol=1e-7,rtol=0)
    np.testing.assert_allclose(t['observations'][:,:,48:][t['active']],0.,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['applied_targets'][1:],np.asarray(r['policy_home'])+t['raw_policy_actions'][1:],atol=1e-6,rtol=0)
    rows=independent(t,p['criteria'],np.asarray(r['joint_limits']),np.deg2rad(p['actual_joint_limit_tolerance_deg']))
    for observed,saved in zip(rows,item['episodes']):
        for key,v in observed.items():
            if v is None or isinstance(v,(bool,int)):assert v==saved[key]
            else:np.testing.assert_allclose(v,saved[key],atol=2e-4,rtol=0)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    replayed=0;maximum=0.
    for step in range(1,451,30):
        for index in np.flatnonzero(t['active'][step]):
            action=session.run(None,{'obs':t['observations'][step,index:index+1]})[0][0]
            np.testing.assert_allclose(action,t['raw_policy_actions'][step,index],atol=2e-5,rtol=2e-5)
            maximum=max(maximum,float(abs(action-t['raw_policy_actions'][step,index]).max()));replayed+=1
    video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
    assert meta['fps']==25 and meta['size']==(960,720) and frames==len(item['video']['frame_times_seconds'])
    k=sum(x['success'] for x in rows);assert len(rows)==32 and k==item['successes']
    z=1.959963984540054;f=k/32;den=1+z*z/32;c=(f+z*z/64)/den;half=z*np.sqrt(f*(1-f)/32+z*z/(4*32**2))/den
    result=dict(status='passed',role=role,episodes=32,successes=k,numeric_gate_passed=f>=.9,
        wilson95=[max(0.,c-half),min(1.,c+half)],rows=rows,evaluator=str(folder.relative_to(ROOT)),
        protocol_sha256=sha(folder/'protocol.json'),report_sha256=sha(folder/'evaluation.json'),
        trace_sha256=sha(path),video_sha256=sha(video),decoded_frames=frames,
        replayed_inputs=replayed,maximum_inference_error=maximum,learning_updates=0,heldout_used=False,
        script_sha256=sha(Path(__file__)))
    write(folder/'verification.json',result);write(OUT/role/'evaluation-analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result


def run(binding):
    assert os.isatty(1),'Use the shared visible task terminal'
    p=read(Path(binding));role=p['role'];out=OUT/role;out.mkdir(exist_ok=True)
    assert not (out/'evaluation-progress.json').exists()
    assert p['frozen_master_sha256']==sha(ROOT/'configs/experiments/roll_top_contact_pair.json')
    assert sha(ROOT/p['model'])==p['model_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest
    run_id=f'duck_roll_top-contact-eval_{role}_s1004_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),role=role,conditions=[],
        model=str(ROOT/p['model']),model_sha256=p['model_sha256'],protocol_sha256=sha(Path(binding)),learning_updates=0,
        started_at=datetime.now(timezone.utc).isoformat())
    def save():write(folder/'evaluation.json',state);write(out/'evaluation-progress.json',state)
    save();shutil.copy2(binding,folder/'protocol.json')
    for name in p['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    from evaluate_standup_reference import Tee
    with (folder/'execution.log').open('a',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:evaluate(p,folder,state);state['status']='completed'
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:state['ended_at']=datetime.now(timezone.utc).isoformat();save();sys.stdout,sys.stderr=oldout,olderr
    return verify(folder)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binding');args=p.parse_args();run(args.binding)
