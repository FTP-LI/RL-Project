"""CPU screen of one predeclared deployable stall detector, no intervention.

Replay measured actor inputs and raw requests from960saved development
episodes. This does not show that the correction would cause recovery. The
independent NumPy predicate and elapsed-time check reject input/timing bugs.
Do not search detector thresholds to make these existing outcomes pass.
"""
from pathlib import Path
import json,sys,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_early_guard import SETTINGS,detect,correction_step
from record_io import write
from s1_reference_inputs import read,sha


def main():
    draft=read(ROOT/'configs/experiments/getup_early_guard_draft.json')
    assert draft['settings']==SETTINGS
    source=read(ROOT/'configs/experiments/getup_early_support_audit.json')
    assert sha(ROOT/'configs/experiments/getup_late_feasibility_pair.json')==source['parent_sha256']
    m=read(ROOT/'configs/experiments/getup_late_feasibility_pair.json');p=m['arms']['original'];neck=p['neck_index']
    cfg=read(ROOT/'records/reports/getup-late-feasibility/preflight.json')['training_config']
    assert cfg['env']['observations']['actor']['terms']['joint_vel']['scale']=='null'
    results={};all_rows=[];torch.set_num_threads(2)
    for role,s in source['sources'].items():
        assert sha(ROOT/s['trace'])==s['sha256']
        proof=read(ROOT/s['proof']);stable=np.asarray(proof['end_hold']['stable_by_environment'],bool)
        with np.load(ROOT/s['trace']) as f:obs=f['observations'];raw=f['requested_targets'][...,neck]
        assert obs.shape==(550,320,61) and raw.shape==(550,320)
        count=torch.zeros(320,dtype=torch.long);latch=torch.zeros(320,dtype=torch.bool)
        np_count=np.zeros(320,np.int64);np_latch=np.zeros(320,bool);when=np.full(320,np.nan)
        eligible_samples=0
        for i in range(550):
            age=torch.full((320,),i,dtype=torch.long);o=torch.from_numpy(obs[i]);v=torch.from_numpy(raw[i])
            previous=latch.clone();count,latch,eligible=detect(count,latch,o,v,age,.02,p['model_home'][neck],neck,p['joint_limits'][neck][0])
            g=obs[i,:,3:6];vertical=-g[:,2]/np.linalg.norm(g,axis=-1)
            expected=(i>=100)&(vertical>=math.cos(math.radians(45)))&(vertical<=math.cos(math.radians(15)))
            expected &= (obs[i,:,6+neck]+p['model_home'][neck]<math.radians(-110))
            expected &= (abs(obs[i,:,20+neck])<=1)&(raw[i]<p['joint_limits'][neck][0])
            np_count=np.where(expected,np.minimum(np_count+1,16),0);np_latch|=np_count>=16
            np.testing.assert_array_equal(eligible.numpy(),expected);np.testing.assert_array_equal(count.numpy(),np_count)
            np.testing.assert_array_equal(latch.numpy(),np_latch)
            new=(latch&~previous).numpy();when[new]=i*.02;eligible_samples+=int(eligible.sum())
            if new.any():
                assert i>=115
                # Every new latch has16past/current observations, not just15.
                assert np_latch[new].all()
        rows=[]
        for i in range(320):
            row=dict(role=role,world=i,end_stable=bool(stable[i]),detected=bool(latch[i]),
                latch_seconds=None if not np.isfinite(when[i]) else float(when[i]));rows.append(row)
        results[role]=dict(episodes=320,successes=int(stable.sum()),failures=int((~stable).sum()),
            detected_successes=int((latch.numpy()&stable).sum()),detected_failures=int((latch.numpy()&~stable).sum()),
            earliest=None if not np.isfinite(when).any() else float(np.nanmin(when)),
            latest=None if not np.isfinite(when).any() else float(np.nanmax(when)),rows=rows,
            eligible_control_samples=eligible_samples)
        all_rows+=rows
        print('EARLY_GUARD_SAVED_SCREEN',role,json.dumps({k:v for k,v in results[role].items() if k!='rows'}),flush=True)
    # Correction must rise by at most0.04rad, vanish when the raw request clears
    # the bound, and never push a request above max(original,-75deg).
    previous=torch.tensor([0.,.2,.5]);active=torch.tensor([True,True,False]);raw=torch.tensor([-3.5,-1.,-3.5])
    correction=correction_step(previous,active,raw,.02)
    torch.testing.assert_close(correction,torch.tensor([.04,0.,0.]),atol=1e-7,rtol=0)
    assert not torch.cuda.is_initialized()
    result=dict(status='passed-cpu-only',roles=results,episodes=len(all_rows),
        script_sha256=sha(Path(__file__)),detector_sha256=sha(ROOT/'scripts/training/getup_early_guard.py'),
        definition_sha256=sha(ROOT/'configs/experiments/getup_early_guard_draft.json'),
        independent_numpy_matches=True,observations_span_seconds=.30,joint_velocity_scale=1.,
        new_physics_steps=0,new_learning_steps=0,actual_recoveries_demonstrated=0,heldout_used=False,
        scope='Development detector overlap only; no causal correction effect or independent reliability evidence')
    out=ROOT/'records/reports/getup-early-support';write(out/'guard-screen.json',result)


if __name__=='__main__':main()
