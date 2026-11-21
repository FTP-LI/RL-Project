"""Read actual saved fixed-policy joint traces; no inference or simulation.

Checks whether hip-roll adduction predates live rendering. Include every active
frame, with no selected time window; samples are correlated within 32 episodes.
Requested targets and measured positions are kept separate. This cannot prove
which reward term caused a learned action preference.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation_run')
    a=p.parse_args()
    folder=ROOT/'records/evaluations'/a.evaluation_run
    assert folder.parent==ROOT/'records/evaluations'
    evaluation=json.loads((folder/'evaluation.json').read_text())
    verified=json.loads((folder/'verification.json').read_text())
    assert verified['status']=='passed' and verified['report_sha256']==sha(folder/'evaluation.json')
    names=evaluation['model_metadata']['joint_names'].split(',')
    # The evaluator records targets in policy HOME, then converts them into
    # simulation actions. Rounded ONNX HOME is distinct from simulation HOME.
    home=np.asarray(evaluation['policy_home'])
    scale=float(evaluation['model_metadata']['action_scale'])
    indices=[names.index(side+'_hip_roll') for side in ('left','right')]
    result=[]
    for condition in ('stop','forward','backward','turn_left','turn_right'):
        path=folder/f'{condition}-trace.npz'
        data=np.load(path,allow_pickle=False)
        bound=next(item for item in verified['checks'] if item['condition']==condition)
        assert bound['trace_sha256']==sha(path)
        assert all(k in data for k in ('joint_positions','requested_joint_targets','applied_actions'))
        active=data['active'].astype(bool)
        actual=data['joint_positions'][active][:,indices]
        target=data['requested_joint_targets'][active][:,indices]
        assert np.isfinite(actual).all() and np.isfinite(target).all()
        expected=home+scale*data['applied_actions']
        assert np.allclose(data['requested_joint_targets'][active],expected[active],atol=1e-5)
        result.append(dict(condition=condition,num_environments=int(active.shape[1]),
            active_samples=int(active.sum()),trace_sha256=sha(path),
            actual_deg_min_median_max=(np.quantile(actual,[0,.5,1],axis=0)*180/np.pi).tolist(),
            target_deg_min_median_max=(np.quantile(target,[0,.5,1],axis=0)*180/np.pi).tolist(),
            fraction_both_inward_beyond_20_deg=float(np.mean((actual[:,0]<-np.deg2rad(20))&(actual[:,1]>np.deg2rad(20)))),
            target_reconstruction_verified=True))
        # Match the existing repair diagnostic's signed body-left foot gap.
        # A negative site projection is not a measured collision/contact force.
        if all(k in data for k in ('root_quaternion','foot_positions')):
            quat=data['root_quaternion']
            np.testing.assert_allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-4)
            w,x,y,z=np.moveaxis(quat,-1,0)
            left=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
            gap=((data['foot_positions'][:,:,0]-data['foot_positions'][:,:,1])*left).sum(axis=-1)
            assert np.isfinite(gap[active]).all()
            result[-1].update(median_signed_foot_gap_mm=float(np.median(gap[active])*1000),
                minimum_signed_foot_gap_mm=float(gap[active].min()*1000),
                episodes_with_crossed_foot_sites=int(((gap<0)&active).any(axis=0).sum()))
    output=ROOT/'records/reports/saved-joint-posture'/a.evaluation_run
    output.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),evaluation_run=a.evaluation_run,
        model_sha256=evaluation['model_sha256'],evaluation_sha256=sha(folder/'evaluation.json'),
        script_sha256=sha(Path(__file__)),joint_order=['left_hip_roll','right_hip_roll'],conditions=result,
        samples_are_independent_episodes=False,all_active_frames_included=True,simulation_started=False,
        scope='Existing fixed-policy actual joint traces; model provenance is distinct from current changing training policy')
    (output/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
