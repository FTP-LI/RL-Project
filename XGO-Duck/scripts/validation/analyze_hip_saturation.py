"""Describe requested-versus-actual hip motion on saved forward episodes.

This is post-hoc diagnosis, not a new acceptance metric or causal proof. No
new rollout or policy inference is performed. Keep episode weighting and bind
the physical ranges, traces and evaluation reports used by the comparison.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha

OUT=ROOT/'records/reports/s2-hip-repair'


def main():
    state=read(OUT/'progress.json');assert state['status']=='passed'
    s1=read(ROOT/'records/reports/s1-evaluation/progress.json')
    models={'official-reference':s1['models']['official-reference'],'parent':s1['models']['reachable_head'],**state['models']}
    record=Path(state['models']['control']['record'])
    meta_path=record/'training-model-metadata.json';meta=read(meta_path)
    names=meta['joint_names'];hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    limits=np.asarray(meta['joint_ranges_rad'])[hips];rows=[]
    inputs={str(meta_path.relative_to(ROOT)):sha(meta_path)}
    for role,entry in models.items():
        folder=Path(entry['five']);report=read(folder/'evaluation.json')
        assert report['model_metadata']['joint_names'].split(',')==names
        assert read(folder/'verification.json')['status']=='passed'
        path=folder/'forward-trace.npz';inputs[str(path.relative_to(ROOT))]=sha(path)
        inputs[str((folder/'evaluation.json').relative_to(ROOT))]=sha(folder/'evaluation.json')
        with np.load(path,allow_pickle=False) as data:
            mask=data['active'][50:];q=data['joint_positions'][50:,:,hips]
            target=data['requested_joint_targets'][50:,:,hips]
            assert mask.sum(axis=0).min()>0
            # Lower stop is inward on the left; upper stop is inward on the right.
            extra=np.stack((np.maximum(limits[0,0]-target[:,:,0],0),
                            np.maximum(target[:,:,1]-limits[1,1],0)),axis=-1)
            per_episode=lambda x:np.asarray([x[:,i][mask[:,i]].mean(axis=0) for i in range(32)])
            exceed=per_episode((extra>0).astype(float));outside=per_episode((extra>0).all(axis=-1).astype(float))
            actual=per_episode(np.rad2deg(q));requested=per_episode(np.rad2deg(target));overshoot=per_episode(np.rad2deg(extra))
            near=(np.abs(q[:,:,0]-limits[0,0])<=np.deg2rad(1))&(np.abs(q[:,:,1]-limits[1,1])<=np.deg2rad(1))
            rows.append(dict(role=role,episodes=32,window='after first second',
                mean_actual_hip_degrees=actual.mean(axis=0).tolist(),
                mean_requested_hip_degrees=requested.mean(axis=0).tolist(),
                mean_inward_target_excess_degrees=overshoot.mean(axis=0).tolist(),
                target_beyond_inward_limit_fraction=exceed.mean(axis=0).tolist(),
                both_targets_beyond_inward_limits_fraction=float(outside.mean()),
                both_actual_hips_within1deg_of_inward_stops_fraction=float(per_episode(near.astype(float)).mean())))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),rows=rows,
        hard_hip_ranges_degrees=np.rad2deg(limits).tolist(),input_sha256=inputs,script_sha256=sha(Path(__file__)),
        threshold_selected_before_training=False,scope='Post-hoc descriptive saturation diagnostic; does not establish which reward or budget would fix learning',
        new_training=False,new_simulation=False,new_inference=False)
    (OUT/'saturation-diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
