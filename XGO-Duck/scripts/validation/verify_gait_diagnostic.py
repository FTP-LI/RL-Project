"""Recompute posture and verify one-factor contracts of the saved diagnosis.

Uses actual joint/foot/quaternion traces and existing independent gait/video
checks. No simulation or policy call. Small fixed-weight interventions describe
sensitivity and cannot certify a repaired trained policy or hardware behavior.
"""
import argparse
import ast
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path);args=parser.parse_args()
    folder=args.batch.resolve();assert folder.parent==ROOT/'records/evaluations'
    batch=read(folder/'batch.json');assert batch['status']=='completed-and-verified'
    plan=read(folder/'gait_mechanism_diagnostic.json')
    assert sha(folder/'gait_mechanism_diagnostic.json')==batch['protocol_sha256']
    assert [r['variant'] for r in batch['runs']]==plan['variants']
    # The interrupted batch continued with one logging fix. Compare executable
    # ASTs so the missing group cannot silently change simulation or inference.
    original=folder/'evaluate_fixed_policy.py'
    recovered=folder/'recovery/evaluate_fixed_policy.py'
    old_tree=ast.parse(original.read_text());new_tree=ast.parse(recovered.read_text())
    class CopyRecordedActions(ast.NodeTransformer):
        changed=False
        def visit_Call(self,node):
            if not self.changed and ast.unparse(node)=='trace[key].append(value)':
                node.args[0]=ast.parse("value.copy() if key == 'actions' else value",mode='eval').body
                self.changed=True
            return self.generic_visit(node)
    fix=CopyRecordedActions();expected_tree=fix.visit(old_tree)
    assert fix.changed and ast.dump(expected_tree)==ast.dump(new_tree), 'Unexpected evaluator change during batch recovery'
    tables=[];inputs={};reference_cfg=None
    for run in batch['runs']:
        record=Path(run['path']);variant=run['variant'];report=read(record/'evaluation.json')
        verification=read(record/'verification.json')
        assert verification['status']=='passed' and verification['report_sha256']==sha(record/'evaluation.json')
        assert sha(record/'evaluate_fixed_policy.py')==report['evaluator_sha256']
        assert report['evaluator_sha256']==sha(recovered if variant=='hip_roll_guard' else original)
        assert report['mechanism_variant']==variant and report['model_sha256']==plan['model_sha256']
        assert (report['seed'],report['num_envs'],report['episode_seconds'])==(1001,8,8.0)
        assert not report['training_started'] and not report['meets_all_reference_criteria']
        cfg=yaml.load((record/'env.yaml').read_text(),Loader=yaml.BaseLoader)
        if reference_cfg is None:reference_cfg=cfg
        else:
            comparable=copy.deepcopy(cfg)
            if variant=='foot_contact':
                for key in ('contype','conaffinity'):
                    comparable['scene']['entities']['robot']['collisions'][0][key]=reference_cfg['scene']['entities']['robot']['collisions'][0][key]
            assert comparable==reference_cfg, 'Unexpected configuration difference: '+variant
        names=report['model_metadata']['joint_names'].split(',')
        roll=[names.index(side+'_hip_roll') for side in ('left','right')]
        home=np.asarray(report['policy_home']);scale=float(report['model_metadata']['action_scale'])
        for item in report['conditions']:
            trace=record/(item['condition']+'-trace.npz');data=np.load(trace,allow_pickle=False)
            check=next(c for c in verification['checks'] if c['condition']==item['condition'])
            assert check['trace_sha256']==sha(trace)
            active=data['active'];raw=home+scale*data['applied_actions']
            np.testing.assert_allclose(data['policy_joint_targets'],raw,atol=1e-5)
            requested=raw.copy()
            if variant=='hip_roll_guard':
                bound=np.deg2rad(plan['hip_roll_target_guard_deg'])
                requested[:,:,roll]=np.clip(requested[:,:,roll],-bound,bound)
            np.testing.assert_allclose(data['requested_joint_targets'],requested,atol=1e-5)
            q=data['root_quaternion'];np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1,atol=1e-4)
            w,x,y,z=np.moveaxis(q,-1,0)
            body_left=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
            delta=data['foot_positions'][:,:,0]-data['foot_positions'][:,:,1]
            gap=np.sum(delta*body_left,axis=-1)
            angles=np.rad2deg(data['joint_positions'][:,:,roll])
            self_contact=data['self_collision'].reshape(*active.shape,-1).any(axis=-1)
            assert np.isfinite(gap[active]).all() and np.isfinite(angles[active]).all()
            row=dict(variant=variant,condition=item['condition'],episodes=int(active.shape[1]),
                active_samples=int(active.sum()),left_roll_median_deg=float(np.median(angles[:,:,0][active])),
                right_roll_median_deg=float(np.median(angles[:,:,1][active])),
                min_lateral_gap_mm=float(gap[active].min()*1000),median_lateral_gap_mm=float(np.median(gap[active])*1000),
                crossed_lateral_sample_fraction=float(np.mean(gap[active]<0)),
                episodes_with_crossed_lateral_sites=int(((gap<0)&active).any(axis=0).sum()),
                both_inward_over20deg_sample_fraction=float(np.mean((angles[:,:,0][active]<-20)&(angles[:,:,1][active]>20))),
                self_collision_sample_fraction=float(np.mean(self_contact[active])),
                full_episode_survival=item['full_episode_survival'],fall_count=item['fall_count_excluding_timeouts'],
                mean_forward_velocity_m_s=float(np.mean((data['velocity'][:,:,0]*active).sum(axis=0)/active.sum(axis=0))),
                linear_xy_mae_m_s=item['linear_xy_mae_m_s'],yaw_mae_rad_s=item['yaw_mae_rad_s'],
                run_id=report['run_id'])
            tables.append(row);inputs[str(trace.relative_to(ROOT))]=sha(trace)
        inputs[str((record/'evaluation.json').relative_to(ROOT))]=sha(record/'evaluation.json')
        inputs[str((record/'env.yaml').relative_to(ROOT))]=sha(record/'env.yaml')
    with (folder/'posture-comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        batch=folder.name,protocol_sha256=batch['protocol_sha256'],analysis_sha256=sha(Path(__file__)),
        one_factor_configuration_verified=True,requested_target_mapping_verified=True,
        evaluator_change_only_action_copy_verified=True,
        inputs_sha256=inputs,results=tables,training_started=False,repaired_gait_verified=False,
        scope='Two conditions, eight episodes each, eight seconds; exploratory fixed-weight diagnosis, not formal or hardware acceptance')
    (folder/'posture-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
