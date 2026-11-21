"""Recompute seven-command screening from saved first-episode traces on CPU.

No inference, learning or threshold tuning. Old five-command pass flags stay
unchanged; report the new per-command screening alongside descriptive posture.
Episode counts, not correlated video frames, define the success interval.
"""
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

from s1_reference_inputs import ROOT, read, sha
sys.path.insert(0, str(ROOT/'scripts/training'))
from training_watch import signed_foot_gap_mm

OUT = ROOT/'records/reports/s1-evaluation'
PROTOCOL = ROOT/'configs/experiments/walking_reference_suite.json'


def wilson(k, n):
    z = 1.959963984540054; p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-radius), min(1., center+radius)]


def main():
    state, protocol = read(OUT/'progress.json'), read(PROTOCOL)
    criteria = protocol['criteria']; inputs = {str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    rows, episodes, config_reference, package_reference = [], [], None, None
    for role, model in state['models'].items():
        assert sha(Path(model['model'])) == model['model_sha256']
        seen = []
        for part in ('five', 'lateral'):
            folder = Path(model[part]); report = read(folder/'evaluation.json')
            verification = read(folder/'verification.json')
            assert verification['status'] == 'passed'
            assert verification['report_sha256'] == sha(folder/'evaluation.json')
            assert report['model_sha256'] == model['model_sha256']
            assert report['source_sha256'] == protocol['training_source_sha256']
            assert (report['seed'], report['num_envs'], report['episode_seconds'], report['step_dt']) == (1001,32,20.,.02)
            assert report['physics_dt'] == .005
            assert report['linear_xy_error_definition'] == 'time_mean_planar_l2_then_episode_mean'
            assert not any(report['action_filter_alpha'])
            assert not report.get('diagnostic_variant') and not report.get('mechanism_variant')
            cfg = yaml.load((folder/'env.yaml').read_text(), Loader=yaml.BaseLoader)
            # Official play configuration retains actor observation noise. Bind
            # its actual setting to the saved equal configuration; do not assume
            # evaluation means noise-free or rewrite any collected trajectory.
            assert report['actor_corruption'] is (cfg['observations']['actor']['enable_corruption']=='true')
            if config_reference is None:
                config_reference, package_reference = cfg, report['packages']
            else:
                assert cfg == config_reference, 'Simulation configuration incompatible with reused traces'
                assert report['packages'] == package_reference, 'Package set changed'
            inputs[str((folder/'evaluation.json').relative_to(ROOT))] = sha(folder/'evaluation.json')
            inputs[str((folder/'env.yaml').relative_to(ROOT))] = sha(folder/'env.yaml')
            names = report['model_metadata']['joint_names'].split(',')
            hips = [names.index(s+'_hip_roll') for s in ('left','right')]
            head = [names.index(n) for n in ('neck_pitch','head_pitch','head_yaw','head_roll')]
            for item in report['conditions']:
                condition = 'stand' if item['condition'] == 'stop' else item['condition'].replace('_','-')
                assert condition not in seen; seen.append(condition)
                command = np.asarray(criteria['commands'][condition], float)
                np.testing.assert_array_equal(item['command'], command)
                path = folder/(item['condition']+'-trace.npz')
                inputs[str(path.relative_to(ROOT))] = sha(path)
                with np.load(path, allow_pickle=False) as d:
                    active = d['active']; counts = active.sum(axis=0); n = active.shape[1]
                    assert n == 32 and np.all(counts > 0)
                    mean = lambda value: (value*active).sum(axis=0)/counts
                    xy = mean(np.linalg.norm(d['velocity']-command[:2], axis=2))
                    yaw = mean(np.abs(d['yaw_rate']-command[2]))
                    survives = np.array([x['outcome']=='horizon_complete' for x in item['episodes']])
                    success = survives & (xy <= criteria['linear_xy_l2_error_max'][condition]) & (yaw <= criteria['yaw_error_max_rad_s'])
                    axis = int(np.argmax(np.abs(command)))
                    signed = mean(d['yaw_rate'] if axis == 2 else d['velocity'][:,:,axis])*np.sign(command[axis])
                    if condition == 'stand':
                        drift = np.array([x['final_displacement_m'] for x in item['episodes']])
                        success &= drift <= criteria['stand_drift_median_max_m']
                        directional_pass = True
                    else:
                        required = criteria['minimum_signed_command_axis_speed'][condition]
                        success &= signed >= required
                        directional_pass = bool(signed.mean() >= required)
                    gap = signed_foot_gap_mm(d['foot_positions'], d['root_quaternion'])
                    angles = np.rad2deg(d['joint_positions'][:,:,hips])
                    inward = (angles[:,:,0] < -20) & (angles[:,:,1] > 20)
                    after = active.copy(); after[:50] = False
                    # Head jitter is centered within each episode, then equally
                    # averaged; it does not treat successive frames as replicates.
                    head_rms = []
                    for i in range(n):
                        use = after[:,i]
                        if use.any(): head_rms.append(np.rad2deg(d['joint_positions'][use,i][:,head]).std(axis=0))
                    np.testing.assert_allclose(xy.mean(), item['linear_xy_mae_m_s'], rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(yaw.mean(), item['yaw_mae_rad_s'], rtol=1e-5, atol=1e-6)
                    row = dict(role=role, condition=condition, surviving=int(survives.sum()), episodes=n,
                        success_count=int(success.sum()), success_fraction=float(success.mean()),
                        success_wilson95=wilson(int(success.sum()), n),
                        linear_xy_l2_error=float(xy.mean()), yaw_error=float(yaw.mean()),
                        mean_signed_command_axis_speed=float(signed.mean()),
                        signed_foot_gap_median_mm=float(np.median(gap[active])),
                        episodes_with_negative_foot_projection=int(((gap<0)&active).any(axis=0).sum()),
                        inward_over20deg_fraction=float(inward[active].mean()),
                        head_centered_rms_deg_after1s=np.mean(head_rms,axis=0).tolist() if head_rms else None,
                        aggregate_tracking_pass=bool(survives.mean() >= criteria['survival_fraction_min']
                            and xy.mean() <= criteria['linear_xy_l2_error_max'][condition]
                            and yaw.mean() <= criteria['yaw_error_max_rad_s'] and directional_pass
                            and (condition != 'stand' or np.median(drift) <= criteria['stand_drift_median_max_m'])),
                        evaluation_record=str(folder.relative_to(ROOT)), source_part=part)
                    rows.append(row)
                    for i in range(n):
                        episodes.append(dict(role=role, condition=condition, episode=i,
                            survived=bool(survives[i]), success=bool(success[i]), linear_error=float(xy[i]),
                            yaw_error=float(yaw[i]), signed_speed=float(signed[i])))
        assert set(seen) == set(criteria['commands'])
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(PROTOCOL), analysis_sha256=sha(Path(__file__)),
        rows=rows, episode_rows=len(episodes), simulation_configs_equal=True,
        actor_observation_noise_enabled=config_reference['observations']['actor']['enable_corruption']=='true',
        packages=package_reference, input_sha256=inputs,
        scope='Development screening with known old five-command evidence; no heldout or multi-seed confirmation',
        posture_scope='Signed site projection and bilateral inward angle are diagnostic; no universal turn/strafe geometry pass is asserted',
        stable_walking_verified=False, automatic_new_training_allowed=False)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name, values in [('summary.csv',rows),('episodes.csv',episodes)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    print(json.dumps(dict(status='passed', conditions=len(rows), episodes=len(episodes),
        tracking_passes={role:[r['condition'] for r in rows if r['role']==role and r['aggregate_tracking_pass']]
                         for role in state['models']}),indent=2))


if __name__ == '__main__':
    main()
