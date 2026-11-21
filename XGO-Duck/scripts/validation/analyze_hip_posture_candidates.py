"""Compare existing posture reward functions on saved actual joint traces.

Why: penalizing requested targets did not remove the measured inward hip pose.
What changes: nothing in training. Run the installed Gaussian pose reward and
two unchanged official joint-position penalties on saved observations, then
independently recompute them in NumPy. A reward-only replacement of the two hip
angles by HOME probes saturation; it is NOT a physical intervention or a new
policy rollout. No weights are selected and no learning benefit is claimed.
"""
from __future__ import annotations

import __future__
import ast
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
OUT=ROOT/'records/reports/hip-posture-candidates'
MODELS={
    'trained_control':'custom_onnx_reference_20261101T190142_s1001',
    'target_penalty':'custom_onnx_reference_20261101T191205_s1001',
    'official_reference':'custom_onnx_reference_20261101T195044_s1001',
}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    source=ROOT/'official_resources/xgoduck_rl/src/mjlab_microduck/tasks/mdp.py'
    assert sha(source)==sha(WORK/'src/mjlab_microduck/tasks/mdp.py')
    functions=['joint_deviation_l1','joint_pos_limit_proximity']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in functions]
    assert len(nodes)==2
    namespace={'torch':torch,'_DEFAULT_ASSET_CFG':SceneEntityCfg('robot')}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
    config_path=ROOT/'records/experiments/head_command_repair_control_20261101T210228_s42/params/env.yaml'
    config_text=config_path.read_text()
    config=yaml.load(config_text,Loader=yaml.BaseLoader)
    # BaseLoader drops !!python/name values. Verify callable names against the
    # original YAML tags, rather than accepting empty parsed function strings.
    callable_tags=re.findall(r'!!python/name:([^\s]+)',config_text)
    assert 'mjlab.tasks.velocity.mdp.rewards.variable_posture' in callable_tags
    assert not any(f'mjlab_microduck.tasks.mdp.{name}' in callable_tags for name in functions)
    pose=config['rewards']['pose'];params=pose['params']
    metadata_path=ROOT/'records/experiments/head_command_repair_control_20261101T213026_s42/training-model-metadata.json'
    metadata=read(metadata_path);names=metadata['joint_names'];limits=np.asarray(metadata['joint_ranges_rad'])
    legs=[i for i,n in enumerate(names) if re.fullmatch(params['asset_cfg']['joint_names'][0],n)]
    hips=[names.index(side+'_hip_roll') for side in ('left','right')]
    assert len(legs)==10 and len(hips)==2
    stds={name:{pattern:float(value) for pattern,value in params[name].items()}
          for name in ('std_standing','std_walking','std_running')}
    thresholds={name:float(params[name]) for name in ('walking_threshold','running_threshold')}
    leg_cfg=SceneEntityCfg('robot',joint_names=tuple(names[i] for i in legs),joint_ids=legs)
    hip_cfg=SceneEntityCfg('robot',joint_names=tuple(names[i] for i in hips),joint_ids=hips)
    margin=.15  # Existing official function's default, not tuned on these data.
    assert np.all(limits[hips,1]-limits[hips,0]>2*margin)
    rows=[];episodes=[];inputs={str(config_path.relative_to(ROOT)):sha(config_path),str(metadata_path.relative_to(ROOT)):sha(metadata_path)}
    provenance={};max_errors={name:0.0 for name in ('pose','home_l1','limit_proximity','hip_home_reward_only')}
    for label,run_id in MODELS.items():
        folder=ROOT/'records/evaluations'/run_id;report=read(folder/'evaluation.json');check=read(folder/'verification.json')
        assert check['status']=='passed' and check['report_sha256']==sha(folder/'evaluation.json')
        assert report['model_metadata']['joint_names'].split(',')==names
        assert (report['num_envs'],report['episode_seconds'],report['seed'])==(32,20.0,1001)
        assert len(report['conditions'])==5
        if label!='official_reference':
            training_run=('gait_repair_control_' if label=='trained_control' else 'gait_repair_target_envelope_')+'20261101T174853_s42'
            audit_path=ROOT/'records/experiments'/training_run/'checkpoint-checks/model_5249.json'
            audit=read(audit_path)
            assert audit['status']=='passed' and audit['final_budget_verified'] and audit['cumulative_completed_updates']==5250
            manifest_path=ROOT/'records/experiments'/training_run/'artifact-manifest.json'
            model=ROOT/'artifacts'/training_run/'policy.onnx'
            assert sha(model)==report['model_sha256']
            inputs[str(audit_path.relative_to(ROOT))]=sha(audit_path)
            inputs[str(manifest_path.relative_to(ROOT))]=sha(manifest_path)
        provenance[label]=dict(run_id=run_id,model_sha256=report['model_sha256'],seed=1001,
            parent_budget='5250 completed updates, one seed' if label!='official_reference' else 'Official model; training history unknown')
        inputs[str((folder/'evaluation.json').relative_to(ROOT))]=sha(folder/'evaluation.json')
        inputs[str((folder/'verification.json').relative_to(ROOT))]=sha(folder/'verification.json')
        for condition in report['conditions']:
            name=condition['condition'];path=folder/f'{name}-trace.npz';data=np.load(path,allow_pickle=False)
            binding=next(c for c in check['checks'] if c['condition']==name)
            assert binding['trace_sha256']==sha(path);inputs[str(path.relative_to(ROOT))]=sha(path)
            q=data['joint_positions'];active=data['active'].astype(bool)
            home=np.broadcast_to(np.asarray(report['simulation_home'],dtype=np.float32),q.shape).copy()
            command=np.asarray(condition['command'],dtype=np.float32)
            assert command.shape==(3,)
            command=np.broadcast_to(command,(*active.shape,3)).copy()
            shape=active.shape
            actual=torch.tensor(q.reshape(-1,14),dtype=torch.float32)
            default=torch.tensor(home.reshape(-1,14),dtype=torch.float32)
            asset=SimpleNamespace(data=SimpleNamespace(joint_pos=actual,default_joint_pos=default,
                joint_pos_limits=torch.tensor(limits,dtype=torch.float32)[None]))
            asset.find_joints=lambda patterns: (legs,[names[i] for i in legs])
            env=SimpleNamespace(scene={'robot':asset},device='cpu',
                command_manager=SimpleNamespace(get_command=lambda _:torch.tensor(command.reshape(-1,3))))
            cfg=RewardTermCfg(func=variable_posture,weight=float(pose['weight']),
                params=dict(asset_cfg=leg_cfg,command_name='twist',**stds,**thresholds))
            reward=variable_posture(cfg,env)
            values={'pose':reward(env,**cfg.params).numpy(),
                'home_l1':namespace['joint_deviation_l1'](env,asset_cfg=hip_cfg).numpy(),
                'limit_proximity':namespace['joint_pos_limit_proximity'](env,asset_cfg=hip_cfg,margin=margin).numpy()}
            # Evaluate the reward formula only; never feed these edited angles
            # to physics, a policy, the frozen worker or the saved trace files.
            original=asset.data.joint_pos
            asset.data.joint_pos=original.clone();asset.data.joint_pos[:,hips]=default[:,hips]
            values['hip_home_reward_only']=reward(env,**cfg.params).numpy()
            asset.data.joint_pos=original
            speed=np.linalg.norm(command[...,:2],axis=-1)+np.abs(command[...,2])
            std=np.empty((*shape,len(legs)))
            for mode,mask in [('std_standing',speed<thresholds['walking_threshold']),
                ('std_walking',(speed>=thresholds['walking_threshold'])&(speed<thresholds['running_threshold'])),
                ('std_running',speed>=thresholds['running_threshold'])]:
                resolved=[]
                for i in legs:
                    matches=[v for pattern,v in stds[mode].items() if re.fullmatch(pattern,names[i])]
                    assert len(matches)==1;resolved.append(matches[0])
                std[mask]=resolved
            error=(q-home)[...,legs];squared=(error/std)**2
            counter=squared.copy()
            counter[...,[legs.index(i) for i in hips]]=0
            direct={'pose':np.exp(-squared.mean(axis=-1)),
                'home_l1':np.abs(q[...,hips]-home[...,hips]).sum(axis=-1),
                'limit_proximity':(np.maximum(limits[hips,0]+margin-q[...,hips],0)+np.maximum(q[...,hips]-limits[hips,1]+margin,0)).sum(axis=-1),
                'hip_home_reward_only':np.exp(-counter.mean(axis=-1))}
            for key in values:
                values[key]=values[key].reshape(shape)
                np.testing.assert_allclose(values[key][active],direct[key][active],atol=2e-6,rtol=2e-5)
                max_errors[key]=max(max_errors[key],float(np.max(np.abs(values[key][active]-direct[key][active]))))
            quat=data['root_quaternion'];np.testing.assert_allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-4)
            w,x,y,z=np.moveaxis(quat,-1,0);left=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
            gap=((data['foot_positions'][:,:,0]-data['foot_positions'][:,:,1])*left).sum(axis=-1)*1000
            inward=(q[...,hips[0]]<-np.deg2rad(20))&(q[...,hips[1]]>np.deg2rad(20))
            hip_exp=(squared[...,[legs.index(i) for i in hips]].sum(axis=-1)/len(legs))
            metrics={**values,'signed_foot_gap_mm':gap,'both_hips_inward_over20deg':inward,
                'pose_below_one_percent':values['pose']<.01,'hip_contribution_to_exponent':hip_exp,
                'limit_penalty_active':values['limit_proximity']>1e-6}
            for window,start in [('full_episode',0),('after_first_second',round(1/report['step_dt']))]:
                mask=active.copy();mask[:start]=False;items=[]
                for index in range(shape[1]):
                    valid=mask[:,index]
                    if not valid.any():continue
                    item=dict(model=label,condition=name,window=window,env=index,samples=int(valid.sum()),
                        **{key:float(v[valid,index].mean()) for key,v in metrics.items()})
                    items.append(item);episodes.append(item)
                assert items
                rows.append(dict(model=label,condition=name,window=window,episodes=len(items),
                    episode_mean={key:float(np.mean([v[key] for v in items])) for key in metrics}))
            data.close()
    for relative,digest in inputs.items():assert sha(ROOT/relative)==digest,relative
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)
    (OUT/'official-functions.py').write_text('\n\n'.join(ast.get_source_segment(source.read_text(),n) for n in nodes)+'\n')
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,hip_joint_names=[names[i] for i in hips],pose_joint_names=[names[i] for i in legs],
        pose_weight=float(pose['weight']),hip_pose_std_rad=.05,limit_proximity_margin_rad=margin,
        limit_proximity_zero_interval_deg=np.rad2deg(limits[hips]+np.array([margin,-margin])).tolist(),
        results=rows,episode_rows=len(episodes),max_numpy_recompute_errors=max_errors,
        installed_pose_source=str(Path(inspect.getfile(variable_posture))),installed_pose_source_sha256=sha(Path(inspect.getfile(variable_posture))),
        official_source_sha256=sha(source),script_sha256=sha(Path(__file__)),
        inputs_sha256=inputs,episode_csv_sha256=sha(OUT/'episodes.csv'),
        unweighted_candidate_costs=True,training_reward_weight_selected=False,training_entry_changed=False,
        hypothetical_pose_only_not_simulated=True,learning_improvement_verified=False,cuda_initialized=False,
        scope='Saved actual first-episode traces and unchanged reward functions on CPU; descriptive diagnostic. Different model histories are not a controlled training comparison.')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs_sha256','results')},indent=2))
    print(json.dumps([v for v in rows if v['condition']=='forward' and v['window']=='after_first_second'],indent=2))


if __name__=='__main__':main()
