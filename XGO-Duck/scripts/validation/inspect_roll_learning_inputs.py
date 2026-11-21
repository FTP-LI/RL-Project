"""Audit the actual roll model and source training recipe before learning.

No environment, optimizer, CUDA context or hypothetical training result.
Record model topology, normalizer/home contract, reward functions and action
offset support so a fresh bounded adaptation does not reuse walking-specific
assumptions or alter the rejected starting-pose experiment silently.
"""
from dataclasses import asdict,fields
from pathlib import Path
import inspect
import json
import onnx
from onnx import numpy_helper
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write


def main():
    import mjlab.tasks
    from mjlab.scripts.train import TrainConfig
    model=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx'
    assert sha(model)=='8adb1e6bc3f5eef751ce4edbf940f6d5a25fdec3ec4a798e1d9ca135c8248e97'
    graph=onnx.load(model);onnx.checker.check_model(graph)
    metadata={x.key:x.value for x in graph.metadata_props}
    cfg=TrainConfig.from_task('Mjlab-Roulade-Flat-XgoDuck')
    action=cfg.env.actions['joint_pos'];obs=cfg.env.observations['actor']
    source_files=[WORK/'src/mjlab_microduck/tasks'/f for f in ('mdp.py','microduck_roulade_env_cfg.py','xgoduck_tricks_env_cfg.py')]
    functions={key:dict(function=r.func.__module__+'.'+r.func.__name__,weight=r.weight,
        params=yaml.load(yaml.dump(r.params),Loader=yaml.BaseLoader),source=inspect.getsource(r.func)) for key,r in cfg.env.rewards.items() if key in ('roulade_progress','roulade_head_pivot','roulade_landing_composite')}
    agent=yaml.load(yaml.dump(asdict(cfg.agent)),Loader=yaml.BaseLoader)
    result=dict(status='source-contract-audited',model_sha256=sha(model),metadata=metadata,
        onnx_operations=[n.op_type for n in graph.graph.node],
        initializers={a.name:list(numpy_helper.to_array(a).shape) for a in graph.graph.initializer},
        configured_actor=agent['actor'],configured_critic=agent['critic'],algorithm=agent['algorithm'],
        scene_num_envs=cfg.env.scene.num_envs,physics_dt=cfg.env.sim.mujoco.timestep,control_dt=cfg.env.decimation*cfg.env.sim.mujoco.timestep,
        episode_seconds=cfg.env.episode_length_s,
        action_config=yaml.load(yaml.dump(asdict(action)),Loader=yaml.BaseLoader),
        action_field_types={f.name:str(f.type) for f in fields(action)},
        actor_terms={k:dict(function=t.func.__module__+'.'+t.func.__name__,params=yaml.load(yaml.dump(t.params),Loader=yaml.BaseLoader)) for k,t in obs.terms.items()},
        joint_observation_source=inspect.getsource(obs.terms['joint_pos'].func),
        rewards=functions,source_sha256={str(p.relative_to(WORK)):sha(p) for p in source_files},
        curriculum=list(cfg.env.curriculum),events=list(cfg.env.events),
        source_actor_importer='Walking importer pins another model hash; do not use it unchanged',
        reset_constraint='Preserve source22-degree reset; model24-degree HOME is an observation/action coordinate adapter, not reset-pose replacement',
        monitor_constraint='Do not apply walking tilt/adduction stop rules to intended roll motion',
        environment_steps=0,learning_updates=0,cuda_initialized=torch.cuda.is_initialized(),script_sha256=sha(Path(__file__)))
    assert not result['cuda_initialized']
    out=ROOT/'records/reports/roll-learning-inputs';out.mkdir(exist_ok=True);write(out/'source-contract.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metadata','joint_observation_source','events','rewards')},indent=2))


if __name__=='__main__':main()
