"""Verify the only actual CUDA reset change is the two foot sliding coefficients."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_nominal_roll_timestep_repeat as repeat
from selective_friction_guard import RESET_FIELDS, STATIC_FIELDS, compare_reset_fields, compare_static, resolve_foot_geometry_ids


parent = repeat.parent
ROOT, PROJECT = repeat.ROOT, repeat.PROJECT
read, rel, sha, write = repeat.read, repeat.rel, repeat.sha, repeat.write
KEY, STAGE = 'nominal_timestep_low_friction_preflight', 'nominal-timestep-low-friction-preflight'
PREVIOUS = 'nominal-roll-timestep-independent-confirmation'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = repeat.REPORT.with_name('closed-loop-timestep-low-friction-preflight')
ENTRY = Path(__file__).resolve()
SOURCE_CONFIG = parent.config
MODE_VALUES = {'nominal':1.0,'lower-friction':0.8}


def configured(plan):
    cfg = SOURCE_CONFIG(plan)
    event = cfg.events['foot_friction']
    assert event.params['ranges'] == (1.,1.)
    expected = plan['composition']['foot_friction']
    assert expected in MODE_VALUES.values()
    event.params['ranges'] = (expected,expected)
    return cfg


def freeze():
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert KEY not in goal and parent.allowed(project,PREVIOUS) and not task['simulation_running']
    heldout, baseline = (goal[key] for key in ('nominal_roll_timestep_confirmation',repeat.KEY))
    assert heldout['status']==baseline['status']=='reviewed' and heldout['passed'] and baseline['qualified']
    verified = ROOT/heldout['result']
    nominal = read(verified,heldout['result_sha256'])
    release = heldout['nominal_simulation_delivery']
    index = ROOT/release['path']
    manifest = read(index,release['sha256'])
    assert nominal['nominal_confirmation_passed'] and manifest['nominal_result']['sha256']==sha(verified)
    source = ROOT/baseline['protocol']
    plan = read(source,baseline['protocol_sha256'])
    d = plan['composition']
    assert plan['environment']['seed']==1257 and d['worlds']==plan['environment']['num_envs']==896
    assert (d['foot_friction'],d['voltage_v'],d['physics_period_s'],d['control_period_s'])==(1.,7.7,.004,.02)
    assert all(row['model_sha256']==manifest['policy_roles'][key]['onnx']['sha256']
               for key,row in plan['models'].items())
    assert d['existing_low_friction_reserved_seed']==2246 and not nominal['low_friction_reserved_seed_used']
    plan['friction_preflight'] = dict(mode_values=MODE_VALUES,scene_seed=1257,worlds=896,
        learning_updates=0,physics_substeps=0,hardware_io=False,reserved_seed=2246)
    plan['references'].update({rel(path):sha(path) for path in (
        ENTRY, source, verified, index, Path(compare_reset_fields.__code__.co_filename).resolve())})
    parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN,plan)
    goal[KEY] = dict(status='frozen',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Prove matching CUDA reset observations and expose only two changed foot sliding coefficients.')
    write(PROJECT,project)
    print('NOMINAL_TIMESTEP_FRICTION_FROZEN',sha(PLAN),flush=True)


def measure(plan, role, friction):
    cfg = dict(plan,composition={**plan['composition'],'foot_friction':friction})
    output = REPORT/role
    watch = parent.MemoryWatch(plan['composition']['observer']['gpu_memory_fraction_limit'])
    with watch.active(), patch.object(parent,'config',configured), patch.object(parent,'PLAN',PLAN):
        env = parent.cuda_env(cfg)
        try:
            watch.sample()
            seed = plan['environment']['seed']
            obs,_ = env.reset(seed=seed)
            assert env.common_step_counter==env._sim_step_counter==0
            actors = parent.load_actors(cfg,env.device,lambda:env.common_step_counter)
            proof = parent.qualify(env,obs,cfg,actors,output)
            assert proof['physical_substeps']==0 and proof['protocol_sha256']==sha(PLAN)
            model = env.sim.mj_model
            ids = resolve_foot_geometry_ids(model)
            static = {name:np.asarray(getattr(model,name)).copy() for name in STATIC_FIELDS}
            expanded = {name:parent.cpu(getattr(env.sim.model,name)) for name in RESET_FIELDS}
            np.testing.assert_array_equal(expanded['geom_friction'][:,ids,0],
                np.full((env.num_envs,2),friction,dtype=expanded['geom_friction'].dtype))
            with np.load(output/'reset.npz',allow_pickle=False) as replay:
                reset = dict(replay)
            np.savez_compressed(output/'physical-fields.npz',**{'static_'+key:value for key,value in static.items()},
                                **{'expanded_'+key:value for key,value in expanded.items()})
            assert env.common_step_counter==env._sim_step_counter==0
            watch.sample()
            return dict(static=static,expanded=expanded,reset=reset,ids=ids,proof=proof,cuda=watch.summary())
        finally:
            env.close()


def verify():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    node = goal[KEY]
    assert parent.allowed(project,STAGE) and node['status']=='frozen' and not task['simulation_running']
    plan = read(PLAN,node['protocol_sha256'])
    parent.base.shared.assert_sources(plan)
    assert plan['friction_preflight']['mode_values']==MODE_VALUES
    snapshots = {role:measure(plan,role,mu) for role,mu in MODE_VALUES.items()}
    a, b = (snapshots[role] for role in MODE_VALUES)
    assert a['ids']==b['ids'] and set(a['reset'])==set(b['reset'])
    assert a['proof']['composition_boundary']==b['proof']['composition_boundary']
    for name in a['reset']:
        np.testing.assert_array_equal(a['reset'][name],b['reset'][name],err_msg=name)
    static = compare_static(b['static'],a['static'])
    expanded = compare_reset_fields(b['expanded'],a['expanded'],foot_geom_ids=a['ids'],
                                    friction=MODE_VALUES['lower-friction'],worlds=plan['environment']['num_envs'])
    proof = REPORT/'analysis.json'
    assert not proof.exists()
    write(proof,dict(status='verified-single-factor-cuda-zero',protocol_sha256=sha(PLAN),seed=plan['environment']['seed'],
        worlds=plan['environment']['num_envs'],compared_reset_fields=sorted(a['reset']),
        static_model=static,expanded_model=expanded,actual_sliding_coefficients=MODE_VALUES,
        arms={role:dict(zero=rel(REPORT/role/'proof.json'),zero_sha256=sha(REPORT/role/'proof.json'),
            fields_sha256=sha(REPORT/role/'physical-fields.npz'),telemetry=item['cuda']) for role,item in snapshots.items()},
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False))
    node.update(status='reviewed',evidence=rel(proof),evidence_sha256=sha(proof))
    info = dict(current_protocol_execution=STAGE+'-verified',current_evidence=rel(proof))
    project.update(info)
    task.update(info,next_gate='Freeze one lower-friction development rollout with the same weights, 2 ms roll and original action gates.')
    write(PROJECT,project)
    print('NOMINAL_TIMESTEP_FRICTION_ZERO_VERIFIED',sha(proof),a['ids'],flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','verify'))
    action = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if action=='freeze' else verify)()
