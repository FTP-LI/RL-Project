"""Verify the proposed roll friction sampling change, without a physics step."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path
from unittest.mock import patch
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))

import numpy as np
import mujoco
import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.scripts.train import TrainConfig
from mjlab.sim import Simulation
from mjlab.utils.torch import configure_torch_backends
from mjlab.envs.mdp.dr._types import uniform
from roll_friction_exposure import LOW_EDGE, RANGE, apply, remap, sample_enriched
from common_scene_contract import model_arrays
from selective_friction_guard import resolve_foot_geometry_ids

TASK = 'Mjlab-Roulade-Flat-XgoDuck'
ROLES = ('official-distribution', 'low-friction-enriched')
REPORT = ROOT / 'records/reports/roll-friction-exposure-preflight'
PLAN = ROOT / 'configs/experiments/roll_friction_exposure_preflight.json'
SOURCE = ROOT / 'records/reports/roll-matched-factor-low-continuation-s1242/policy-visible-support-screen.json'
CUDA_ZERO_SEED = 1244
ZERO_WORLDS = 256


def hash_file(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def current():
    p = read(ROOT/'project.json')
    task, goal = p['staged_training_request'], p['goal_execution']
    require(goal['status'] == 'active' and not goal['pause_requested_by_user']
            and task['current_stage'] == 'roll-friction-exposure-preflight'
            and not any(task[k] for k in ('simulation_running', 'learning_running', 'pause_requested', 'deployment_authorized_now'))
            and not goal['roll_friction_exposure']['physics_authorized']
            and not goal['roll_friction_exposure']['learning_authorized'], 'Not an authorized zero-step phase')
    return p


def freeze():
    p = current()
    screen = read(SOURCE)
    require(not any(v['passes_all_development_arms'] for v in screen['screen'].values())
            and p['goal_execution']['roll_policy_visible_support']['result_sha256'] == hash_file(SOURCE),
            'The recorded observation screen changed')
    installed = ['mjlab.envs.mdp.dr._types', 'mjlab.envs.mdp.dr.geom', 'mjlab.envs.mdp.dr._core',
                 'mjlab.envs.manager_based_rl_env', 'mjlab.sim.sim', 'mjlab.managers.event_manager',
                 'mjlab_microduck.tasks.mdp', 'mjlab_microduck.tasks.xgoduck_tricks_env_cfg',
                 'mjlab_microduck.tasks.microduck_roulade_env_cfg']
    sources = [Path(__file__), ROOT/'scripts/training/roll_friction_exposure.py',
               ROOT/'scripts/validation/selective_friction_guard.py', ROOT/'scripts/validation/common_scene_contract.py',
               ROOT/'scripts/setup/project_environment.sh', SOURCE,
               ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx']
    resolved = [Path(importlib.import_module(name).__file__).resolve() for name in installed]
    work = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl')
    require(all(f.is_relative_to(work.resolve()) for f in resolved), 'A source is not from the frozen training copy')
    REPORT.mkdir()
    put(PLAN, dict(status='frozen-no-physics-no-learning', roles=list(ROLES), distribution_range=RANGE,
        nominal_low_probability=.25, enriched_low_probability=.50, low_threshold=LOW_EDGE,
        worlds=ZERO_WORLDS, seed=CUDA_ZERO_SEED, maximum_physics_steps=0, learning_updates=0,
        hardware_io=False, holdout_used=False, automatic_retry=False, gpu_fraction_maximum=.85,
        source_sha256={f.relative_to(ROOT).as_posix():hash_file(f) for f in sources},
        local_training_sha256={str(f):hash_file(f) for f in resolved}))
    print('ROLL_EXPOSURE_ZERO_FROZEN', hash_file(PLAN), flush=True)


def checked():
    current()
    spec = read(PLAN)
    require(spec['status']=='frozen-no-physics-no-learning' and spec['seed']==CUDA_ZERO_SEED and spec['worlds']==ZERO_WORLDS,
            'Frozen zero-step scope changed')
    for base, key in ((ROOT, 'source_sha256'), (Path('/'), 'local_training_sha256')):
        require(all(hash_file(base / name) == value for name,value in spec[key].items()), 'Frozen source changed: '+key)
    return spec


def make(role, worlds, seed):
    cfg = TrainConfig.from_task(TASK)
    apply(cfg.env, role)
    cfg.env.seed = cfg.agent.seed = seed
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = worlds
    return cfg


def check_cpu(spec):
    original = make(ROLES[0], 1, spec['seed'])
    enriched = make(ROLES[1], 1, spec['seed'])
    # Event manager and learner are otherwise exactly the same. asdict keeps
    # callable identity, which detects a silent reward or reset substitution.
    # EnvironmentSpec contains separately constructed lambdas and array-valued
    # dataclasses. Check a serialized before/after snapshot of the same config.
    prior = make(ROLES[0], 1, spec['seed'])
    before = asdict(prior)
    apply(prior.env, ROLES[1])
    after = asdict(prior)
    distribution = after['env']['events']['foot_friction']['params'].pop('distribution')
    require(distribution['sample'] is sample_enriched and yaml.dump(before) == yaml.dump(after),
            'Unexpected difference in task or learner config')
    a = Scene(deepcopy(original.env.scene), device='cpu').compile()
    b = Scene(deepcopy(enriched.env.scene), device='cpu').compile()
    feet = resolve_foot_geometry_ids(a)
    require(feet == resolve_foot_geometry_ids(b), 'Foot geometry differed')
    floor = np.flatnonzero(a.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
    require(len(floor) == 1 and all(int(a.geom_priority[j]) > int(a.geom_priority[floor[0]]) for j in feet),
            'The foot must take priority over terrain at dynamic contact')
    require(not any(any(int(j) in feet for j in row)
                    for row in zip(a.pair_geom1, a.pair_geom2, strict=True)), 'Preset pair overrides foot friction')
    for name,value in model_arrays(a).items():
        np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)

    n = 30000
    lo,hi = (torch.tensor([x]) for x in RANGE)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec['seed'])
        left = uniform.sample(lo, hi, (n,1), 'cpu')
        saved = torch.get_rng_state().clone()
        torch.manual_seed(spec['seed'])
        right = sample_enriched(lo, hi, (n,1), 'cpu')
        torch.testing.assert_close(right, remap(left,lo,hi), atol=0,rtol=0)
        require(torch.equal(saved, torch.get_rng_state()), 'Number or order of RNG draws changed')
    values = [float((x < LOW_EDGE).float().mean()) for x in (left,right)]
    require(.235 <= values[0] <= .265 and .485 <= values[1] <= .515 and not torch.cuda.is_initialized(),
            'CPU distribution check failed')
    return dict(status='passed-cpu-configuration-and-sampler', compared_config_fields='all except distribution',
                equal_static_fields=len(model_arrays(a)), foot_geom_ids=feet, terrain_geom_id=floor[0],
                foot_priority=[int(a.geom_priority[i]) for i in feet], terrain_priority=int(a.geom_priority[floor[0]]),
                no_explicit_foot_contact_pair=True, draws=n, below_0p85_fractions=dict(zip(ROLES,values,strict=True)),
                same_rng_draw_count_and_order=True, cuda_initialized=False, physics_substeps=0, learning_updates=0)


def check_cuda(spec, cpu):
    # Block direct and indirect physical steps before creating either environment.
    def forbidden(*_, **__):
        raise RuntimeError('Physical stepping is not allowed in the reset-only check')
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    samples = {}
    with patch.object(ManagerBasedRlEnv,'step',forbidden), patch.object(Simulation,'step',forbidden):
        for role in ROLES:
            torch.manual_seed(spec['seed'])
            torch.cuda.manual_seed_all(spec['seed'])
            cfg = make(role,spec['worlds'],spec['seed'])
            env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            try:
                env.seed(spec['seed'])
                observed,_ = env.reset()
                require(env.common_step_counter == env._sim_step_counter == 0, 'Reset advanced physics')
                feet = resolve_foot_geometry_ids(env.sim.mj_model)
                require(list(feet) == cpu['foot_geom_ids'], 'Foot IDs changed after CUDA construction')
                fields = {key:getattr(env.sim.model,key).detach().cpu().numpy().copy()
                          for key in sorted(env.sim.expanded_fields)}
                require('geom_friction' in fields and all(np.isfinite(x).all() for x in fields.values()), 'Invalid model fields')
                use, static = fields['geom_friction'][:,feet,0], fields.copy()
                require(use.shape == (spec['worlds'],2), 'Missing per-world foot readings')
                np.testing.assert_array_equal(use[:,0], use[:,1])
                require((use>=RANGE[0]).all() and (use<=RANGE[1]).all(), 'Unexpected physical friction')
                static['geom_friction'] = fields['geom_friction'].copy()
                static['geom_friction'][:,feet,0] = np.float32(1.0)
                obs = observed['actor'].detach().cpu().numpy().copy()
                require(obs.shape==(spec['worlds'],61) and np.isfinite(obs).all(), 'Actor observation invalid')
                free,total = torch.cuda.mem_get_info()
                require((total-free)/total <= spec['gpu_fraction_maximum'], 'CUDA memory ceiling')
                samples[role]=dict(fields=static, feet=use[:,0].copy(), qpos=env.sim.data.qpos.detach().cpu().numpy().copy(),
                    qvel=env.sim.data.qvel.detach().cpu().numpy().copy(), actor=obs,
                    rng_cpu=torch.get_rng_state().numpy().copy(), rng_cuda=torch.cuda.get_rng_state().numpy().copy(),
                    used_mib=round((total-free)/2**20,1), controls=int(env.common_step_counter))
            finally:
                env.close()
    a,b = (samples[r] for r in ROLES)
    require(a['fields'].keys() == b['fields'].keys(), 'Expanded field inventory changed')
    for name in a['fields']:
        np.testing.assert_array_equal(a['fields'][name], b['fields'][name], err_msg=name)
    expected = remap(torch.from_numpy(a['feet']),torch.tensor([RANGE[0]]),torch.tensor([RANGE[1]])).numpy()
    np.testing.assert_allclose(expected,b['feet'],atol=1e-7,rtol=0)
    for name in ('rng_cpu','rng_cuda','qpos','qvel','actor'):
        np.testing.assert_array_equal(a[name],b[name],err_msg=name)
    arrays = REPORT/'zero-step-foot-friction.npz'
    with arrays.open('xb') as output:
        np.savez_compressed(output, official=a['feet'], enriched=b['feet'])
    return dict(status='passed-cuda-reset-only', worlds=spec['worlds'], seed=spec['seed'],
                equal_expanded_fields_outside_two_foot_sliding_columns=sorted(a['fields']),
                same_left_and_right=True, same_reset_qpos_qvel_actor_and_rng=True,
                per_arm={role:dict(below_0p85_worlds=int((v['feet']<LOW_EDGE).sum()),
                                  minimum=float(v['feet'].min()), maximum=float(v['feet'].max()),
                                  memory_used_mib=v['used_mib'], controls=v['controls']) for role,v in samples.items()},
                paired_foot_friction_sha256=hash_file(arrays), physical_substeps=0, learning_updates=0,
                hardware_io=False, protocol_sha256=hash_file(PLAN))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','cpu','cuda'))
    action = parser.parse_args().mode
    if action=='freeze':
        return freeze()
    spec = checked()
    target = REPORT/('cpu.json' if action=='cpu' else 'cuda-zero.json')
    require(not target.exists(), 'Preserve existing preflight')
    if action=='cpu':
        result=check_cpu(spec)
        result['protocol_sha256']=hash_file(PLAN)
    else:
        cpu=read(REPORT/'cpu.json')
        require(cpu['status']=='passed-cpu-configuration-and-sampler' and cpu['protocol_sha256']==hash_file(PLAN),
                'CPU preflight missing')
        result=check_cuda(spec,cpu)
    put(target,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
