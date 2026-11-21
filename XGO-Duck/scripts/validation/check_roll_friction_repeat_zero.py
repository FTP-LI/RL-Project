"""Repeat the exact CUDA reset with two nominal references and one enriched draw."""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import importlib
from unittest.mock import patch

import check_roll_friction_exposure as original

np, torch = original.np, original.torch
ROOT, REPORT = original.ROOT, original.REPORT
PLAN = ROOT/'configs/experiments/roll_friction_exposure_repeat_zero.json'
ENTRY = Path(__file__).resolve()
ROLES = ('official-distribution', 'official-repeat', 'low-friction-enriched')
DERIVED = {'body_invweight0', 'dof_invweight0'}
DERIVED_REL_TOL = 1e-6
DERIVED_ABS_TOL = 1e-8
read, put, sha, require = original.read, original.put, original.hash_file, original.require


def checked():
    project = read(ROOT/'project.json')
    state, goal = project['staged_training_request'], project['goal_execution']
    require(state['current_stage']=='roll-friction-exposure-repeated-zero' and goal['status']=='active'
            and not goal['pause_requested_by_user'] and not any(state[k] for k in (
                'simulation_running','learning_running','pause_requested','deployment_authorized_now'))
            and not goal['roll_friction_exposure']['physics_authorized']
            and not goal['roll_friction_exposure']['learning_authorized'], 'Project does not allow repeated zero-step check')
    prior = read(original.PLAN)
    for base, key in ((ROOT,'source_sha256'),(Path('/'),'local_training_sha256')):
        require(all(sha(base/name)==h for name,h in prior[key].items()), 'Old frozen input changed: '+key)
    stop, cpu = (read(REPORT/name) for name in ('strict-cuda-zero-stop.json','cpu.json'))
    require(stop['plan_sha256']==cpu['protocol_sha256']==sha(original.PLAN)
            and stop['field']=='body_invweight0' and stop['physics_substeps']==0
            and cpu['status']=='passed-cpu-configuration-and-sampler' and prior['maximum_physics_steps']==0,
            'Previous evidence missing')
    return prior, cpu


def freeze():
    _, _ = checked()
    derivation = Path(importlib.import_module('mujoco_warp._src.io').__file__).resolve()
    require(derivation.is_relative_to(Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl')),
            'Unexpected installed solver implementation')
    files = (original.PLAN, ENTRY, REPORT/'cpu.json', REPORT/'strict-cuda-zero-stop.json', derivation,
             ROOT/'scripts/validation/common_scene_contract.py')
    put(PLAN, dict(status='frozen-reset-only', written_at=datetime.now(timezone.utc).isoformat(), roles=ROLES,
        tolerance={'fields':sorted(DERIVED),'rtol':DERIVED_REL_TOL,'atol':DERIVED_ABS_TOL},
        scope='No tolerance for physical input fields; only documented solver-derived inverse-weight fields.',
        criterion='Record observed nonzero difference during same-setup nominal repeat. Derived differences in enriched arm must stay below the predeclared numerical tolerance.',
        source_sha256={str(f):sha(f) for f in files},
        physics_substeps=0, learning_updates=0, hardware_io=False, automatic_retry=False))
    print('ROLL_REPEAT_ZERO_FROZEN', sha(PLAN), flush=True)


def diff(actual, expected, *, tolerates=False):
    a,b = np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),
            'Unexpected array shape, type or nonfinite value')
    differs = a != b
    if not differs.any():
        return None
    error = abs(a.astype(np.float64)-b.astype(np.float64))
    peak = np.maximum(abs(a.astype(np.float64)),abs(b.astype(np.float64)))
    eps = DERIVED_ABS_TOL+DERIVED_REL_TOL*peak
    return dict(changed=int(differs.sum()), max_absolute=float(error.max()),
                max_relative=float(np.divide(error, peak, where=peak!=0, out=np.zeros_like(error)).max()),
                meets_contract=bool(tolerates and np.all(error<=eps)))


def sample(prior, role, cpu):
    seeds = prior['seed']
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    cfg = original.make(original.ROLES[0] if role == ROLES[1] else role, prior['worlds'], seeds)
    env = original.ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    try:
        obs,_ = env.reset(seed=seeds)
        require(env.common_step_counter==env._sim_step_counter==0, 'Nonzero reset counter')
        feet=original.resolve_foot_geometry_ids(env.sim.mj_model)
        require(list(feet)==cpu['foot_geom_ids'], 'CUDA foot mapping changed')
        cp=lambda x:x.detach().cpu().numpy().copy()
        fields={k:cp(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)}
        f=fields['geom_friction'][:,feet,0].copy()
        require(f.shape==(prior['worlds'],2) and np.isfinite(f).all() and (f>=.7).all() and (f<=1.3).all(),
                'Invalid measured friction')
        np.testing.assert_array_equal(f[:,0],f[:,1])
        fields['geom_friction'][:,feet,0]=np.float32(1.)
        robot=env.scene['robot']
        inputs=dict(qpos=cp(env.sim.data.qpos),qvel=cp(env.sim.data.qvel),actor=cp(obs['actor']),critic=cp(obs['critic']),
            rng_cpu=cp(torch.get_rng_state()),rng_cuda=cp(torch.cuda.get_rng_state()),
            actuator_vin=cp(robot.actuators[0].vin_tensor),actuator_drop_gain=cp(robot.actuators[0].vin_drop_gain))
        require(inputs['actor'].shape==(prior['worlds'],61), 'Actor contract changed')
        used,total=torch.cuda.mem_get_info()
        require(1.-used/total<=prior['gpu_fraction_maximum'], 'CUDA memory ceiling')
        return fields, inputs, f[:,0], int((total-used)/2**20)
    finally:
        env.close()


def run():
    prior,cpu=checked()
    spec=read(PLAN)
    require(spec['status']=='frozen-reset-only' and spec['roles']==list(ROLES)
            and all(sha(Path(f))==h for f,h in spec['source_sha256'].items()), 'Repeat protocol or source changed')
    target, arrays=REPORT/'cuda-repeat-zero.json', REPORT/'cuda-repeat-foot-friction.npz'
    require(not target.exists() and not arrays.exists(), 'Keep previous repeat run evidence')
    original.configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)

    def forbidden(*_args, **_kw):
        raise RuntimeError('A reset-only protocol cannot advance physics')

    with patch.object(original.ManagerBasedRlEnv,'step',forbidden),patch.object(original.Simulation,'step',forbidden):
        captured={r:sample(prior,r,cpu) for r in ROLES}
    ref,states,a_f,_=captured[ROLES[0]]
    reports,violations={},[]
    for role in ROLES[1:]:
        fields,other,friction,_ = captured[role]
        require(set(fields)==set(ref) and set(other)==set(states), 'Model or sensor field inventory changed')
        matched_derived={}
        differences={}
        for key in fields:
            d=diff(fields[key],ref[key],tolerates=key in DERIVED)
            if d is not None:
                differences[key]=d
                if key in DERIVED:
                    matched_derived[key]=d
                if not d['meets_contract']:
                    violations.append(role+'/field/'+key)
        for key in states:
            d=diff(other[key],states[key])
            if d:
                differences[key]=d
                violations.append(role+'/state/'+key)
        expected=a_f if role==ROLES[1] else original.remap(torch.from_numpy(a_f),torch.tensor([.7]),torch.tensor([1.3])).numpy()
        err=float(abs(expected-friction).max())
        if (err!=0. if role==ROLES[1] else err>1e-7):
            violations.append(role+'/friction-sampler')
        reports[role]=dict(differing_fields=differences,differing_solver_derived=matched_derived,
                           friction_mapping_max_absolute=err)
    if not reports[ROLES[1]]['differing_solver_derived']:
        violations.append('no-derived-variation-measured-in-official-repeat')
    with arrays.open('xb') as f:
        np.savez_compressed(f, **{r:t[2] for r,t in captured.items()})
    summary=dict(status='passed-repeated-cuda-reset-only' if not violations else 'stopped-review-required',
        protocol_sha256=sha(PLAN), controls=0, physics_substeps=0, learning_updates=0, hardware_io=False,
        worlds=prior['worlds'], seed=prior['seed'], primary_fields=[k for k in ref if k not in DERIVED],
        reset_arrays=list(states), arms={r:dict(below_0p85_worlds=int((t[2]<.85).sum()),used_mib=t[3]) for r,t in captured.items()},
        comparisons=reports, violations=violations, raw_friction_sha256=sha(arrays))
    put(target,summary)
    print('ROLL_REPEAT_ZERO',summary['status'], summary['arms'], reports, violations, flush=True)
    return 0 if not violations else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','run'))
    if parser.parse_args().action=='freeze':
        freeze()
    else:
        raise SystemExit(run())
