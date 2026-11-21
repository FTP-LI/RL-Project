"""Verify native full and selective resets with the difficult backward releases."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.utils.torch import configure_torch_backends
import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from collect_walk_backward_onset import prior
from common_scene_contract import DERIVED, compare_arrays, model_arrays, nominal_primary
from common_walk_startup import make_actor, td
from common_walk_targets import prepare
from record_io import write
from walk_backward_start_reset import DifficultBackwardReleases, add_difficult_backward_event
from walking_command_controller import create_controller


PLAN = ROOT/'configs/experiments/common_walk_backward_start_reset_readback.json'
OUT = ROOT/'records/reports/common-walk-backward-start-curriculum/reset-readback'
RESET_CODE = ROOT/'scripts/training/walk_backward_start_reset.py'
HISTORY = ('_action','_prev_action','_prev_prev_action')
TERM_HISTORY = ('_raw_actions','requested_target','projected_destination')


def host(tensor):
    return tensor.detach().cpu().numpy().copy()


def fingerprint(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bank(plan, cases, base):
    qpos,qvel = [],[]
    origins = {}
    default_xy = {}
    greatest_xy = 0.
    for cohort in cases['cohorts']:
        n = cohort['source_worlds']
        if n not in origins:
            cfg = prepare(dict(base,parallel_worlds=n))
            assert cfg.scene.terrain.terrain_type=='plane'
            root_reset = cfg.events['reset_base']
            assert root_reset.mode=='reset' and root_reset.func.__name__=='reset_root_state_uniform'
            ranges = [list(root_reset.params['pose_range'][axis]) for axis in ('x','y')]
            assert ranges==plan['source_root_xy_range_m']
            default_xy[n] = np.asarray(cfg.scene.entities['robot'].init_state.pos[:2])
            origins[n] = host(Scene(cfg.scene,device='cpu').env_origins)
            assert origins[n].shape==(n,3) and not origins[n][:,2].any()
        path = ROOT/cohort['physical_initial']
        assert sha(path)==cohort['physical_initial_sha256']
        with np.load(path) as data:
            pos,vel = data['qpos'],data['qvel']
        for case in cohort['rows']:
            world = case['world']
            raw = pos[world].copy()
            signature = sha256(raw[2:].tobytes()+vel[world].tobytes()).hexdigest()
            assert signature==case['canonical_qpos_qvel_sha256']
            raw[:3] -= origins[n][world]
            offset = raw[:2]-default_xy[n]
            tolerance = plan['coordinate_tolerance_m']
            for value,(low,high) in zip(offset,plan['source_root_xy_range_m'],strict=True):
                assert low-tolerance<=value<=high+tolerance
            greatest_xy = max(greatest_xy,float(abs(offset).max()))
            qpos.append(raw)
            qvel.append(vel[world])
    assert len(qpos)==len(qvel)==plan['case_slots']
    return np.stack(qpos),np.stack(qvel),greatest_xy


def histories(env):
    manager = env.action_manager
    term = manager.get_term('joint_pos')
    return {key:host(getattr(owner,key)) for owner,keys in ((manager,HISTORY),(term,TERM_HISTORY)) for key in keys}


def reset_readback(env, raw_obs, p, desired, actor, onnx, full_reset):
    robot = env.scene['robot']
    ids,names = robot.find_joints(p['joint_names'],preserve_order=True)
    assert names==p['joint_names']
    physical = dict(qpos=host(env.sim.data.qpos),qvel=host(env.sim.data.qvel),ctrl=host(env.sim.data.ctrl))
    observation = host(raw_obs['actor'])
    assert observation.shape==(env.num_envs,61) and np.isfinite(observation).all()
    command = env.command_manager.get_command('twist')
    command.copy_(torch.as_tensor(desired,device=env.device))
    adapted = observation.copy()
    adapted[:,6:20] += host(robot.data.default_joint_pos[:,ids])-np.asarray(p['model_home'],np.float32)
    adapted[:,48:51] = create_controller(env.num_envs).map(desired,adapted)
    assert np.isfinite(adapted).all()
    with torch.no_grad():
        actual = host(actor(td(torch.as_tensor(adapted,device=env.device))))
    expected = np.concatenate([onnx.run(None,{'obs':row[None]})[0] for row in adapted])
    expanded = {k:host(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)}
    if full_reset:
        nominal_primary(expanded,env.num_envs)
    rng = {'cpu':fingerprint(host(torch.get_rng_state()))}
    if str(env.device).startswith('cuda'):
        rng['cuda'] = fingerprint(host(torch.cuda.get_rng_state(env.device)))
    return dict(physical=physical,observation=observation,adapted=adapted,policy=actual,onnx=expected,
                joints=host(robot.data.joint_pos[:,ids]),joint_vel=host(robot.data.joint_vel[:,ids]),
                histories=histories(env),rng=rng,expanded=expanded)


def sample_arm(plan, p, expected_scene, device, local_qpos, bank_qvel, hard, use_bank, desired, onnx):
    cfg = prepare(p)
    assert not cfg.auto_reset and cfg.scene.terrain.terrain_type=='plane'
    if use_bank:
        add_difficult_backward_event(cfg)
    env = ManagerBasedRlEnv(cfg=cfg,device=device,render_mode=None)
    try:
        compare_arrays(model_arrays(env.sim.mj_model),expected_scene)
        if use_bank:
            env.difficult_backward_releases = DifficultBackwardReleases(env,hard,local_qpos,bank_qvel)
        actor = make_actor(p,device)
        actor_before = {name:fingerprint(host(value)) for name,value in actor.state_dict().items()}
        origins = host(env.scene.env_origins)
        with patch.object(env.sim,'step',side_effect=AssertionError('Reset readback cannot integrate physics')):
            first,_ = env.reset(seed=plan['full_reset_seed'])
            initial = reset_readback(env,first,p,desired,actor,onnx,full_reset=True)
            # Fill each history position with a different value before native partial reset.
            action = env.action_manager
            for value in (.03125,-.0625,.09375):
                action.process_action(torch.full((env.num_envs,14),value,device=device))
            dirty = histories(env)
            assert all(np.any(dirty[key]) for key in HISTORY)
            partial = np.asarray(plan['partial_reset_ids'])
            raw,_ = env.reset(seed=plan['partial_reset_seed'],env_ids=torch.as_tensor(partial,device=device))
            selected = reset_readback(env,raw,p,desired,actor,onnx,full_reset=False)
        assert not env.common_step_counter and not env._sim_step_counter and not host(env.sim.data.time).any()
        assert actor_before=={name:fingerprint(host(value)) for name,value in actor.state_dict().items()}
        return initial,selected,dirty,origins,actor_before
    finally:
        env.close()


def compare(plan, control, candidate, qpos, qvel, hard):
    c0,c1,cd,c_orig,c_actor = control
    h0,h1,hd,h_orig,h_actor = candidate
    np.testing.assert_array_equal(c_orig,h_orig)
    assert c_actor==h_actor
    partial = np.asarray(plan['partial_reset_ids'])
    n = plan['parallel_worlds']
    complement = np.setdiff1d(np.arange(n),partial)
    regular = np.setdiff1d(np.arange(n),hard)
    partial_hard = np.intersect1d(hard,partial)
    dtype = h0['physical']['qpos'].dtype
    target = qpos.astype(dtype,copy=True)
    target[:,:3] += h_orig[hard]
    errors = {}
    computed = {}
    for name,c,h,ids,reset_ids in (('full',c0,h0,hard,np.arange(n)),('partial',c1,h1,partial_hard,partial)):
        assert c['rng']==h['rng'] and c['expanded'].keys()==h['expanded'].keys()
        native_dynamic = set(plan['native_dynamic_bam_fields'])
        assert native_dynamic<=c['expanded'].keys()
        tolerance = plan['native_computed_field_atol']
        computed[name] = {}
        for key,value in c['expanded'].items():
            indices = regular if key in DERIVED or (name=='partial' and key in native_dynamic) else slice(None)
            a,b = value[indices],h['expanded'][key][indices]
            if key in DERIVED or key in native_dynamic:
                np.testing.assert_allclose(a,b,atol=tolerance,rtol=0,err_msg=key)
                computed[name][key] = float(np.max(abs(a-b)))
            else:
                np.testing.assert_array_equal(a,b,err_msg=key)
        for key in ('qpos','qvel'):
            np.testing.assert_array_equal(c['physical'][key][regular],h['physical'][key][regular])
        a,b = c['physical']['ctrl'][regular],h['physical']['ctrl'][regular]
        np.testing.assert_allclose(a,b,atol=tolerance,rtol=0)
        computed[name]['ctrl'] = float(np.max(abs(a-b)))
        np.testing.assert_array_equal(h['physical']['qpos'][ids],target[ids])
        np.testing.assert_array_equal(h['physical']['qvel'][ids],qvel[ids])
        np.testing.assert_allclose(h['physical']['qpos'][ids,:3]-h_orig[ids],qpos[ids,:3],
                                   atol=plan['coordinate_tolerance_m'],rtol=0)
        np.testing.assert_array_equal(c['observation'][regular],h['observation'][regular])
        for term,columns in (('joints',slice(6,20)),('joint_vel',slice(20,34))):
            measured = h['observation'][ids,columns]-c['observation'][ids,columns]
            expected = h[term][ids]-c[term][ids]
            np.testing.assert_allclose(measured,expected,atol=plan['observation_difference_tolerance'],rtol=0)
        for arm in (c,h):
            for values in arm['histories'].values():
                assert not values[reset_ids].any()
            assert not arm['observation'][reset_ids,34:48].any()
            np.testing.assert_allclose(arm['policy'],arm['onnx'],atol=plan['onnx_atol'],rtol=plan['onnx_rtol'])
        errors[name] = max(float(abs(arm['policy']-arm['onnx']).max()) for arm in (c,h))
    for first,after,dirty in ((c0,c1,cd),(h0,h1,hd)):
        # Native partial reset refreshes the batch's actuator outputs. The
        # unselected generalized state and policy memory must stay intact.
        for key in ('qpos','qvel'):
            np.testing.assert_array_equal(after['physical'][key][complement],first['physical'][key][complement])
        for key,value in dirty.items():
            np.testing.assert_array_equal(after['histories'][key][complement],value[complement])
        np.testing.assert_array_equal(after['observation'][complement,34:48],dirty['_action'][complement])
    assert int(np.any(h0['physical']['qpos'][hard]!=c0['physical']['qpos'][hard],axis=1).sum())==len(hard)
    recomputed = {key:int(np.any(c1['expanded'][key]!=c0['expanded'][key],axis=1).sum())
                  for key in plan['native_dynamic_bam_fields']}
    return errors,recomputed,computed


def main():
    cli = ArgumentParser()
    cli.add_argument('device',choices=('cpu','cuda'))
    args = cli.parse_args()
    device = 'cpu' if args.device=='cpu' else 'cuda:0'
    result = OUT/f'{args.device}.json'
    assert not result.exists() and not torch.cuda.is_initialized()
    plan = read(PLAN)
    assert plan['status']=='frozen-zero-step-reset-readback'
    assert not any(plan[key] for key in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==plan['stage'] and ROOT/project['current_protocol']==PLAN
    assert project['current_protocol_frozen'] and not project['simulation_running']
    paths = {name:ROOT/path for name,(path,digest) in plan['sources'].items() if sha(ROOT/path)==digest}
    assert paths.keys()==plan['sources'].keys()
    if device!='cpu':
        receipt = read(OUT/'cpu.json')
        assert receipt['status']=='passed-zero-step-reset-readback' and receipt['protocol_sha256']==sha(PLAN)
        assert receipt['script_sha256']==sha(Path(__file__)) and receipt['reset_code_sha256']==sha(RESET_CODE)
    parent,cases = read(paths['parent']),read(paths['cases'])
    prior.check_sources(parent)
    assert cases['reset_readback_admitted'] and not cases['learning_admitted'] and not cases['heldout_used']
    base = parent['environment']
    n = plan['parallel_worlds']
    p = dict(base,seed=plan['full_reset_seed'],parallel_worlds=n)
    hard = np.arange(plan['case_slots'],dtype=np.int64)
    assert len(hard)<plan['backward_slots']<n
    desired = np.zeros((n,3),dtype=np.float32)
    desired[:plan['backward_slots']] = base['commands']['backward']
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    local_qpos,qvel,xy = bank(plan,cases,base)
    with np.load(paths['scene']) as archive:
        expected_scene = {key:archive[key] for key in archive.files}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    onnx = ort.InferenceSession(str(ROOT/p['model']),sess_options=opts,providers=['CPUExecutionProvider'])
    arms = [sample_arm(plan,p,expected_scene,device,local_qpos,qvel,hard,changed,desired,onnx) for changed in (False,True)]
    errors,recomputed,computed = compare(plan,*arms,local_qpos,qvel,hard)
    assert torch.cuda.is_initialized()==(device!='cpu')
    OUT.mkdir(exist_ok=True)
    saved = OUT/f'{args.device}-readback.npz'
    arrays = {}
    for name,(full,partial,_,origin,_) in zip(('control','candidate'),arms,strict=True):
        for key,stage in (('full',full),('partial',partial)):
            for field,value in {**stage['physical'],**stage['histories'],**{k:stage[k] for k in ('observation','adapted','policy','onnx')}}.items():
                arrays[f'{name}_{key}_{field}'] = value
            for field,value in stage['expanded'].items():
                arrays[f'{name}_{key}_model_{field}'] = value
        arrays[f'{name}_origins'] = origin
    np.savez_compressed(saved,**arrays,source_local_qpos=local_qpos,source_qvel=qvel)
    write(result,dict(status='passed-zero-step-reset-readback',device=device,protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)),reset_code_sha256=sha(RESET_CODE),physical_scene_fields_exact=True,
        source_case_count=len(hard),source_local_xy_max_m=xy,parallel_worlds=n,partial_ids=plan['partial_reset_ids'],
        partial_hard_count=len(np.intersect1d(hard,plan['partial_reset_ids'])),unselected_state_and_histories_exact=True,
        native_first_observation_checked=True,expanded_randomization_exact=True,
        native_bam_fields_recomputed_rows=recomputed,native_computed_field_max_abs_difference=computed,
        cpu_and_device_rng_equal_between_arms=True,
        actor_fixed=True,onnx_inputs=4*n,maximum_onnx_error=errors,readback_sha256=sha(saved),
        physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_RESET_READBACK',device,len(hard),len(plan['partial_reset_ids']),errors,flush=True)


if __name__=='__main__':
    main()
