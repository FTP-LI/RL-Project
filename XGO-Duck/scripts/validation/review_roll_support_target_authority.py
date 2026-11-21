"""Measure delayed motor authority and head-clearance direction at archived roll releases."""

import argparse
import inspect
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from bam.feetech.actuator import STS3215Actuator
from bam.mjlab import BamActuator

from common_scene_contract import model_arrays
from locate_roll_endpoint_support_loss import ROOT, cohort_stats, load, read, sha
from review_roll_head_motor_envelope import check_target_replay, reconstruct_internal_target
from review_roll_head_support_geometry import Geometry
from review_roll_lateral_pitch_authority import torque
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL
from run_continuous_roll_stand_walk import configuration


PLAN = ROOT/'configs/experiments/roll_forward_input_integration/head-recontact-load-recovery/target-authority-review.json'
KEY = 'roll_support_target_authority'


def release_frames(motion, clock, first, last, shared, roll_start, threshold, phase):
    start, end = first-shared, last-shared+1
    prefix = motion['qpos'][roll_start-shared:start+1,:,3:7]
    q = np.concatenate((prefix,clock['after']))
    angle = np.rad2deg(progress(dict(quaternion=q,ground_contact=np.ones(q.shape[:2],bool)))[0])[len(prefix)-1:]
    band = np.maximum(np.minimum(angle[1:],phase[1])-np.maximum(angle[:-1],phase[0]),0.)
    rotated = band.reshape(last-first,5,band.shape[1]).sum(axis=1)
    force = clock['normal_force_n'].reshape(last-first,5,band.shape[1],4)
    endpoint_unloaded = ~motion['ground_contact'][start+1:end] & (motion['phase_next'][start:end-1]==ROLL)
    measured = endpoint_unloaded[:-1] & (rotated[:-1]>0.) & (force[:-1,-1,:,1]>threshold) & (force[1:,0,:,0]<=threshold)
    frames, worlds = np.where(measured)
    if not len(frames) or (frames==0).any():
        raise ValueError('A release lacks the archived preceding electrical step')
    return frames+first+1, worlds, angle[frames*5,worlds], rotated[frames,worlds]


def read_role(folder, role, case, receipt, protocol, sources, first, last):
    branch = folder/'clips'/role
    meta = receipt['branches'][role]
    motion = load(branch/'trace.npz',meta['trace_sha256']['trace.npz'],sources,
                  arrays=('qpos','qvel','ground_contact','phase_next'))
    clock_meta = branch/'contact-clock.json'
    clock = load(branch/'contact-clock.npz',protocol['sources_sha256'][(branch/'contact-clock.npz').relative_to(ROOT).as_posix()],
                 sources,arrays=('substep','after','normal_force_n','motor_control','joint_actuator_torque_nm'))
    clock_proof = load(clock_meta,protocol['sources_sha256'][clock_meta.relative_to(ROOT).as_posix()],sources)
    assert clock_proof['complete'] and clock_proof['joint_names']==case['walking']['joint_names']
    np.testing.assert_array_equal(clock['substep'],np.arange(first*5+1,last*5+1))
    pre = load(folder/'shared/commands.npz',receipt['prefix_sha256']['commands.npz'],
               sources,arrays=('actual_motor_target',))['actual_motor_target']
    suffix = load(branch/'commands.npz', meta['trace_sha256']['commands.npz'],sources,
                  arrays=('actual_motor_target',))['actual_motor_target']
    commands = np.concatenate((np.zeros_like(pre[:1]),pre,suffix[:last-len(pre)])).astype(np.float64)
    return motion, clock, commands, clock_proof


def retarget(motor, at, worlds, q, dq, commands, previous, limits, voltage, dt, joint, amount, lead):
    begin = at-lead
    smooth = previous[begin-1,worlds].copy()
    delayed = commands[begin-1,worlds].copy()
    max_change = motor.model.max_velocity.value*dt
    for control in range(lead):
        modified = commands[begin+control,worlds].copy()
        modified[:,joint] = np.clip(modified[:,joint]+amount,*limits[joint])
        smooth = np.clip(delayed,smooth-max_change,smooth+max_change)
        for _ in range(4):
            smooth = np.clip(modified,smooth-max_change,smooth+max_change)
        delayed = modified
    value,_ = torque(motor,voltage,smooth,delayed,q,dq,dt)
    return value, delayed[:,joint]-commands[at-1,worlds,joint]


def generalized(model, act, values):
    actual = np.where(model.actuator_ctrllimited[act],np.clip(values,*model.actuator_ctrlrange[act].T),values)
    actual = np.where(model.actuator_forcelimited[act],np.clip(actual,*model.actuator_forcerange[act].T),actual)
    return actual*model.actuator_gear[act,0]


def position_response(geometry, qpos, qvel, joints, amounts):
    n = len(qpos)
    inverse_response = np.empty((n,len(joints)))
    height = np.empty((n,len(joints),len(amounts)))
    head_state = np.empty((n,2))
    model = geometry.model
    mass = np.empty((model.nv,model.nv))
    for i,(pose,velocity) in enumerate(zip(qpos,qvel,strict=True)):
        state = geometry.measure(pose,velocity)
        head_state[i] = state['clearance_mm'],state['normal_velocity_mm_s']
        normal = geometry.data.geom_xmat[geometry.floor].reshape(3,3)[:,2]
        jacobian = normal@geometry.jac
        mujoco.mj_fullM(model,geometry.data,mass)
        inverse_response[i] = np.linalg.solve(mass,jacobian)[geometry.dofs]
        for j,address in enumerate(geometry.addresses):
            for k,offset in enumerate(amounts):
                changed = pose.copy()
                changed[address] = np.clip(pose[address]+offset,*geometry.ranges[j])
                height[i,j,k] = geometry.measure(changed,velocity)['clearance_mm']-state['clearance_mm']
    return head_state,inverse_response,height


def review_role(geometry, wrapper, dt, casespec, study, role, motion, clock, commands, clock_proof, found):
    first, last = (casespec['force_capture'][name] for name in ('first_control','last_control'))
    split = casespec['checkpoint_selection']['plant']['shared_snapshot_control']
    roll_start = casespec['initial_stand_controls']
    threshold = clock_proof['force_threshold_n']
    selector = study['selection']
    assert (selector['last_substep_head_load_above_nominal_weight_fraction'] ==
            selector['next_substep_ground_load_must_not_exceed_fraction'] ==
            casespec['force_capture']['load_threshold_nominal_weight_fraction'])
    assert threshold == clock_proof['nominal_weight_n']*selector['next_substep_ground_load_must_not_exceed_fraction']
    at, worlds, angle, uncredited = release_frames(motion,clock,first,last,split,roll_start,threshold,
                                                   selector['positive_geometric_phase_deg'])
    n = len(at)
    common = casespec['walking']['joint_names']
    model = geometry.model
    limits = np.asarray(casespec['walking']['joint_limits'])
    ids = np.asarray([model.joint('robot/'+name).id for name in common])
    assert len(ids)==model.nu==14 and all(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)
    column_for_joint = {int(joint): i for i,joint in enumerate(model.actuator_trnid[:,0])}
    act = [column_for_joint[joint] for joint in ids]
    qpos = motion['qpos'][at-1-split,worlds].astype(np.float64)
    qvel = motion['qvel'][at-1-split,worlds].astype(np.float64)
    q, dq = qpos[:,model.jnt_qposadr[ids]],qvel[:,model.jnt_dofadr[ids]]
    servo = casespec['parameter_factors']['vin_range_v']
    assert servo==7.7 and wrapper.cfg.vin_min==7.0 and wrapper.cfg.vin_drop_gain_range==(.1,.1)
    motor = wrapper._bam_model.actuator
    smooth = reconstruct_internal_target(commands,motor.model.max_velocity.value*dt)
    motor.vin = servo
    check_target_replay(commands,smooth,motor,dt)
    sub = (at-first-1)*5
    previous_motor = clock['motor_control'][sub-1,worlds]
    voltage = np.maximum(servo-.1*np.abs(previous_motor).sum(axis=1,keepdims=True),wrapper.cfg.vin_min)
    reference,_ = torque(motor,voltage,smooth[at-1,worlds],commands[at-1,worlds],q,dq,dt)
    measured_motor = clock['motor_control'][sub,worlds]
    measured_joint = clock['joint_actuator_torque_nm'][sub,worlds]
    np.testing.assert_allclose(generalized(model,act,measured_motor),measured_joint,atol=1e-6,rtol=0)
    raw_error = np.abs(reference-measured_motor)
    tolerance = study['raw_motor_replay_atol_nm']
    if raw_error.max() > tolerance:
        worst = np.unravel_index(raw_error.argmax(),raw_error.shape)
        raise ValueError(f'{role} original BAM replay error {raw_error.max():.6g} Nm at '
                         f'control {at[worst[0]]}, world {worlds[worst[0]]}, joint {common[worst[1]]}')
    ref = generalized(model,act,reference)
    joints = study['joints']
    assert joints==['neck_pitch','head_pitch']
    ranges = study['trial_offsets_rad']
    head, gain, displacement = position_response(geometry,qpos,qvel,joints,ranges)
    focused = np.isin(worlds,found)
    cohorts = dict(all_release_intervals=np.ones(n,dtype=bool),paired_regression_worlds=focused)
    variants = {}
    for j,name in enumerate(joints):
        column = common.index(name)
        for index,amount in enumerate(ranges):
            for lead in study['trial_lead_controls']:
                values, offset = retarget(motor,at,worlds,q,dq,commands,smooth,limits,voltage,dt,column,amount,lead)
                delta = (generalized(model,act,values)-ref)[:,column]
                target_change = displacement[:,j,index]
                normal_drive = gain[:,j]*delta
                changed = np.abs(delta)>tolerance
                closes = (target_change < -1e-5) & (normal_drive < -1e-5) & changed
                variants[f'{name}:{amount:+.2f}rad:{lead}lead'] = {group:dict(
                    worlds=len(np.unique(worlds[select])), samples=int(select.sum()), applied_target_changed=int((select&(offset!=0.)).sum()),
                    measured_above_motor_readback_floor=int((select&changed).sum()),
                    static_head_lowered=int((select&(target_change < -1e-5)).sum()),
                    downward_free_acceleration=int((select&(normal_drive < -1e-5)).sum()), combined=int((select&closes).sum()),
                    clearance_change_mm=cohort_stats(target_change[select]),motor_torque_change_nm=cohort_stats(delta[select]),
                    free_normal_acceleration_change_m_s2=cohort_stats(normal_drive[select])) for group,select in cohorts.items()}
    return dict(samples=n,worlds=len(np.unique(worlds)),paired_regression_samples=int(focused.sum()),
        maximum_motor_readback_error_nm=float(raw_error.max()), head_clearance_at_control_start_mm=cohort_stats(head[:,0]),
        head_normal_speed_at_control_start_mm_s=cohort_stats(head[:,1]), event_entry_angle_deg=cohort_stats(angle),
        variants=variants, samples_by_world=[dict(world=int(w),control=int(k),start_phase_deg=float(a),uncredited_band_angle_deg=float(b))
                    for w,k,a,b in zip(worlds,at,angle,uncredited,strict=True)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    state = read(ROOT/'project.json')['goal_execution'][KEY]
    study = read(PLAN)
    assert sha(PLAN)==state['protocol_sha256']
    local = [PLAN,Path(__file__).resolve()]
    local.extend(Path(__file__).with_name(name+'.py') for name in ('review_roll_head_motor_envelope','review_roll_head_support_geometry',
                                                                'review_roll_lateral_pitch_authority','roll_metrics'))
    sources = {path.relative_to(ROOT).as_posix():sha(path) for path in local}
    master,loads,audit = (load(ROOT/study[name],study[name+'_sha256'],sources) for name in ('load_master','force_summary','clock_audit'))
    assert audit['source_sha256']==study['force_summary_sha256'] and master['reserved_seed']==study['reserved_seed']==2246
    master_path = ROOT/study['load_master']
    outcomes = {}
    installed = {inspect.getfile(value):sha(Path(inspect.getfile(value))) for value in (BamActuator,STS3215Actuator)}
    for seed in study['development_seeds']:
        key = f'seed-{seed}'
        casespec = load(master_path.with_name(key+'.json'),master['protocols'][key],sources)
        case = loads['cases'][key]
        assert casespec['seed']==seed and casespec['worlds']==128
        folder = ROOT/'records/evaluations'/case['run_id']
        receipt_path = folder/'guarded-evaluation.json'
        receipt = load(receipt_path,loads['sources_sha256'][receipt_path.relative_to(ROOT).as_posix()],sources)
        cfg = configuration(casespec)
        compiled = Scene(cfg.scene,device='cpu')
        model = compiled.compile()
        wrapper, = compiled['robot'].actuators
        servo, = cfg.scene.entities['robot'].articulation.actuators
        dt = cfg.sim.mujoco.timestep
        assert isinstance(wrapper._bam_model.actuator,STS3215Actuator) and dt==.004 and cfg.decimation==5
        assert servo.delay_min_lag==servo.delay_max_lag==1
        static = folder/'scene-static-fields.npz'
        contract = read(folder/'scene-static-contract.json')
        sources[static.relative_to(ROOT).as_posix()] = sha(static)
        assert contract['status']=='passed' and contract['common_step_counter']==0
        actual = model_arrays(model)
        with np.load(static,allow_pickle=False) as saved:
            for field in ('body_mass','body_inertia','body_pos','body_quat','jnt_type','jnt_range','dof_armature',
                          'geom_type','geom_pos','geom_quat','actuator_trnid','actuator_gear'):
                np.testing.assert_array_equal(actual[field],saved[field],err_msg=field)
        geometry = Geometry(model)
        first,last = (casespec['force_capture'][name] for name in ('first_control','last_control'))
        selected = case['cohorts']['new_matched_regressions']['worlds']
        variants = {}
        for role in case['original_gate_passes']:
            values = read_role(folder,role,casespec,receipt,loads,sources,first,last)
            variants[role] = review_role(geometry,wrapper,dt,casespec,study,role,*values,selected)
            print('ROLL_ELECTRICAL_REPLAY',seed,role,variants[role]['samples'],
                  variants[role]['maximum_motor_readback_error_nm'],flush=True)
        outcomes[key] = variants
    assert not torch.cuda.is_initialized()
    out = (ROOT/study['force_summary']).with_name('target-authority-review.json')
    data = dict(seeds=outcomes,sources_sha256=sources,installed_sources_sha256=installed,
                scope='Static geometry, delayed electrical readback using measured voltage sag, and unconstrained acceleration at saved poses',
                new_physical_steps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False)
    encoded = (json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if args.check:
        if out.read_bytes()!=encoded:
            raise ValueError('Saved authority report failed byte-for-byte verification')
    else:
        with out.open('xb') as stream:
            stream.write(encoded)
    print('ROLL_TARGET_AUTHORITY_VERIFIED',sha(out),flush=True)


if __name__=='__main__':
    main()
