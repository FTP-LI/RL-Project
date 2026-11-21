"""Check head contact leverage and delayed servo authority from saved poses."""

from collections import Counter
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
import torch

from record_io import write
from review_roll_forward_input_stop import ROOT, arrays, read, sha
from review_roll_head_motor_envelope import check_target_replay, reconstruct_internal_target
from review_roll_head_support_geometry import Geometry
from review_roll_lateral_pitch_authority import torque
from roll_metrics import progress
from run_continuous_roll_stand_walk import configuration


REPORT = ROOT/'records/reports/roll-forward-input-integration/support-window-force'
JOINTS = ('neck_pitch','head_pitch','head_roll')
REQUEST_RAD = .02


def main():
    torch.set_num_threads(2)
    evidence, timing = (REPORT/name for name in ('force-readback.json','contact-sample-timing.json'))
    original, interval = read(evidence),read(timing)
    assert sha(evidence)==interval['source_sha256'] and not original['heldout_used']
    assert all(sha(ROOT/name)==digest for name,digest in original['source_sha256'].items())
    folder = ROOT/'records/evaluations'/original['run_id']
    protocol, commands_path, receipt_path = (folder/name for name in ('protocol.json','commands.npz','guarded-evaluation.json'))
    assert sha(protocol)==original['protocol_sha256'] and sha(commands_path)==read(receipt_path)['command_sha256']
    plan, trace, clock = read(protocol),arrays(folder/'trace.npz'),arrays(folder/'contact-clock.npz')
    cfg = configuration(plan)
    scene = Scene(cfg.scene,device='cpu')
    model = scene.compile()
    geom = Geometry(model)
    motor = scene['robot'].actuators[0]._bam_model.actuator
    servo, = cfg.scene.entities['robot'].articulation.actuators
    dt = cfg.sim.mujoco.timestep
    assert dt==.004 and cfg.decimation==5 and servo.delay_min_lag==servo.delay_max_lag==1
    order = plan['walking']['joint_names']
    columns = [order.index(name) for name in JOINTS]
    dofs = np.asarray([model.joint(f'robot/{name}').dofadr.item() for name in JOINTS])
    qpos_addresses = np.asarray([model.joint(f'robot/{name}').qposadr.item() for name in JOINTS])
    bounds = np.asarray(plan['walking']['joint_limits'])[columns]

    first,last = (plan['force_capture'][name] for name in ('first_control','last_control'))
    k_roll = plan['initial_stand_controls']
    q_angle,_,_ = progress({name:trace[name][k_roll:] for name in ('quaternion','ground_contact')})
    valid = trace['active'][first+1:last+1] & (trace['phase_next'][first+1:last+1]==1)
    missing = valid & ~trace['ground_contact'][first+1:last+1] & (np.diff(q_angle,axis=0)[first-k_roll:last-k_roll]>0.)
    head = (clock['normal_force_n'].reshape(last-first,5,128,4)[...,1]>original['threshold_n']).any(axis=1)
    cleared = (trace['phase_next'][k_roll:]==2).any(axis=0)
    scored = interval['groups']
    assert np.count_nonzero(missing & head & ~cleared[None,:])==scored['roll-fail']['head_load_interval_count']
    assert np.count_nonzero(missing & head & cleared[None,:])==scored['roll-pass']['head_load_interval_count']
    times,worlds = np.nonzero(missing&head)
    controls = times+first
    velocity = trace['qvel'][controls,worlds].astype(np.float64)
    pose = trace['qpos'][controls,worlds].astype(np.float64)
    target = arrays(commands_path)['actual_motor_target'][:last,:,columns].astype(np.float64)
    targets = np.concatenate((np.zeros_like(target[:1]),target))
    smooth = reconstruct_internal_target(targets,motor.model.max_velocity.value*dt)
    check_target_replay(targets,smooth,motor,dt)

    gradient, poses = [], []
    for k,w,q,v in zip(controls,worlds,pose,velocity,strict=True):
        state = geom.measure(q,v)
        normal = geom.data.geom_xmat[geom.floor].reshape(3,3)[:,2]
        gradient.append((normal@geom.jac)[dofs])
        poses.append(dict(world=int(w),before_control=int(k),roll_pass=bool(cleared[w]),
                          head_clearance_mm=state['clearance_mm'],head_vertical_speed_mm_s=state['normal_velocity_mm_s']))
    gradient = np.asarray(gradient)
    q,v = pose[:,qpos_addresses],velocity[:,dofs]
    voltage = plan['parameter_factors']['vin_range_v']
    voltage = voltage[0] if isinstance(voltage,list) else voltage
    assert voltage==7.7
    _, delayed = torque(motor,voltage,smooth[controls,worlds],targets[controls,worlds],q,v,dt)
    base = targets[controls+1,worlds]
    baseline,_ = torque(motor,voltage,delayed,base,q,v,dt)
    results = {}
    for i,name in enumerate(JOINTS):
        for sign in (-1,1):
            alt = base.copy()
            alt[:,i] = np.clip(base[:,i]+sign*REQUEST_RAD,*bounds[i])
            offsets = alt[:,i]-base[:,i]
            assert np.all(offsets*sign>=-1e-6) and np.abs(offsets).max()<=REQUEST_RAD+1e-6
            changed,_ = torque(motor,voltage,delayed,alt,q,v,dt)
            difference = changed[:,i]-baseline[:,i]
            gap = gradient[:,i]*offsets*1000
            roles = {}
            for label,selected in (('roll-pass',cleared[worlds]),('roll-fail',~cleared[worlds])):
                mask = selected & (gap < 0.) & (difference*sign>1e-6)
                roles[label] = dict(samples=int(selected.sum()), original_worlds=len(set(worlds[selected].tolist())),
                    closer_to_plane=int((selected&(gap<0.)).sum()), torque_in_requested_direction=int((selected&(difference*sign>1e-6)).sum()),
                    both=int(mask.sum()), median_linearized_clearance_change_mm=float(np.median(gap[selected])),
                    median_torque_change_n_m=float(np.median(difference[selected])))
            results[f'{name}:{sign:+d}'] = roles
    assert geom.data.time==0. and not torch.cuda.is_initialized()
    target_file = REPORT/'head-contact-authority.json'
    write(target_file,dict(hardware_io=False,new_physical_steps=0,learning_updates=0,scope='Fixed recorded states, not a response rollout',
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (evidence,timing,protocol,commands_path,Path(__file__))},
        voltage_v=voltage, actuator_delay_substeps=1, trial_target_rad=REQUEST_RAD,
        intervals=dict(Counter('roll-pass' if item['roll_pass'] else 'roll-fail' for item in poses)), variants=results,poses=poses))
    print('ROLL_HEAD_AUTHORITY_REVIEWED', {name:(row['roll-fail']['both'],row['roll-fail']['samples']) for name,row in results.items()})


if __name__=='__main__':
    main()
