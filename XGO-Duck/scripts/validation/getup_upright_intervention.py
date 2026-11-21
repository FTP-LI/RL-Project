"""Bounded legs-only getup correction on the original physical plant."""
from collections import deque
from unittest.mock import patch

import numpy as np
import torch

from check_getup_supine_upright_geometry import targets
from collect_getup_supine_evaluation import collect
import collect_getup_supine_evaluation as collection
from s1_reference_inputs import ROOT, read
from record_io import write
from standup_metrics import classify_release, projected_gravity


def cpu(value):
    return value.detach().cpu().numpy().copy()


class Trigger:
    def __init__(self,pose,entry):
        self.entry = entry
        self.supine = np.asarray(pose)==entry['actual_release_pose']
        self.step = np.full(len(pose),-1,dtype=np.int32)
        self.history = deque(maxlen=entry['consecutive_samples'])

    def observe(self,control,quaternion,feet,head):
        if control<=100 or control>round(self.entry['last_trigger_seconds']/.02):
            return np.empty(0,dtype=np.int64)
        g = projected_gravity(quaternion)
        tilt = np.rad2deg(np.arccos(np.clip(-g[:,2],-1.,1.)))
        good = (self.supine&feet.all(axis=-1)&head
                &(tilt>self.entry['minimum_tilt_deg_exclusive'])&(tilt<=self.entry['maximum_tilt_deg_inclusive']))
        self.history.append((good,tilt))
        if len(self.history)<self.history.maxlen:
            return np.empty(0,dtype=np.int64)
        sustained = np.logical_and.reduce([row[0] for row in self.history])
        slow = (self.history[0][1]-tilt)<self.entry['maximum_recent_tilt_reduction_deg_exclusive']
        selected = np.flatnonzero(sustained&slow&(self.step<0))
        self.step[selected] = control
        return selected


def pulse_request(original,correction,lower,upper):
    """Do not change a preexisting out-of-range request."""
    return torch.where((original>=lower)&(original<=upper),
        torch.clamp(original+correction,min=lower,max=upper),original)


def run_collection(env,actors,memory,meta,spec,record,state,publish,check_active):
    design,p = spec['intervention'],spec['environment']
    geometry = read(ROOT/design['geometry_protocol'])
    weights = torch.tensor(design['weights'],dtype=torch.float32,device=env.device)
    limits = torch.tensor(p['joint_limits'],dtype=torch.float32,device=env.device)
    names = meta['joint_names'].split(',')
    term = env.action_manager.get_term('joint_pos')
    original = term.process_actions
    robot = env.scene['robot']
    feet = env.scene['feet_ground_contact']
    events,trigger,active = {},None,None
    offsets = torch.zeros(env.num_envs,14,dtype=torch.float32,device=env.device)

    def relay(path,value):
        nonlocal trigger,active
        write(path,value)
        if path==record/'release-readback.json':
            active = next(reversed(value))
            assert active not in events and env.common_step_counter==spec['prefix_controls']==100
            q = cpu(robot.data.root_link_quat_w)
            pose = classify_release(projected_gravity(q),p['criteria']['release_alignment_min'])
            trigger = Trigger(pose,design['entry'])
            offsets.zero_()
            events[active] = []

    def process(actions):
        original(actions)
        if trigger is None:
            return
        decision = int(env.common_step_counter)
        if 100<decision<=round(design['entry']['last_trigger_seconds']/.02):
            selected = trigger.observe(decision,cpu(robot.data.root_link_quat_w),
                cpu(feet.data.found).reshape(env.num_envs,2,-1).any(axis=-1),cpu(env._getup_contact['contact']))
            if selected.size:
                qpos = cpu(env.sim.data.qpos) if active==spec['branches'][-1] else None
                for world in selected:
                    info = dict(world=int(world),decision_control=decision)
                    if qpos is not None:
                        target = targets(env.sim.mj_model,dict(qpos=qpos[world]),geometry,False)
                        info['geometry'] = target
                        if target['feasible']:
                            axes = [names.index(name) for name in target['joint_names']]
                            offsets[world,axes] = torch.tensor(target['joint_delta_rad'],
                                                              device=env.device,dtype=torch.float32)
                    events[active].append(info)
        if active!=spec['branches'][-1]:
            return
        ages = decision-trigger.step
        selected = np.flatnonzero((trigger.step>=0)&(ages<len(weights)))
        if selected.size:
            index = torch.as_tensor(ages[selected],device=env.device,dtype=torch.long)
            original_target = term._processed_actions[selected].clone()
            correction = offsets[selected]*weights[index,None]
            modified = pulse_request(original_target,correction,limits[:,0],limits[:,1])
            assert torch.abs(modified-original_target).max()<=design['target']['maximum_joint_delta_rad']+1e-7
            term._processed_actions[selected] = modified
            term.previous_target[selected] = modified

    try:
        with patch.object(collection,'write',relay),patch.object(term,'process_actions',process):
            collect(env,actors,memory,meta,spec,record,state,publish,check_active)
    finally:
        write(record/'upright-trigger-events.json',events)
