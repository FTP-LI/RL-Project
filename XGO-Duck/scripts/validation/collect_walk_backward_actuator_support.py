"""Apply a smooth, bounded leg target at a matched backward stall."""
from copy import deepcopy
from datetime import datetime
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import collect_walk_backward_onset as onset
import collect_walk_backward_lateral_probe as lateral
from check_walk_actuator_support_targets import DRAFT,apply_support_target,pulse_weight
from check_walk_bounded_lateral_kinematics import lsq_linear,target
from check_walk_lateral_kinematics import addresses
from common_walk_forward_phase_snapshot import WalkingSnapshot
from record_io import write
from s1_reference_inputs import ROOT,read,sha
import training_live_page as live


PLAN = ROOT/'configs/experiments/common_walk_backward_actuator_support.json'
OUT = ROOT/'records/reports/common-walk-backward-actuator-support'
STAGE = 'S2-backward-actuator-support-development'
ROLES,SIGNS = lateral.ROLES,lateral.SIGNS
SEEDS = (1077,1078)
BASE = onset.prior.collect


def page(seed):
    return ROOT/f'records/walk-backward-actuator-support-s{seed}-live.html'


def pulse(spec,step):
    shape = spec['support']
    return pulse_weight(step,shape['start_control'],shape['ramp_up_controls'],shape['hold_controls'],shape['ramp_down_controls'])


def zero_probe(env,spec):
    native = WalkingSnapshot(env,spec['maximum_warp_snapshot_bytes'])
    initial = native.verify()
    assert initial['counters']==[0,0]
    term = env.action_manager.get_term('joint_pos')
    raw = torch.zeros_like(term._raw_actions)
    env.action_manager.process_action(raw)
    before = term._processed_actions.clone()
    move = torch.zeros_like(before[:4])
    move[:,1] = torch.tensor([.03,-.03,.03,-.03],device=env.device)
    limits = torch.tensor(spec['environment']['joint_limits'],device=env.device,dtype=before.dtype)
    changed = apply_support_target(before[:4],move,limits,1.)
    assert (changed-before[:4]).abs().max()<=.0300001
    term._processed_actions[:4] = changed
    term.projected_destination[:4] = changed
    torch.testing.assert_close(term._raw_actions,raw,atol=0,rtol=0)
    torch.testing.assert_close(term._processed_actions[4:],before[4:],atol=0,rtol=0)
    assert native.restore()==initial
    return initial


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft = read(DRAFT)
    gate_file = lateral.OUT/'actuator-target-preflight.json'
    gate = read(gate_file)
    assert gate['status']=='passed-saved-actuator-targets' and gate['draft_sha256']==sha(DRAFT)
    source = ROOT/draft['source']
    assert sha(source)==draft['source_sha256']
    assert read(source)['status']=='passed-saved-bounded-kinematics'
    parent = read(lateral.PLAN)
    onset.prior.check_sources(parent)
    assert tuple(draft['fresh_development_seeds'])==SEEDS
    assert not any(f'_s{seed}_' in name.name for seed in SEEDS for name in (ROOT/'records/evaluations').iterdir())
    spec = deepcopy(parent)
    p = spec['environment']
    spec.pop('lateral')
    p.update(seed=SEEDS[0],stage=STAGE)
    stamp = datetime.now(onset.TZ).strftime('%Y%m%dT%H%M%S')
    fields = ('start_control','ramp_up_controls','hold_controls','ramp_down_controls',
              'maximum_relative_joint_target_rad','geometry_target_body_shift_mm')
    spec.update(stage=STAGE,runs={str(seed):f'duck_walk_backward-actuator-support_paired_s{seed}_{stamp}' for seed in SEEDS},
        support={k:draft['design'][k] for k in fields}|dict(support_deadline_controls=30),
        scope='One bounded actuator waveform from a double-support, low-activity state. Both directions share their pre-action scene; original motion gates remain intact.')
    assert spec['support']['start_control']==spec['phase']['warmup_controls']==100
    assert [pulse(spec,k) for k in range(101,125)]==gate['weights']
    for path in (DRAFT,gate_file,source,lateral.PLAN,lateral.OUT/'analysis.json',lateral.OUT/'cpu-scene-fields.npz'):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    names = ('collect_walk_backward_actuator_support.py','verify_walk_backward_actuator_support.py',
             'check_walk_actuator_support_targets.py','check_walk_bounded_lateral_kinematics.py','check_walk_lateral_kinematics.py')
    for name in names:
        path = Path(__file__).with_name(name)
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    solver = Path(inspect.getsourcefile(lsq_linear))
    spec['library_source_sha256'][solver.as_posix()] = sha(solver)
    onset.prior.check_sources(spec)
    torch.set_num_threads(4)
    cfg,_ = onset.config(p)
    env = ManagerBasedRlEnv(cfg=cfg,device='cpu',render_mode=None)
    try:
        env.reset(seed=p['seed'])
        _,desired = onset.prior.groups_and_commands(p)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        scene = onset.model_arrays(env.sim.mj_model)
        with np.load(lateral.OUT/'cpu-scene-fields.npz') as stored:
            onset.compare_arrays(scene,{k:stored[k] for k in stored.files})
        snapshot = zero_probe(env,spec)
    finally:
        env.close()
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz',**scene)
    scene_path = OUT/'cpu-scene-fields.npz'
    spec['reference_sha256'][scene_path.relative_to(ROOT).as_posix()] = sha(scene_path)
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),
        snapshot=snapshot,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_ACTUATOR_SUPPORT_FROZEN',sha(PLAN),spec['runs'],flush=True)


def geometry(model,p,positions,eligible):
    legs,coords,vel,limits = addresses(model,p)
    actuator_ids = [p['joint_names'].index(name) for name in legs]
    sites = np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,f'robot/{side}_foot')
                      for side in ('left','right')])
    assert (sites>=0).all()
    chosen = []
    shifts = []
    for i in np.flatnonzero(eligible):
        q = positions[i]
        if not ((q[coords]>=limits[:,0])&(q[coords]<=limits[:,1])).all():
            continue
        pair = [target(model,q,sites,coords,vel,limits,sign) for sign in (1.,-1.)]
        if not all(case['feasible'] for case in pair):
            continue
        chosen.append(i)
        shifts.append([case['target_rad'] for case in pair])
    offsets = np.zeros((len(chosen),2,14),dtype=np.float32)
    if shifts:
        offsets[:,:,actuator_ids] = shifts
    return np.asarray(chosen,dtype=np.int64),offsets


def collect(env,actor,spec,record,state,desired):
    prior = onset.prior
    p = spec['environment']
    low,fork = spec['trigger']['window']
    terms = env.action_manager.get_term('joint_pos')
    process,save = terms.process_actions,prior.write
    branch = 'shared-prefix'
    step = 0
    active = None
    instance = None
    targets = None
    entries = []
    limits = torch.tensor(p['joint_limits'],dtype=torch.float32,device=env.device)
    cap = spec['support']['maximum_relative_joint_target_rad']

    class Controller(onset.BackwardController):
        def __init__(self,plan,settings,commands):
            nonlocal instance
            super().__init__(plan,settings,commands)
            self.activity = np.zeros(len(commands),dtype=np.float64)
            self.alive = np.ones(len(commands),dtype=bool)
            instance = self

        def reset(self,ids):
            super().reset(ids)
            self.alive[ids] = False

        def snapshot(self):
            return super().snapshot(),self.activity.copy(),self.alive.copy()

        def restore(self,saved):
            super().restore(saved[0])
            self.activity[:],self.alive[:] = saved[1:]

        def map(self,observed,control,candidate,alive):
            nonlocal step,active
            step,active = control,alive.copy()
            if low<=step<=fork:
                self.activity += abs(observed[:,spec['trigger']['observation_indices']]).mean(axis=1)
                self.alive &= alive
            return super().map(observed,step,False,alive)

    def at_fork(path,value):
        nonlocal branch,targets
        save(path,value)
        if path!=record/'fork-readback.json':
            return
        branch = next(reversed(value))
        assert env.common_step_counter==fork
        if targets is not None:
            return
        support = prior.cpu(env.scene['feet_ground_contact'].data.found).reshape(env.num_envs,2,-1).any(axis=2)
        low_activity = instance.activity/(fork-low+1)<=spec['trigger']['threshold_rad_s']
        eligible = instance.alive & low_activity & support.all(axis=1)
        qpos = prior.cpu(env.sim.data.qpos)
        ids,offsets = geometry(env.sim.mj_model,p,qpos,eligible)
        targets = ids,torch.from_numpy(offsets).to(env.device)
        np.savez_compressed(record/'support-targets.npz',ids=ids,offsets=offsets,eligible=eligible,qpos=qpos)
        print('BACKWARD_SUPPORT_TARGETS',len(ids),'of',int(eligible.sum()),flush=True)

    def adjusted(actions):
        process(actions)
        weight = pulse(spec,step)
        if branch not in SIGNS or weight<=0.:
            return
        ids,offsets = targets
        chosen = active[ids]
        if not chosen.any():
            return
        ix = torch.from_numpy(ids[chosen]).to(env.device)
        shift = offsets[torch.from_numpy(np.flatnonzero(chosen)).to(env.device),0 if branch=='body-left' else 1]
        nominal = terms._processed_actions[ix].clone()
        output = apply_support_target(nominal,shift,limits,weight)
        assert (output-nominal).abs().max()<=cap+1e-7
        terms._processed_actions[ix] = output
        terms.projected_destination[ix] = output
        torch.testing.assert_close(terms._raw_actions,actions,atol=0,rtol=0)
        entries.append(dict(role=branch,step=step,worlds=ids[chosen].tolist(),weight=weight))

    state['actuator_zero_restore'] = zero_probe(env,spec)
    try:
        with patch.object(terms,'process_actions',adjusted),patch.object(prior,'ROLES',ROLES), \
             patch.object(prior,'ForwardPhase',Controller),patch.object(prior,'write',at_fork):
            BASE(env,actor,spec,record,state,desired)
    finally:
        write(record/'support-actuation.json',entries)


def sample(seed):
    assert seed in SEEDS
    base_publish = live.publish
    spec = read(PLAN)
    maximum = spec['maximum_control_transitions']//spec['environment']['parallel_worlds']

    def publish(record,state):
        base_publish(record,dict(state,maximum_controls=maximum,purpose='后退执行器支撑转移 · 固定策略'))

    with patch.object(onset,'PLAN',PLAN),patch.object(onset,'OUT',OUT),patch.object(onset,'STAGE',STAGE), \
         patch.object(onset,'page',page),patch.object(onset.prior,'collect',collect),patch.object(live,'publish',publish):
        try:
            onset.sample(seed)
        finally:
            folder = ROOT/'records/evaluations'/spec['runs'][str(seed)]
            entry = folder/'README.md'
            if entry.is_file():
                entry.write_text(entry.read_text(encoding='utf-8').replace('后退启动','后退执行器支撑转移')
                                 .replace('common_walk_backward_onset.json',PLAN.name),encoding='utf-8')


if __name__=='__main__':
    onset.check_project()
    if sys.argv[1:]==['--freeze']:
        freeze()
    else:
        assert len(sys.argv)==3 and sys.argv[1]=='--sample'
        sample(int(sys.argv[2]))
