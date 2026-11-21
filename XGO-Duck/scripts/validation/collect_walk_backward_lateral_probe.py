"""Fork one real backward release and apply opposite horizontal impulses."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

import collect_walk_backward_onset as onset
import collect_walk_backward_start_pulse as old
from check_walk_backward_lateral_impulse import DRAFT, OUT as CPU_PROOF, lateral_velocity_delta
from common_walk_forward_phase_snapshot import WalkingSnapshot
from getup_release_fork_state import signature
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live


PLAN = ROOT/'configs/experiments/common_walk_backward_lateral_probe.json'
OUT = ROOT/'records/reports/common-walk-backward-lateral-probe'
STAGE = 'S2-backward-lateral-causality-development'
ROLES = ('backward','reference-repeat','body-left','body-right')
SIGNS = {'body-left':1.,'body-right':-1.}
SEEDS = (1075,1076)
BASE_COLLECT = onset.prior.collect


def page(seed):
    return ROOT/f'records/walk-backward-lateral-s{seed}-live.html'


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft, proof, parent = (read(path) for path in (DRAFT,CPU_PROOF,old.PLAN))
    assert proof['status']=='passed-real-cpu-zero-step' and proof['draft_sha256']==sha(DRAFT)
    fixture = Path(lateral_velocity_delta.__code__.co_filename)
    assert proof['source_sha256']==sha(fixture)
    library = Path(proof['installed_velocity_source'])
    assert sha(library)==proof['installed_velocity_source_sha256']
    onset.prior.check_sources(parent)
    geometry = ROOT/draft['evidence']
    assert read(geometry)['status']=='passed-saved-geometry-review'
    for seed in SEEDS:
        assert not any(f'_s{seed}_' in folder.name for folder in (ROOT/'records/evaluations').iterdir())
    spec = deepcopy(parent)
    p = spec['environment']
    n,h,fork = p['parallel_worlds'],p['maximum_control_steps'],spec['phase']['warmup_controls']
    assert (n,h,fork)==(224,300,100)
    controls = fork+len(ROLES)*(h-fork)
    stamp = datetime.now(onset.TZ).strftime('%Y%m%dT%H%M%S')
    p.update(seed=SEEDS[0],stage=STAGE,maximum_transitions=controls*n)
    pulse = spec.pop('pulse')
    spec.update(stage=STAGE,roles=list(ROLES),runs={str(s):f'duck_walk_backward-lateral_paired_s{s}_{stamp}' for s in SEEDS},
        trigger={key:pulse[key] for key in ('threshold_rad_s','observation_indices','window')},
        lateral=dict(magnitude_m_s=draft['design']['dose_draft_delta_lateral_m_s'],signs=SIGNS,
                     support_deadline_controls=30,same_foot_controls=4),
        maximum_control_transitions=controls*n,maximum_physics_substeps=controls*5,
        maximum_sampling_wall_seconds=480,full_budget=controls*n*len(SEEDS),
        scope='One isolated lateral velocity change. Mechanism gate requires the same direction across both new development cohorts, no lost starts and no unsafe intervention.')
    for path in (old.PLAN,DRAFT,CPU_PROOF,geometry,old.OUT/'cpu-scene-fields.npz'):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    validators = ('verify_walk_backward_lateral_probe.py','review_walk_backward_support_transfer.py',
                  'verify_walk_live_page.py')
    for path in (fixture,Path(__file__).resolve(),*(Path(__file__).with_name(name) for name in validators)):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    spec['library_source_sha256'][library.as_posix()] = sha(library)
    onset.prior.check_sources(spec)
    torch.set_num_threads(4)
    cfg,_ = onset.config(p)
    env = ManagerBasedRlEnv(cfg=cfg,device='cpu',render_mode=None)
    try:
        env.reset(seed=p['seed'])
        _, desired = onset.prior.groups_and_commands(p)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        scene = onset.model_arrays(env.sim.mj_model)
        with np.load(old.OUT/'cpu-scene-fields.npz') as stored:
            onset.compare_arrays(scene,{key:stored[key] for key in stored.files})
        zero = reversible_zero(env,spec)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz',**scene)
    scene_file = OUT/'cpu-scene-fields.npz'
    spec['reference_sha256'][scene_file.relative_to(ROOT).as_posix()] = sha(scene_file)
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),
         zero_restore=zero,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_LATERAL_FROZEN',sha(PLAN),spec['runs'],flush=True)


def isolated_write(env,ids,speed):
    term = env.action_manager.get_term('joint_pos')
    robot = env.scene['robot']
    vadr = robot.data.indexing.free_joint_v_adr[:2]
    preserved = {'qpos':env.sim.data.qpos.clone(),'ctrl':env.sim.data.ctrl.clone(),
        'raw':term._raw_actions.clone(),'processed':term._processed_actions.clone(),
        'action':env.action_manager._action.clone(),'obs':env.obs_buf['actor'].clone(),
        'command':env.command_manager.get_command('twist').clone()}
    before = env.sim.data.qvel.clone()
    intended,yaw = lateral_velocity_delta(env,ids,speed)
    expected = before.clone()
    expected[ids[:,None],vadr] += intended
    after = env.sim.data.qvel.clone()
    torch.testing.assert_close(after,expected,atol=0,rtol=0)
    current = {'qpos':env.sim.data.qpos.clone(),'ctrl':env.sim.data.ctrl.clone(),
        'raw':term._raw_actions,'processed':term._processed_actions,'action':env.action_manager._action,
        'obs':env.obs_buf['actor'],'command':env.command_manager.get_command('twist')}
    for name,value in preserved.items():
        torch.testing.assert_close(value,current[name],atol=0,rtol=0)
    cpu = onset.prior.cpu
    return {k:cpu(v) for k,v in dict(ids=ids,yaw=yaw,vadr=vadr,before=before,after=after,
                                   qpos=preserved['qpos']).items()}


def reversible_zero(env,spec):
    snapshot = WalkingSnapshot(env,spec['maximum_warp_snapshot_bytes'])
    expected = snapshot.verify()
    assert expected['counters']==[0,0]
    ids = torch.tensor([0,2,4,6],device=env.device)
    speed = torch.tensor([1.,-1.,1.,-1.],device=env.device)*spec['lateral']['magnitude_m_s']
    isolated_write(env,ids,speed)
    assert snapshot.live_signatures()!=snapshot.expected
    assert snapshot.restore()==expected
    return expected


def collect(env,actor,spec,record,state,desired):
    prior = onset.prior
    trigger = spec['trigger']
    low,fork = trigger['window']
    retained, receipts = {}, {}
    original_write = prior.write

    class Controller(onset.BackwardController):
        def __init__(self,p,settings,commands):
            super().__init__(p,settings,commands)
            self.activity = np.zeros(len(commands),dtype=np.float64)
            self.eligible = np.ones(len(commands),dtype=bool)
            retained['instance'] = self

        def reset(self,ids):
            super().reset(ids)
            self.eligible[ids] = False

        def snapshot(self):
            return super().snapshot(),self.activity.copy(),self.eligible.copy()

        def restore(self,value):
            super().restore(value[0])
            self.activity[:],self.eligible[:] = value[1:]

        def map(self,observed,step,candidate,active):
            if low<=step<=fork:
                self.activity += abs(observed[:,trigger['observation_indices']]).mean(axis=1)
                self.eligible &= active
            return super().map(observed,step,False,active)

    def at_fork(path,data):
        original_write(path,data)
        if path!=record/'fork-readback.json':
            return
        role = next(reversed(data))
        assert role not in receipts and env.common_step_counter==fork and env._sim_step_counter==fork*5
        controller = retained['instance']
        ids = np.flatnonzero(controller.eligible & (controller.activity/(fork-low+1)<=trigger['threshold_rad_s']))
        if role not in SIGNS:
            ids = ids[:0]
        idx = torch.as_tensor(ids,device=env.device)
        speeds = torch.full((len(ids),),spec['lateral']['signs'].get(role,0.)*spec['lateral']['magnitude_m_s'],device=env.device)
        # The actor still sees the common pre-action observation. The next
        # physics control receives the isolated change to native velocity.
        result = isolated_write(env,idx,speeds)
        receipts[role] = sha(path)
        np.savez_compressed(record/f'impulse-{role}.npz',**result)
        print('BACKWARD_LATERAL_AT_FORK',role,len(ids),flush=True)

    state['lateral_zero_restore'] = reversible_zero(env,spec)
    with patch.object(prior,'ROLES',ROLES),patch.object(prior,'ForwardPhase',Controller),patch.object(prior,'write',at_fork):
        BASE_COLLECT(env,actor,spec,record,state,desired)
    assert set(receipts)==set(ROLES)


def sample(seed):
    assert seed in SEEDS
    base_publish = live.publish
    spec = read(PLAN)
    maximum = spec['maximum_control_transitions']//spec['environment']['parallel_worlds']

    def publish(record,state):
        base_publish(record,dict(state,maximum_controls=maximum,purpose='后退双向侧移 · 固定策略'))

    with patch.object(onset,'PLAN',PLAN),patch.object(onset,'OUT',OUT),patch.object(onset,'STAGE',STAGE), \
         patch.object(onset,'page',page),patch.object(onset.prior,'collect',collect),patch.object(live,'publish',publish):
        try:
            onset.sample(seed)
        finally:
            record = ROOT/'records/evaluations'/spec['runs'][str(seed)]
            entry = record/'README.md'
            if entry.is_file():
                entry.write_text(entry.read_text(encoding='utf-8').replace('后退启动','后退双向侧移').replace(
                    'common_walk_backward_onset.json',PLAN.name),encoding='utf-8')


if __name__=='__main__':
    onset.check_project()
    if sys.argv[1:]==['--freeze']:
        freeze()
    else:
        assert len(sys.argv)==3 and sys.argv[1]=='--sample'
        sample(int(sys.argv[2]))
