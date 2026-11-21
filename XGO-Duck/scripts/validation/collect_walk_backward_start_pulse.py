"""One short actuator correction for a low-activity backward start."""
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

import collect_walk_backward_onset as onset
from common_walk_targets import ProjectedWalkAction
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import projected_feedback_target


PLAN = ROOT/'configs/experiments/common_walk_backward_start_pulse.json'
DRAFT = ROOT/'configs/experiments/common_walk_backward_start_pulse_draft.json'
OUT = ROOT/'records/reports/common-walk-backward-start-pulse'
STAGE = 'S2-backward-start-pulse-development'
ROLES = ('backward','reference-repeat','start-pulse')
original_collect = onset.prior.collect


def page(seed):
    return ROOT/f'records/walk-backward-pulse-s{seed}-live.html'


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT)
    source, review = (ROOT/draft[key] for key in ('source','observables'))
    spec, saved = read(source), read(review)
    assert saved['protocol_sha256'] == sha(source) and not saved['heldout_used']
    assert draft['roles'] == list(ROLES)
    assert draft['fit_seed'] == 1071 and draft['check_seed'] == 1072
    onset.prior.check_sources(spec)
    cohorts = []
    for data in saved['cohorts']:
        seed = data['seed']
        proof = onset.OUT/f'seed{seed}-analysis.json'
        assert sha(proof) == data['source_sha256']
        rows = read(proof)['analysis']['worlds']
        assert len(rows) == len(data['activity_rad_s'])
        assert [row['world'] for row in rows] == list(range(len(rows)))
        activity = np.asarray(data['activity_rad_s'])
        early = np.asarray([row['onset_block_end_seconds'] in (1,2) for row in rows])
        stalls = np.asarray(data['observed_stalls'])
        delayed = np.asarray([(row['onset_block_end_seconds'] or 0)>2 for row in rows])
        assert activity[stalls].max() < activity[early].min()
        if seed == draft['fit_seed']:
            threshold = float((activity[stalls].max()+activity[early].min())/2)
        selected = (activity <= threshold)&np.asarray(data['eligible'])
        assert (selected&stalls).sum() == stalls.sum() and not (selected&early).any()
        cohorts.append(dict(seed=seed,near_static=int((selected&stalls).sum()),delayed=int((selected&delayed).sum()),
                            already_started=int((selected&early).sum()), total=int(selected.sum())))
        spec['reference_sha256'][proof.relative_to(ROOT).as_posix()] = sha(proof)
    assert [row['seed'] for row in cohorts] == [1071,1072]
    n, fork, h = (draft[key] for key in ('worlds_per_seed','shared_prefix_controls','controls_per_episode'))
    seeds = draft['new_development_seeds']
    assert not any(f'_s{seed}_' in f.name for seed in seeds for f in (ROOT/'records/evaluations').iterdir())
    controls = fork+len(ROLES)*(h-fork)
    assert draft['maximum_control_transitions'] == controls*n*len(seeds)
    assert draft['maximum_physics_substeps'] == controls*5*len(seeds)
    stamp = datetime.now(onset.TZ).strftime('%Y%m%dT%H%M%S')
    p = spec['environment']
    p.update(seed=seeds[0],parallel_worlds=n,episodes_per_command=n,groups={'backward':list(range(n))},stage=STAGE)
    p['maximum_transitions'] = controls*n
    assert p['maximum_control_steps'] == h and fork == 100
    spec.update(stage=STAGE,roles=list(ROLES),runs={str(s):f'duck_walk_backward-pulse_paired_s{s}_{stamp}' for s in seeds},
                phase=dict(warmup_controls=fork),maximum_control_transitions=controls*n,maximum_physics_substeps=controls*5,
                pulse=dict(threshold_rad_s=threshold,observation_indices=saved['cohorts'][0]['indices'],window=[51,100],
                           length=draft['pulse_controls'],input_m_s=draft['internal_pulse_m_s'],
                           cap_rad=draft['maximum_joint_target_change_rad']),
                full_budget=draft['maximum_control_transitions'],scope=draft['admission'])
    for path in (source, review, DRAFT):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__).resolve(),Path(__file__).resolve().with_name('verify_walk_backward_start_pulse.py'),
                 Path(__file__).resolve().with_name('walking_actuator_yaw_feedback.py')):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    onset.prior.check_sources(spec)
    torch.set_num_threads(4)
    cfg, _ = onset.config(p)
    env = ManagerBasedRlEnv(cfg=cfg,device='cpu',render_mode=None)
    try:
        env.reset(seed=seeds[0])
        _, desired = onset.prior.groups_and_commands(p)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        receipt = onset.prior.snapshot_probe(env,spec['maximum_warp_snapshot_bytes'])
        fields = onset.model_arrays(env.sim.mj_model)
        with np.load(onset.OUT/'cpu-scene-fields.npz') as saved_fields:
            onset.compare_arrays(fields,{key:saved_fields[key] for key in saved_fields.files})
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields)
    spec['reference_sha256'][(OUT/'cpu-scene-fields.npz').relative_to(ROOT).as_posix()] = sha(OUT/'cpu-scene-fields.npz')
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),snapshot=receipt,
          early_starters_protected=True,archive_coverage=cohorts,physics_substeps=0,heldout_used=False))
    print('WALK_BACKWARD_PULSE_FROZEN',sha(PLAN),spec['runs'],flush=True)


def collect(env,actor,spec,record,state,desired):
    term = env.action_manager.get_term('joint_pos')
    assert isinstance(term,ProjectedWalkAction) and term.cfg.scale == 1. and not term.cfg.use_default_offset
    limits = torch.as_tensor(term.cfg.physical_limits,device=env.device,dtype=torch.float32)
    process = term.process_actions
    plan = spec['pulse']
    low, fork = plan['window']
    pending, rows = None, []

    class Controller(onset.BackwardController):
        def __init__(self,p,settings,commands):
            super().__init__(p,settings,commands)
            self.activity = np.zeros(len(commands),dtype=np.float64)

        def reset(self,ids):
            super().reset(ids)
            self.activity[ids] = 0.

        def snapshot(self):
            return super().snapshot(),self.activity.copy()

        def restore(self,state):
            super().restore(state[0])
            self.activity[:] = state[1]

        def map(self,observed,step,candidate,active):
            nonlocal pending
            assert pending is None
            command = super().map(observed,step,candidate,active)
            if low <= step <= fork:
                self.activity += abs(observed[:,plan['observation_indices']]).mean(axis=1)
            if candidate and fork < step <= fork+plan['length']:
                enabled = active & (self.activity/(fork-low+1) <= plan['threshold_rad_s'])
                ids = np.flatnonzero(enabled)
                if ids.size:
                    alternate = observed[ids].copy()
                    alternate[:,48:51] = command[ids]
                    alternate[:,48] = plan['input_m_s']
                    out = actor(onset.td(torch.from_numpy(alternate).to(env.device)))
                    pending = step,ids,out
            return command

    def process_once(actions):
        nonlocal pending
        process(actions)
        if pending is None:
            return
        step, ids, raw = pending
        idx = torch.as_tensor(ids,device=env.device)
        home = term._offset if term._offset.ndim == 1 else term._offset[idx]
        target, fraction = projected_feedback_target(term._processed_actions[idx],raw,home,limits,plan['cap_rad'])
        term._processed_actions[idx] = target
        term.projected_destination[idx] = target
        rows.append((np.full(len(ids),step),ids,onset.prior.cpu(raw),onset.prior.cpu(fraction)))
        pending = None

    try:
        with patch.object(term,'process_actions',process_once), patch.object(onset.prior,'ROLES',ROLES), \
             patch.object(onset.prior,'ForwardPhase',Controller):
            original_collect(env,actor,spec,record,state,desired)
        assert pending is None
    finally:
        receipt = [np.concatenate(col) for col in zip(*rows,strict=True)] if rows else [
            np.empty(0,int),np.empty(0,int),np.empty((0,14),np.float32),np.empty(0,np.float32)]
        np.savez_compressed(record/'pulse-readback.npz',**dict(zip(('step','world','alternate_raw','fraction'),receipt,strict=True)))


def sample(seed):
    with patch.object(onset,'PLAN',PLAN), patch.object(onset,'OUT',OUT), patch.object(onset,'STAGE',STAGE), \
         patch.object(onset,'page',page), patch.object(onset.prior,'collect',collect):
        try:
            onset.sample(seed)
        finally:
            record = ROOT/'records/evaluations'/read(PLAN)['runs'][str(seed)]
            entry = record/'README.md'
            if entry.exists():
                entry.write_text(entry.read_text(encoding='utf-8').replace('后退启动','后退启动助推').replace(
                    'common_walk_backward_onset.json',PLAN.name),encoding='utf-8')


if __name__ == '__main__':
    onset.check_project()
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == '--sample'
        sample(int(sys.argv[2]))
