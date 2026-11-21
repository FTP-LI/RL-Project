"""Run a bounded matched-fork test of the stance-preserving roll targets."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import inspect
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import check_walk_backward_roll_target as geometry_proof
import collect_walk_backward_actuator_support as runner
import verify_walk_backward_actuator_support as verifier
from check_walk_lateral_kinematics import addresses
from record_io import write
from review_roll_foot_transfer import SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_roll_pair.json'
OUT = ROOT/'records/reports/common-walk-backward-roll-pair'
STAGE = 'S2-backward-roll-pair-development'
SEEDS = (1103,1104)
ROLES = runner.ROLES


def page(seed):
    return ROOT/f'records/walk-backward-roll-s{seed}-live.html'


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert ROOT/stage['current_protocol']==geometry_proof.PLAN and stage['current_stage']=='S2-backward-roll-target'
    static = read(geometry_proof.PLAN)
    proof_path = geometry_proof.OUT/'analysis.json'
    proof = read(proof_path)
    assert proof['status']=='passed-roll-target-geometry' and proof['protocol_sha256']==sha(geometry_proof.PLAN)
    assert proof['script_sha256']==sha(Path(geometry_proof.__file__))
    assert all(proof['geometry_admission'].values()) and not proof['heldout_used']
    parent,old_sha = static['parent_protocol']
    assert sha(ROOT/parent)==old_sha
    spec = deepcopy(read(ROOT/parent))
    runner.onset.prior.check_sources(spec)
    assert not any(f'_s{seed}_' in f.name for seed in SEEDS for f in (ROOT/'records/evaluations').iterdir())
    stamp = datetime.now(runner.onset.TZ).strftime('%Y%m%dT%H%M%S')
    runs = {str(seed):f'duck_walk_backward-roll_paired_s{seed}_{stamp}' for seed in SEEDS}
    witnesses = []
    for seed in static['seeds']:
        old = read(runner.OUT/f'seed{seed}-analysis.json')
        witness = ROOT/old['record']/'windows-view-verification.json'
        assert read(witness)['status']=='passed'
        witnesses.append(witness)
    spec['environment'].update(seed=SEEDS[0],stage=STAGE)
    assert spec['environment']['parallel_worlds']==224 and spec['environment']['maximum_control_steps']==300
    assert spec['phase']['warmup_controls']==100 and spec['maximum_control_transitions']==201600
    assert spec['maximum_physics_substeps']==4500 and list(ROLES)==spec['roles']
    assert spec['support']['maximum_relative_joint_target_rad']==static['design']['maximum_joint_offset_rad']
    spec.update(stage=STAGE,runs=runs,roll_design=static['design'],windows_references=[relative(path) for path in witnesses],
                first_seed_gate=dict(common_stalls_minimum=3,measured_roll_deg_minimum=.3,
                                     measured_roll_gain_over_reference_mean_deg_minimum=.1),
                second_seed_condition='Run the same pair only if seed1103 admits a direction. Replication requires the same direction, matched rescue, no losses, no unsafe samples, and verified Windows observation.',
                scope='Fixed actor; roll offsets only on eligible low-activity bilateral stalls. Retain full raw history, target limits and original backward scoring.')
    for path in (ROOT/parent,geometry_proof.PLAN,proof_path,*witnesses):
        spec['reference_sha256'][relative(path)] = sha(path)
    for path in (Path(__file__),Path(geometry_proof.__file__),
                 Path(__file__).with_suffix('.sh'),Path(__file__).with_name('verify_walk_live_page.py')):
        spec['source_sha256'][relative(path)] = sha(path)
    optimizer = Path(inspect.getsourcefile(geometry_proof.least_squares))
    spec['library_source_sha256'][optimizer.as_posix()] = sha(optimizer)
    torch.set_num_threads(4)
    p = dict(spec['environment'],run_id=runs[str(SEEDS[0])])
    cfg,_ = runner.onset.config(p)
    env = ManagerBasedRlEnv(cfg=cfg,device='cpu',render_mode=None)
    try:
        env.reset(seed=SEEDS[0])
        _,desired = runner.onset.prior.groups_and_commands(p)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        fields = runner.onset.model_arrays(env.sim.mj_model)
        with np.load(runner.OUT/'cpu-scene-fields.npz') as old:
            runner.onset.compare_arrays(fields,{key:old[key] for key in old.files})
        snapshot = runner.zero_probe(env,spec)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    scene = OUT/'cpu-scene-fields.npz'
    np.savez_compressed(scene,**fields)
    spec['reference_sha256'][relative(scene)] = sha(scene)
    runner.onset.prior.check_sources(spec)
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),
          snapshot=snapshot,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_ROLL_PAIR_FROZEN',sha(PLAN),runs,flush=True)


def offsets(model,p,positions,eligible):
    design = read(PLAN)['roll_design']
    legs,coords,_,limits = addresses(model,p)
    actuator_ids = [p['joint_names'].index(name) for name in legs]
    sites = np.array([model.site(f'robot/{side}_foot').id for side in geometry_proof.SIDES])
    geometry = SoleGeometry(model)
    chosen,shifts = [],[]
    for i in np.flatnonzero(eligible):
        pair = [geometry_proof.target(geometry,sites,positions[i],coords,limits,k,design)
                for k in range(len(geometry_proof.SIDES))]
        if not all(item['feasible'] for item in pair):
            continue
        chosen.append(i)
        shifts.append([item['joint_delta_rad'] for item in pair])
    result = np.zeros((len(chosen),2,14),dtype=np.float32)
    if shifts:
        result[:,:,actuator_ids] = shifts
    return np.asarray(chosen,dtype=np.int64),result


@contextmanager
def adapted():
    with patch.multiple(runner,PLAN=PLAN,OUT=OUT,STAGE=STAGE,SEEDS=SEEDS,page=page,geometry=offsets), patch.multiple(
        verifier,PLAN=PLAN,OUT=OUT,SEEDS=SEEDS,page=page,geometry=offsets,
    ):
        yield


def sample(seed):
    assert seed in SEEDS
    spec = read(PLAN)
    entry = read(OUT/'windows-entry.json')
    assert entry['status']=='passed' and entry['protocol_sha256']==sha(PLAN) and page(seed).is_file()
    if seed==SEEDS[1]:
        first = read(OUT/f'seed{SEEDS[0]}-decision.json')
        assert first['passed_directions'] and first['protocol_sha256']==sha(PLAN)
        prior = ROOT/first['record']/'windows-view-verification.json'
        assert read(prior)['status']=='passed'
    with adapted():
        runner.sample(seed)


def verify(seed):
    with adapted():
        verifier.verify(seed)
    spec = read(PLAN)
    path = OUT/f'seed{seed}-analysis.json'
    proof = read(path)
    record = ROOT/proof['record']
    fork = spec['phase']['warmup_controls']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['shared_prefix_sha256']
    with np.load(prefix) as data:
        initial = Rotation.from_quat(np.roll(data['root_quaternion'][-1],-1,axis=-1))
    by_role = {}
    for role in ROLES:
        trace = record/f'trace-{role}.npz'
        assert sha(trace)==proof['trace_sha256'][role]
        with np.load(trace) as data:
            by_role[role] = data['root_quaternion'][:spec['support']['support_deadline_controls']]
    gate = spec['first_seed_gate']
    directions = {}
    for i,role in enumerate(ROLES[2:]):
        side = geometry_proof.SIDES[i]
        sign = 1 if side=='right' else -1
        outcome = proof['directions'][role]
        eligible = []
        details = []
        for world in outcome['linked_support_and_onset']:
            support = proof['support'][role][str(world)]
            index = support['control']-fork-1
            assert 0<=index<len(by_role[role])
            measured = {}
            for label in (role,*ROLES[:2]):
                position = Rotation.from_quat(np.roll(by_role[label][index,world],-1))
                measured[label] = float(np.rad2deg((initial[world].inv()*position).as_rotvec()[0])*sign)
            gain = measured[role]-float(np.mean([measured[label] for label in ROLES[:2]]))
            passed = (support['side']==side and measured[role]>=gate['measured_roll_deg_minimum'] and
                      gain>=gate['measured_roll_gain_over_reference_mean_deg_minimum'])
            details.append(dict(world=world,expected_stance=side,observed_stance=support['side'],
                           control=support['control'],measured_roll_toward_stance_deg=measured,gain_deg=gain,passed=passed))
            if passed:
                eligible.append(world)
        directions[role] = dict(linked_with_measured_roll=eligible,details=details,
             passed=bool(outcome['passed'] and eligible and len(proof['common_near_static'])>=gate['common_stalls_minimum']))
    admitted = [role for role,detail in directions.items() if detail['passed']]
    if seed==SEEDS[1]:
        prior = read(OUT/f'seed{SEEDS[0]}-decision.json')
        admitted = [role for role in admitted if role in prior['passed_directions']]
    write(OUT/f'seed{seed}-decision.json',dict(status='passed-integrity-decision',protocol_sha256=sha(PLAN),
        analysis_sha256=sha(path),record=proof['record'],common_stalls=len(proof['common_near_static']),
        directions=directions,passed_directions=admitted,heldout_used=False,learning_updates=0))
    print('BACKWARD_ROLL_PAIR_DECISION',seed,len(proof['common_near_static']),admitted,flush=True)


if __name__=='__main__':
    runner.onset.check_project()
    if sys.argv[1:]==['--freeze']:
        freeze()
    else:
        command,seed = sys.argv[1:]
        assert command in ('--sample','--verify') and int(seed) in SEEDS
        (sample if command=='--sample' else verify)(int(seed))
