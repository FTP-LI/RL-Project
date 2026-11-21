"""Check 2 ms and 1 ms CUDA clocks on actual saved low-friction roll entries."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

from mjlab.scene import Scene
from mjlab.sim import Simulation
import numpy as np
import torch

from local_getup_scale import gpu_sample
from preflight_nominal_timestep_low_friction import configured
from qualify_nominal_roll_cuda_timestep import SAME_FIELDS, measured
from review_nominal_skill_composition_handoff import read
from roll_stand_walk_sequence import ROLL
from run_nominal_skill_composition import ROOT, PROJECT, allowed, rel, sha, write
from selective_friction_guard import resolve_foot_geometry_ids


KEY = 'nominal_low_friction_roll_clock'
STAGE = 'nominal-low-friction-roll-clock-preflight'
PREVIOUS = 'nominal-supine-reward-friction-trial'
LOW = 'nominal_timestep_low_friction_development'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-roll-clock'


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert allowed(project,PREVIOUS) and not task['simulation_running'] and KEY not in goal
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    earlier = goal[LOW]
    source = ROOT/earlier['protocol']
    raw = read(ROOT/earlier['collection'],earlier['collection_sha256'])
    plan = read(source,earlier['protocol_sha256'])
    review = goal[LOW+'_mechanism_review']
    mechanisms = read(ROOT/review['evidence'],review['evidence_sha256'])
    paired = goal['nominal_supine_reward_friction_trial']
    assert not read(ROOT/paired['result'],paired['result_sha256'])['advance_to_integrated_trial']
    branch = raw['branches']['lower-friction']
    begin = plan['composition']['roll_start_control']
    start, = (row for row in branch['chunks'] if row['first']<=begin<=row['last'])
    motor, = (row for row in branch['boundaries'] if row['control']==begin)
    developed = (ROOT/earlier['collection']).parent/'lower-friction'
    sources = {rel(path):sha(path) for path in (
        Path(__file__).resolve(),Path(measured.__code__.co_filename).resolve(),
        Path(configured.__code__.co_filename).resolve(),source,ROOT/earlier['collection'],
        ROOT/review['evidence'],ROOT/paired['result'])}
    for row in (start,motor):
        path = developed/row['file']
        assert sha(path)==row['sha256']
        row['path'] = rel(path)
        sources[row['path']] = row['sha256']
    losses = [row['world'] for row in mechanisms['rolling']]
    admitted = sorted(branch['command_entry_worlds'])[:10]
    assert len(set(losses+admitted))==16 and begin==600
    assert plan['composition']['foot_friction']==.8
    REPORT.mkdir()
    write(PLAN,dict(stage=STAGE,source=rel(source),source_sha256=sources,seed=1257,reserved_seed=2246,
        reference=.002,candidate=.001,motor_interval_s=.004,archived_control=begin,start=start,motor=motor,
        archived_support_losses=losses,retained_route_examples=admitted,maximum_solver_launches=30,
        maximum_scene_solver_steps=480,learning_updates=0,hardware_io=False))
    goal[KEY] = dict(status='frozen',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    task.update(current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Check equal 4 ms motor holds at 2 ms and 1 ms on saved rolling-entry inputs.')
    project.update(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    write(PROJECT,project)
    print('LOW_FRICTION_ROLL_CLOCK_FROZEN',sha(PLAN),flush=True)


def check():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    state = goal[KEY]
    assert allowed(project,STAGE) and not task['simulation_running'] and state['status']=='frozen'
    plan = read(PLAN,state['protocol_sha256'])
    assert all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items())
    original = read(ROOT/plan['source'])
    ids = np.asarray(plan['archived_support_losses']+plan['retained_route_examples'])
    source = plan['start']
    with np.load(ROOT/source['path'],allow_pickle=False) as raw:
        k = plan['archived_control']-source['first']
        assert raw['control'][k]==plan['archived_control'] and (raw['roll_phase'][k,ids]==ROLL).all()
        start = tuple(torch.as_tensor(raw[name][k,ids],device='cuda:0') for name in ('qpos','qvel'))
    with np.load(ROOT/plan['motor']['path'],allow_pickle=False) as raw:
        inputs = torch.as_tensor(raw['motor'][:,ids],device='cuda:0')
    assert inputs.shape==(5,16,14) and torch.isfinite(inputs).all()
    cfg = configured(original)
    model = Scene(cfg.scene,device='cpu').compile()
    feet = list(resolve_foot_geometry_ids(model))
    # The isolated simulator has no scene event manager. Apply exactly the
    # two verified slide coefficients before the model is uploaded to CUDA.
    model.geom_friction[feet,0] = .8
    sim = Simulation(num_envs=len(ids),cfg=cfg.sim,model=model,device='cuda:0')
    active_friction = sim.wp_model.geom_friction.numpy()[...,feet,0]
    np.testing.assert_array_equal(active_friction,np.full(active_friction.shape,.8,dtype=active_friction.dtype))
    current = gpu_sample()
    assert current['used_mib']/current['total_mib']<=original['composition']['observer']['gpu_memory_fraction_limit']
    samples = [measured(sim,period,plan['motor_interval_s'],start,inputs)
               for period in (plan['reference'],plan['candidate'])]
    for name in SAME_FIELDS:
        torch.testing.assert_close(samples[0][0][name],samples[1][0][name],atol=0,rtol=0)
    clocks = [row for _,row in samples]
    count = sum(row['executed_solver_steps'] for row in clocks)
    assert count==plan['maximum_solver_launches'] and count*len(ids)==plan['maximum_scene_solver_steps']
    cached = REPORT/'clock-readback.json'
    write(cached,dict(status='verified-cuda-clock',protocol_sha256=sha(PLAN),worlds=ids.tolist(),
        same_readback_fields=list(SAME_FIELDS),sliding_friction=.8,actuator_hold_ms=4,intervals=5,
        closed_loop_evaluated=False,closed_loop_starts_from_serialized_state=False,
        added_solver_passes=count,added_scene_solver_steps=count*len(ids),
        modes=clocks,cuda_sample=current,learning_updates=0,reserved_seed_used=False,hardware_io=False))
    state.update(status='reviewed',evidence=rel(cached),evidence_sha256=sha(cached))
    info = dict(current_protocol_execution=STAGE+'-verified',current_evidence=rel(cached))
    task.update(info,next_gate='Compare 2 ms repeats and 1 ms only during the first 1.20 s under the original rolling and command gates.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_ROLL_CLOCK_VERIFIED',sha(cached),count,flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','check'))
    action = parser.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if action=='freeze' else check)()
