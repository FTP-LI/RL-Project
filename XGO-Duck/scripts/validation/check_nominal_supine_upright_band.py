"""Verify that bounded upright credit favors the intended motion on saved pairs."""

from pathlib import Path

import numpy as np
import torch

from getup_upright_band_reward import upright_band_increment, upright_credit
from review_nominal_skill_composition_handoff import read
from roll_metrics import rotation
from run_nominal_supine_early_load import PROJECT, REPORT, ROOT, parent, rel, sha, write


KEY = 'nominal_supine_upright_band'
OUT = REPORT/'support-progress'


def settings(spec, stage):
    release = stage['release_control']
    milliseconds_per_control = 1000 * stage['controller_period_s']
    start, stop = [release+round(k/milliseconds_per_control) for k in spec['relative_release_window_ms']]
    return dict(start_age=start, stop_age=stop, full_credit_tilt_deg=spec['full_credit_tilt_deg'],
                credit_starts_tilt_deg=spec['credit_starts_below_tilt_deg'])


def fixtures(config):
    # Exercise the exact component used for archive replay and prospective
    # training: early credit, genuine progression, loss, and reset exclusion.
    start, stop = config['start_age'],config['stop_age']
    before = torch.cos(torch.deg2rad(torch.tensor([10.,25.,20.,12.,9.,11.])))
    after = torch.cos(torch.deg2rad(torch.tensor([10.,25.,15.,34.,9.,11.])))
    ages = torch.tensor([start,start,start+1,start+3,stop,start])
    supine = torch.tensor([True,True,True,True,True,False])
    scored = upright_band_increment(before,after,ages,supine,**config)
    torch.testing.assert_close(scored,torch.tensor([1.,1/3,1/3,-1.,0.,0.]),atol=2e-6,rtol=0.)
    return scored.tolist()


def review(seed, pairs, config, load, spec):
    archive = ROOT/seed['output_clock']
    assert sha(archive) == seed['clock_sha256'] == pairs['clock_sha256']
    steps_per_control = round(load['controller_period_s']/load['physics_period_s'])
    controls = np.arange(config['start_age'],config['stop_age']+1)
    controls_physics = controls*steps_per_control
    missed = [row['missed_world'] for row in pairs['worlds']]
    admitted = [row['admitted_world'] for row in pairs['worlds']]
    assert len(missed)==pairs['distinct_pairs']==len(admitted)
    with np.load(archive,allow_pickle=False) as rows:
        offset = np.searchsorted(rows['substep'],controls_physics)
        np.testing.assert_array_equal(rows['substep'][offset],controls_physics)
        up = rotation(rows['after'][offset][:,missed+admitted])[...,2,2]
    x = torch.from_numpy(up)
    age = torch.from_numpy(controls[:-1])[:,None]
    contributing = upright_band_increment(x[:-1],x[1:],age,torch.ones_like(x[1:],dtype=torch.bool),**config)
    levels = upright_credit(x,config['full_credit_tilt_deg'],config['credit_starts_tilt_deg'])
    torch.testing.assert_close(contributing.sum(dim=0),levels[-1],atol=1e-14,rtol=0.)
    discount = spec['discount']**torch.arange(len(contributing),dtype=x.dtype)
    scores = (discount@contributing).numpy()
    n = len(missed)
    wins = scores[n:]-scores[:n]
    return dict(seed=seed['seed'],pairs=n,admitted_higher=int((wins>0).sum()),tied=int((wins==0).sum()),
                admitted_median=float(np.median(scores[n:])),missed_median=float(np.median(scores[:n])),
                paired_median_admitted_minus_missed=float(np.median(wins)),
                candidate_step_abs_max=float(contributing.abs().max()),
                final_full_credit_admitted=int((levels[-1,n:]==1.).sum()),
                final_full_credit_missed=int((levels[-1,:n]==1.).sum()),
                per_pair=[dict(missed_world=i,admitted_world=j,missed_return=float(s),admitted_return=float(t))
                          for i,j,s,t in zip(missed,admitted,scores[:n],scores[n:],strict=True)])


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    design = ROOT/node['design']
    spec = read(design)
    assert node['status']=='archive-screen-ready' and parent.allowed(project,spec['stage'])
    assert not project['staged_training_request']['simulation_running']
    upstream = project['goal_execution']
    signed = read(ROOT/upstream['nominal_supine_support_progress']['preflight'],spec['signed_progress_preflight_sha256'])
    assert not signed['replay_screen_passed']
    alignment = read(ROOT/upstream['nominal_supine_support_rotation_alignment']['analysis'],spec['source_alignment_sha256'])
    node_load = upstream['nominal_supine_early_load']
    load = read(ROOT/node_load['analysis'],node_load['analysis_sha256'])
    original = read(ROOT/node_load['design'],node_load['design_sha256'])
    configuration = settings(spec,original)
    assert spec['full_credit_tilt_deg'] == 15 and spec['credit_starts_below_tilt_deg']==2*spec['full_credit_tilt_deg']
    assert signed['parent']['widths'] == dict(actor=spec['actor_observation_width'],critic=spec['critic_observation_width'])
    torch.set_num_threads(2)
    synthetic = fixtures(configuration)
    results = [review(seed,paired,configuration,original,spec)
               for seed,paired in zip(load['seed_results'],alignment['seed_results'],strict=True)]
    eligible = all(row['paired_median_admitted_minus_missed']>0 and
                   row['admitted_higher']/row['pairs']>=spec['minimum_admitted_higher_fraction_each_seed'] for row in results)
    path = OUT/'upright-band-preflight.json'
    assert not path.exists() and not torch.cuda.is_initialized()
    component = ROOT/'scripts/training/getup_upright_band_reward.py'
    write(path,dict(proposal_sha256=sha(design),component_sha256=sha(component),script_sha256=sha(Path(__file__)),
        parent_preflight_sha256=spec['signed_progress_preflight_sha256'],term=configuration,
        synthetic_edge_values=synthetic,cohorts=results,qualifies_cuda_start=eligible,
        new_physics_steps=0,learning_updates=0,hardware_io=False))
    node.update(status='archive-screen-reviewed',design_sha256=sha(design),preflight=rel(path),
                preflight_sha256=sha(path),qualifies_cuda_start=eligible)
    task = project['staged_training_request']
    task.update(current_evidence=rel(path),current_protocol_execution='nominal-supine-upright-band-archive-reviewed')
    project.update(current_evidence=rel(path),current_protocol_execution=task['current_protocol_execution'])
    write(PROJECT,project)
    for row in results:
        print('SUPINE_UPRIGHT_BAND_COHORT',{k:v for k,v in row.items() if k!='per_pair'},flush=True)
    print('SUPINE_UPRIGHT_BAND_PREFLIGHT',sha(path),eligible,flush=True)


if __name__=='__main__':
    main()
