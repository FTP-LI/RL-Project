"""Measure solved contact loads and motor effort in a common low-friction release."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import recover_nominal_low_friction_support_effort_zero as qualified_effort
import verify_nominal_low_friction_loaded_upright_ppo as prior


comparison = prior.comparison
collector, driver = comparison.collector, comparison.driver
recorder, parent = collector.recorder, prior.parent
ROOT, PROJECT, read, rel, sha, write = prior.ROOT, prior.PROJECT, prior.read, prior.rel, prior.sha, prior.write
KEY, STAGE = 'nominal_low_friction_forward_support', 'nominal-low-friction-forward-support'
PLAN = prior.PLAN.with_stem(KEY)
REPORT = parent.REPORT / 'low-friction-forward-support'
ROLES = (prior.ROLES[0], 'parent-policy-repeat', *prior.ROLES[1:])
REPEAT = 'getup-supine-repeat'
MODEL_ROLES = dict(prior.MODEL_ROLES, **{'parent-policy-repeat': REPEAT})
LABELS = dict(prior.LABELS, **{'parent-policy-repeat': '父策略同权重复测'})
START, END, WORLDS = 100, 200, 896


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert parent.allowed(project, prior.STAGE) and not task['simulation_running']
    assert KEY not in goal and not (PLAN.exists() or REPORT.exists()) and not torch.cuda.is_initialized()
    reviewed = goal[prior.KEY]
    assert reviewed['status'] == 'reviewed' and not reviewed['eligible_for_full_development']
    timing = reviewed['forward_recovery_readback']
    targets = read(ROOT/timing['path'], timing['sha256'])
    focus = reviewed['next_forward_development_worlds']
    assert focus == [439,779] and targets['forward_repeatability']['differing_forward_worlds'] == [283,727]
    plan = read(ROOT/reviewed['protocol'], reviewed['protocol_sha256'])
    d, env = plan['composition'], plan['environment']
    assert (env['seed'], d['worlds'], d['foot_friction'], d['release_control'], d['roll_start_control']) == (1257,WORLDS,.8,START,600)
    assert d['existing_low_friction_reserved_seed'] == 2246 and d['maximum_controls'] == 2125
    assert plan['roll_route']['standing_dwell_controls'] == 25 and plan['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
    assert set(plan['seven_command']['minimum_by_command'].values()) == {122} and plan['seven_command']['minimum_overall'] == 852
    load = goal['nominal_low_friction_support_effort_limit_readback']
    assert load['status'] == 'reviewed'
    pairs = read(ROOT/load['result'], load['result_sha256'])['release_support']['matches']
    matches = {row['missed']['world']: row['admitted']['world'] for row in pairs}
    assert len(matches) == 13 and all(world in matches for world in focus)
    selected = sorted(set(plan['seven_command']['groups']['forward']) | set(matches) | set(matches.values()) | {672,863})
    pristine = ROOT/plan['standing_pair']['archived_initial']
    assert sha(pristine.with_name('reset.npz')) == read(pristine)['reset_sha256']
    sources = (collector, driver, recorder, comparison.measured, qualified_effort, qualified_effort.original)
    files = [Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in sources),
             Path(__file__).with_name('roll_contact_clock_capture.py')]
    for path, checksum in ((reviewed['protocol'],reviewed['protocol_sha256']), (reviewed['result'],reviewed['result_sha256']),
                           (timing['path'],timing['sha256']), (load['result'],load['result_sha256'])):
        assert sha(ROOT/path) == checksum
        plan['references'][path] = checksum
    plan['source_sha256'].update({rel(path):sha(path) for path in files})
    plan['models'][REPEAT] = dict(plan['models']['getup-supine'])
    plan['policy_home'][REPEAT] = plan['policy_home']['getup-supine']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-forward-support-friction080-s1257-{stamp}'
    env['run_id'] = run
    controls = START + len(ROLES)*(END-START)
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', maximum_unique_controls=controls,
        maximum_physics_substeps=5*controls, maximum_control_transitions=controls*WORLDS,
        forward_support=dict(windows=[dict(name='release',start=START,stop=END)], selected_worlds=selected,
            focus_worlds=focus, matched_archive_worlds={str(world):matches[world] for world in focus},
            repeatability_worlds=[283,727], nominal_load_fraction=.02, both_feet_fraction=.5, minimum_load_solves=5,
            diagnostic_upright_threshold_deg=15, comparison_window_ms=80, physics_dt=.004, heldout_seed=2246,
            action_score_source=reviewed['result'], original_route_protocol=reviewed['protocol'],
            scope='Four identical releases for measurement through control 200; future acceptance and hardware unchanged.',
            expected_learning_updates=0))
    plan['upright_band_comparison'].update(roles=list(ROLES),model_by_role=MODEL_ROLES,end_control=END,
        evaluation_stage='contact and motor measurement only through control 200')
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    evidence = REPORT/'preflight.json'
    write(evidence,dict(protocol_sha256=sha(PLAN),observer=observer,zero_physical_steps=0,
        selected_worlds=len(selected),retained_original_gates=True,maximum_transitions=plan['maximum_control_transitions']))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,preflight=rel(evidence),hardware_io=False)
    prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',evidence,
        'Collect solved foot and head loads and actual motor effort over the four shared-release branches.')
    write(PROJECT,project)
    print('FORWARD_SUPPORT_FROZEN',sha(PLAN),run,len(selected),flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    scope = plan['forward_support']
    meter = qualified_effort.SupportEffortWithControlRange(env,scope)
    physics = env.sim.step
    base_fork = recorder.StandingFork
    capture = np.asarray(scope['selected_worlds'])
    names = iter(ROLES)
    info = {}
    role = None

    def finish_role():
        samples = meter.rows['release']
        if samples:
            assert role is not None and role not in info
            info[role] = meter.save(folder/role)
        samples.clear()
        meter.maximum_readback_error = 0.

    class CaptureFork(base_fork):
        def restore(self):
            nonlocal role
            finish_role()
            state = super().restore()
            role = next(names)
            return state

    def step():
        rows = meter.rows['release']
        previous = len(rows)
        meter.step(physics)
        if len(rows) != previous:
            last = rows[-1]
            rows[-1] = {key:value[capture] if isinstance(value,np.ndarray) else value for key,value in last.items()}

    try:
        with (patch.object(recorder,'StandingFork',CaptureFork),
              patch.object(meter.inner,'compute_control',meter.motor),
              patch.object(env.sim,'step',step)):
            comparison.measured.collect(env,plan,actors,folder,observer,watch,ledger)
    finally:
        finish_role()
        ledger['forward_support'] = dict(worlds=capture.tolist(),roles=info)
    assert list(info) == list(ROLES)
    assert all(item['sampled_substeps'] == 5*(END-START) for item in info.values())


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample'))
    option = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.multiple(parent,config=comparison.measured.friction.configured,PLAN=PLAN),
          patch.multiple(collector,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES,LABELS=LABELS),
          patch.object(collector.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample}[option]()


if __name__ == '__main__':
    main()
