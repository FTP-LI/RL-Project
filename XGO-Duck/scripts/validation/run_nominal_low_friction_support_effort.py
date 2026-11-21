"""Measure simultaneous floor load and servo effort on the unmodified low-friction route."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from getup_release_fork_state import signature
from recover_nominal_skill_composition_zero import isolated_components
from review_nominal_skill_composition_handoff import read
from roll_contact_clock_capture import ContactClock, LABELS
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent
from preflight_nominal_timestep_low_friction import configured, RESET_FIELDS


ROOT, PROJECT = parent.ROOT, parent.PROJECT
rel, sha, write, cpu = parent.rel, parent.sha, parent.write, parent.cpu
KEY, STAGE = 'nominal_low_friction_support_effort', 'nominal-low-friction-support-effort'
PREVIOUS, PREVIOUS_KEY = 'nominal-low-friction-upright-early-return', 'nominal_low_friction_upright_early_return'
PLAN, REPORT = parent.PLAN.with_name(f'{KEY}.json'), parent.REPORT/'low-friction-support-effort'
ENTRY = Path(__file__).resolve()
QUALIFY, COLLECT, RENDER = parent.qualify, parent.collect, parent.render_status
CONTROLS = (('release', 100, 175), ('before-deadline', 250, 400))


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert parent.allowed(project, PREVIOUS) and not task['simulation_running']
    assert KEY not in goal and not (PLAN.exists() or REPORT.exists()) and not torch.cuda.is_initialized()
    prior = goal[PREVIOUS_KEY]
    source, score = (ROOT/prior[field] for field in ('protocol', 'result'))
    plan, result = (read(path, prior[field+'_sha256']) for path, field in ((source, 'protocol'), (score, 'result')))
    assert prior['status']=='reviewed' and not result['ready_for_full_roll_comparison']
    tracking = REPORT.with_name('low-friction-upright-early-return')/'target-tracking.json'
    assert tracking.is_file()
    plant, route = plan['environment'], plan['composition']
    assert (plant['seed'], route['worlds'], route['foot_friction'], route['existing_low_friction_reserved_seed']) == (1257, 896, .8, 2246)
    criteria = plant['criteria']
    end, horizon, dt = route['roll_start_control'], route['maximum_controls'], route['control_period_s']
    deadline = round(criteria['deadline_seconds']/dt)
    assert (end, horizon, dt, deadline, route['getup_to_stand_control']) == (600, 2125, .02, 400, 550)
    assert route['release_control']==CONTROLS[0][1] and deadline==CONTROLS[-1][-1]
    init = ROOT/plan['standing_pair']['archived_initial']
    fields = ROOT/plan['friction_development']['startup_fields']
    reset = init.with_name('reset.npz')
    assert sha(reset)==read(init)['reset_sha256']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-support-effort-friction080-s1257-{stamp}'
    plant['run_id'] = run
    plan['composition']['maximum_controls'] = end
    for key in ('standing_pair', 'upright_recovery', 'return_timing_comparison', 'solver_comparison'):
        plan.pop(key, None)
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', maximum_unique_controls=end,
        maximum_control_transitions=end*route['worlds'], maximum_physics_substeps=5*end,
        support_effort=dict(windows=[dict(name=name, start=start, stop=stop) for name,start,stop in CONTROLS],
            fixed_policy=True, same_startup=rel(init), environment_horizon_controls=horizon,
            physics_dt=.004, nominal_load_fraction=.02, both_feet_fraction=.5, minimum_load_solves=5,
            saturation_of_limit_fraction=.995, settled_controls=[350, 400], independent_seed=2246))
    plan['references'].update({rel(path):sha(path) for path in (source, score, tracking, init, fields, reset)})
    plan['source_sha256'].update({rel(path):sha(path) for path in (
        ENTRY, ENTRY.with_name('roll_contact_clock_capture.py'),
        ENTRY.with_name('review_nominal_low_friction_support_effort.py'))})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN, plan)
    preflight = REPORT/'preflight.json'
    write(preflight, dict(observer=observer, protocol_sha256=sha(PLAN), physical_steps=0,
                          maximum_control_transitions=plan['maximum_control_transitions'], learning_updates=0))
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), preflight=rel(preflight),
                     run_id=run, hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen', current_evidence=rel(preflight))
    task.update(info, current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Measure actual head and foot loads, firmware voltage duty and applied torque before the original recovery deadline.')
    project.update(info)
    write(PROJECT, project)
    print('LOW_FRICTION_SUPPORT_EFFORT_FROZEN', sha(PLAN), run, flush=True)


def config(plan):
    route = dict(plan['composition'], maximum_controls=plan['support_effort']['environment_horizon_controls'])
    return configured(dict(plan, composition=route))


class SupportEffort:
    def __init__(self, env, spec):
        self.env = env
        assert env.physics_dt == spec['physics_dt']
        self.ranges = [(5*row['start'], 5*row['stop'], row['name']) for row in spec['windows']]
        clock_plan = dict(capture_seconds=[self.ranges[0][0]*env.physics_dt, self.ranges[-1][1]*env.physics_dt],
                          maximum_batched_substeps=self.ranges[-1][1],
                          load_threshold_nominal_weight_fraction=spec['nominal_load_fraction'])
        self.clock = ContactClock(env, clock_plan)
        self.rows = {name: [] for _,_,name in self.ranges}
        self.window_by_substep = {n: name for first,last,name in self.ranges for n in range(first+1,last+1)}
        self.pending = None
        self.maximum_readback_error = 0.
        self.actuator, = env.scene['robot'].actuators
        self.inner = self.actuator._bam_model.actuator
        assert type(self.inner).__name__=='STS3215Actuator' and self.inner.max_current is None
        self.original = self.inner.compute_control
        self.ids = self.actuator._global_ctrl_ids
        self.names = list(self.actuator._target_names)
        model = env.sim.mj_model
        ids = cpu(self.ids)
        assert model.nu==len(self.names)==14 and sorted(ids)==list(range(model.nu))
        assert model.actuator_forcelimited[ids].all() and not model.actuator_ctrllimited[ids].any()
        limits = model.actuator_forcerange[ids]
        assert (limits[:,1]>0.).all() and np.array_equal(limits[:,1], -limits[:,0])
        assert np.array_equal(model.actuator_gainprm[ids,0], np.ones(model.nu))
        self.limits = torch.as_tensor(limits[:,1], device=env.device, dtype=torch.float32)
        self.cap = float(self.inner.max_pwm)
        self.kt = float(self.actuator._bam_model.kt.value)
        self.resistance = float(self.actuator._bam_model.R.value)

    def motor(self, target, position, velocity, dt):
        voltage = self.original(target, position, velocity, dt)
        # The official actuator performs delay, filtering, back-EMF and duty clipping.
        # Keep the original tensor on the write path and read it on the same solver call.
        if self.env._sim_step_counter in self.window_by_substep:
            assert self.pending is None
            self.pending = voltage, self.inner.vin, velocity
        return voltage

    def step(self, original):
        n = self.env._sim_step_counter
        name = self.window_by_substep.get(n)
        if name is None:
            original()
            return
        voltage, supply, velocity = self.pending
        self.pending = None
        sim = self.env.sim.data
        root_before = cpu(sim.qpos[:,3:7])
        command = sim.ctrl[:,self.ids].clone()
        requested = (voltage-self.kt*velocity)*(self.kt/self.resistance)
        original()
        _, normal = self.clock.read_contacts()
        actual = sim.actuator_force[:,self.ids]
        duty = voltage/supply
        errors = torch.stack(((requested-command).abs().max(), (command-self.actuator._prev_motor_torque).abs().max(),
                              (actual-command.clamp(-self.limits,self.limits)).abs().max()))
        errors = cpu(errors)
        assert errors.max()<=5e-6, errors
        self.maximum_readback_error = max(self.maximum_readback_error, float(errors.max()))
        values = cpu(torch.cat((duty, command, actual, supply, sim.qpos[:,3:7]), dim=1))
        assert normal.shape==(self.env.num_envs, len(LABELS)) and np.isfinite(values).all()
        assert np.abs(values[:,:14]).max()<=self.cap+1e-6
        self.rows[name].append(dict(substep=n, normal_force_n=normal.astype(np.float32), orientation_before=root_before,
            orientation_after=values[:,43:], voltage_duty=values[:,:14], requested_torque_nm=values[:,14:28],
            applied_torque_nm=values[:,28:42], supply_v=values[:,42]))

    def save(self, folder):
        manifests = []
        for first, last, name in self.ranges:
            rows = self.rows[name]
            if not rows:
                continue
            archive = folder/f'support-effort-{name}.npz'
            np.savez_compressed(archive, **{key:np.stack([row[key] for row in rows]) for key in rows[0]})
            manifests.append(dict(window=name, start_substep=first+1, planned_last_substep=last,
                reached_substep=rows[-1]['substep'], rows=len(rows), file=archive.name, sha256=sha(archive)))
        manifest = folder/'support-effort.json'
        write(manifest, dict(windows=manifests, contact_channels=LABELS, nominal_weight_n=self.clock.weight,
            threshold_n=self.clock.threshold, joint_order=self.names, pwm_limit=self.cap,
            solver_torque_limit_nm=cpu(self.limits).tolist(), torque_constant_nm_per_a=self.kt,
            resistance_ohm=self.resistance, maximum_torque_readback_error_nm=self.maximum_readback_error))
        return dict(path=manifest.name, sha256=sha(manifest), sampled_substeps=sum(len(rows) for rows in self.rows.values()))


def qualify(env, obs, plan, actors, folder):
    receipt = QUALIFY(env, obs, plan, actors, folder)
    initial = ROOT/plan['support_effort']['same_startup']
    original = read(initial)
    assert receipt['composition_boundary']==original['composition_boundary']
    assert sha(initial.with_name('reset.npz'))==original['reset_sha256']
    with np.load(initial.with_name('reset.npz'), allow_pickle=False) as expected, \
         np.load(folder/'reset.npz', allow_pickle=False) as actual:
        assert actual.files==expected.files
        for key in expected.files:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
    with np.load(ROOT/plan['friction_development']['startup_fields'], allow_pickle=False) as saved:
        for field in sorted(RESET_FIELDS):
            np.testing.assert_array_equal(cpu(getattr(env.sim.model, field)), saved['expanded_'+field], err_msg=field)
    before = [cpu(getattr(env.sim.data,key)) for key in ('qpos','qvel','ctrl')]
    rng = torch.get_rng_state(), torch.cuda.get_rng_state(env.device)
    meter = SupportEffort(env, plan['support_effort'])
    points, load = meter.clock.read_contacts()
    assert points.shape==load.shape==(env.num_envs,len(LABELS))
    assert np.isfinite(load).all() and (load>=0.).all()
    assert cpu(env.sim.data.actuator_force).shape==(env.num_envs, 14)
    for key, value in zip(('qpos','qvel','ctrl'), before, strict=True):
        np.testing.assert_array_equal(cpu(getattr(env.sim.data,key)),value,err_msg=key)
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state(env.device))
    assert env.common_step_counter==env._sim_step_counter==0
    write(folder/'support-effort-zero.json',dict(reset_and_rng_preserved=True, prior_low_friction_reset_identical=True,
        physical_steps=0, channels=LABELS, nominal_weight_n=meter.clock.weight, joints=meter.names,
        pwm_limit=meter.cap, torque_cap_nm=cpu(meter.limits).tolist()))
    return receipt


def status(observer, camera, folder, run, count, horizon, worlds, progress, mode):
    mode = '低摩擦起身｜足底与头部载荷、电机输出'
    return RENDER(observer, camera, folder, run, count, horizon, worlds, progress, mode)


def collect(env, plan, actors, folder, observer, watch, ledger):
    meter = SupportEffort(env, plan['support_effort'])
    physical = env.sim.step
    before = signature({role: actor.state_dict() for role,actor in actors.items()})
    try:
        with (patch.object(env.sim,'step',lambda:meter.step(physical)),
              patch.object(meter.inner,'compute_control',meter.motor),
              patch.object(env,'_reset_idx',side_effect=AssertionError('Unexpected episode reset'))):
            COLLECT(env, plan, actors, folder, observer, watch, ledger)
    finally:
        ledger['support_effort'] = meter.save(folder)
    for first, last, name in meter.ranges:
        assert len(meter.rows[name])==last-first
    assert signature({role: actor.state_dict() for role,actor in actors.items()})==before
    ledger['actor_weights_unchanged'] = True
    observer.publish(folder,dict(read(folder/'state.json'),latest_clip=ledger['video']['file']))


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze','zero','sample'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(),
          patch.multiple(parent, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT, config=config,
                         qualify=qualify, collect=collect, render_status=status),
          patch.object(feedback,'components',isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'zero':parent.zero,'sample':parent.sample}[action]()
