"""Compare the parent and both trained policies from one physical release."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import collect_nominal_pre_roll_stand_pair as recorder
import routed_getup_sequence as feedback
import run_nominal_pre_roll_stand_pair as driver
from review_nominal_skill_composition_handoff import read
from routed_seven_command_score import recovery_and_standing
from run_nominal_roll_closed_loop_timestep import CollectorLedger
from run_nominal_supine_upright_band_candidate import KEY as TRAINING
from verify_nominal_skill_composition import PHYSICAL, STATE_FIELDS, STAND, Sequence, check_video, read_trace
from roll_stand_walk_sequence import ROLL


parent = driver.parent
ROOT, PROJECT = parent.ROOT, parent.PROJECT
rel, sha, write = parent.rel, parent.sha, parent.write
KEY = 'nominal_supine_upright_band_action_comparison'
STAGE = 'nominal-supine-upright-band-action-comparison'
PLAN = ROOT / 'configs/experiments' / (KEY + '.json')
REPORT = parent.REPORT/'early-getup-load/support-progress/action-comparison'
ENTRY = Path(__file__).resolve()
ROLES = ('parent-policy', 'original-reward', 'upright-band-reward')
MODEL_ROLES = dict(zip(ROLES, ('getup-supine','getup-supine-control','getup-supine-candidate'), strict=True))
LABELS = dict(zip(('shared',*ROLES), ('共同跌落与释放','仰卧父策略','原奖励对照','站稳带奖励候选'), strict=True))


def freeze():
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    training = goal[TRAINING]
    assert training['status']=='training-verified-action-comparison-pending' and not task['simulation_running']
    assert parent.allowed(project,task['current_stage']) and KEY not in goal
    receipt = ROOT / training['verification']
    verified = read(receipt, training['verification_sha256'])
    assert verified['status']=='passed-matched-training-artifacts' and not verified['heldout_used']
    baseline = goal['nominal_skill_composition_independent']
    source = ROOT / baseline['protocol']
    plan = read(source, baseline['protocol_sha256'])
    zero = ROOT / baseline['zero']
    original_zero = read(zero, baseline['zero_sha256'])
    reset = zero.with_name('reset.npz')
    assert original_zero['reset_sha256']==sha(reset)
    composition = plan['composition']
    seed, n = plan['environment']['seed'], composition['worlds']
    assert (seed,n) == (1257,896)
    protected = (2256,2246)
    assert seed not in protected
    start, end = (composition[name] for name in ('release_control','roll_start_control'))
    assert (start,end)==(100,600) and plan['dual_route']['release_counter']==start
    assert plan['roll_route']['standing_dwell_controls']==25
    assert plan['roll_route']['roll']['criteria']['tilt_max_deg']==15
    assert len(set(plan['seven_command']['minimum_by_command'].values()))==1

    paths = [ENTRY, source, zero, reset, receipt, ROOT/training['protocol'],
             Path(recovery_and_standing.__code__.co_filename).resolve()]
    home = plan['policy_home']['getup-supine']
    assert home==plan['environment']['model_home']
    for name, role in zip(('control','candidate'), ROLES[1:], strict=True):
        delivery = verified['arms'][name]['onnx']
        model, checkpoint = (ROOT / delivery[field] for field in ('model','checkpoint'))
        assert sha(model)==delivery['model_sha256'] and sha(checkpoint)==delivery['checkpoint_sha256']
        plan['models'][MODEL_ROLES[role]] = dict(model=rel(model), model_sha256=sha(model), seed=seed)
        plan['policy_home'][MODEL_ROLES[role]] = home
        paths.extend((model,checkpoint))

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-upright-band-action-comparison-s{seed}-{stamp}'
    plan['environment']['run_id'] = run
    controls = start+len(ROLES)*(end-start)
    plan.update(stage=STAGE, record=f'records/evaluations/{run}',
                maximum_unique_controls=controls, maximum_physics_substeps=5*controls,
                maximum_control_transitions=controls*n,
                standing_pair=dict(common_snapshot_control=start, archived_initial=rel(zero)),
                upright_band_comparison=dict(roles=list(ROLES), model_by_role=MODEL_ROLES, end_control=end,
                    minimum_forward_gain_against_each_reference=2, require_no_other_pose_entry_losses=True,
                    seed_reserved_for_confirmation=protected, evaluation_stage='release to original pre-roll standing gate'))
    plan['references'].update({rel(path):sha(path) for path in paths})
    modules = (driver, recorder)
    plan['source_sha256'].update({rel(Path(item.__file__).resolve()):sha(Path(item.__file__).resolve()) for item in modules})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN, plan)
    evidence = REPORT/'preflight.json'
    write(evidence, dict(protocol_sha256=sha(PLAN), observer=observer, matched_release=start,
                         maximum_controls=controls, maximum_transitions=controls*n, hardware_io=False))
    goal[KEY] = dict(status='sample-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
                    preflight=rel(evidence), run_id=run, hardware_io=False)
    info = dict(current_protocol_execution='upright-band-action-comparison-frozen', current_evidence=rel(evidence))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Compare the matched parent and both trained networks at the original recovery gates.')
    project.update(info)
    write(PROJECT, project)
    print('UPRIGHT_BAND_ACTION_COMPARISON_FROZEN', sha(PLAN), run, flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    start = plan['standing_pair']['common_snapshot_control']
    end = plan['upright_band_comparison']['end_control']
    short = dict(plan, composition={**plan['composition'],'maximum_controls':end})
    route, names = actors['getup'], iter(ROLES)
    original = route.parent.candidate.forward
    fork_base = recorder.StandingFork
    phase = 'shared'

    def select_policy(observations):
        if phase in ROLES[1:]:
            return actors[MODEL_ROLES[phase]](observations)
        return original(observations)

    class SharedRelease(fork_base):
        def __init__(self, plant, controller, actor, memory):
            # The existing fork serializer requires populated latch tensors.
            # Latch from the unchanged release observation without applying
            # any actor action to the plant or advancing its clock.
            assert plant.common_step_counter==start and actor is route
            assert actor.calls==0 and actor.right is None and actor.parent.supine_at_release is None
            actor(parent.base.shared.td('actor',plant.obs_buf['actor']))
            actor.calls = 0
            super().__init__(plant,controller,actor,memory)
            np.savez_compressed(folder/'release-routing.npz',
                gravity=recorder.cpu(actor.first_gravity), right=recorder.cpu(actor.right),
                latched_supine=recorder.cpu(actor.parent.supine_at_release))
            ledger['release_routing_sha256'] = sha(folder/'release-routing.npz')

        def restore(self):
            nonlocal phase
            state = super().restore()
            phase = next(names)
            return state

    native = observer.publish

    def publish(path, values):
        native(path, dict(values,purpose=f'仰卧起身奖励评估｜{LABELS[values["phase"]]}'))

    counters = CollectorLedger(ledger)
    try:
        with (patch.multiple(recorder,ROLES=ROLES,LABELS=LABELS,CANDIDATE=None,StandingFork=SharedRelease),
              patch.object(route.parent.candidate,'forward',select_policy), patch.object(observer,'publish',publish)):
            recorder.collect(env, short, actors, folder, observer, watch, counters)
    finally:
        ledger.update(counters)


def score():
    project, plan = driver.checked('score-pending')
    node, task = project['goal_execution'][KEY], project['staged_training_request']
    folder = ROOT / plan['record']
    collection = read(folder/'collection.json', node['collection_sha256'])
    assert collection['transitions']==plan['maximum_control_transitions'] and collection['resets']==1
    assert collection['actor_weights_unchanged'] and not collection['learning_updates']
    start = plan['standing_pair']['common_snapshot_control']
    end = plan['upright_band_comparison']['end_control']
    d, p = plan['composition'], plan['environment']
    world_count = d['worlds']
    initial_path = folder/'initial.npz'
    routes = folder/'release-routing.npz'
    assert sha(initial_path)==collection['initial_sha256'] and sha(routes)==collection['release_routing_sha256']
    assert sha(folder/'release-poses.npz')==collection['release_sha256']
    with np.load(initial_path,allow_pickle=False) as data, np.load(routes,allow_pickle=False) as routed, +         np.load(folder/'release-poses.npz',allow_pickle=False) as releases:
        initial = dict(data)
        observed = dict(routed)
        poses = releases['poses']
    selected = observed['latched_supine'] & ~observed['right']
    assert selected.any() and (~selected).any()
    fields = tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'original_feet','roll_phase')))
    common = read_trace(folder/'shared',collection['prefix']['chunks'],1,start,fields,sha)
    fork = read(folder/'fork.json',collection['fork_sha256'])
    forward = np.asarray(plan['seven_command']['groups']['forward'])
    joints = collection['qpos_joint_addresses']
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    results, entries, first = {}, {}, None
    for role in ROLES:
        record = collection['branches'][role]
        path = folder / role
        assert read(path/'restored-fork.json',record['restore_sha256']) == {k:fork[k] for k in ('physical','controller_sha256')}
        mark, = (x for x in record['boundaries'] if x['control']==start)
        assert sha(path/mark['file'])==mark['sha256']
        with np.load(path/mark['file'],allow_pickle=False) as boundary:
            action = dict(boundary)
        np.testing.assert_array_equal(action['actor_input'][:,3:6],observed['gravity'])
        if first is None:
            first = action
        for key in ('observed','actor_input','mode','previous_target'):
            np.testing.assert_array_equal(action[key],first[key],err_msg=f'{role}/{key}')
        np.testing.assert_array_equal(action['raw'][~selected],first['raw'][~selected])
        session = parent.onnx_session(ROOT/plan['models'][MODEL_ROLES[role]]['model'])
        expected = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in action['actor_input'][selected]])
        np.testing.assert_allclose(action['raw'][selected],expected,atol=2e-5,rtol=2e-5)
        if role != ROLES[0]:
            assert np.max(abs(action['raw'][selected]-first['raw'][selected]))>1e-5
        leaf = read_trace(path,record['chunks'],start+1,end,fields,sha)
        trace = {k:np.concatenate((common[k],leaf[k])) for k in fields}
        q = trace['qpos'][...,joints]
        envelope = ((q>=limits[:,0]-tolerance)&(q<=limits[:,1]+tolerance)).all(axis=-1)
        physical = dict(trace,foot_contact=trace['original_feet'],joint_positions=q,joint_envelope_ok=envelope)
        getup, standing, original_score = recovery_and_standing(plan,folder,trace,initial,physical)
        sequence = Sequence(plan['roll_route'])
        for control in range(d['getup_to_stand_control'],end+1):
            frame = {key:trace[key][control-1] for key in STATE_FIELDS}
            if control==end:
                frame['active'] = frame['active'] & (trace['mode'][control-1]==STAND)
            sequence.update(control-d['getup_to_stand_control'],frame)
            np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][control-1])
        entries[role] = sequence.phase==ROLL
        results[role] = dict(getup=original_score, timely_and_terminal_getup=int(getup.sum()),
            pre_roll_hold=int(standing.sum()), roll_entry=int(entries[role].sum()),
            forward_group_roll_entry=int(entries[role][forward].sum()),
            actual_release_roll_entry={pose:dict(passed=int(entries[role][poses==pose].sum()),total=int((poses==pose).sum()))
                for pose in sorted(set(poses))},
            maximum_release_onnx_error=float(abs(action['raw'][selected]-expected).max()),
            video=check_video(path,dict(record,camera_command=collection['camera_command']),trace,initial))
    candidate = entries[ROLES[-1]]
    comparison = {}
    for role in ROLES[:2]:
        gained, lost = candidate & ~entries[role], entries[role] & ~candidate
        comparison[role] = dict(gained_worlds=np.flatnonzero(gained).tolist(),lost_worlds=np.flatnonzero(lost).tolist(),
            net_roll_entry=int(gained.sum()-lost.sum()),net_forward_roll_entry=int(gained[forward].sum()-lost[forward].sum()),
            other_route_losses=int(lost[~selected].sum()))
    threshold = plan['upright_band_comparison']['minimum_forward_gain_against_each_reference']
    advances = results[ROLES[-1]]['getup']['functional_coverage_gate'] and all(
        item['net_forward_roll_entry']>=threshold and item['net_roll_entry']>=0 and item['other_route_losses']==0
        for item in comparison.values())
    result = REPORT/'analysis.json'
    assert not result.exists()
    write(result, dict(status='reviewed',seed=p['seed'],worlds_per_arm=world_count,
        protocol_sha256=sha(PLAN),collection_sha256=node['collection_sha256'],
        release_fork_control=start, original_roll_entry_control=end,
        batched_physical_substeps=collection['physics_substeps'],transitions=collection['transitions'],
        observed_supine_policy_worlds=int(selected.sum()),field_inventory_count=len(fork['fields']),
        roles=results, paired_comparison=comparison, minimum_forward_gain=threshold,
        advance_to_integrated_trial=bool(advances),learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed', analysis=rel(result), analysis_sha256=sha(result), advance_to_integrated_trial=bool(advances))
    info = dict(current_protocol_execution='upright-band-action-comparison-reviewed',current_evidence=rel(result))
    project.update(info)
    task.update(info)
    write(PROJECT,project)
    print('UPRIGHT_BAND_ACTION_COMPARISON_SCORED',sha(result),{key:value['roll_entry'] for key,value in results.items()},advances,flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','sample','score'))
    action = parser.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.object(parent,'PLAN',PLAN), patch.object(feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__=='__main__':
    main()
