"""Recheck the completed training pair and score it on matched low-friction releases."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import preflight_nominal_timestep_low_friction as friction
import run_nominal_supine_upright_band_comparison as trained
import score_nominal_supine_upright_band_comparison as saved_score
import verify_nominal_supine_upright_band_training as verify


ROOT, PROJECT = trained.ROOT, trained.PROJECT
read, rel, sha, write = trained.read, trained.rel, trained.sha, trained.write
driver, parent, roles = trained.driver, trained.parent, trained.ROLES
KEY = 'nominal_supine_reward_friction_trial'
STAGE = 'nominal-supine-reward-friction-trial'
LOW = 'nominal_timestep_low_friction_development'
MECHANISMS = LOW+'_mechanism_review'
PREVIOUS = 'nominal-timestep-low-friction-mechanism-review'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = trained.REPORT.with_name('low-friction-action-comparison')
PAIR_RECEIPT = REPORT/'matched-training-readback.json'


def readback_score(project, plan, output):
    # Run the frozen scorer against the original inputs. Capture its two
    # writes so this audit cannot replace the archived score or project state.
    staged = deepcopy(project)
    node = staged['goal_execution'][saved_score.KEY]
    recorded = {}

    def record(path, value):
        assert path in (output, PROJECT) and path not in recorded
        recorded[path] = value

    # The scorer hashes its result after visiting its writer. Supply that
    # digest from the serialized value without touching the original output.
    def digest(path):
        if path==output:
            return node.get('analysis_sha256','captured-during-readback')
        return sha(path)

    with (patch.object(driver,'checked',return_value=(staged,plan)),
          patch.multiple(saved_score,REPORT=output.parent,write=record,sha=digest)):
        saved_score.score()
    assert set(recorded)=={output,PROJECT}
    return recorded[output]


def verify_finished_pair(project):
    goal = project['goal_execution']
    learning = goal[verify.KEY]
    training_path = ROOT/learning['verification']
    archived = read(training_path,learning['verification_sha256'])
    plan = read(verify.PLAN,learning['protocol_sha256'])
    verify.sources_checked(plan)
    parent_checkpoint = verify.checkpoint(ROOT/plan['models']['checkpoint'])
    torch.set_num_threads(4)
    arms, rows = {}, {}
    for role in verify.ROLES:
        arms[role], rows[role] = verify.check_arm(role,plan,learning,parent_checkpoint)
    for a,b in zip(*(rows[role] for role in verify.ROLES),strict=True):
        for field in ('anchor_batch_sha256','retention_batch_sha256'):
            assert a[field]==b[field]
    release = verify.compare_training_release([ROOT/arms[role]['record'] for role in verify.ROLES])
    assert arms==archived['arms'] and release==archived['actual_training_release_difference']
    old = goal[trained.KEY]
    old_score = read(ROOT/old['analysis'],old['analysis_sha256'])
    old_plan = read(ROOT/old['protocol'],old['protocol_sha256'])
    parent.base.shared.assert_sources(old_plan)
    assert readback_score(project,old_plan,REPORT/'analysis.json')==old_score
    assert not torch.cuda.is_initialized()
    write(PAIR_RECEIPT,dict(status='verified-saved-training-and-nominal-action-pair',
        training=rel(training_path),training_sha256=sha(training_path),arms=arms,
        actual_training_release_difference=release,nominal_score=old['analysis'],
        nominal_score_sha256=old['analysis_sha256'],nominal_roles=old_score['roles'],
        nominal_advance=old_score['advance_to_integrated_trial'],learning_updates_in_readback=0,
        added_physical_steps=0,hardware_io=False))
    print('TRAINED_PAIR_RECHECKED',sha(PAIR_RECEIPT),flush=True)


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert KEY not in goal and parent.allowed(project,PREVIOUS) and not task['simulation_running']
    assert not PLAN.exists() and not (REPORT/'preflight.json').exists()
    assert not torch.cuda.is_initialized()
    REPORT.mkdir(exist_ok=True)
    if not PAIR_RECEIPT.exists():
        verify_finished_pair(project)
    prior = goal[trained.KEY]
    nominal_plan = ROOT/prior['protocol']
    plan = read(nominal_plan,prior['protocol_sha256'])
    prior_low = goal[LOW]
    lower_path = ROOT/prior_low['protocol']
    lower = read(lower_path,prior_low['protocol_sha256'])
    mechanisms = ROOT/goal[MECHANISMS]['evidence']
    audit = read(mechanisms,goal[MECHANISMS]['evidence_sha256'])
    assert audit['standing_original_entry']==882 and audit['original_roll_completed']==859
    d, original = plan['composition'], lower['composition']
    seed, worlds = plan['environment']['seed'], d['worlds']
    assert (seed,worlds,d['foot_friction'],original['foot_friction'])==(1257,896,1.,.8)
    assert plan['environment']['split']==lower['environment']['split']
    assert plan['seven_command']==lower['seven_command']
    assert all(d[name]==original[name] for name in ('control_period_s','physics_period_s','voltage_v',
        'release_control','getup_to_stand_control','roll_start_control','existing_low_friction_reserved_seed'))
    assert plan['roll_route']==lower['roll_route']
    assert all(plan['models'][name]==model for name,model in lower['models'].items())
    zero = ROOT/lower['standing_pair']['archived_initial']
    fields = ROOT/lower['friction_development']['startup_fields']
    reset = zero.with_name('reset.npz')
    assert sha(reset)==read(zero)['reset_sha256']
    freeze_files = (Path(__file__).resolve(),Path(trained.__file__).resolve(),Path(verify.__file__).resolve(),
        Path(saved_score.__file__).resolve(),Path(friction.__file__).resolve(),nominal_plan,lower_path,
        mechanisms,PAIR_RECEIPT,zero,fields,reset)
    plan['references'].update({rel(path):sha(path) for path in freeze_files})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    d['foot_friction'] = .8
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-upright-reward-friction080-s{seed}-{stamp}'
    plan['environment']['run_id'] = run
    plan['standing_pair']['archived_initial'] = rel(zero)
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',low_friction_trial=dict(
        startup_fields=rel(fields),standing_loss_worlds=[row['world'] for row in audit['standing']],
        nominal_readback=rel(PAIR_RECEIPT),reserved_seed=2246,stop_control=600,
        minimum_forward_gain=2,require_left_turn_nondecrease=True,simulation_only=True))
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(status='frozen',protocol_sha256=sha(PLAN),run_id=run,observer=observer,
        arms=list(roles),worlds=worlds,physics_steps=0,updates=0,original_release=100,
        stop_control=600,maximum_transitions=plan['maximum_control_transitions']))
    goal[KEY] = dict(status='sample-ready',run_id=run,preflight=rel(receipt),protocol=rel(PLAN),
        protocol_sha256=sha(PLAN),hardware_io=False)
    goal['latest_candidate_resume'].update(status='trained-pair-verified-low-friction-comparison-frozen',
        readback=rel(PAIR_RECEIPT),new_protocol=rel(PLAN))
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Compare only the supine policy after the shared release at friction 0.8 and verify the original standing gate.')
    write(PROJECT,project)
    print('SUPINE_REWARD_FRICTION_FROZEN',sha(PLAN),run,flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    assert env.common_step_counter==env._sim_step_counter==0
    path = ROOT/plan['low_friction_trial']['startup_fields']
    with np.load(path,allow_pickle=False) as expected:
        for name in sorted(friction.RESET_FIELDS):
            actual = parent.cpu(getattr(env.sim.model,name))
            np.testing.assert_array_equal(actual,expected['expanded_'+name],err_msg=name)
    ledger['verified_low_friction_startup_sha256'] = sha(path)
    trained.collect(env,plan,actors,folder,observer,watch,ledger)


def score():
    project,plan = driver.checked('score-pending')
    node = project['goal_execution'][KEY]
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['verified_low_friction_startup_sha256']==sha(ROOT/plan['low_friction_trial']['startup_fields'])
    output = REPORT/'analysis.json'
    with patch.object(saved_score,'KEY',KEY), patch.object(saved_score,'PLAN',PLAN):
        result = readback_score(project,plan,output)
    masks = {}
    for role in roles:
        part = raw['branches'][role]['chunks'][-1]
        assert part['last']==plan['low_friction_trial']['stop_control']
        path = folder/role/part['file']
        assert sha(path)==part['sha256']
        with np.load(path,allow_pickle=False) as data:
            masks[role] = data['roll_phase'][-1]==trained.ROLL
        assert int(masks[role].sum())==result['roles'][role]['roll_entry']
    groups = plan['seven_command']['groups']
    per_group = {role:{name:int(mask[worlds].sum()) for name,worlds in groups.items()} for role,mask in masks.items()}
    candidate = per_group[roles[-1]]
    only_recovery = plan['low_friction_trial']['standing_loss_worlds']
    necessary = (sum(candidate.values())>=plan['seven_command']['minimum_overall'] and
        all(candidate[name]>=minimum for name,minimum in plan['seven_command']['minimum_by_command'].items()))
    left_retained = all(candidate['turn-left']>=per_group[role]['turn-left'] for role in roles[:-1])
    eligible = necessary and left_retained and result['advance_to_integrated_trial']
    result.update(foot_sliding_friction=.8,pre_roll_passes_by_future_command=per_group,
        recovered_previously_tilt_limited_worlds={role:[world for world in only_recovery if mask[world]]
            for role,mask in masks.items()},necessary_command_pools_at_pre_roll=necessary,
        left_turn_pre_roll_counts_retained=left_retained,advance_to_integrated_trial=eligible,
        original_nominal_selection_rule_met=bool(necessary and left_retained and
            all(row['net_forward_roll_entry']>=plan['low_friction_trial']['minimum_forward_gain']
                for row in result['paired_comparison'].values())),
        peak_gpu_mib=max(row['used_mib'] for row in raw['gpu_samples']))
    write(output,result)
    details = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output),eligible_for_full_development=eligible)
    project['goal_execution']['latest_candidate_resume'].update(status='trained-and-paired-analysis-verified',
        result=rel(output),eligible_for_full_development=eligible)
    project['staged_training_request'].update(details,next_gate=
        'Verify the complete low-friction route before using the untouched seed.' if eligible else
        'Keep the existing delivery weights and use the recorded 20 ms support losses and tilted getup traces to select a distinct next comparison.')
    project.update(details)
    write(PROJECT,project)
    print('SUPINE_REWARD_FRICTION_SCORED',sha(output),per_group,eligible,flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','sample','score'))
    action = parser.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=roles,collect=collect),
          patch.multiple(parent,config=friction.configured,PLAN=PLAN),
          patch.object(trained.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__=='__main__':
    main()
