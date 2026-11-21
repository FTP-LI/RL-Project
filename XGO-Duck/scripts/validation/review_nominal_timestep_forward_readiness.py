"""Read which previously blocked forward scenes have new pre-roll evidence."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_stand_walk_sequence import ROLL


PROJECT = ROOT/'project.json'
COMPARISON = 'nominal_supine_upright_band_action_comparison'
PRIOR = 'nominal_roll_timestep_closed_loop'
KEY = 'nominal_timestep_forward_readiness'
STAGE = 'nominal-timestep-forward-readiness'
OUT = ROOT/'records/reports/nominal-skill-composition/early-getup-load/support-progress/action-comparison/timestep-forward-readiness.json'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def record_for(node):
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    collection = read(ROOT/node['collection'],node['collection_sha256'])
    return plan, collection, ROOT/plan['record']


def main():
    assert not OUT.exists()
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert goal['status']=='active' and not goal['pause_requested_by_user'] and not task['pause_requested']
    assert not any(task[key] for key in ('simulation_running','learning_running','deployment_authorized_now'))
    assert KEY not in goal
    latest, older = (goal[key] for key in (COMPARISON,PRIOR))
    assert latest['status']==older['status']=='reviewed'
    goal[KEY] = dict(status='saved-data-check',hardware_io=False)
    task.update(current_stage=STAGE,
                next_gate='Match initial scene IDs and check the archived five forward pre-roll blocks.')
    write(PROJECT,project)

    recent, recent_data, recent_folder = record_for(latest)
    old, old_data, old_folder = record_for(older)
    assert recent['environment']['seed']==old['environment']['seed']==1257
    assert recent['seven_command']['groups']==old['seven_command']['groups']
    assert recent['models']['getup-supine']==old['models']['getup-supine']
    end = recent['upright_band_comparison']['end_control']
    assert end==recent['composition']['roll_start_control']==old['composition']['roll_start_control']==600
    before_path = ROOT/older['forward_pre_roll_admission']
    before = read(before_path,older['forward_pre_roll_admission_sha256'])
    prior_gate = read(ROOT/older['support_and_throughput'],older['support_and_throughput_sha256'])
    forward = prior_gate['forward_admission']['solver-2ms-during-roll']
    cases = [case['world'] for case in before['evaluated_forward_assignments']]
    assert cases==[case['world'] for case in forward['blocked_scenes'] if case['stage']=='no-roll-entry']
    assert set(cases)<=set(recent['seven_command']['groups']['forward'])
    other = [row['world'] for row in forward['blocked_scenes'] if row['stage']!='no-roll-entry']
    assert len(cases)==5 and len(other)==3 and forward['admitted']==120 and forward['minimum']==122
    for data, folder in ((recent_data,recent_folder),(old_data,old_folder)):
        assert sha(folder/'initial.npz')==data['initial_sha256']
        assert sha(folder/'release-poses.npz')==data['release_sha256']
    with (
        np.load(recent_folder/'initial.npz',allow_pickle=False) as new,
        np.load(old_folder/'initial.npz',allow_pickle=False) as previous,
        np.load(recent_folder/'release-poses.npz',allow_pickle=False) as released,
        np.load(old_folder/'release-poses.npz',allow_pickle=False) as old_released,
    ):
        exact = bool(np.array_equal(new['qpos'],previous['qpos']))
        np.testing.assert_array_equal(new['qpos'][cases],previous['qpos'][cases])
        np.testing.assert_array_equal(released['poses'][cases],old_released['poses'][cases])
        assert (released['poses'][cases]=='supine').all()
    scored = read(ROOT/latest['analysis'],latest['analysis_sha256'])
    comparisons = {}
    for role, branch in recent_data['branches'].items():
        entry = branch['chunks'][-1]
        path = recent_folder/role/entry['file']
        assert entry['last']==end and sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as samples:
            assert int(samples['control'][-1])==end
            admitted = samples['roll_phase'][-1]==ROLL
        assert int(admitted.sum())==scored['roles'][role]['roll_entry']
        assert int(admitted[recent['seven_command']['groups']['forward']].sum())==scored['roles'][role]['forward_group_roll_entry']
        comparisons[role] = dict(admitted_at_original_roll_entry=[x for x in cases if admitted[x]],
                                 stayed_blocked=[x for x in cases if not admitted[x]])
    result = dict(seed=1257,roll_entry_control=end,all_initial_qpos_identical=exact,
        five_blocked_cases_initial_qpos_and_release_classes_matched=True, prior_forward_entrants=forward['admitted'],
        prior_forward_minimum=forward['minimum'],prior_later_roll_failures=other,earlier_blocked_worlds=cases,
        new_pre_roll_evidence=comparisons, scope='Separate archived physical rollouts, paired by initial scene IDs.',
        source_sha256={rel(path):sha(path) for path in (ROOT/latest['analysis'],ROOT/latest['collection'],
            before_path,ROOT/older['support_and_throughput'],Path(__file__).resolve())},
        new_physical_steps=0,new_learning_updates=0, heldout_used=False,
        action_completion_counters_from_prior_rollout_only=True)
    write(OUT,result)
    goal[KEY].update(status='reviewed',evidence=rel(OUT),evidence_sha256=sha(OUT))
    info = dict(current_protocol_execution='nominal-timestep-forward-readiness-reviewed',current_evidence=rel(OUT))
    project.update(info)
    task.update(info,next_gate='Use paired fixed-policy repeats to estimate natural variation in the nominal 2 ms route before new learning.')
    write(PROJECT,project)
    print('NOMINAL_TIMESTEP_FORWARD_READINESS',sha(OUT),comparisons,flush=True)


if __name__=='__main__':
    main()
