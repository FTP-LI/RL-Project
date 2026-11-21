"""Attribute command entry shortfalls using only the recorded scene histories."""

from collections import Counter
from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


PROJECT = ROOT/'project.json'
LOW = 'nominal_timestep_low_friction_development'
REPEAT = 'nominal_roll_timestep_repeat'
KEY = LOW+'_entry_review'


def trial(node):
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    raw = read(ROOT/node['collection'],node['collection_sha256'])
    score = read(ROOT/node['result'],node['result_sha256'])
    assert score['collection_sha256']==node['collection_sha256']
    return plan,raw,score,ROOT/plan['record']


def main():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert goal['status']=='active' and not goal['pause_requested_by_user'] and not task['pause_requested']
    assert not any(task[k] for k in ('simulation_running','learning_running','deployment_authorized_now'))
    current, reference = (goal[name] for name in (LOW,REPEAT))
    assert current['status']==reference['status']=='reviewed' and not current['qualified']
    assert KEY not in goal
    plan,raw,score,folder = trial(current)
    old, old_raw, original, old_folder = trial(reference)
    assert plan['environment']['seed']==old['environment']['seed']==score['seed']==1257
    groups = plan['seven_command']['groups']
    assert groups==old['seven_command']['groups'] and plan['models']==old['models']
    assert all(item['passed_development_gate'] for item in original['results'].values())
    entry = plan['composition']['command_entry_control']
    assert raw['controls']==entry and score['result']['complete'] is None
    report = ROOT/current['result']
    out = report.with_name('command-entry-readback.json')
    assert not out.exists()
    goal[KEY] = dict(status='saved-trace-review',hardware_io=False)
    task.update(current_stage='nominal-timestep-low-friction-entry-review',
                next_gate='Read the exact upstream loss reasons for forward and left turn.')
    write(PROJECT,project)

    for path,record in ((folder,raw),(old_folder,old_raw)):
        assert sha(path/'initial.npz')==record['initial_sha256']
        assert sha(path/'release-poses.npz')==record['release_sha256']
    with (
        np.load(folder/'initial.npz',allow_pickle=False) as low,
        np.load(old_folder/'initial.npz',allow_pickle=False) as base,
        np.load(folder/'release-poses.npz',allow_pickle=False) as pose,
    ):
        np.testing.assert_array_equal(low['qpos'],base['qpos'])
        np.testing.assert_array_equal(low['requested_pose'],base['requested_pose'])
        poses = pose['poses']

    role, = raw['branches']
    branch = raw['branches'][role]
    chunks = [dict(item,file=f'{label}/{item["file"]}')
              for label, data in (('shared',raw['prefix']),(role,branch)) for item in data['chunks']]
    _, entered, passed, reason, peak, _ = replay(plan,folder,dict(chunks=chunks))
    assert int(passed.sum())==score['result']['roll_complete']
    assert int(entered.sum())==score['roll_entry']
    reached = set(branch['command_entry_worlds'])
    reference_sets = [set(row['command_entry_worlds']) for row in old_raw['branches'].values()]
    both_nominal = set.intersection(*reference_sets)
    final = folder/role/branch['chunks'][-1]['file']
    assert sha(final)==branch['chunks'][-1]['sha256']
    with np.load(final,allow_pickle=False) as data:
        assert int(data['control'][-1])==entry
        roll_ready, stand = data['roll_ready'][-1], data['base_stand_reads'][-1]
    commands = {}
    for command, group in groups.items():
        admitted = len(reached.intersection(group))
        required = plan['seven_command']['minimum_by_command'][command]
        assert admitted==branch['command_admission'][command]
        rows = []
        for scene in group:
            if scene in reached:
                continue
            if passed[scene]:
                cause = 'base-stand-dwell' if roll_ready[scene] else 'post-roll-stand-handoff'
                assert not roll_ready[scene] or stand[scene]<plan['roll_route']['standing_dwell_controls']
            else:
                cause = str(reason[scene])
            rows.append(dict(world=scene,actual_release_pose=str(poses[scene]),stage=cause,
                peak_supported_rotation_deg=float(peak[1,scene]),admitted_in_both_nominal_repeats=scene in both_nominal))
        commands[command] = dict(assigned=len(group),admitted=admitted,required=required,shortfall=max(required-admitted,0),
            blocked_by_stage=dict(Counter(item['stage'] for item in rows)),blocked_in_low_only=[
                item['world'] for item in rows if item['admitted_in_both_nominal_repeats']],scenes=rows)
    assert {name for name,row in commands.items() if row['shortfall']}=={'forward','turn-left'}
    assert commands['forward']['blocked_by_stage']==score['forward_entry_reasons']['blocked_by_stage']
    write(out,dict(status='scored-from-original-traces',seed=1257,worlds=896,entry_control=entry,
        source_collection_sha256=current['collection_sha256'],source_score_sha256=current['result_sha256'],
        same_initial_scene_bank=True,reference_collection_sha256=reference['collection_sha256'],
        script_sha256=sha(Path(__file__).resolve()),commands=commands,physical_substeps=0,learning_updates=0,
        reserved_seed_used=False,hardware_io=False))
    goal[KEY].update(status='reviewed',evidence=out.relative_to(ROOT).as_posix(),evidence_sha256=sha(out))
    info = dict(current_protocol_execution='nominal-timestep-low-friction-entry-reviewed',current_evidence=out.relative_to(ROOT).as_posix())
    project.update(info)
    task.update(info,next_gate='Diagnose support and contact transitions on the fixed forward and left-turn low-friction cases.')
    write(PROJECT,project)
    print('NOMINAL_FRICTION_COMMAND_ENTRY_READBACK',sha(out),{x:commands[x]['blocked_by_stage'] for x in ('forward','turn-left')},flush=True)


if __name__=='__main__':
    main()
