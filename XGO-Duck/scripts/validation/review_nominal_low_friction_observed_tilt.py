"""Read back the settled-tilt marker on the original noisy policy inputs."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_low_friction_head_support import OUTPUT as PHYSICAL, summarize
from review_nominal_low_friction_roll_entry import KEY, OUT, PROJECT, ROOT, first_mark, load_case, read, sha
from review_nominal_timestep_friction_mechanisms import standing_states, standing_trace


OUTPUT = OUT.with_name('observed-tilt-readback.json')


def recorded_branch(plan, folder, collection, role, marker):
    rec = collection['branches'][role]
    start = plan['standing_pair']['common_snapshot_control']
    end = plan['composition']['roll_start_control']
    assert (start,end)==(100,600)
    archive = folder/role/'pre-roll-standing.npz'
    assert sha(archive)==rec['standing_sha256']
    frames = np.arange(start,end)
    lo,hi = marker['controls']
    look = (frames>=lo)&(frames<=hi)
    with np.load(archive,allow_pickle=False) as stored:
        samples, mode = stored['actor_input'], stored['mode']
    assert samples.shape[:2]==mode.shape==(end-start,896) and samples.shape[2]>=6
    for control in (start,plan['composition']['getup_to_stand_control']):
        entry, = (row for row in rec['boundaries'] if row['control']==control)
        path = folder/role/entry['file']
        assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as picked:
            original = picked['observed']
            np.testing.assert_array_equal(samples[control-start,:,:6],original[:,:6])
            np.testing.assert_array_equal(mode[control-start],picked['mode'])
    actual = standing_trace(folder/role,rec['chunks'],start+1,end)
    np.testing.assert_array_equal(mode,actual['mode'])
    entered = standing_states(plan,actual)['gate']
    gyro = samples[look,:,:3].astype(np.float64)
    gravity = samples[look,:,3:6].astype(np.float64)
    assert np.isfinite(gyro).all() and np.isfinite(gravity).all()
    assert (np.linalg.norm(gravity,axis=-1)>.5).all()
    pitch = np.rad2deg(np.arctan2(gravity[...,0],np.linalg.norm(gravity[...,1:],axis=-1)))
    pitch_lo,pitch_hi = marker['signed_pitch_degrees']
    qualified = ((pitch>=pitch_lo)&(pitch<=pitch_hi)&
        (np.linalg.norm(gyro,axis=-1)<=marker['maximum_angular_speed_rad_s']))
    observed = first_mark(frames[look],qualified)
    return observed, entered, archive


def main():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert node['status']=='reviewed' and goal['status']=='active' and not goal['pause_requested_by_user']
    assert not any(task[key] for key in ('simulation_running','learning_running','pause_requested','deployment_authorized_now'))
    audit = read(OUT,node['entry_chronology_sha256'])
    physical = read(PHYSICAL,node['head_support_readback_sha256'])
    cases = {row['collection']:load_case(row) for row in (
        goal['nominal_timestep_low_friction_development'],goal['nominal_roll_timestep_repeat'])}
    marker = physical['criterion']
    assert marker['successive_controls']==25
    scores = {}
    for name, origin in audit['sources'].items():
        # The newest shared prefix did not archive full per-step actor inputs;
        # only the earlier low-friction rollout and both nominal arms did.
        role = origin['branch']
        if role is None:
            continue
        plan,folder,collection,_ = cases[origin['collection']]
        assert collection['protocol_sha256'] and sha(ROOT/origin['collection'])==origin['sha256']
        observed, entered, archive = recorded_branch(plan,folder,collection,role,marker)
        assert int((~entered).sum())==physical['positives'][name]['head_and_feet_all']['all_entry_losses']
        left = np.asarray(plan['seven_command']['groups']['turn-left'])
        scores[name] = dict(all=summarize(observed,entered),left=summarize(observed[left],entered[left]),
            observed_npz=archive.relative_to(ROOT).as_posix(),observed_npz_sha256=sha(archive),
            marked_controls={int(i):int(observed[i]) for i in np.flatnonzero(observed>=0)})
    assert len(scores)==3 and not OUTPUT.exists()
    passed = all(row['all']['marked_entry_losses']==row['all']['all_entry_losses'] and
        row['all']['marked_entry_passes']==0 for row in scores.values())
    write(OUTPUT,dict(status='verified-observed-tilt-readback',seed=1257,
        detector=dict(controls=marker['controls'],successive_controls=25,signed_pitch_degrees=marker['signed_pitch_degrees'],
            maximum_angular_speed_observation_norm=marker['maximum_angular_speed_rad_s'],
            uses_archived_actor_projected_gravity_and_gyro_only=True),
        arms=scores,current_shared_prefix_full_observations_available=False,
        all_marker_checks_passed=passed,source_head_support_sha256=sha(PHYSICAL),
        script_sha256=sha(Path(__file__).resolve()),learning_updates=0,new_physical_steps=0,reserved_seed_used=False))
    node.update(observed_tilt_readback=OUTPUT.relative_to(ROOT).as_posix(),observed_tilt_readback_sha256=sha(OUTPUT))
    task.update(next_gate='Preflight a bounded recovery target on the observed settled-tilt state and compare using original gates.'
                if passed else 'Resolve the measured observation coverage gap before choosing a recovery target.')
    project['next_gate'] = task['next_gate']
    project['current_objective'] = task['next_gate']
    write(PROJECT,project)
    print('LOW_FRICTION_OBSERVED_TILT_READBACK',sha(OUTPUT),passed,
          {name:(row['all'],row['left']) for name,row in scores.items()},flush=True)


if __name__=='__main__':
    main()
