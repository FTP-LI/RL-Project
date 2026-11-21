"""Locate the original standing predicates preventing roll entry."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_getup_sequence import STAND
from verify_nominal_skill_composition import read_trace


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_roll_timestep_closed_loop']
    collection = ROOT / state['collection']
    ledger = read(collection, state['collection_sha256'])
    plan = read(ROOT / state['protocol'], state['protocol_sha256'])
    readback = ROOT / state['support_and_throughput']
    result = read(readback, state['support_and_throughput_sha256'])
    assert result['collection_sha256'] == state['collection_sha256']
    candidate = plan['solver_comparison']['roles'][-1]
    missing = [row for row in result['forward_admission'][candidate]['blocked_scenes'] if row['stage'] == 'no-roll-entry']
    worlds = [row['world'] for row in missing]
    assert len(worlds) == 5 and all(row['actual_release_pose'] == 'supine' for row in missing)
    end = plan['composition']['roll_start_control']
    duration = plan['roll_route']['standing_dwell_controls']
    sequence = Sequence(plan['roll_route'])
    fields = ('mode', 'roll_phase', *STATE_FIELDS)
    trace = read_trace(collection.parent / 'shared', ledger['prefix']['chunks'], 1, end, fields, sha)
    values = {key: trace[key][-duration:] for key in fields}
    up_z = rotation(values['quaternion'])[..., 2, 2]
    tilt = np.rad2deg(np.arccos(np.clip(up_z, -1., 1.)))
    speed = np.linalg.norm(values['angular_velocity'], axis=-1)
    height_error = np.abs(values['height'] - sequence.criteria['nominal_height_m'])
    c = sequence.criteria
    conditions = dict(active=values['active'], feet=values['foot_contact'].all(axis=-1),
        joints=values['joint_envelope_ok'], tilt=up_z >= np.cos(np.deg2rad(c['tilt_max_deg'])),
        height=height_error <= c['height_tolerance_m'], speed=speed <= c['angular_speed_max_rad_s'])
    stable = np.logical_and.reduce(tuple(conditions.values()))
    original = np.stack([sequence.stable({k: values[k][i] for k in STATE_FIELDS}) for i in range(duration)])
    np.testing.assert_array_equal(stable, original)
    entry = (values['mode'][-1] == STAND) & stable.all(axis=0)
    np.testing.assert_array_equal(entry, values['roll_phase'][-1] == ROLL)
    assert not entry[worlds].any()
    rows = []
    for world in worlds:
        failures = np.flatnonzero(~stable[:, world])
        dwell = duration if not len(failures) else duration - 1 - int(failures[-1])
        rows.append(dict(world=world, standing_actor_at_roll_entry=bool(values['mode'][-1, world] == STAND),
            consecutive_stable_reads_at_entry=dwell,
            failed_reads_by_predicate={k:int((~v[:, world]).sum()) for k,v in conditions.items()},
            final_control_failed_predicates=[k for k,v in conditions.items() if not v[-1,world]],
            final_control_tilt_deg=float(tilt[-1,world]), final_control_height_m=float(values['height'][-1,world]),
            final_control_angular_speed_rad_s=float(speed[-1,world])))
    output = readback.with_name('forward-pre-roll-admission.json')
    assert not output.exists()
    write(output, dict(protocol_sha256=state['protocol_sha256'], collection_sha256=state['collection_sha256'],
        support_readback_sha256=state['support_and_throughput_sha256'], script_sha256=sha(Path(__file__).resolve()),
        seed=plan['environment']['seed'], gate_control=end, last_standing_read_window=[end-duration+1,end],
        original_standing_dwell_reads=duration, total_roll_entry=int(entry.sum()), original_entry_matches_all_worlds=True,
        minimum_next_forward_gain=2, evaluated_forward_assignments=rows, new_physical_steps=0, new_learning_updates=0))
    print('ROLL_ENTRY_ADMISSION_READBACK', sha(output), [(row['world'], row['consecutive_stable_reads_at_entry'],
        row['final_control_failed_predicates']) for row in rows], flush=True)


if __name__ == '__main__':
    main()
