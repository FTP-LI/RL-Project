"""Separate genuine matched roll-admission changes from reference variation."""

from pathlib import Path

import numpy as np

from preflight_nominal_supine_unloaded_foot import chunk_at
from roll_stand_walk_sequence import ROLL
from run_nominal_supine_foot_authority import KEY, PROJECT, REPORT, ROLES, ROOT, read, rel, sha, write


def difference(ref, candidate):
    return dict(gains=np.flatnonzero(~ref & candidate).tolist(), losses=np.flatnonzero(ref & ~candidate).tolist())


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status'] == 'reviewed' and not project['staged_training_request']['simulation_running']
    plan = read(ROOT/node['protocol'], node['protocol_sha256'])
    analysis = read(ROOT/node['analysis'], node['analysis_sha256'])
    source = ROOT/node['collection']
    collection = read(source, node['collection_sha256'])
    end = plan['foot_authority']['end_control']
    labels, boundary_sources = [], {}
    for arm in ROLES:
        chunks = collection['branches'][arm]['chunks']
        row, boundary_sources[arm] = chunk_at(source.parent/arm, chunks, end, ('roll_phase',))
        labels.append(row['roll_phase'] == ROLL)
        assert labels[-1].sum() == analysis['roles'][arm]['roll_admitted']
    a, b, c = labels
    arrivals = dict(reference_repeat_difference=difference(a,b), candidate_vs_reference_a=difference(a,c),
                    candidate_vs_reference_b=difference(b,c), concordant_gains=np.flatnonzero(~a & ~b & c).tolist(),
                    concordant_losses=np.flatnonzero(a & b & ~c).tolist())
    forward = set(plan['seven_command']['groups']['forward'])
    readback = analysis['response']
    live = {row['world'] for row in readback if row['target_changed']}
    affected = {k:[world for world in values if world in live] for k,values in arrivals.items() if isinstance(values,list)}
    forward_changes = {k:[world for world in values if world in forward] for k,values in arrivals.items() if isinstance(values,list)}
    observed = np.array([row['before_touch_reads'] for row in readback if row['target_changed']])
    gain = [row['relative_gain_mm'] - row['reference_gap_mm'] for row in readback if row['before_touch_reads']]
    assert len(gain) == analysis['common_unloaded_response_count'] == (observed>0).sum()
    day = REPORT/'matched-admission-and-foot-response.json'
    assert not day.exists()
    write(day, dict(source=rel(ROOT/node['analysis']), source_sha256=node['analysis_sha256'],
                    script_sha256=sha(Path(__file__)), boundary_sources=boundary_sources,
                    all_scenario_admission_differences=arrivals, actuated_scenarios_concordant=affected,
                    forward_scenarios_concordant=forward_changes, loaded_on_first_common_read=int((observed==0).sum()),
                    common_unloaded_reads_at_most_one=int((observed<=1).sum()),
                    before_touch_peak_gain_above_reference_repeat_mm=dict(n=len(gain), median=float(np.median(gain)),
                        min=float(min(gain)), max=float(max(gain))), gpu_memory_peak_mib=max(row['used_mib'] for row in collection['gpu_samples']),
                    simultaneous_scenario_count=plan['composition']['worlds'], new_physics_steps=0, learning_updates=0))
    node.update(effect_readback=rel(day), effect_readback_sha256=sha(day), advance_to_integrated_trial=False)
    project['staged_training_request']['current_evidence'] = project['current_evidence'] = rel(day)
    write(PROJECT, project)
    print('SUPINE_FOOT_AUTHORITY_READBACK', sha(day), arrivals, affected, forward_changes, flush=True)


if __name__ == '__main__':
    main()
