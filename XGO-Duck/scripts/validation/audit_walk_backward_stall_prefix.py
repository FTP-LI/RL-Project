"""Audit the yield of capturing real backward stalls after the first two seconds."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from audit_walk_backward_initial_cases import FIELDS, archive_for, checked_array
from record_io import write
from verify_walk_backward_onset import score


PLAN = ROOT/'configs/experiments/common_walk_backward_stall_prefix_audit.json'
OUT = ROOT/'records/reports/common-walk-backward-stall-prefix/analysis.json'
PAIR_PLAN = ROOT/'configs/experiments/common_walk_backward_start_enrichment_recovery.json'


def source(plan, proof_path, digest):
    assert sha(proof_path)==digest
    proof = read(proof_path)
    assert proof['status']=='passed-integrity' and not proof['heldout_used'] and not proof['learning_updates']
    record = archive_for(proof)
    protocol = record/'protocol.json'
    assert sha(protocol)==proof['protocol_sha256']
    spec = read(protocol)
    p = spec['environment']
    paired = 'scores' in proof
    role = plan['reference_role']
    trace_path = record/f'trace-{role}.npz'
    digest = proof['trace_sha256'][role] if paired else proof['source_sha256']
    trace = checked_array(trace_path,digest,FIELDS)
    if paired:
        prefix = checked_array(record/'trace-shared-prefix.npz',proof['shared_prefix_sha256'],FIELDS)
        trace = {key:np.concatenate((prefix[key],trace[key])) for key in FIELDS}
    assert spec['block_seconds']==plan['prefix_block_seconds']
    limit = plan['maximum_absolute_prefix_block_speed_m_s']
    assert spec['near_static_max_absolute_block_speed_m_s']==limit
    block = p['criteria']['block_steps']
    assert 2*block==plan['prefix_controls']
    recalculated = score(trace,p,spec)
    expected = proof['scores'][role] if paired else proof['analysis']
    assert recalculated['categories']==expected['categories']
    assert recalculated['near_static_worlds']==expected['near_static_worlds']
    assert recalculated['worlds']==expected['worlds']
    stopped,selected = [],[]
    for row in recalculated['worlds']:
        first = row['signed_speed_by_second'][:2]
        if any(value is None for value in first) or max(abs(value) for value in first)>limit:
            continue
        world = row['world']
        selected.append(world)
        if not row['survived']:
            outcome = 'ended_early'
        elif row['near_static']:
            outcome = 'near_static_at_six_seconds'
        elif row['onset_block_end_seconds']:
            assert row['onset_block_end_seconds']>2
            outcome = 'later_started'
        else:
            outcome = 'moved_without_qualified_start'
        stopped.append(dict(world=world,outcome=outcome,prefix_speed_m_s=first))
    categories = {key:sum(row['outcome']==key for row in stopped)
                  for key in ('ended_early','near_static_at_six_seconds','later_started','moved_without_qualified_start')}
    assert categories['near_static_at_six_seconds']==recalculated['categories']['near_static']
    return dict(seed=proof['seed'],record=record.relative_to(ROOT).as_posix(),all_worlds=len(recalculated['worlds']),
                selected_worlds=selected,outcomes=categories,rows=stopped)


def summarize(rows):
    total = sum(row['all_worlds'] for row in rows)
    chosen = sum(len(row['selected_worlds']) for row in rows)
    outcomes = {key:sum(row['outcomes'][key] for row in rows) for key in rows[0]['outcomes']}
    stalls = outcomes['near_static_at_six_seconds']
    return dict(total=total,selected=chosen,source_seeds_with_selection=sum(bool(row['selected_worlds']) for row in rows),
                outcomes=outcomes,selected_fraction=chosen/total,
                stall_fraction=stalls/chosen if chosen else 0.,all_world_stall_fraction=stalls/total,
                prevalence_lift=total/chosen if chosen and stalls else 0.)


def main():
    assert not OUT.exists()
    plan = read(PLAN)
    assert plan['status']=='frozen-saved-trajectory-capture-screen'
    assert not any(plan[key] for key in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    active = read(ROOT/'project.json')['staged_training_request']
    assert active['current_stage']==plan['stage'] and ROOT/active['current_protocol']==PLAN
    archive,digest = plan['archive_index']
    decision,pair_digest = plan['new_pair']
    assert sha(ROOT/archive)==digest and sha(ROOT/decision)==pair_digest
    index = read(ROOT/archive)
    pair = read(ROOT/decision)
    assert not index['heldout_used'] and not pair['heldout_used'] and not pair['replication_admitted']
    assert pair['protocol_sha256']==sha(PAIR_PLAN)
    earlier = [source(plan,ROOT/row['source'],row['source_sha256']) for row in index['cohorts']]
    assert pair['seed'] not in {row['seed'] for row in earlier}
    newest_path = ROOT/decision
    report = newest_path.parent/'control'/f"seed{pair['seed']}-analysis.json"
    latest = source(plan,report,pair['score_sha256']['control'])
    assert latest['record']==pair['records']['control']
    control = ROOT/latest['record']
    state_path = control/'two-second-state.npz'
    state = read(control/'two-second-state.json')
    assert state['control']==plan['prefix_controls'] and sha(state_path)==state['sha256'] and state['raw_history']
    with np.load(state_path) as stored,np.load(control/'trace-backward.npz') as trace:
        k = state['control']-1
        chosen = latest['selected_worlds']
        assert trace['active'][k,chosen].all()
        np.testing.assert_array_equal(stored['qpos'][chosen],trace['qpos'][k,chosen])
        np.testing.assert_array_equal(stored['_action'][chosen],trace['actions'][k,chosen])
        assert np.isfinite(stored['qvel'][chosen]).all()
    historical,prospective = summarize(earlier),summarize([latest])
    gate = dict(development_count=historical['selected']>=plan['development_minimum_selected_worlds'],
                development_seeds=historical['source_seeds_with_selection']>=plan['development_minimum_contributing_seeds'],
                development_lift=historical['prevalence_lift']>=plan['development_minimum_stall_prevalence_lift'],
                prospective_stalls=prospective['outcomes']['near_static_at_six_seconds']>=plan['prospective_minimum_confirmed_stalls'])
    OUT.parent.mkdir()
    write(OUT,dict(status='passed-stall-prefix-integrity',protocol_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        historical=historical,prospective=prospective,historical_rows=earlier,prospective_rows=latest,gate=gate,
        prospective_compact_state_sha256=state['sha256'],full_state_capture_trial_admitted=all(gate.values()),
        criterion_included_in_six_second_label=True,learning_updates=0,physical_substeps=0,heldout_used=False))
    print('BACKWARD_STALL_PREFIX',historical,prospective,gate,flush=True)


if __name__=='__main__':
    main()
