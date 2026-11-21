"""Locate the current supine gate failure from verified saved trajectories."""
from pathlib import Path

from record_io import write
from review_getup_nominal_supine import review
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_requested_supine_failure_review.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo/evaluation/requested-supine-review.json'


def main():
    assert not OUT.exists()
    spec = read(PLAN)
    assert spec['status'] == 'frozen-saved-traces-only'
    assert not any(spec[key] for key in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == spec['stage'] and ROOT/stage['current_protocol'] == PLAN
    source, previous = ROOT/spec['source'], ROOT/spec['previous_review']
    proof = read(source)
    reference = read(previous)
    assert reference['status'] == 'passed-saved-trace-review' and not reference['heldout_used']
    eval_spec = read(ROOT/proof['record']/'protocol.json')
    assert eval_spec['environment']['seed'] == spec['seed']
    chosen = proof['selection']['right_side_candidate']
    right = chosen['by_actual_pose']['control']['right-side']
    maximum = right['worlds']-right['reference_successes']
    minimum = eval_spec['choice']['actual_right_net_gain_over_control_min']
    replay = review(source.parent.relative_to(ROOT/'records/reports').as_posix())
    assert replay['evaluation_sha256'] == sha(source)
    assert set(replay['arms']['control']['failures']) == set(replay['all_three_failed'])
    write(OUT,dict(status='passed-saved-trace-review',protocol_sha256=sha(PLAN),**replay,
        previous_review_sha256=sha(previous),comparison_seeds=sorted(reference['batches']),
        right_gain=dict(actual=right['net'],maximum_possible=maximum,required=minimum,
                        gate_feasible_at_observed_control=maximum>=minimum),
        physical_substeps=0,learning_updates=0,heldout_used=False))
    print('REQUESTED_SUPINE_TRACE_REVIEW',replay['all_three_failed'],flush=True)


if __name__ == '__main__':
    main()
