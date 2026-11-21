"""Compare fixed replay terms with the original bounded-update CPU sequence."""
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import check_getup_success_step_control_sequence as sequence
import getup_recovery_retention_ppo as retention_ppo
from getup_targeted_replay import side_replay
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_targeted_replay.json'


def replay_variant(variant, spec, joints, expected):
    factory = sequence.learner_factory
    objective = retention_ppo.anchored_objective
    replay_loss = None
    calls = 0

    def make(*args, **kwargs):
        nonlocal replay_loss, calls
        assert calls % 2 == 0
        learner = factory(*args, **kwargs)
        loss = side_replay(learner, joints, spec['terms'], variant)
        replay_loss = lambda: loss(learner.plan['initial_std'])
        return learner

    def add_targeted(policy, auxiliary, weight):
        nonlocal calls
        # The frozen learner calls this once for official replay and once
        # for parent replay. The extra term belongs only to official replay.
        if calls % 2 == 0 and variant != 'original':
            assert weight == expected['anchor_weight']
            auxiliary = auxiliary+replay_loss()
        calls += 1
        return objective(policy, auxiliary, weight)

    start = perf_counter()
    with patch.object(sequence, 'learner_factory', make), \
         patch.object(retention_ppo, 'anchored_objective', add_targeted), \
         patch.object(sequence, 'write') as capture:
        sequence.main()
    capture.assert_called_once()
    _, result = capture.call_args.args
    assert calls > 0 and calls % 2 == 0
    assert result['limits_rad'] == expected['limits_rad']
    assert result['physical_substeps'] == result['optimizer_updates_saved'] == 0
    assert not result['heldout_used'] and not torch.cuda.is_initialized()
    if variant == 'original':
        for role, arm in result['arms'].items():
            prior = expected['prior']['arms'][role]
            assert arm['accepted_updates'] == prior['accepted_updates']
            assert arm['reduced_rate_updates'] == prior['reduced_rate_updates']
            assert arm['first_rejection']['update'] == prior['first_rejection']['update']
            np.testing.assert_allclose(arm['final_rad'], prior['final_rad'], atol=2e-6, rtol=0)
    return dict(wall_seconds=perf_counter()-start, arms=result['arms'])


def main():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-cpu-component-screen'
    assert not spec['dynamic_execution_authorized'] and not spec['heldout_used']
    for relative, digest in spec['source_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    prior = read(ROOT/spec['prior_result'])
    frozen = read(sequence.PLAN)
    joints = frozen['evaluation']['model_interface']['joint_names'].split(',')
    expected = dict(prior=prior, limits_rad=prior['limits_rad'],
                    anchor_weight=frozen['training']['control']['anchor_weight'])
    variants = {}
    for variant in spec['variants']:
        variants[variant] = replay_variant(variant, spec, joints, expected)
        print('TARGETED_REPLAY', variant,
              {k: (v['accepted_updates'], v['final_rad'], v['passed'])
               for k, v in variants[variant]['arms'].items()}, flush=True)
    admitted = all(arm['passed'] for arm in variants[spec['admission_candidate']]['arms'].values())
    out = ROOT/spec['result']
    write(out, dict(status='passed-primary-cpu-screen' if admitted else 'below-primary-cpu-admission',
                    protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
                    variants=variants, admission_candidate=spec['admission_candidate'],
                    physical_substeps=0, optimizer_updates_saved=0, heldout_used=False,
                    dynamic_execution_authorized=False))


if __name__ == '__main__':
    main()
