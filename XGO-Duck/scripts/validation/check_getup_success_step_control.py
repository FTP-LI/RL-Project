"""Exercise bounded updates on saved CPU batches and check exact rollback."""
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from check_getup_recovery_retention_ppo import fixtures, read_replay
from getup_recovery_retention_pair import OUT as CLOSED, PLAN
from getup_recovery_retention_ppo import RecoveryRetentionPPO
from getup_success_step_control import ReplayStepRejected, SuccessStepControlledPPO
from getup_supine_learning import load_parent
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_teacher_anchor_train import exact


OUT = ROOT/'records/reports/getup-success-step-control'
SCALES = (1., .5, .25)
BOUNDARY_RMS = (.0496, .04975, .04985, .04993)


def boundary_actor(learner, stopped, target):
    """Make a disposable near-limit fixture from two archived actor states."""
    middle = {key: value.clone() for key, value in learner.actor.state_dict().items()}
    lower, upper = 0., 1.
    for _ in range(18):
        blend = (lower+upper)/2
        learner.actor.load_state_dict({key: torch.lerp(value, stopped[key], blend) if key.startswith('mlp.') else value
                                       for key, value in middle.items()})
        if learner.replay_drift()[0] > target:
            upper = blend
        else:
            lower = blend
    assert abs(learner.replay_drift()[0]-target) < 2e-6
    return learner.actor.state_dict()


def learner_factory(path, plan, anchor, retention, replay_limits, weight, bounded, scales=SCALES):
    constructor = partial(SuccessStepControlledPPO, replay_limits=replay_limits, step_scales=scales) if bounded else RecoveryRetentionPPO
    constructor = partial(constructor, focus_weight=plan['focus_weight'], retention_replay=retention,
                          retention_weight=weight, retention_seed=plan['retention_seed'],
                          retention_samples_per_cell=plan['retention_samples_per_cell'])
    return load_parent(path, plan, anchor, 'cpu', constructor)[0]


def assert_baseline(checked, metrics, reference, batch, last, scale):
    rates = [group['lr'] for group in reference.actor_optimizer.param_groups]
    for group, rate in zip(reference.actor_optimizer.param_groups, rates, strict=True):
        group['lr'] = rate*scale
    expected = reference.update(batch, last)
    for group, rate in zip(reference.actor_optimizer.param_groups, rates, strict=True):
        group['lr'] = rate
    exact(expected, {key: metrics[key] for key in expected})
    expected_state = (reference.actor.state_dict(), reference.critic.state_dict(),
                      reference.actor_optimizer.state_dict(), reference.critic_optimizer.state_dict(),
                      reference.generator.get_state(), reference.anchor_generator.get_state(),
                      reference.actor_steps, reference.critic_steps)
    exact(checked._snapshot_update(), expected_state)


def check_rollback(make, actor, batch, last, injected=False):
    learner = make(True, (1.,))
    learner.actor.load_state_dict(actor)
    before = learner._snapshot_update()
    if injected:
        step = learner.actor_optimizer.step

        def fail_after_step():
            step()
            raise RuntimeError('test failure after actor Adam step')

        with patch.object(learner.actor_optimizer, 'step', fail_after_step):
            try:
                learner.update(batch, last)
            except RuntimeError as error:
                assert str(error) == 'test failure after actor Adam step'
            else:
                raise AssertionError('Injected Adam failure did not propagate')
    else:
        try:
            learner.update(batch, last)
        except ReplayStepRejected:
            pass
        else:
            raise AssertionError('Unsafe unscaled proposal was accepted')
    exact(learner._snapshot_update(), before)


def main():
    torch.set_num_threads(4)
    rng = torch.get_rng_state().clone()
    proof_path = CLOSED/'stop-verification.json'
    proof, frozen = read(proof_path), read(PLAN)
    assert proof['protocol_sha256'] == sha(PLAN) and proof['control_updates'] == 32
    assert proof['status'] == 'verified-guard-stop-saved-data-only' and not torch.cuda.is_initialized()
    record = ROOT/proof['control_record']
    plan = frozen['training']['control']
    middle, stopped = record/'model-25.pt', record/'stopped-partial.pt'
    assert sha(middle) == proof['checkpoint_cpu_drift']['update_25']['checkpoint_sha256']
    assert sha(stopped) == proof['checkpoint_cpu_drift']['stopped']['checkpoint_sha256']
    limits = proof['rms_limit_rad'], proof['absolute_limit_rad']
    anchor = read_replay(ROOT/plan['anchor_data'], ROOT/plan['anchor_cells'], 'teacher_targets')
    retention = read_replay(ROOT/plan['retention_data'], ROOT/plan['retention_cells'], 'targets')
    saved_actor = torch.load(stopped, map_location='cpu', weights_only=True)['actor_state_dict']
    probe = ROOT/plan['probe_dataset']
    assert sha(probe) == plan['probe_sha256']
    checks = {}
    all_passed = True
    for role in ('control', 'candidate'):
        weight = frozen['training'][role]['retention_weight']

        def make(bounded, scales=SCALES):
            return learner_factory(middle, plan, anchor, retention, limits, weight, bounded, scales)

        normal = make(True)
        _, batch, last = fixtures(probe, plan, normal)[2]
        result = normal.update(batch, last)
        assert result['actor_step_scale'] == 1.
        assert_baseline(normal, result, make(False), batch, last, 1.)
        inactive = dict(batch, active=torch.zeros_like(batch['active']), early_supine=torch.zeros_like(batch['early_supine']))
        learner = make(True)
        before = learner._snapshot_update()
        idle = learner.update(inactive, last)
        assert idle['actor_gradient_steps'] == 0 and idle['actor_step_scale'] == 1.
        after = learner._snapshot_update()
        exact((before[0], before[2], before[6]), (after[0], after[2], after[6]))

        attempts = []
        backtrack = None
        for threshold in BOUNDARY_RMS:
            boundary = make(True)
            actor = boundary_actor(boundary, saved_actor, threshold)
            for age, batch, last in fixtures(probe, plan, boundary):
                learner = make(True)
                learner.actor.load_state_dict(actor)
                try:
                    metrics = learner.update(batch, last)
                except ReplayStepRejected as error:
                    attempts.append(dict(target_rms_rad=threshold, saved_age=age, rejected=str(error)))
                    continue
                scale = metrics['actor_step_scale']
                attempts.append(dict(target_rms_rad=threshold, saved_age=age, accepted_scale=scale,
                                     attempts=metrics['replay_step_attempts']))
                if scale < 1.:
                    reference = make(False)
                    reference.actor.load_state_dict(actor)
                    assert_baseline(learner, metrics, reference, batch, last, scale)
                    check_rollback(make, actor, batch, last)
                    check_rollback(make, actor, batch, last, injected=True)
                    backtrack = dict(attempts[-1], exact_vs_single_lower_rate=True,
                                     exhausted_and_faulted_actor_critic_adam_rng_restored=True)
                    break
            if backtrack is not None:
                break
        all_passed &= backtrack is not None
        checks[role] = dict(weight=weight, nominal_exact=True, inactive_actor_and_adam_exact=True,
                            attempts=attempts, backtrack=backtrack)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    result = dict(status='passed-cpu-fixture-only' if all_passed else 'no-safe-reduced-scale-demonstrated',
                  original_limits_rad=limits, step_scales=SCALES, boundary_fixture_rms_rad=BOUNDARY_RMS,
                  tested=checks, physical_substeps=0, optimizer_updates_saved=0, heldout_used=False,
                  dynamic_execution_authorized=False, stopped_trial_evidence_sha256=sha(proof_path),
                  saved_batch_sha256=sha(probe), script_sha256=sha(Path(__file__)),
                  optimizer_source_sha256=sha(ROOT/'scripts/training/getup_success_step_control.py'))
    OUT.mkdir(exist_ok=True)
    write(OUT/'preflight.json', result)
    assert all_passed, 'The saved near-limit fixtures did not demonstrate the reduced-rate path for both arms'
    print('SUCCESS_STEP_CONTROL_CPU_VERIFIED', {role: entry['backtrack']['accepted_scale'] for role, entry in checks.items()})


if __name__ == '__main__':
    main()
