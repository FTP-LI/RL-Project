"""Score the matched pulse arms under the original roll and forward gates."""

from unittest.mock import patch

import numpy as np

import run_roll_active_lateral as legacy
import run_roll_head_support_pulse as run
import verify_continuous_roll_stand_walk as policy_check
import verify_roll_active_lateral as prior
from verify_roll_head_support_control import audit


def verify():
    root,read,write,sha = run.ROOT,run.read,run.write,run.sha
    plan = read(run.PLAN)
    shared = plan['perturbation']['shared']
    record = root/plan['record']
    independent = run.OUT/'independent'
    independent.mkdir(exist_ok=True)
    target = run.OUT/'analysis.json'
    assert not target.exists()
    raw = independent/'analysis.json'

    def replay(commands, scene, spec, phase):
        np.testing.assert_array_equal(commands['phase_used'],phase)
        enabled = bool(commands['pulse_enabled'].any())
        return audit(commands,scene,spec,shared,enabled)

    with patch.multiple(legacy,PLAN=run.PLAN,OUT=independent,SEED=run.SEED,STAGE=run.STAGE), \
         patch.object(prior,'PLAN',run.PLAN), patch.object(policy_check,'check_policy',replay):
        prior.verify()
    proof = read(raw)
    if proof['status']=='verified-shared-stop':
        write(target,dict(proof,pulse_development_passed=False))
        return
    assert proof['status']=='passed-independent-verification' and len(proof['branches'])==6
    assert proof['same_first_observation_input_action_target_and_delayed_motor']
    assert proof['first_physics_qvel_matches_injection']
    fork = read(record/'fork.json')
    from roll_head_support_control import PULSE_FIELDS

    assert {f'controller.pulse.{name}' for name in PULSE_FIELDS} <= set(fork['field_names'])
    pairs = {}
    depths = {}
    for role in proof['branches']:
        source = record/'clips'/role/'trace.npz'
        assert sha(source) == proof['source_sha256'][source.relative_to(root).as_posix()]
        with np.load(source,allow_pickle=False) as data:
            values = data['head_penetration_m']
        assert np.isfinite(values).all() and (values >= 0.).all()
        depths[role] = dict(peak_m=float(values.max()),penetrating_worlds=int((values > 0).any(axis=0).sum()),
                            maximum_by_world_m=values.max(axis=0).tolist())
    for arm in ('reference','left-push','right-push'):
        control,pulse = (proof['branches'][role] for role in (arm,'pulse-'+arm))
        assert control['perturbation_m_s']==pulse['perturbation_m_s']
        assert not control['policy']['pulse_applied_controls'] and pulse['policy']['pulse_applied_controls']
        a,b = (np.asarray(row['success_by_world'],dtype=bool) for row in (control,pulse))
        pairs[arm] = dict(
            control_roll=control['roll_completed'],pulse_roll=pulse['roll_completed'],
            control_complete=control['original_successes'],pulse_complete=pulse['original_successes'],
            pulse_gate=pulse['passed_gate'],gained=np.flatnonzero(b & ~a).tolist(),
            lost=np.flatnonzero(a & ~b).tolist())
    nominal = pairs['reference']
    no_regression = nominal['pulse_roll'] >= nominal['control_roll'] and nominal['pulse_complete'] >= nominal['control_complete']
    passed = proof['observer_verified'] and no_regression and all(row['pulse_gate'] for row in pairs.values())
    write(target,dict(proof,legacy_verification_sha256=sha(raw),pulse_pairs=pairs,
        head_penetration=depths,nominal_no_regression=no_regression,pulse_development_passed=passed))
    print('ROLL_HEAD_SUPPORT_PULSE_VERIFIED',pairs,passed,flush=True)


if __name__=='__main__':
    verify()
