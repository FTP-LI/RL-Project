"""Bound the effect of sole-contact timestamps on the last weighted getup batch."""
import numpy as np

from getup_coverage_metrics import POSES
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from verify_getup_anchor_same_release import independent_holds


def main():
    source = ROOT/'records/reports/getup-supine-weighted-learning/analysis.json'
    proof = read(source)
    assert proof['status'] == 'passed-independent-verification' and not proof['heldout_used']
    role = 'supine-weighted-continuation'
    arm = proof['arms'][role]
    folder = ROOT/proof['record']
    spec = read(folder/'protocol.json')
    prefix, suffix = folder/'trace-shared-prefix.npz', folder/f'trace-{role}.npz'
    assert sha(prefix) == proof['prefix_sha256'] and sha(suffix) == arm['trace_sha256']
    fields = ('height', 'quaternion', 'angular_velocity', 'foot_contact', 'joint_envelope_ok', 'active')
    with np.load(prefix) as a, np.load(suffix) as b:
        trace = {key: np.concatenate((a[key], b[key])) for key in fields}
    criteria = spec['environment']['criteria']
    early, final = independent_holds(trace, criteria)
    cases = arm['timing']['episodes']
    np.testing.assert_array_equal(early, [row['hold_start_seconds'] is not None for row in cases])
    np.testing.assert_array_equal(final, arm['end_hold']['stable_by_environment'])
    actual, requested = (np.array([row[key] for row in cases]) for key in ('release_pose', 'requested_pose'))
    valid = np.isin(actual, POSES)
    original = early & final & valid
    assert original.sum() == arm['combined_successes'] == 310
    ideal_contact = dict(trace, foot_contact=np.ones_like(trace['foot_contact'], dtype=bool))
    relaxed_early, relaxed_final = independent_holds(ideal_contact, criteria)
    ceiling = relaxed_early & relaxed_final & valid
    assert (original <= ceiling).all()

    def groups(labels):
        return {pose: dict(episodes=int((labels == pose).sum()),
                          original=int((original & (labels == pose)).sum()),
                          permissive_bound=int((ceiling & (labels == pose)).sum())) for pose in POSES}

    out = ROOT/'records/reports/getup-contact-sensitivity'
    out.mkdir(exist_ok=True)
    target = out/'analysis.json'
    assert not target.exists()
    write(target, dict(status='passed-saved-data-only', seed=spec['environment']['seed'], role=role,
        source_sha256=sha(source), prefix_sha256=sha(prefix), arm_sha256=sha(suffix),
        original_successes=int(original.sum()), permissive_bound=int(ceiling.sum()),
        newly_possible_worlds=np.flatnonzero(~original & ceiling).tolist(),
        requested=groups(requested), actual=groups(actual), heldout_used=False, new_physical_steps=0,
        scope='Recorded-pose ceiling under all sole flags true. The original hold, contact, release and deployment requirements stay in force.'))
    print('GETUP_CONTACT_SENSITIVITY', int(original.sum()), int(ceiling.sum()), groups(requested), groups(actual), flush=True)


if __name__ == '__main__':
    main()
