"""Find usable parent-success observations for two underperforming getup groups."""
from pathlib import Path

import numpy as np

import getup_shared_release as shared
from getup_coverage_metrics import POSES
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release
from verify_getup_shared_release import load_arrays


OUT = ROOT/'records/reports/getup-supine-weighted-learning'
HISTORY = ROOT/'records/reports/getup-anchor-same-release'
AGE_STEPS = [100, 110, 120, 130, 140, 149]


def groups(arm):
    rows = arm['timing']['episodes']
    release = np.array([row['release_pose'] for row in rows])
    requested = np.array([row['requested_pose'] for row in rows])
    timely = np.array([row['hold_start_seconds'] is not None for row in rows])
    end = np.array(arm['end_hold']['stable_by_environment'])
    return release, requested, timely & end & np.isin(release, POSES)


def evidence(folder, parent_role):
    path = folder/'analysis.json'
    proof, state = read(path), read(folder/'coordinator.json')
    assert state['status'] == 'completed' and state['analysis_sha256'] == sha(path) and not proof['heldout_used']
    record = ROOT/proof['record']
    spec = read(record/'protocol.json')
    assert sha(record/'protocol.json') == proof.get('evaluation_protocol_sha256', proof['protocol_sha256'])
    arm = proof['arms'][parent_role]
    branch = record/f'trace-{parent_role}.npz'
    assert sha(branch) == arm['trace_sha256']
    release, requested, stable = groups(arm)
    assert int(stable.sum()) == arm['combined_successes']
    idx = {name: np.flatnonzero(group & stable) for name, group in (
        ('actual-right-side', release == 'right-side'),
        ('requested-supine', requested == 'supine'),
    )}
    assert not set(idx['actual-right-side']) & set(idx['requested-supine'])
    with np.load(branch) as data:
        first_active = np.asarray(AGE_STEPS) - spec['prefix_controls']
        inputs = {name: data['actor_input'][first_active][:, ids].copy() for name, ids in idx.items()}
        targets = {name: data['raw_policy_actions'][first_active][:, ids].copy() for name, ids in idx.items()}
    for name in idx:
        assert inputs[name].shape == (len(AGE_STEPS), len(idx[name]), 61)
        assert targets[name].shape == (len(AGE_STEPS), len(idx[name]), 14)
    return proof, spec, record, idx, inputs, targets, sha(path)


def main():
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    reports = {role: evidence(folder, policy) for role, folder, policy in (
        ('previous', HISTORY, 'success-anchor'), ('current', OUT, 'fixed-parent'),
    )}
    prev, current = reports['previous'], reports['current']
    previous_plan, current_plan = prev[1]['policies']['success-anchor'], current[1]['policies']['fixed-parent']
    assert previous_plan['model'] == current_plan['model']
    assert previous_plan['model_sha256'] == current_plan['model_sha256']
    parent = shared.actor_model(current_plan)
    weighted_role = 'supine-weighted-continuation'
    weighted = shared.actor_model(current[1]['policies'][weighted_role])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/current_plan['model']), sess_options=options, providers=['CPUExecutionProvider'])
    summary, audits = {}, {}
    with torch.no_grad():
        for name, (proof, spec, record, ids, inputs, targets, digest) in reports.items():
            audits[name] = dict(seed=spec['environment']['seed'], evaluation_sha256=digest, groups={})
            for group, samples in inputs.items():
                x = samples.reshape(-1, 61)
                y = targets[group].reshape(-1, 14)
                a = parent(shared.td('actor', torch.from_numpy(x))).numpy()
                b = weighted(shared.td('actor', torch.from_numpy(x))).numpy()
                np.testing.assert_allclose(a, y, atol=2e-5, rtol=2e-5)
                native = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in x[::11]])
                np.testing.assert_allclose(a[::11], native, atol=2e-5, rtol=2e-5)
                audits[name]['groups'][group] = dict(worlds=len(ids[group]), inputs=len(x),
                    environment_indices=ids[group].tolist(), cpu_parent_match_error=float(np.abs(a-y).max()),
                    weighted_action_difference_rms_rad=float(np.sqrt(np.mean((b-a)**2))),
                    weighted_action_difference_max_rad=float(np.abs(b-a).max()))
                summary[group] = summary.get(group, 0)+len(ids[group])
    p, spec, record = current[:3]
    release, requested, parent_ok = groups(p['arms']['fixed-parent'])
    _, _, weighted_ok = groups(p['arms'][weighted_role])
    first = load_arrays(record/'first-control-fixed-parent.npz')
    alignment = spec['environment']['criteria']['release_alignment_min']
    observed = np.array(classify_release(first['actor_input'][:, 3:6], alignment))
    right = release == 'right-side'
    requested_supine = requested == 'supine'
    eligible = right | requested_supine
    lost = eligible & parent_ok & ~weighted_ok
    # A saved-result bound only: retaining these parent successes would help
    # both gates, but an actual learned policy must reproduce them afresh.
    retained = weighted_ok | lost
    routed = np.where(observed == 'right-side', parent_ok, weighted_ok)
    assert not torch.cuda.is_initialized()
    result = dict(status='passed-saved-development-only', source_sha256=sha(Path(__file__)),
        heldout_used=False, physical_substeps=0, optimizer_updates=0, parent_model_sha256=current_plan['model_sha256'],
        early_actor_input_ages=AGE_STEPS, data_audit=audits, usable_parent_success_worlds=summary,
        observed_right=dict(actual_right=int(right.sum()), observed_right=int((observed == 'right-side').sum()),
                            correctly_observed_right=int((right & (observed == 'right-side')).sum())),
        saved_outcome_bound=dict(
            weighted_total=int(weighted_ok.sum()), right_losses_from_parent=np.flatnonzero(lost & right).tolist(),
            requested_supine_losses_from_parent=np.flatnonzero(lost & requested_supine).tolist(),
            if_retained_total=int(retained.sum()), if_retained_actual_right=int((retained & right).sum()),
            if_retained_requested_supine=int((retained & requested_supine).sum()),
            observed_right_selection_actual_right=int((routed & right).sum()),
            observed_right_selection_requested_supine=int((routed & requested_supine).sum())),
    )
    write(OUT/'recovery-retention-audit.json', result)
    print('GETUP_RETENTION_AUDIT', summary, result['observed_right'], result['saved_outcome_bound'], flush=True)


if __name__ == '__main__':
    main()
