"""Replay the frozen early signal on the gravity input actually seen by PPO actors."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from record_io import write
from review_getup_supine_support_timing import first_run
from screen_getup_supine_early_geometry import score
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity


PLAN = ROOT/'configs/experiments/getup_supine_early_signal_readback.json'
OUT = ROOT/'records/reports/getup-supine-early-signal-readback'


def tilt(gravity):
    return np.rad2deg(np.arccos(np.clip(-gravity[...,2],-1.,1.)))


def normalized(gravity):
    raw = gravity.astype(np.float64)
    length = np.linalg.norm(raw,axis=-1,keepdims=True)
    assert np.isfinite(length).all() and (length>0.).all()
    return raw/length


def read_batch(seed, plan, stage, rule):
    source,expected = plan['archives'][str(seed)]
    path = ROOT/source
    assert sha(path)==expected
    proof = read(path)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    protocol = read(record/'protocol.json')
    criteria = protocol['environment']['criteria']
    assert protocol['environment']['seed']==seed and criteria['step_dt']==.02
    start,end = stage['geometry']['jam_window_controls']
    c = rule['control']
    gravity_slice = slice(*stage['actor_gravity_columns'])
    # The actor input saved on the next control uses the observation just
    # produced by this control. The release sample is before control 101.
    policy_indices = [0,c-2+1-start,c+1-start]
    native_indices = [c-2-start,c-start]
    role_reports = {}
    for role in stage['roles']:
        arm = proof['arms'][role]
        branch = record/f'trace-{role}.npz'
        assert sha(branch)==arm['trace_sha256']
        with np.load(branch) as saved:
            times = saved['time']
            np.testing.assert_allclose(times[policy_indices],np.array([start,c-1,c+1])*.02,atol=1e-8,rtol=0)
            quats = saved['quaternion'][:end-start+1]
            native = tilt(projected_gravity(quats))
            feet = saved['foot_contact'][:end-start+1].all(axis=-1)
            head = saved['native_head_leg_contact'][:end-start+1]
            inputs = saved['actor_input'][policy_indices]
            sensed = saved['observations'][policy_indices]
            assert inputs.shape==sensed.shape==(3,protocol['environment']['num_envs'],61)
            np.testing.assert_array_equal(inputs[:,:,gravity_slice],sensed[:,:,gravity_slice])
        observed = normalized(inputs[:,:,gravity_slice])
        actual = np.array([row['release_pose'] for row in arm['timing']['episodes']])
        observed_release = np.array(classify_release(observed[0],criteria['release_alignment_min']))
        true_supine,actor_supine = actual=='supine',observed_release=='supine'
        assert true_supine.any()
        low,high = stage['geometry']['jam_tilt_deg']
        contact = (native>low)&(native<=high)&feet&head
        jam = [first_run(contact[:,i],stage['geometry']['jam_consecutive_samples']) for i in range(len(actual))]
        onset = np.array([pair[0]+start if pair else -1 for pair in jam])
        early = (onset>=0)&(onset-c>=stage['geometry']['minimum_lead_controls'])
        passed = np.array([row['hold_start_seconds'] is not None for row in arm['timing']['episodes']])
        passed &= np.array(arm['end_hold']['stable_by_environment'])
        assert int(passed[actual!='other'].sum())==arm['combined_successes']
        native_change = native[native_indices[1]]-native[native_indices[0]]
        observed_change = tilt(observed[2])-tilt(observed[1])
        threshold = rule['threshold']
        assert rule['feature']=='tilt_change_40ms_deg' and rule['direction']=='at-least'
        native_active = true_supine&(native_change>=threshold)
        actor_active = actor_supine&(observed_change>=threshold)
        native_counts = score((native_change[true_supine],passed[true_supine],early[true_supine]),rule['direction'],threshold)
        actor_eligible_change = np.where(actor_supine[true_supine],observed_change[true_supine],-np.inf)
        actor_counts = score((actor_eligible_change,passed[true_supine],early[true_supine]),rule['direction'],threshold)
        rows = []
        for world in np.flatnonzero(true_supine|actor_active):
            rows.append(dict(world=int(world),actual_release=str(actual[world]),
                observed_release=str(observed_release[world]),passed=bool(passed[world]),
                jam_onset=None if onset[world]<0 else int(onset[world]),native_change_deg=float(native_change[world]),
                actor_change_deg=float(observed_change[world]),native_active=bool(native_active[world]),
                actor_active=bool(actor_active[world])))
        role_reports[role] = dict(trace_sha256=arm['trace_sha256'],native=native_counts,actor=actor_counts,
            actor_activations_outside_true_supine=np.flatnonzero(actor_active&~true_supine).tolist(),
            release_classification_mismatch=np.flatnonzero(observed_release!=actual).tolist(),
            native_actor_decision_mismatch=np.flatnonzero(native_active!=actor_active).tolist(),
            absolute_signal_error_deg=dict(median=float(np.median(abs(native_change-observed_change)[true_supine])),
                                           maximum=float(max(abs(native_change-observed_change)[true_supine]))),
            rows=rows)
    return dict(seed=seed,policy_read_controls=[start,c-1,c+1],roles=role_reports)


def main():
    assert not OUT.exists()
    stage = read(PLAN)
    assert stage['status']=='frozen-saved-readback-only'
    assert not any(stage[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io','rule_adjustment'))
    state = read(ROOT/'project.json')['staged_training_request']
    assert state['current_stage']==stage['stage'] and ROOT/state['current_protocol']==PLAN
    source,expected = stage['geometry_evidence']
    assert sha(ROOT/source)==expected
    proof = read(ROOT/source)
    assert proof['status']=='passed-saved-geometry-screen' and proof['rule_locked'] and proof['signal_admitted']
    spec_path = ROOT/stage['geometry_plan']
    assert proof['protocol_sha256']==sha(spec_path)
    stage['geometry'] = read(spec_path)
    plan = read(ROOT/stage['geometry']['timing_plan'])
    rule = proof['ranked_reference_rules'][0]
    with ThreadPoolExecutor(max_workers=len(stage['seeds'])) as pool:
        batches = list(pool.map(lambda seed:read_batch(seed,plan,stage,rule),stage['seeds']))
    primary,control = stage['roles']
    expected_scores = [*rule['references'],proof['transfer_score']]
    assert [row['roles'][primary]['native'] for row in batches]==expected_scores
    assert stage['gate_uses_primary_and_latest_control']
    selected = [batch['roles'][primary] for batch in batches]+[batches[-1]['roles'][control]]

    def eligible(item,kind):
        metrics = item[kind]
        return (metrics['useful_fraction']>=stage['geometry']['minimum_useful_failure_coverage'] and
                metrics['success_activation_rate']<=stage['geometry']['maximum_success_activation_rate'] and
                (kind=='native' or not item['actor_activations_outside_true_supine']))

    admission = {kind:all(eligible(row,kind) for row in selected) for kind in ('native','actor')}
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-signal-readback',protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),frozen_rule=rule,batches=batches,signal_gate=admission,
          candidate_dynamic_execution_authorized=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('SUPINE_EARLY_SIGNAL_READBACK',[(row['seed'], {role:{
        kind:[report[kind]['useful_failures'],report[kind]['failed'],report[kind]['success_activations'],report[kind]['successful']]
        for kind in admission} for role,report in row['roles'].items()}) for row in batches],admission,flush=True)


if __name__=='__main__':
    main()
