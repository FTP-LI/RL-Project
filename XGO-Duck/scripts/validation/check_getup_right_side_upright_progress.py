"""Screen a time-bounded upright progress signal on archived detections."""
from pathlib import Path
import json

import numpy as np

from review_getup_right_side_trigger_transfer import COHORTS, DEST as TRIGGER, ROOT, load, sha, trigger
from standup_metrics import classify_release, projected_gravity


PLAN = ROOT/'configs/experiments/getup_right_side_upright_progress_draft.json'
DEST = TRIGGER.with_name('right-side-upright-progress-screen.json')


def potential(quaternion,feet):
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(quaternion)[...,2],-1.,1.)))
    return np.clip((45.-tilt)/30.,0.,1.)*feet.mean(axis=-1)


def main():
    assert not DEST.exists()
    draft,evidence = load(PLAN),load(TRIGGER)
    assert draft['status']=='draft-not-executable' and not draft['heldout_used'] and not draft['learning_updates']
    assert evidence['status']=='verified-saved-trigger-transfer' and not evidence['heldout_used']
    for path,digest in evidence['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    horizon = draft['proposed_reward_window_controls']
    assert horizon==40
    entry = evidence['reused_entry']
    rows,scenes,comparison = [],0,{}
    for label,proof_file,roles in COHORTS:
        proof = load(proof_file)
        record = ROOT/proof['record']
        names = ('quaternion','foot_contact','native_head_leg_contact','active')
        with np.load(record/'trace-shared-prefix.npz') as prefix:
            initial = {name:prefix[name] for name in names}
        pose = np.asarray(classify_release(projected_gravity(initial['quaternion'][-1])))
        eligible = pose=='right-side'
        for role in roles:
            scenes += len(pose)
            outcomes = proof['arms'][role]
            scores = np.asarray([row['hold_start_seconds'] is not None for row in outcomes['timing']['episodes']])
            scores &= outcomes['end_hold']['stable_by_environment']
            with np.load(record/f'trace-{role}.npz') as archive:
                data = {name:np.concatenate((initial[name],archive[name]),axis=0) for name in names}
            q,feet,head = (data[name] for name in names[:3])
            decision = trigger(q,feet,head,eligible,entry)
            expected = evidence['cohorts'][label]['references'][role]['trigger_control_by_world']
            assert {str(w):int(decision[w]) for w in np.flatnonzero(decision>=0)}==expected
            p = potential(q,feet)
            increments = np.zeros_like(p)
            for world in np.flatnonzero(decision>=0):
                start = int(decision[world])-1
                stop = start+horizon
                assert stop<len(p) and data['active'][start:stop+1,world].all()
                changes = np.diff(p[start:stop+1,world])
                increments[start+1:stop+1,world] = changes
                net = float(changes.sum())
                np.testing.assert_allclose(net,p[stop,world]-p[start,world],atol=1e-12,rtol=0)
                rows.append(dict(cohort=label,role=role,world=int(world),trigger_control=int(decision[world]),
                    original_pass=bool(scores[world]),signed_progress=net,initial_potential=float(p[start,world]),
                    final_potential=float(p[stop,world]),head_leg_clear_at_end=bool(not head[stop,world]),
                    peak_step_progress=float(np.abs(changes).max())))
            assert not increments[:,decision<0].any() and not increments[:,~eligible].any()
        comparison[label] = {key:sum(row['original_pass']==passed for row in rows if row['cohort']==label)
                             for key,passed in (('passed_triggered',True),('failed_triggered',False))}
    success = [row for row in rows if row['original_pass']]
    failure = [row for row in rows if not row['original_pass']]
    assert len(success)==2 and len(failure)==26
    report = dict(status='verified-saved-upright-progress-only',roles_by_cohort=comparison,physical_substeps=0,
        heldout_used=False,learning_updates=0,total_reference_trajectories=scenes,triggered_passes=len(success),
        triggered_failures=len(failure),passed_net_progress=[row['signed_progress'] for row in success],
        failed_net_progress_quantiles=np.quantile([row['signed_progress'] for row in failure],[.1,.5,.9]).tolist(),
        triggered_failures_clear_head_leg_at_end=sum(row['head_leg_clear_at_end'] for row in failure),
        passing_examples_receive_nonnegative_progress=all(row['signed_progress']>=0 for row in success),
        untriggered_and_other_pose_signal_exactly_zero=True,rows=rows,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (PLAN,TRIGGER,Path(__file__))},
        scope='Saved-state shaping audit only. Online reward latch, reset, value qualification and actual learning remain pending.')
    DEST.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('RIGHT_SIDE_PROGRESS_SCREEN',report['passed_net_progress'],report['failed_net_progress_quantiles'],
          report['triggered_failures_clear_head_leg_at_end'],flush=True)


if __name__=='__main__':
    main()
