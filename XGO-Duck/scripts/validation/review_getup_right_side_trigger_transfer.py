"""Check whether the existing early supine trigger transfers to right-side release."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from standup_metrics import classify_release, projected_gravity


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'records/reports/getup-upright-recovery-pair'
PLAN = ROOT/'configs/experiments/getup_upright_recovery_pair.json'
DEST = OUT/'right-side-trigger-transfer.json'
COHORTS = (
    ('early-support',ROOT/'records/reports/getup-early-support-ppo-functional/analysis.json',('fixed-weighted','control')),
    ('expanded-replay',ROOT/'records/reports/getup-side-to-supine-replay/evaluation/analysis.json',('fixed-weighted','control')),
    ('bounded-legs',OUT/'seed1093-analysis.json',('reference-a','reference-b')),
)


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def trigger(q,feet,head,eligible,entry):
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(q)[...,2],-1.,1.)))
    limit = round(entry['last_trigger_seconds']/.02)
    samples = entry['consecutive_samples']
    active = feet.all(axis=2)&head&eligible[None]
    active &= (tilt>entry['minimum_tilt_deg_exclusive'])&(tilt<=entry['maximum_tilt_deg_inclusive'])
    first = np.full(len(eligible),-1,dtype=np.int32)
    for k in range(100+samples,limit+1):
        start = k-samples
        slow = tilt[start]-tilt[k-1]<entry['maximum_recent_tilt_reduction_deg_exclusive']
        selected = (first<0)&active[start:k].all(axis=0)&slow
        first[selected] = k
    return first


def main():
    assert not DEST.exists()
    p = load(PLAN)
    entry = p['specs']['1093']['intervention']['entry']
    minimum = p['specs']['1093']['environment']['criteria']['release_alignment_min']
    assert entry['actual_release_pose']=='supine' and not p['heldout_used']
    source_sha = {path.relative_to(ROOT).as_posix():sha(path) for path in (PLAN,Path(__file__))}
    cases = {}
    for label,proof_path,roles in COHORTS:
        proof = load(proof_path)
        assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
        record = ROOT/proof['record']
        prefix = record/'trace-shared-prefix.npz'
        source_sha[proof_path.relative_to(ROOT).as_posix()] = sha(proof_path)
        source_sha[prefix.relative_to(ROOT).as_posix()] = sha(prefix)
        assert source_sha[prefix.relative_to(ROOT).as_posix()]==proof['prefix_sha256']
        names = ('quaternion','foot_contact','native_head_leg_contact')
        with np.load(prefix) as archive:
            initial = [archive[name] for name in names]
        pose = np.asarray(classify_release(projected_gravity(initial[0][-1]),minimum))
        right = pose=='right-side'
        local = {}
        for role in roles:
            a = proof['arms'][role]
            outcomes = a['timing']['episodes']
            np.testing.assert_array_equal(pose,[row['release_pose'] for row in outcomes])
            passed = np.asarray([row['hold_start_seconds'] is not None for row in outcomes])&a['end_hold']['stable_by_environment']
            archive = record/f'trace-{role}.npz'
            source_sha[archive.relative_to(ROOT).as_posix()] = sha(archive)
            assert source_sha[archive.relative_to(ROOT).as_posix()]==a['trace_sha256']
            with np.load(archive) as branch:
                q,feet,head = (np.concatenate((pre,branch[name]),axis=0) for name,pre in zip(names,initial,strict=True))
            if label=='bounded-legs':
                old = trigger(q,feet,head,pose=='supine',entry)
                events = record/'upright-trigger-events.json'
                source_sha[events.relative_to(ROOT).as_posix()] = sha(events)
                assert source_sha[events.relative_to(ROOT).as_posix()]==proof['event_sha256']
                expected = sorted((row['world'],row['decision_control']) for row in load(events)[role])
                assert [(int(w),int(old[w])) for w in np.flatnonzero(old>=0)]==expected
            selected = trigger(q,feet,head,right,entry)
            yes = selected>=0
            failures = right&~passed
            successes = right&passed
            local[role] = dict(failures=int(failures.sum()),failures_triggered=int((failures&yes).sum()),
                successes=int(successes.sum()),successes_triggered=int((successes&yes).sum()),
                trigger_control_by_world={str(w):int(selected[w]) for w in np.flatnonzero(yes)},
                failure_worlds=np.flatnonzero(failures).tolist())
        cases[label] = dict(actual_right=int(right.sum()),references=local)
    writeup = dict(status='verified-saved-trigger-transfer',source_sha256=source_sha,
        reused_entry={**entry,'actual_release_pose':'right-side'},cohorts=cases,original_supine_events_reproduced=True,
        heldout_used=False,physical_substeps=0,learning_updates=0,
        scope='Identical timing, tilt and contact thresholds; change only the eligible actual release pose. Counts are observational.')
    DEST.write_text(json.dumps(writeup,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,case in cases.items():
        summary = {role:{key:row[key] for key in ('failures','failures_triggered','successes','successes_triggered')}
                   for role,row in case['references'].items()}
        print('GETUP_RIGHT_TRIGGER_TRANSFER',name,case['actual_right'],summary,flush=True)


if __name__=='__main__':
    main()
