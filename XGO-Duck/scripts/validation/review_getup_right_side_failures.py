"""Localize common right-side recovery failures without using a held-out seed."""
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from standup_metrics import projected_gravity


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'records/reports/getup-upright-recovery-pair'
PROOF = OUT/'seed1093-analysis.json'
DEST = OUT/'seed1093-right-side-localization.json'
ROLES = ('reference-a','reference-b','bounded-legs')
TIME = (2.,2.5,3.,4.,5.,8.,11.)


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def observations(data,world):
    gravity = projected_gravity(data['quaternion'][:,world])
    tilt = np.rad2deg(np.arccos(np.clip(-gravity[:,2],-1.,1.)))
    active = data['active'][:,world]
    feet = data['foot_contact'][:,world].all(axis=1)&active
    bilateral = np.convolve(feet[100:].astype(int),np.ones(10,dtype=int),mode='valid')==10
    first = np.flatnonzero(bilateral)
    head = data['native_head_leg_contact'][:,world]&active
    frames = [round(seconds/.02)-1 for seconds in TIME]
    return dict(tilt_deg=tilt[frames].tolist(),trunk_height_m=data['height'][frames,world].tolist(),
        first_bilateral_200ms_start_s=float(2+first[0]*.02) if len(first) else None,
        head_leg_fraction_2_to_5_s=float(head[100:250].mean()),
        head_leg_fraction_8_to_11_s=float(head[400:550].mean()))


def main():
    assert not DEST.exists()
    proof = json.loads(PROOF.read_text(encoding='utf-8'))
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    record = ROOT/proof['record']
    sources = {PROOF.relative_to(ROOT).as_posix():sha(PROOF),Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))}
    prefix = record/'trace-shared-prefix.npz'
    sources[prefix.relative_to(ROOT).as_posix()] = sha(prefix)
    assert proof['prefix_sha256']==sources[prefix.relative_to(ROOT).as_posix()]
    names = ('quaternion','height','foot_contact','native_head_leg_contact','active')
    with np.load(prefix) as archive:
        initial = {name:archive[name] for name in names}
    results,pose_rows = {},[]
    for role in ROLES:
        scores = proof['arms'][role]
        trace = record/f'trace-{role}.npz'
        sources[trace.relative_to(ROOT).as_posix()] = sha(trace)
        assert scores['trace_sha256']==sources[trace.relative_to(ROOT).as_posix()]
        episodes = scores['timing']['episodes']
        pose_rows.append([(row['requested_pose'],row['release_pose']) for row in episodes])
        with np.load(trace) as archive:
            fields = {name:np.concatenate((initial[name],archive[name])) for name in names}
            assert fields['active'].shape==(550,320)
            results[role] = [dict(passed=row['hold_start_seconds'] is not None and scores['end_hold']['stable_by_environment'][world],
                                  **observations(fields,world))
                             for world,row in enumerate(episodes)]
    assert pose_rows[0]==pose_rows[1]==pose_rows[2] and len(pose_rows[0])==320
    right = [w for w,pair in enumerate(pose_rows[0]) if 'right-side' in pair]
    rows = [dict(world=w,requested=pose_rows[0][w][0],release=pose_rows[0][w][1],
                 branches={role:results[role][w] for role in ROLES}) for w in right]
    assert sum(row['requested']=='right-side' for row in rows)==128
    assert sum(row['release']=='right-side' for row in rows)==60
    strata = {}
    for requested,release in sorted(set(pose_rows[0][w] for w in right)):
        group = [row for row in rows if (row['requested'],row['release'])==(requested,release)]
        strata[f'{requested}/{release}'] = dict(worlds=len(group),passes={role:sum(row['branches'][role]['passed']
                                                                                for row in group) for role in ROLES})
    actual = [row for row in rows if row['release']=='right-side']
    summary = {}
    for kind,pass_pair in (('both-pass',[True,True]),('both-fail',[False,False])):
        cohort = [row for row in actual if [row['branches'][role]['passed'] for role in ROLES[:2]]==pass_pair]
        flat = [row['branches'][role] for row in cohort for role in ROLES[:2]]
        assert flat
        both = [row['first_bilateral_200ms_start_s'] for row in flat if row['first_bilateral_200ms_start_s'] is not None]
        summary[kind] = dict(worlds=len(cohort),world_ids=[row['world'] for row in cohort],
            median_tilt_deg=np.median([row['tilt_deg'] for row in flat],axis=0).tolist(),
            median_trunk_height_m=np.median([row['trunk_height_m'] for row in flat],axis=0).tolist(),
            bilateral_200ms_references=len(both),median_bilateral_200ms_start_s=float(np.median(both)) if both else None,
            median_head_leg_fraction_2_to_5_s=float(np.median([row['head_leg_fraction_2_to_5_s'] for row in flat])),
            median_head_leg_fraction_8_to_11_s=float(np.median([row['head_leg_fraction_8_to_11_s'] for row in flat])))
    result = dict(status='verified-saved-right-side-recovery',worlds_related_to_right=len(rows),time_seconds=list(TIME),
        source_sha256=sources,heldout_used=False,physical_substeps=0,learning_updates=0,
        release_strata=strata,actual_right_reference_agreement=summary,rows=rows)
    DEST.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(strata=strata,actual_right_reference_agreement=summary),indent=2),flush=True)


if __name__=='__main__':
    main()
