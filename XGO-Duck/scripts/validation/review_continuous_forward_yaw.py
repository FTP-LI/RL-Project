"""Localize the remaining continuous forward yaw gap on verified trajectories."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/continuous-roll-projected-entry'


def aggregate(rows,upper):
    fields = ('original_mean_absolute_rad_s','first_second_mean_absolute_rad_s',
              'seconds_2_to_20_mean_absolute_rad_s','last_ten_seconds_mean_absolute_rad_s',
              'mean_signed_rad_s','sign_changes','internal_cap_fraction','target_cap_fraction')
    medians = {field:float(np.median([row[field] for row in rows])) for field in fields}
    return dict(worlds=len(rows),medians=medians,
        final_19s_above_original_threshold=sum(row[fields[2]]>upper for row in rows),
        final_10s_above_original_threshold=sum(row[fields[3]]>upper for row in rows))


def main():
    source = OUT/'analysis.json'
    proof = read(source)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    assert proof['counts']['continuous_successes']==111 and proof['counts']['finished_forward_window']==126
    folder = ROOT/proof['record']
    plan,trace_path,cmd_path = (folder/file for file in ('protocol.json','trace.npz','commands.npz'))
    assert sha(plan)==proof['protocol_sha256']
    for path in (trace_path,cmd_path):
        assert sha(path)==proof['source_sha256'][path.relative_to(ROOT).as_posix()]
    p = read(plan)
    upper = p['walking']['criteria']['yaw_error_max']
    assert p['forward_controls']==1000 and p['walking']['commands']['forward'][2]==0. and upper==.2
    with np.load(trace_path) as data:
        yaw = data['angular_velocity'][...,2]
    with np.load(cmd_path) as data:
        feedback,request,alternative = (data[key] for key in ('feedback_yaw','requested_targets','alternate_action'))
    lo,hi = np.asarray(p['walking']['joint_limits'],np.float32).T
    home = np.asarray(p['walking']['model_home'],np.float32)
    original = np.clip(request,lo,hi)
    alternate = np.clip(alternative+home,lo,hi)
    cap = p['feedback']['maximum_joint_correction_rad']
    rows = []
    for episode in proof['forward']:
        if not episode['entered_forward']:
            assert episode['failed_checks']==['no-forward-entry']
            continue
        world,begin = episode['world'],episode['forward_start']
        end = begin+p['forward_controls']
        section = yaw[begin+1:end+1,world]
        assert section.shape==(1000,)
        signed = float(section.mean())
        absolute = np.abs(section)
        np.testing.assert_allclose(absolute.mean(),episode['yaw_error_rad_s'],atol=2e-7,rtol=0)
        assert episode['success'] or episode['failed_checks']==['yaw_error']
        on = feedback[begin:end,world]
        candidate = alternate[begin:end,world]-original[begin:end,world]
        rows.append(dict(world=world,passed=episode['success'],
            original_mean_absolute_rad_s=float(absolute.mean()),first_second_mean_absolute_rad_s=float(absolute[:50].mean()),
            seconds_2_to_20_mean_absolute_rad_s=float(absolute[50:].mean()),
            last_ten_seconds_mean_absolute_rad_s=float(absolute[-500:].mean()),mean_signed_rad_s=signed,
            sign_changes=int(np.count_nonzero(np.signbit(section[1:])!=np.signbit(section[:-1]))),
            internal_cap_fraction=float((np.abs(on)>=p['feedback']['maximum_yaw_input_rad_s']-1e-6).mean()),
            target_cap_fraction=float((np.abs(candidate).max(axis=-1)>cap).mean()),
            per_second_mean_absolute_rad_s=absolute.reshape(20,50).mean(axis=-1).tolist()))
    passed = [row for row in rows if row['passed']]
    failed = [row for row in rows if not row['passed']]
    assert len(passed)==111 and len(failed)==15
    groups = {name:aggregate(items,upper) for name,items in (('passed',passed),('yaw_only_failures',failed))}
    output = OUT/'forward-yaw-localization.json'
    assert not output.exists()
    write(output,dict(status='passed-saved-forward-yaw-localization',groups=groups,original_threshold_rad_s=upper,
        episodes=rows,sources={f.relative_to(ROOT).as_posix():sha(f) for f in (source,plan,trace_path,cmd_path,Path(__file__))},
        scope='Causal timing summaries on completed forward windows. Original scores and gate are unchanged.',
        physical_substeps=0,learning_updates=0,heldout_used=False))
    print('CONTINUOUS_FORWARD_YAW',groups,flush=True)


if __name__=='__main__':
    main()
