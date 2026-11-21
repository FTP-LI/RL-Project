"""Project the saved legs-only correction into the original joint envelope."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'
CAP = .08


def main():
    source = OUT/'nominal-supine-upright-geometry.json'
    proof = read(source)
    assert proof['status']=='passed-saved-geometry-review' and not proof['heldout_used']
    # The first and last sample preserve the unchanged policy target.
    rise = np.linspace(0.,1.,8,dtype=np.float32)
    weights = np.concatenate((rise,np.ones(7,dtype=np.float32),rise[-2::-1]))
    assert len(weights)==22 and weights[0]==weights[-1]==0.
    rows = []
    for name, items in proof['results'].items():
        analysis = ROOT/'records/reports'/name/'analysis.json'
        report = read(analysis)
        record = ROOT/report['record']
        spec = read(record/'protocol.json')
        names = spec['model_interface']['joint_names'].split(',')
        limits = np.asarray(spec['environment']['joint_limits'],np.float32)
        for row in items:
            geometry = row['legs']
            assert geometry['feasible']
            role, world = row['role'],row['world']
            path = record/f'trace-{role}.npz'
            assert sha(path)==row['source_sha256']==report['arms'][role]['trace_sha256']
            columns = [names.index(joint) for joint in geometry['joint_names']]
            lo,hi = limits[columns].T
            with np.load(path) as data:
                k = int(np.argmin(abs(data['time']-row['seconds'])))
                assert abs(data['time'][k]-row['seconds'])<1e-7
                base = data['applied_targets'][k+1:k+1+len(weights),world][:,columns]
            assert base.shape==(22,len(columns))
            request = weights[:,None]*np.asarray(geometry['joint_delta_rad'],np.float32)
            inside = (base>=lo)&(base<=hi)
            target = np.where(inside,np.clip(base+request,lo,hi),base)
            applied = target-base
            np.testing.assert_array_equal(target[[0,-1]],base[[0,-1]])
            np.testing.assert_array_equal(target[~inside],base[~inside])
            assert ((target[inside]>=np.broadcast_to(lo,base.shape)[inside])&
                    (target[inside]<=np.broadcast_to(hi,base.shape)[inside])).all()
            assert float(abs(applied).max())<=CAP+1e-7
            preserved = float(np.sum(applied*request)/np.sum(request**2))
            rows.append(dict(batch=name,role=role,world=world,start_time_seconds=row['seconds']+.02,
                maximum_delta_rad=float(abs(applied).max()),outside_original_targets=int((~inside).sum()),
                directional_retention=preserved,trace_sha256=sha(path)))
    output = OUT/'nominal-supine-upright-target-projection.json'
    assert not output.exists() and len(rows)==6
    write(output,dict(status='passed-static-target-projection',rows=rows,source_sha256=sha(source),
        script_sha256=sha(Path(__file__)),weights=weights.tolist(),target_cap_rad=CAP,
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Actuator request arithmetic on existing frames. No new observation, contact or torque outcome.'))
    print('GETUP_UPRIGHT_TARGET_PROJECTION',[(x['world'],x['role'],x['outside_original_targets'],
          round(x['directional_retention'],6)) for x in rows],flush=True)


if __name__=='__main__':
    main()
