"""Repeat the admitted continuous feedback controller on a fresh development seed."""
import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
import run_continuous_forward_yaw_pair as pair
import run_continuous_roll_projected_entry as projected
import verify_continuous_roll_projected_entry as verifier


PLAN = ROOT/'configs/experiments/continuous_forward_yaw_replication.json'
DRAFT = PLAN.with_name('continuous_forward_yaw_replication_draft.json')
OUT = ROOT/'records/reports/continuous-forward-yaw-replication'
PAGE = ROOT/'records/continuous-forward-yaw-replication-live.html'
baseline = projected.baseline


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft,p = read(DRAFT),projected.checked()
    pair_plan = pair.checked()
    source = ROOT/draft['source']
    proof = read(source)
    assert draft['status']=='draft-not-executable' and proof['status']=='passed-independent-verification'
    assert proof['passed_development_gate'] and proof['same_first_observation_policy_input_raw_and_delayed_motor']
    assert sha(pair.PLAN)==proof['protocol_sha256']
    assert draft['candidate']==pair.ROLES[1] and not proof['heldout_used'] and not proof['learning_updates']
    assert not any(draft[key] for key in ('heldout_used','hardware_io','learning_updates','automatic_retry','automatic_extension'))
    chosen = proof['branches'][draft['candidate']]
    cap = draft['feedback_input_rad_s']
    assert chosen['original_gate'] and chosen['feedback_cap_rad_s']==pair_plan['pair']['roles'][draft['candidate']]==cap
    assert proof['paired']['net_success_gain']>=1 and not proof['paired']['new_safety_failures']
    record = ROOT/proof['record']
    windows = record/'windows-view-verification.json'
    readback = read(windows)
    assert readback['status']=='passed' and readback['observer_host']=='Windows'
    assert readback['running_snapshots_with_original_frame']>0 and readback['snapshots_with_complete_clip']>0
    for name,expected in proof['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    for key in ('worlds','maximum_controls','maximum_control_transitions','maximum_physics_substeps',
                'minimum_roll_completion','minimum_continuous_successes'):
        assert draft[key]==p[key],key
    assert p['forward_controls']==draft['unchanged_forward_controls']
    assert p['walking']['criteria']['yaw_error_max']==draft['unchanged_yaw_error_max_rad_s']
    assert p['feedback']['maximum_joint_correction_rad']==draft['unchanged_joint_correction_max_rad']
    seed = draft['development_seed']
    assert seed!=pair_plan['seed'] and not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    files = [DRAFT,source,windows,pair.PLAN,projected.PLAN,Path(__file__)]
    sources = dict(pair_plan['frozen_sha256'])
    sources.update({file.relative_to(ROOT).as_posix():sha(file) for file in files})
    trial = dict(p,stage=draft['stage'],seed=seed,roll=dict(p['roll'],seed=seed),replication=draft,
        run_id=f'duck_switch_forward-yaw-replication_development_s{seed}_{datetime.now(baseline.TZ):%Y%m%dT%H%M%S}',
        feedback=dict(p['feedback'],maximum_yaw_input_rad_s=cap),frozen_sha256=sources)
    OUT.mkdir()
    write(PLAN,trial)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),physical_substeps=0,
        source_pair_successes=chosen['continuous_successes'],source_pair_sha256=sha(source),seed=seed,
        frozen_files=len(sources),observation=PAGE.relative_to(ROOT).as_posix()))
    print('CONTINUOUS_FORWARD_YAW_REPLICATION_FROZEN',trial['run_id'],sha(PLAN),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','zero','sample','verify'))
    args = parser.parse_args()
    if args.command=='freeze':
        return freeze()
    with patch.multiple(projected,PLAN=PLAN,OUT=OUT,PAGE=PAGE):
        spec = projected.checked()
        if args.command=='verify':
            with patch.multiple(verifier,PLAN=PLAN,OUT=OUT,PAGE=PAGE,checked=lambda:spec):
                return verifier.main()
        with patch.multiple(baseline,PLAN=PLAN,OUT=OUT,PAGE=PAGE,checked=lambda:spec,collect=projected.collect),torch.no_grad():
            baseline.run(args.command=='zero')


if __name__=='__main__':
    main()
