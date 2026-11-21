"""Fixed random-drop coverage of the archived official recovery policy.

Change from the stratified reference: use the source task's random Euler/drop
distribution, without resampling outcomes. Classify tilt at 2 s BEFORE policy
activation; already-upright releases cannot inflate the primary recovery rate.
The immutable reference collector and independent stable-hold verifier are
reused. Their requested-stratum flags are diagnostic only for random drops.
No learning, no hardware, and no implicit extension of an existing attempt.
"""
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import shutil
import sys
import numpy as np
import evaluate_standup_reference as core
from evaluate_standup_reference import ROOT, WORK, read, sha, write, Tee

PLAN = ROOT/'configs/experiments/random_recovery_reference.json'
OUT = ROOT/'records/reports/random-recovery'
BASE_CONFIG = core.config


def random_drop(env, env_ids, pose, drop_z_min, drop_z_max, euler_abs_max):
    """The pose argument is a collector label, not an orientation constraint."""
    from mjlab_microduck.tasks.mdp import set_random_drop_state
    assert pose == 'random-drop'
    set_random_drop_state(env, env_ids, drop_z_min=drop_z_min,
                         drop_z_max=drop_z_max, euler_abs_max=euler_abs_max)


def config(plan):
    cfg = BASE_CONFIG(plan)
    cfg.events['set_ground_state'].func = random_drop
    cfg.events['set_ground_state'].params = dict(pose='random-drop',
        drop_z_min=.234158, drop_z_max=.244158, euler_abs_max=float(np.pi))
    return cfg


def freeze():
    import torch
    assert not PLAN.exists()
    evidence = ROOT/'records/reports/lateral-recovery/coverage-analysis.json'
    assert read(evidence)['all_primary_conditions_passed']
    plan = read(ROOT/'configs/experiments/standup_reference.json')
    plan.update(frozen_at=datetime.now(timezone.utc).isoformat(),
        episodes_per_pose=256, seed=1003, poses=['random-drop'],
        scope='Development random-drop coverage; independent hold verification; no transition or hardware acceptance',
        random_drop_gate=dict(release_tilt_min_deg=45., minimum_fallen_episodes=128,
                             success_fraction_min=.95),
        release_policy='All 256 fixed proposals retained; primary fallen group uses tilt >45 degrees at 2 s before policy activation',
        rationale_evidence_sha256=sha(evidence))
    plan['source_sha256'].update({str(p.relative_to(ROOT)): sha(p) for p in
        (Path(__file__), ROOT/'scripts/validation/verify_standup_reference.py')})
    cfg = config(plan)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    write(PLAN, plan)
    write(OUT/'preflight.json', dict(status='passed', protocol_sha256=sha(PLAN),
        cuda_initialized=False, proposals=256, learning_updates=0,
        physics_step_seconds=cfg.sim.mujoco.timestep))
    print(json.dumps(plan, indent=2))


def proportion(entries):
    n = len(entries); k = sum(e['recovered_and_held'] for e in entries)
    if not n: return dict(attempts=0, successes=0, success_fraction=None, wilson_95=None)
    p = k/n; z = 1.959963984540054; den = 1+z*z/n
    middle = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(attempts=n, successes=k, success_fraction=p,
                wilson_95=[middle-half, middle+half])


def analyze():
    import verify_standup_reference as verifier
    verifier.OUT = OUT; verifier.main()
    state = read(OUT/'progress.json'); folder = Path(state['path']); plan = read(PLAN)
    assert sha(PLAN) == sha(folder/'protocol.json')
    report = read(folder/'evaluation.json'); condition = report['conditions'][0]
    release = round(plan['criteria']['freeze_seconds']/plan['criteria']['step_dt'])-1
    with np.load(folder/'random-drop-trace.npz') as trace:
        q = trace['quaternion'][release]
        tilt = np.rad2deg(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2), -1, 1)))
        active = trace['active'][release]
    gate = plan['random_drop_gate']; groups = defaultdict(list); details = []
    for i, e in enumerate(condition['episodes']):
        # A terminated release counts in the primary denominator as failure.
        fallen = tilt[i] > gate['release_tilt_min_deg'] or not active[i]
        row = dict(env_index=i, release_tilt_deg=float(tilt[i]), release_active=bool(active[i]),
            release_pose=e['release_pose'], fallen=bool(fallen),
            recovered_and_held=e['hold_start_seconds'] is not None and bool(active[i]),
            hold_start_seconds=e['hold_start_seconds'])
        details.append(row); groups['fallen' if fallen else 'not-fallen'].append(row)
    primary = proportion(groups['fallen']); secondary = proportion(groups['not-fallen'])
    passed = (primary['attempts'] >= gate['minimum_fallen_episodes'] and
              primary['success_fraction'] >= gate['success_fraction_min'])
    by_pose = {pose: proportion([e for e in details if e['release_pose']==pose])
               for pose in sorted({e['release_pose'] for e in details})}
    assert len(details) == 256 and len({e['env_index'] for e in details}) == 256
    result = dict(status='passed', primary_gate_passed=passed, primary_fallen=primary,
        secondary_not_fallen=secondary, release_pose_diagnostics=by_pose, episodes=details,
        all_proposals_retained=True, protocol_sha256=sha(PLAN),
        report_sha256=sha(folder/'evaluation.json'), script_sha256=sha(Path(__file__)),
        learning_updates=0, scope=plan['scope'],
        next_stage='Recovery-to-standing transition and independent confirmation' if passed else
                   'Analyze saved failed trajectories; no automatic rerun or learning')
    write(OUT/'random-analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}, indent=2))


def run():
    assert os.isatty(1), 'Use the single visible VS Code terminal'
    plan = read(PLAN)
    for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name)==digest, name
    for name,digest in plan['training_source_sha256'].items(): assert sha(WORK/name)==digest, name
    assert sha(core.MODEL)==plan['model_sha256']
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    assert not (OUT/'progress.json').exists(), 'Existing attempt retained; no automatic rerun'
    run_id='duck_getup_random-drop_reference_s1003_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    output=ROOT/'records/evaluations'/run_id; output.mkdir(); (output/'videos').mkdir()
    log=(output/'execution.log').open('a', buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(output),conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),
        model_sha256=sha(core.MODEL),learning_updates=0)
    write(OUT/'progress.json',state);write(output/'evaluation.json',state)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(output),run_id=run_id,pid=os.getpid()))
    shutil.copy2(PLAN,output/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,output/Path(name).name)
    core.OUT=OUT;core.config=config
    try:
        core.evaluate(plan,output,state);state['status']='completed'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat()
        write(output/'evaluation.json',state);write(OUT/'progress.json',state)
        print('RANDOM_RECOVERY:',state['status'],str(output),flush=True)
    analyze()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.analyze: analyze()
    else: run()
