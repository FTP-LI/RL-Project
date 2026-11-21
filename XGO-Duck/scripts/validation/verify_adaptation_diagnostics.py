"""Independently check single-factor scope, runtime math and saved diagnostics.

Reads saved traces and runs CPU ONNX only. The generic verifier must already
have recomputed metrics and decoded videos. This adds exact config-difference
checks, raw-action history, runtime Policy.infer replay and requested targets.
Reported changes are same-seed diagnostics, not training or hardware effects.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--resume-verification',action='store_true',
                        help='Recheck completed runs after a recorded verifier-only interruption; never rerun simulation.')
    args=parser.parse_args()
    batch_dir=args.batch.resolve()
    batch=read(batch_dir/'batch.json'); plan=read(batch_dir/'plan.json')
    recovering=batch['status']=='verification-interrupted'
    assert batch['status']=='completed' or (args.resume_verification and recovering)
    if recovering:
        assert batch.get('verification_interruption')
        # WSL may stop between visible tasks and reuse PIDs on its next boot.
        # A numeric PID alone cannot identify the old supervisor.
        old_cmd=Path(f'/proc/{batch["pid"]}/cmdline')
        if old_cmd.exists():
            try:
                words=old_cmd.read_bytes().decode(errors='replace').split('\0')
            except FileNotFoundError:
                words=[]
            assert not any(Path(word).name=='run_adaptation_diagnostics.py' for word in words if word),'Original supervisor is still live'
        assert all(r['status']=='verified' and r['exit_code']==0 and r['verification_exit_code']==0 for r in batch['runs'])
        shutil.copy2(__file__,batch_dir/'verify_adaptation_diagnostics.recovery.py')
    assert sha(batch_dir/'plan.json')==batch['plan_sha256']
    expected=['runtime_filter'] if batch['mode']=='preflight' else list(plan['variants'])
    assert [r['variant'] for r in batch['runs']]==expected
    runtime=ROOT/'official_resources/xgoduck_runtime_arduino/python'
    # Resolve the pinned model within this copy of the project so saved WSL
    # records can be inspected after copying the project to native Ubuntu.
    model=runtime/'xgoduck_walk.onnx'
    sys.path.insert(0,str(runtime))
    import rl_core
    from policy import Policy
    assert Path(rl_core.__file__).parent==runtime
    checks=[]; rows=[]; base_cfg=None; base_initial={}
    for entry in batch['runs']:
        run=ROOT/entry['path']; report=read(run/'evaluation.json')
        verified=read(run/'verification.json')
        assert entry['status']=='verified' and verified['status']=='passed'
        assert report['model_sha256']==plan['model_sha256']==sha(model)
        assert report['diagnostic_plan_sha256']==batch['plan_sha256']
        assert report['protocol_sha256']==plan['formal_protocol_sha256']
        assert report['diagnostic_settings']==plan['variants'][entry['variant']]
        assert not report['training_started'] and not report['hardware_verified']
        assert not any(r['formal_acceptance_eligible'] for r in report['conditions'])
        settings=report['diagnostic_settings']
        # Raw YAML comparison also preserves callable tags, not only values.
        # The gain diagnostic must have exactly one text substitution.
        cfg=(run/'env.yaml').read_bytes()
        if entry['variant']=='reference_repeat': base_cfg=cfg
        differences=[]
        if base_cfg is not None:
            wanted=base_cfg
            if entry['variant']=='actuator_kp6':
                assert base_cfg.count(b'kp_fw: 5.0')==1
                wanted=base_cfg.replace(b'kp_fw: 5.0',b'kp_fw: 6.0')
                differences=['scene.entities.robot.articulation.actuators[0].kp_fw: 5.0 -> 6.0']
            assert cfg==wanted,f'Unexpected environment change: {entry["variant"]}'
        policy=Policy()
        ok,msg=policy.load(str(model),use_model_home=settings['home']=='model'); assert ok,msg
        if settings['action_filter']=='none': policy.set_action_alphas(leg=0,head=0)
        home=np.deg2rad(policy.home_deg)
        np.testing.assert_allclose(home,report['policy_home'],atol=1e-7,rtol=0)
        expected_alpha=[policy.action_alpha]*5+[policy.action_alpha_head]*5+[policy.action_alpha]*4
        np.testing.assert_allclose(report['action_filter_alpha'],expected_alpha,atol=2e-8,rtol=0)
        obs_error=0.; applied_error=0.; replay_steps=0; trace_hashes={}
        for item,video_check in zip(report['conditions'],verified['checks']):
            condition=item['condition']; trace_path=run/f'{condition}-trace.npz'
            assert sha(trace_path)==video_check['trace_sha256']
            trace_hashes[condition]=sha(trace_path)
            # NpzFile.__getitem__ decompresses the entire member on every call.
            # Load each member once before per-step replay; values/checks stay
            # identical, and the archive is closed before the CPU analysis.
            with np.load(trace_path) as archive:
                d={key:archive[key] for key in archive.files}
            live=d['active']
            for key in ('observations','actions','applied_actions','requested_joint_targets','joint_positions','joint_velocities'):
                assert np.isfinite(d[key][live]).all()
            np.testing.assert_allclose(d['observations'][0,:,34:48],0,atol=0,rtol=0)
            pairs=live[1:] & live[:-1]
            np.testing.assert_allclose(d['observations'][1:,:,34:48][pairs],d['actions'][:-1][pairs],atol=0,rtol=0)
            requested=home+policy.action_scale*d['applied_actions']
            np.testing.assert_allclose(requested[live],d['requested_joint_targets'][live],atol=2e-7,rtol=0)
            # Independently apply the archived runtime to every recorded env-0
            # input through its FIRST episode. This is a replay of saved states,
            # not a new closed-loop rollout or an imitation of firmware sensors.
            policy.reset_action_filter()
            for step in range(len(live)):
                if not live[step,0]: break
                x=d['observations'][step,0]
                angles=np.zeros(15,dtype=np.float32); vels=np.zeros(15,dtype=np.float32)
                angles[rl_core.POLICY_INDEX]=np.rad2deg(x[6:20]+home)
                vels[rl_core.POLICY_INDEX]=np.rad2deg(x[20:34])
                policy.prev_action[:]=x[34:48]
                actual=policy.infer(x[3:6],x[:3],angles,vels,x[48:51],x[51:55])
                obs_error=max(obs_error,float(np.max(np.abs(policy._obs-x))))
                applied_error=max(applied_error,float(np.max(np.abs(actual-d['applied_actions'][step,0]))))
                replay_steps+=1
            # All environments: independent EMA and requested targets, including
            # every still-live first episode; reset-only samples stay excluded.
            filtered=np.zeros_like(d['actions'][0]); alpha=np.asarray(expected_alpha,dtype=np.float32)
            for step in range(len(live)):
                filtered=alpha*filtered+(1-alpha)*d['actions'][step]
                np.testing.assert_allclose(filtered[live[step]],d['applied_actions'][step][live[step]],atol=1e-7,rtol=0)
            if entry['variant']=='reference_repeat':
                base_initial[condition]=(d['initial_xy'].copy(),d['initial_joint_positions'].copy())
            elif base_initial:
                # Initial kinematics must match; stochastic streams may later
                # diverge after a termination, so no stronger pairing is claimed.
                np.testing.assert_allclose(d['initial_xy'],base_initial[condition][0],atol=0,rtol=0)
                np.testing.assert_allclose(d['initial_joint_positions'],base_initial[condition][1],atol=0,rtol=0)
            residual=d['requested_joint_targets']-d['joint_positions']
            row=dict(variant=entry['variant'],condition=condition,
                     full_survival=item['full_episode_survival'],falls=item['fall_count_excluding_timeouts'],
                     xy_error=item['linear_xy_mae_m_s'],yaw_error=item['yaw_mae_rad_s'],
                     displacement_median=item['final_displacement_m_median'],
                     raw_action_delta_rms=item['action_delta_rms'],foot_slip=item['contact_foot_slip_m_s'],
                     requested_target_tracking_rms_rad=float(np.sqrt(np.mean(residual[live]**2))),
                     joint_speed_rms_rad_s=float(np.sqrt(np.mean(d['joint_velocities'][live]**2))),
                     meets_numeric_criteria=item['meets_numeric_criteria'],run=entry['path'])
            rows.append(row)
        assert obs_error<2e-6 and applied_error<2e-5,(obs_error,applied_error)
        checks.append(dict(variant=entry['variant'],run=entry['path'],config_differences=differences,
                           runtime_replay_steps=replay_steps,max_observation_error=obs_error,
                           max_applied_action_error=applied_error,trace_sha256=trace_hashes,
                           evaluation_sha256=sha(run/'evaluation.json'),verification_sha256=sha(run/'verification.json')))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),batch_id=batch['batch_id'],
                command=[sys.executable,*sys.argv],exit_code=0,checks=checks,rows=rows,
                source_sha256={str(p):sha(p) for p in (Path(__file__),model,runtime/'rl_core.py',runtime/'policy.py',batch_dir/'plan.json')},
                scope='Same-seed fixed-weight simulation diagnostics; no held-out or hardware acceptance',
                verification_resumed_after_interruption=recovering,
                root_cause_resolved=False,training_started=False,hardware_verified=False)
    (batch_dir/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (batch_dir/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if recovering:
        # Preserve the interruption record and original batch snapshot. Mark
        # completed only after every original check has succeeded on saved data.
        batch.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat(),
                     adaptation_verification_command=[sys.executable,*sys.argv],
                     adaptation_verification_exit_code=0,
                     adaptation_verification_source_sha256=sha(Path(__file__)))
        (batch_dir/'batch.json').write_text(json.dumps(batch,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','rows','checks')},indent=2))
    for c in checks: print(c['variant'],'runtime steps',c['runtime_replay_steps'],'max applied error',c['max_applied_action_error'])
    for r in rows: print(r['variant'],r['condition'],'survival',r['full_survival'],'xy',round(r['xy_error'],5),'yaw',round(r['yaw_error'],5),'pass',r['meets_numeric_criteria'])


if __name__=='__main__': main()
