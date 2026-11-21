"""Run frozen, fixed-weight adaptation diagnostics sequentially in one terminal.

No training or optimizer is invoked. Each variant gets a new simulator and its
own unchanged protocol, trajectories and video. Failure stops the batch instead
of hiding it or selecting a more favorable run. A preflight checks only the
runtime-filter path at 2 environments x 2 seconds; it is never gait acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    plan_path=ROOT/'configs/experiments/reference_adaptation_diagnostics.json'
    plan=json.loads(plan_path.read_text())
    mode='preflight' if args.preflight else 'comparison'
    batch_id=f'adaptation_{mode}_{datetime.now().strftime("%Y%m%dT%H%M%S")}_s{plan["seed"]}'
    output=ROOT/'records/evaluations'/batch_id
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(plan_path,output/'plan.json')
    shutil.copy2(__file__,output/Path(__file__).name)
    for name in ('verify_fixed_evaluation.py','verify_adaptation_diagnostics.py'):
        shutil.copy2(SCRIPTS/name,output/name)
    report=dict(batch_id=batch_id,status='running',mode=mode,pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),runs=[],
                training_started=False,hardware_verified=False)
    save(output/'batch.json',report)
    save(ROOT/'records/evaluations/current-adaptation-diagnostics.json',dict(path=str(output),pid=os.getpid(),batch_id=batch_id))
    variants=['runtime_filter'] if args.preflight else list(plan['variants'])
    try:
        for variant in variants:
            cmd=[sys.executable,'-B',str(SCRIPTS/'evaluate_fixed_policy.py'),'--diagnostic-variant',variant]
            if args.preflight: cmd.append('--preflight')
            print(f'\nDIAGNOSTIC {variant} ({len(report["runs"])+1}/{len(variants)}) | fixed weights | no training',flush=True)
            entry=dict(variant=variant,command=cmd,status='running')
            report['runs'].append(entry); save(output/'batch.json',report)
            # Inherit stdout/stderr from the visible VS Code task. No detached
            # worker, hidden terminal, service or alternate training process.
            process=subprocess.Popen(cmd)
            entry['pid']=process.pid
            save(output/'batch.json',report)
            try:
                entry['exit_code']=process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=20)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
            current=json.loads((ROOT/'records/evaluations/current-evaluation.json').read_text())
            assert current['pid']==process.pid,'Current evaluation does not belong to this foreground child'
            run=Path(current['path'])
            entry['path']=str(run.relative_to(ROOT))
            evaluation=json.loads((run/'evaluation.json').read_text())
            assert evaluation['diagnostic_variant']==variant
            assert entry['exit_code']==0 and evaluation['status']=='completed',evaluation.get('error')
            verify_cmd=[sys.executable,'-B',str(SCRIPTS/'verify_fixed_evaluation.py'),str(run)]
            entry['verification_command']=verify_cmd
            entry['verification_exit_code']=subprocess.run(verify_cmd).returncode
            assert entry['verification_exit_code']==0,'Independent metric/video verification failed'
            entry['status']='verified'
            save(output/'batch.json',report)
        report['status']='completed'
        save(output/'batch.json',report)
        # Simulation and independent per-run metrics are complete. The last
        # check compares raw configs and replays the actual archived runtime.
        check_cmd=[sys.executable,'-B',str(SCRIPTS/'verify_adaptation_diagnostics.py'),str(output)]
        report['adaptation_verification_command']=check_cmd
        report['adaptation_verification_exit_code']=subprocess.run(check_cmd).returncode
        assert report['adaptation_verification_exit_code']==0,'Adaptation verification failed'
    except BaseException as exc:
        report['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        report['error']=f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        save(output/'batch.json',report)
        print(f'DIAGNOSTICS {report["status"]}: {output}',flush=True)
    raise SystemExit(0 if report['status']=='completed' else 1)


if __name__=='__main__': main()
