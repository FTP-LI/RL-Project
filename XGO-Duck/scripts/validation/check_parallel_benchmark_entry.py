"""Check actual scaling configs and rejection paths without GPU simulation."""
import ast
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
import benchmark_parallel_envs as entry


def main():
    import torch
    import yaml
    cases=[]
    for n in (1024,2048,4096):
        cfg,protocol,plan,evidence=entry.prepare(n,'scale_check_only')
        assert plan['expected_final_observation_count']==122880000+40*24*n
        cases.append(dict(num_envs=n,changes=evidence['configuration_changes'],plan=plan))
    after=yaml.load(yaml.dump({'env':asdict(cfg.env),'agent':asdict(cfg.agent)}),Loader=yaml.BaseLoader)
    before=copy.deepcopy(after)
    before['agent']['max_iterations']='27'
    before['env']['scene']['num_envs']=before['env']['scene']['terrain']['num_envs']='1024'
    rejected=[]
    edits=[('reward',lambda v:v['env']['rewards']['track_linear_velocity'].__setitem__('weight','3.3')),
           ('seed',lambda v:v['agent'].__setitem__('seed','43')),
           ('budget',lambda v:v['agent'].__setitem__('max_iterations','41')),
           ('terrain-count',lambda v:v['env']['scene']['terrain'].__setitem__('num_envs','2048'))]
    for name,mutate in edits:
        changed=copy.deepcopy(after)
        mutate(changed)
        try:
            entry.validate_changes(before,changed,4096,40)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Undeclared change accepted: '+name)
    with tempfile.TemporaryDirectory(prefix='xgo-scale-pause-') as temporary:
        root=Path(temporary)
        (root/'project.json').write_text(json.dumps({'formal_training':{'status':'paused-by-user'}}))
        try:
            entry.require_training_enabled(root)
        except SystemExit:
            rejected.append('paused-project')
        else:
            raise AssertionError('Paused project accepted')
    telemetry=entry.telemetry_sample()
    assert telemetry.get('gpu_exit_code')==0 and 'gpu' in telemetry and 'gpu_error' not in telemetry
    tasks=json.loads((ROOT/'.vscode/tasks.json').read_text())
    task=next(t for t in tasks['tasks'] if t['label'].startswith('XGO-Duck: 23 '))
    assert tasks['presentation']['panel']=='shared' and tasks['presentation']['reveal']=='always'
    assert task['runOptions']['instanceLimit']==1
    shell=ROOT/'scripts/validation/run_parallel_benchmark.sh'
    subprocess.run(['bash','-n',str(shell)],check=True)
    paths=[Path(entry.__file__),Path(__file__),ROOT/'scripts/validation/verify_parallel_benchmark.py']
    for path in paths:
        ast.parse(path.read_text())
    assert not torch.cuda.is_initialized()
    output=ROOT/'records/reports/parallel-entry-check'
    output.mkdir(exist_ok=True)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),actual_configs=cases,
                rejected= rejected,telemetry_probe=telemetry,shared_task_checked=True,
                cuda_initialized=False,simulation_created=False,optimizer_updates=0,
                protocol_sha256=entry.sha(entry.PROTOCOL),
                source_sha256={str(p.relative_to(ROOT)):entry.sha(p) for p in [*paths,shell,ROOT/'.vscode/tasks.json']})
    entry.write(output/'check.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
