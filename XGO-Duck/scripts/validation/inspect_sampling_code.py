"""Archive the installed sampling path and the saved continuation configuration.

Static inspection only: no simulator, CUDA initialization, training or profiler.
Code locations identify what to measure next, not where wall time was spent.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    record = root/'records/experiments'/args.run_id
    assert record.parent == root/'records/experiments' and record.is_dir()
    request = json.loads((record/'training-request.json').read_text())
    live = json.loads((record/'resume-live-check.json').read_text())
    params = Path(live['log_dir'])/'params'
    site = work/'.venv/lib/python3.12/site-packages'
    assert sha(work/'src/mjlab_microduck/tasks/__init__.py') == request['task_source_sha256']
    sources = {
        'runner.py':site/'rsl_rl/runners/on_policy_runner.py',
        'logger.py':site/'rsl_rl/utils/logger.py',
        'rsl_utils.py':site/'rsl_rl/utils/utils.py',
        'environment.py':site/'mjlab/envs/manager_based_rl_env.py',
        'simulation.py':site/'mjlab/sim/sim.py',
        'task_mdp.py':work/'src/mjlab_microduck/tasks/mdp.py',
        'resume_official.py':root/'scripts/training/resume_official.py',
        'env.yaml':params/'env.yaml', 'agent.yaml':params/'agent.yaml',
    }
    env = yaml.load(sources['env.yaml'].read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load(sources['agent.yaml'].read_text(), Loader=yaml.BaseLoader)
    assert agent['run_name'] == args.run_id
    assert (int(env['decimation']),int(agent['num_steps_per_env']),int(env['scene']['num_envs'])) == (4,24,1024)
    assert 'projected_gravity' in env['observations']['actor']['terms']
    assert 'raw_accelerometer' not in env['observations']['actor']['terms']
    assert 'raw_accelerometer' not in env['observations']['critic']['terms']
    output = record/'sampling-code-audit'
    output.mkdir(exist_ok=True)
    manifest = {}
    for name, source in sources.items():
        shutil.copy2(source, output/name)
        assert sha(source) == sha(output/name)
        manifest[name] = dict(original_path=str(source),sha256=sha(source))

    def function(name, method):
        text = sources[name].read_text()
        nodes = [node for node in ast.walk(ast.parse(text)) if isinstance(node,ast.FunctionDef) and node.name == method]
        assert len(nodes) == 1, (name,method,len(nodes))
        node = nodes[0]
        return dict(file=name,function=method,line_start=node.lineno,line_end=node.end_lineno,
                    source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno]))

    excerpts = [function(name,method) for name,method in (
        ('runner.py','learn'),('logger.py','process_env_step'),('rsl_utils.py','check_nan'),
        ('environment.py','step'),('environment.py','render'),('simulation.py','step'),
        ('simulation.py','sense'),('simulation.py','_should_use_cuda_graph'),('task_mdp.py','raw_accelerometer'))]
    learn = excerpts[0]['source']
    phrases = ['actions = self.alg.act(obs)','self.env.step(','check_nan(obs, rewards, dones)',
               'self.alg.process_env_step(','self.logger.process_env_step(','collect_time = stop - start']
    positions = [learn.index(phrase) for phrase in phrases]
    assert positions == sorted(positions)
    assert 'self.cfg.get("check_for_nan", True)' in learn
    logger = excerpts[1]['source']
    assert 'nonzero(as_tuple=False)' in logger and '.cpu().numpy().tolist()' in logger
    step = excerpts[3]['source']
    assert 'range(self.cfg.decimation)' in step and 'self.sim.step()' in step
    assert 'self.reset_buf.nonzero(as_tuple=False)' in step
    assert all(phrase in step for phrase in ('self.reward_manager.compute','self.sim.forward()',
                                           'self.sim.sense()','self.observation_manager.compute'))
    assert 'wp.capture_launch(self.step_graph)' in excerpts[5]['source']
    (output/'source-excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
    result = dict(status='static-source-and-config-checked', checked_at=datetime.now(timezone.utc).isoformat(),
                  run_id=args.run_id, source_files=manifest, excerpts_sha256=sha(output/'source-excerpts.json'),
                  rollout_environment_steps=24, physics_substeps_per_environment_step=4,
                  batched_physics_steps_per_rollout=96, parallel_environments=1024,
                  runner_nan_check_enabled=str(agent.get('check_for_nan','true')).lower()=='true',
                  actor_gravity_term='projected_gravity', raw_accelerometer_term_active=False,
                  collection_includes_policy_environment_nan_checks_storage_and_logger=True,
                  code_supports_cuda_graphs=True, live_cuda_graph_state_observed=False,
                  cpu_copy_and_nonzero_sites_present=True, measured_component_timings=False,
                  bottleneck_identified=False, controlled_speedup_verified=False,
                  primary_api_reference='https://docs.pytorch.org/docs/2.9/generated/torch.nonzero.html',
                  next_measurement='After baseline completion, profile a separate visible run by component; preserve the frozen host benchmark for throughput comparisons and record profiler overhead.',
                  scope='Installed source and saved configuration only; no runtime profiling or performance intervention')
    (output/'check.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,output/Path(__file__).name)
    print(json.dumps({key:value for key,value in result.items() if key!='source_files'},indent=2))


if __name__ == '__main__':
    main()
