"""Exercise profiler boundaries/restoration with CPU toy calls, not simulation.

The actual checkpoint/configuration preflight is supplied separately. This
check does not construct a robot, train a policy, or claim GPU attribution.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
import profile_components as entry


def fake_runner():
    import torch
    token = object()
    def small_cpu_call(*args, **kwargs):
        return torch.ones(2)+0.5
    body = SimpleNamespace(sim=SimpleNamespace(step=small_cpu_call, forward=small_cpu_call, sense=small_cpu_call),
                           reward_manager=SimpleNamespace(compute=small_cpu_call),
                           termination_manager=SimpleNamespace(compute=small_cpu_call),
                           observation_manager=SimpleNamespace(compute=small_cpu_call), _reset_idx=small_cpu_call)
    def step(action):
        for _ in range(4):
            body.sim.step()
        body.sim.forward()
        body.sim.sense()
        body.reward_manager.compute()
        body.termination_manager.compute()
        body.observation_manager.compute()
        return action
    env = SimpleNamespace(unwrapped=body, step=step)
    alg = SimpleNamespace(act=small_cpu_call, process_env_step=small_cpu_call,
                          compute_returns=small_cpu_call, update=small_cpu_call)
    logger = SimpleNamespace(process_env_step=small_cpu_call, log=lambda **kw: token)
    module = SimpleNamespace(check_nan=small_cpu_call)
    runner = SimpleNamespace(env=env, alg=alg, logger=logger)
    objects = [alg, env, body, body.sim, body.reward_manager, body.termination_manager,
               body.observation_manager, logger, module]
    return runner, module, token, objects


def main():
    import torch
    out = ROOT/'records/reports/profile-entry-check'
    out.mkdir(parents=True, exist_ok=True)
    readiness = entry.read(out/'input-readiness.json')
    assert readiness['status'] == 'inputs-verified-only' and not readiness['cuda_initialized']
    assert readiness['configuration_matches'] == {'env': True, 'agent': True}
    for key, expected in readiness['source_sha256'].items():
        assert entry.sha(Path(readiness['source_paths'][key])) == expected
    protocol = entry.read(entry.PROTOCOL)
    folder = out/('cpu-profiler-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir()
    runner, module, token, objects = fake_runner()
    before = [(obj, vars(obj).copy()) for obj in objects]
    ticks = []
    options = protocol['profiler']
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
          schedule=torch.profiler.schedule(**{k: options[k] for k in ('wait','warmup','active','repeat')}),
          on_trace_ready=lambda prof: entry.save_trace(prof, folder, protocol, require_cuda=False)) as prof:
        with entry.instrument(runner, module, prof, lambda info: ticks.append(info['it'])):
            for iteration in range(5000, 5027):
                for _ in range(24):
                    action = runner.alg.act(None)
                    assert runner.env.step(action) is action
                    module.check_nan(None)
                    runner.alg.process_env_step(None)
                    runner.logger.process_env_step(None)
                runner.alg.compute_returns(None)
                runner.alg.update()
                assert runner.logger.log(it=iteration) is token
    assert ticks == list(range(5000, 5027))
    for obj, attributes in before:
        assert vars(obj) == attributes
    summary = entry.read(folder/'profile-summary.json')
    assert summary['status'] == 'passed' and summary['cuda_event_count'] == 0
    events = entry.read(folder/'trace.json')['traceEvents']
    counts = Counter(event.get('name') for event in events if event.get('ph') == 'X')
    expected = protocol['expected_captured_calls']
    assert {key: counts[key] for key in expected} == expected

    class ExpectedFailure(Exception):
        pass
    def fail(*args, **kwargs):
        raise ExpectedFailure('injected fixture error')
    runner, module, _, objects = fake_runner()
    runner.env.unwrapped.sim.step = fail
    before = [(obj, vars(obj).copy()) for obj in objects]
    try:
        with entry.instrument(runner, module, SimpleNamespace(step=lambda: None), lambda info: None):
            runner.env.step(None)
    except ExpectedFailure:
        pass
    else:
        raise AssertionError('Fixture exception was swallowed')
    for obj, attributes in before:
        assert vars(obj) == attributes

    class Inherited:
        def call(self):
            return token
    obj = Inherited()
    try:
        with entry.replaced(obj, 'call', fail):
            obj.call()
    except ExpectedFailure:
        pass
    assert 'call' not in vars(obj) and obj.call() is token
    assert not torch.cuda.is_initialized()
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  checks=['actual-parent-config-preflight', 'cpu-profiler-window-and-raw-scope-counts',
                          'return-values-and-normal-restoration', 'exception-and-inherited-method-restoration'],
                  captured_scope_counts={key: counts[key] for key in expected},
                  cpu_trace=str((folder/'trace.json').relative_to(ROOT)), cpu_trace_sha256=entry.sha(folder/'trace.json'),
                  input_readiness_sha256=entry.sha(out/'input-readiness.json'),
                  entry_sha256=entry.sha(ROOT/'scripts/training/profile_components.py'),
                  verifier_sha256=entry.sha(Path(__file__)), cuda_initialized=False,
                  simulation_created=False, optimizer_steps_executed=0,
                  gpu_profile_verified=False, real_profile_started=False)
    entry.write(out/'check.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
