"""Freeze and run a bounded forward-startup pair without policy learning."""

from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile, disk_usage
import sys
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_routed_getup_sequence as parent
from routed_forward_entry_ramp import ROLES, collect, command_table


base, route = parent.base, parent.route
ROOT, read, sha, write = parent.ROOT, parent.read, parent.sha, parent.write
DRAFT = ROOT / 'configs/experiments/continuous_routed_forward_entry_ramp_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_forward_entry_ramp')
OUT = ROOT / 'records/reports/continuous-routed-forward-entry-ramp'
PAGE = ROOT / 'records/continuous-routed-forward-entry-ramp-live.html'
STAGE, SEED = 'S5-routed-forward-entry-ramp', 1171


def phase(protocol):
    parent.require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['development_seed'] == SEED and SEED not in d['excluded_for_tuning']
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io',
                                       'automatic_retry', 'automatic_extension'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in item.name for item in (ROOT / folder).iterdir())
    source, scoring, review = [route.pinned(d[key]) for key in ('source_protocol', 'development_score', 'failure_localization')]
    assert source == parent.PLAN and read(scoring)['protocol_sha256'] == read(review)['protocol_sha256'] == sha(source)
    assert read(review)['analysis_sha256'] == sha(scoring) and not read(scoring)['score']['passed_development_gate']
    spec = read(source)
    p, s = spec['environment'], spec['sequence']
    n, count = d['worlds'], d['shared_controls'] + len(ROLES) * d['forward_controls_per_branch']
    assert n == p['num_envs'] == sum(d['requested_pose_counts']) and count == d['maximum_unique_controls']
    assert n * count == d['maximum_transitions'] and count * 5 == d['maximum_physics_substeps']
    assert s['boundary_controls'] == [100, 550, d['shared_controls']] and d['control_hz'] == 50
    assert d['requested_pose_counts'] == spec['dual_route']['requested_pose_counts']
    assert all(d['branches'][key]['external_forward_m_s'] == s['controller']['forward_command_m_per_s'] for key in ROLES)
    a, b = [command_table(d['branches'][key], d['forward_controls_per_branch']) for key in ROLES]
    ramp = d['branches'][ROLES[1]]['ramp_controls']
    assert d['branches'][ROLES[0]]['ramp_controls'] == 0 and ramp == 25
    np.testing.assert_array_equal(a[ramp-1:], b[ramp-1:])
    assert (b[:ramp-1, 0] < a[:ramp-1, 0]).all() and (a[:, 1:] == 0).all() and (b[:, 1:] == 0).all()
    assert disk_usage(ROOT).free >= 8 * 1024 ** 3
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    run_id = f'duck_switch_forward-entry-ramp_development_s{SEED}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=groups))
    spec['dual_route']['seed'] = SEED
    for policy in (*spec['policies'].values(), spec['common_fallback'], *spec['switch_policies'].values()):
        policy['seed'] = SEED
    spec.update(status='frozen-routed-forward-entry-ramp', stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
        maximum_unique_controls=count, maximum_physics_substeps=d['maximum_physics_substeps'],
        maximum_sampling_wall_seconds=540, pair=d)
    spec['draft']['gpu_memory_fraction_cap'] = d['maximum_gpu_memory_fraction']
    spec['references'].update({base.rel(path): sha(path) for path in (DRAFT, source, scoring, review)})
    for path in (Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve() for name in (
            'routed_forward_entry_ramp.py', 'verify_routed_forward_entry_ramp.py', 'verify_walk_live_page.py',
            'getup_release_scene_state.py', 'getup_release_fork_state.py')]):
        spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    cfg, _, _ = parent.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and not cfg.env.auto_reset
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    torch.set_num_threads(4)
    boundaries = parent.boundary_fixture(spec, parent.switch_models(spec, 'cpu'), 'cpu')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN), physical_substeps=0,
        controls_per_arm=d['forward_controls_per_branch'], shared_controls=d['shared_controls'], cpu_scene_exact=True,
        switch_fixtures=boundaries, mapped_step=a[:ramp+1].tolist(), mapped_ramp=b[:ramp+1].tolist()))
    (record / 'README.md').write_text('# 前进起步对照\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-forward-entry-ramp-live.html)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='同一站立状态下的前进起步对照', run_id=run_id,
        controls=0, maximum_controls=count, transitions=0, updates=[], learning_updates=0))
    print('ROUTED_ENTRY_RAMP_FROZEN', base.rel(record), sha(PLAN), flush=True)


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    phase(PLAN)
    spec = read(PLAN)
    base.shared.assert_sources(spec)
    record = ROOT / spec['record']
    assert not (record / 'evaluation.json').exists() and not (OUT / 'native-zero.json').exists()
    assert sha(PLAN) == sha(record / 'protocol.json') == read(OUT / 'freeze.json')['protocol_sha256']
    state, ready = [read(record / name) for name in ('state.json', 'windows-observer-ready.json')]
    assert ready['status'] == 'starting-page-observed' and ready['protocol_sha256'] == sha(PLAN)
    assert ready['page_sha256'] == sha(PAGE) and ready['initial_state_sha256'] == sha(record / 'state.json')
    assert state['status'] == 'starting' and disk_usage(ROOT).free >= 8 * 1024 ** 3
    p = spec['environment']
    state.update(actual=dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=0),
        protocol_sha256=sha(PLAN), phase='gpu-zero', heldout_used=False, hardware_io=False)
    watch = base.MemoryWatch(spec['pair']['maximum_gpu_memory_fraction'])
    last, timestamp = 0, monotonic()

    def guard():
        phase(PLAN)
        watch.check()

    def publish():
        nonlocal last, timestamp
        k, now = state['actual']['controls'], monotonic()
        if k > last:
            state['updates'].append(dict(control=k, transitions_per_second=(k-last)*p['num_envs']/(now-timestamp)))
            last, timestamp = k, now
        state.update(controls=k, transitions=k*p['num_envs'])
        write(record / 'evaluation.json', state)
        base.publish(record, PAGE, state)

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = parent.prepare(p)
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (record / 'execution.log').open('x', buffering=1) as log, \
                redirect_stdout(base.Tee(sys.stdout, log)), redirect_stderr(base.Tee(sys.stderr, log)):
            write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), protocol_sha256=sha(PLAN)))
            env = None
            try:
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=SEED)
                state['actual']['resets'] = 1
                memory = base.shared.make_memory(p, meta, env.device)
                actors, proof = route.prior.qualify(env, obs, memory, spec, record)
                independent = read(OUT / 'independent-zero/proof.json')
                assert proof['sequence_boundaries'] == independent['sequence_boundaries']
                guard()
                watch.sample()
                write(OUT / 'native-zero.json', dict(status='passed-native-zero', proof=proof, gpu=watch.summary(),
                    qualification_sha256=sha(record / 'zero-qualification.json')))
                walking = parent.switch_models(spec, env.device)
                state['status'] = 'running'
                base.shared.project_running(True)
                publish()
                collect(env, actors, walking, memory, spec, record, state, publish, guard)
                count, substeps = spec['maximum_unique_controls'], spec['maximum_physics_substeps']
                assert state['actual'] == dict(controls=count, attempted_controls=count, substeps=substeps,
                    attempted_substeps=substeps, resets=1)
                guard()
                state.update(status='completed', phase='independent-score-pending')
            except BaseException as error:
                state.update(status='stopped-review-required', error=repr(error))
                raise
            finally:
                if env is not None:
                    env.close()
                state['gpu'] = watch.summary()
                write(record / 'gpu-telemetry.json', watch.rows)
                base.shared.project_running(False)
                publish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    with patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=parent.snapshot), \
         patch.object(base.shared, 'prepare', parent.prepare), patch.object(base.shared, 'actor_model', route.guarded_actor):
        (route.prior.zero if mode == 'zero' else sample)()


if __name__ == '__main__':
    main()
