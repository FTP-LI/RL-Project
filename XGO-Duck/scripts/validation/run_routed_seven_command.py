"""Freeze and execute one routed getup–stand–seven-command development batch."""

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from math import ceil, cos, radians
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_routed_seven_command_cpu import DRAFT, OUT, ROOT, prior, read, sha, sources, write
from check_routed_seven_command_cuda import target_fixture
from routed_getup_sequence import FORWARD
from routed_seven_command_controller import SevenCommandSequenceController, balanced_groups


original, base, route = prior.original, prior.base, prior.route
PLAN = DRAFT.with_stem('continuous_routed_seven_command')
PAGE = ROOT/'records/continuous-routed-seven-command-live.html'
STAGE = 'S5-routed-seven-command'
ORIGINAL_SWITCH = original.switch_models
ORIGINAL_SNAPSHOT = original.snapshot


def phase(protocol):
    original.require_stage(protocol, STAGE+'-preparation' if protocol == DRAFT else STAGE)


def route_models(spec, device):
    standing = route.guarded_actor(spec['switch_policies']['stand'], device)
    bank = {'base': standing}
    for role, policy in spec['seven_command']['policies'].items():
        if role != 'base':
            bank[role] = route.guarded_actor(policy, device)
    return dict(stand=standing, forward=torch.nn.ModuleDict(bank))


def freeze():
    from mjlab.scene import Scene
    from batched_velocity_confirmation import prepare as prepare_walking

    phase(DRAFT)
    d, spec, delivery, paths = sources()
    cpu, cuda = (OUT/name for name in ('preflight.json', 'cuda-zero.json'))
    evidence = read(cuda)
    assert evidence['status'] == 'passed-cuda-seven-route-zero' and evidence['physical_substeps'] == 0
    assert evidence['worlds'] == d['worlds'] and evidence['development_seed'] == d['development_seed']
    assert all(sha(ROOT/name) == digest for name, digest in evidence['source_sha256'].items())
    assert not any(path.exists() for path in (PLAN, PAGE, OUT/'freeze.json'))
    assert not torch.cuda.is_initialized() and disk_usage(ROOT).free >= 12*1024**3
    seed, n, total = (d[key] for key in ('development_seed', 'worlds', 'maximum_unique_controls'))
    assert n*total == d['maximum_transitions'] and total*5 == d['maximum_batched_physics_substeps']
    for name in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/name).iterdir())
    poses = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    pose_groups = {pose: np.flatnonzero(poses == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    commands = delivery['commands']
    groups = balanced_groups(poses, commands, seed+70000)
    witness = read(cpu)
    assert witness['command_order'] == list(groups) and witness['counts'] == list(map(len, groups.values()))
    assert witness['requested_pose_counts_by_command'] == [np.bincount(poses[ids], minlength=4).tolist() for ids in groups.values()]
    run_id = f'duck_switch_routed-seven-command_development_s{seed}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT/'records/evaluations'/run_id
    p, s = spec['environment'], spec['sequence']
    assert d['boundaries'] == s['boundary_controls'] + [total]
    p.update(seed=seed, stage=STAGE, run_id=run_id, split=dict(pose_codes=poses.tolist(), groups=pose_groups))
    spec['dual_route']['seed'] = seed
    for policy in (*spec['policies'].values(), spec['common_fallback'], *spec['switch_policies'].values()):
        policy['seed'] = seed
    policies = {}
    for role, details in delivery['models'].items():
        model = details['files']['onnx']
        route.pinned((model['path'], model['sha256']))
        policies[role] = dict(model=model['path'], model_sha256=model['sha256'], seed=seed,
                              home=spec['switch_policies']['stand']['home'])
    assert policies['base'] == spec['switch_policies']['stand'] and policies['official'] == spec['switch_policies']['forward']
    spec.pop('holdout')
    walk_case, = read(paths[-1])['cases'].values()
    walking_fall = prepare_walking(walk_case['environment']).terminations['fell_over']
    assert walking_fall.func.__name__ == 'bad_orientation'
    assert abs(walking_fall.params['limit_angle']-radians(70.)) < 1e-9
    spec.update(status='frozen-before-dynamics', stage=STAGE, data_scope='new-development', reserved_holdout=False,
                record=base.rel(record), page=base.rel(PAGE), seven_command=dict(groups=groups, commands=commands,
                routes=delivery['route_by_command'], policies=policies,
                minimum_by_command={name:ceil(d['minimum_complete_fraction_per_command']*len(ids)) for name, ids in groups.items()},
                minimum_overall=ceil(d['minimum_complete_fraction_overall']*n),
                maximum_transitions=d['maximum_transitions'], stop_if_entry_upper_bound_below_gate=True,
                minimum_root_up_z_for_walk_survival=cos(walking_fall.params['limit_angle']),
                scoring='Original 20 s per-command gates on reached worlds. Full sequence gate includes all assigned worlds.',
                auto_retry=False, auto_extend=False, video_command=next(name for name, ids in groups.items() if 0 in ids)))
    assert spec['draft']['gpu_memory_fraction_cap'] == d['maximum_gpu_memory_fraction']
    spec['references'].update({base.rel(path): sha(path) for path in (*paths, cpu, cuda)})
    for name in ('run_routed_seven_command.py', 'routed_seven_command_controller.py',
                 'check_routed_seven_command_cuda.py', 'verify_routed_seven_command.py', 'routed_seven_command_score.py'):
        path = Path(__file__).with_name(name).resolve()
        spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    cfg, _, _ = original.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and cfg.env.decimation == 5
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    assert not torch.cuda.is_initialized()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record/'protocol.json')
    write(OUT/'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN), physical_substeps=0,
        command_sizes=witness['counts'], cpu_scene_exact=True, development_seed=seed))
    (record/'README.md').write_text('# 起身接七指令开发评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-seven-command-live.html) · '
        '[执行日志](execution.log) · [实验索引](../../reports/continuous-routed-seven-command/README.md)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身接七指令连续评估', run_id=run_id,
        controls=0, maximum_controls=total, transitions=0, updates=[], learning_updates=0))
    print('ROUTED_SEVEN_COMMAND_FROZEN', base.rel(record), sha(PLAN), flush=True)


def snapshot(env, obs, memory, spec, target):
    with patch.object(original, 'switch_models', ORIGINAL_SWITCH):
        actors, proof, fields, inputs = ORIGINAL_SNAPSHOT(env, obs, memory, spec, target)
    delivery = read(ROOT/read(DRAFT)['walking_delivery'][0])
    check = target_fixture(env, obs['actor'], spec, delivery, spec['environment']['seed'])
    assert check == read(OUT/'cuda-zero.json')['proof']
    proof['seven_command_targets'] = check
    assert env.common_step_counter == env._sim_step_counter == 0
    return actors, proof, fields, inputs


def collect(env, actors, walking, meta, spec, record, state, publish, guard):
    d = spec['seven_command']
    ramp = spec['ramped_forward']['ramp']
    state['terminal_commands'] = {name: len(ids) for name, ids in d['groups'].items()}
    state['recorded_world_command'] = d['video_command']

    class AdmissionController(SevenCommandSequenceController):
        def advance(self, control, sensed):
            x, bridge = super().advance(control, sensed)
            if control == self.forward_at:
                entered = base.shared.cpu(self.mode == FORWARD)
                counts = {name: int(entered[ids].sum()) for name, ids in d['groups'].items()}
                insufficient = [name for name, count in counts.items() if count < d['minimum_by_command'][name]]
                write(record/'handoff-entries.json', dict(control=control, entrants=counts, below_minimum=insufficient))
                state['handoff_entries'] = counts
                if insufficient or sum(counts.values()) < d['minimum_overall']:
                    state['early_stop'] = 'complete-sequence-upper-bound-below-original-gate'
                    raise RuntimeError(f'Incomplete handoff makes the frozen complete gate unreachable: {insufficient}')
            return x, bridge

    factory = partial(AdmissionController, commands=d['commands'], groups=d['groups'], routes=d['routes'], ramp=ramp)
    with patch.object(prior.acquisition, 'SequenceController', factory):
        prior.acquisition.collect(env, actors, walking, meta, spec, record, state, publish, guard)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    phase(PLAN)
    seed = read(PLAN)['environment']['seed']
    with patch.multiple(original, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=seed, phase=phase,
                        collect=collect, switch_models=route_models), \
         patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=seed, phase=phase, snapshot=snapshot), \
         patch.object(base.shared, 'prepare', original.prepare), \
         patch.object(base.shared, 'actor_model', route.guarded_actor):
        (route.prior.zero if mode == 'zero' else original.sample)()


if __name__ == '__main__':
    main()
