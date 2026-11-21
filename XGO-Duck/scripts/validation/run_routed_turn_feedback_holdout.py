"""Freeze and consume the single independent seven-command holdout."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from math import ceil
import os
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import collect_routed_turn_feedback_pair as acquisition
import run_routed_turn_feedback_pair as pair
from routed_seven_command_controller import balanced_groups


ROOT, read, sha, write = pair.ROOT, pair.read, pair.sha, pair.write
base, route, original = pair.base, pair.route, pair.original
DRAFT = ROOT/'configs/experiments/continuous_routed_turn_feedback_holdout_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_turn_feedback_holdout')
OUT = ROOT/'records/reports/continuous-routed-turn-feedback-holdout'
PAGE = ROOT/'records/continuous-routed-turn-feedback-holdout-live.html'
STAGE = 'S5-routed-turn-feedback-holdout'
ARM = 'restored-gyro'


def phase(protocol):
    original.require_stage(protocol, STAGE+'-preparation' if protocol == DRAFT else STAGE)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['status'] == 'reserved-no-physical-execution' and d['stage'] == STAGE+'-preparation'
    assert not any(d[key] for key in ('physical_substeps_authorized','holdout_consumed','automatic_retry',
                                      'automatic_extension','hardware_io','learning_updates'))
    for name, expected in d['pinned'].items():
        assert sha(ROOT/name) == expected, name
    proof = read(pair.OUT/'analysis.json')
    assert proof['restored_feedback_admitted'] and not proof['heldout_used']
    assert proof['scores'][ARM]['passed_development_gate'] and proof['protocol_sha256'] == sha(pair.PLAN)
    spec = read(pair.PLAN)
    p,s,seven = (spec[key] for key in ('environment','sequence','seven_command'))
    seed,n,steps = (d[key] for key in ('seed','worlds','controls'))
    assert seed == 2175 and n == p['num_envs'] and steps == spec['per_branch_horizon']
    assert d['requested_pose_counts'] == spec['dual_route']['requested_pose_counts']
    assert d['maximum_transitions'] == n*steps and d['maximum_batched_physics_substeps'] == steps*5
    assert d['trace_zip_level'] == spec['turn_pair']['trace_zip_level']
    assert d['minimum_fraction_each_command'] == d['minimum_fraction_overall'] == .95
    assert disk_usage(ROOT).free >= 8*1024**3
    for folder in ('records/evaluations','records/experiments','records/training-archive/evaluations'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/folder).iterdir())
    poses = np.random.default_rng(seed).permutation(np.repeat(np.arange(4),d['requested_pose_counts']))
    groups = balanced_groups(poses,seven['commands'],seed+70000)
    original_camera_group = next(name for name, ids in groups.items() if 0 in ids)
    if original_camera_group != 'turn-left':
        replacement = min(i for i in groups['turn-left'] if poses[i] == poses[0])
        groups[original_camera_group] = sorted(replacement if i == 0 else i for i in groups[original_camera_group])
        groups['turn-left'] = sorted(0 if i == replacement else i for i in groups['turn-left'])
    assert sorted(i for ids in groups.values() for i in ids) == list(range(n))
    run_id = f'duck_switch_routed-turn-gyro_holdout_s{seed}_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT/'records/evaluations'/run_id
    p.update(seed=seed,stage=STAGE,run_id=run_id,split=dict(pose_codes=poses.tolist(),
        groups={pose:np.flatnonzero(poses == i).tolist() for i,pose in enumerate(base.shared.POSES)}))
    spec['dual_route']['seed'] = seed
    for policy in (*spec['policies'].values(),spec['common_fallback'],*spec['switch_policies'].values(),
                   *seven['policies'].values()):
        policy['seed'] = seed
    seven.update(groups=groups,minimum_by_command={name:ceil(.95*len(ids)) for name,ids in groups.items()},
                 minimum_overall=ceil(.95*n),video_command='turn-left')
    limit = spec['turn_pair']['maximum_gpu_memory_fraction']
    assert limit == spec['draft']['gpu_memory_fraction_cap']
    # The archived sampler reads the cap from this key. Freeze the adapter
    # as part of the protocol so launch cannot fail after claiming the seed.
    spec.update(status='frozen-single-use-holdout',stage=STAGE,reserved_holdout=True,data_scope='reserved-holdout',
        record=base.rel(record),page=base.rel(PAGE),maximum_unique_controls=steps,maximum_physics_substeps=steps*5,
        maximum_sampling_wall_seconds=660,pair={'maximum_gpu_memory_fraction':limit},
        final_holdout=dict(d,arm=ARM,camera_original_group=original_camera_group))
    spec['references'].update(d['pinned'])
    spec['references'][base.rel(DRAFT)] = sha(DRAFT)
    for name in ('run_routed_turn_feedback_holdout.py','verify_routed_turn_feedback_holdout.py'):
        source = Path(__file__).with_name(name).resolve()
        spec['source_sha256'][base.rel(source)] = sha(source)
    base.shared.assert_sources(spec)
    cfg,_,_ = original.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and not cfg.env.auto_reset
    base.exact_scene(Scene(cfg.env.scene,device='cpu').compile())
    torch.set_num_threads(4)
    fixture = pair.feedback_fixture(spec,'cpu')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,record/'protocol.json')
    write(OUT/'freeze.json',dict(status='passed-cpu-freeze',protocol_sha256=sha(PLAN),physical_substeps=0,
        seed=seed,exact_cpu_scene=True,turn_fixture=fixture,record=spec['record']))
    (record/'README.md').write_text('# 转向反馈独立留出\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-turn-feedback-holdout-live.html)\n',encoding='utf-8')
    base.publish(record,PAGE,dict(status='starting',purpose='七指令独立留出',run_id=run_id,
        controls=0,maximum_controls=steps,transitions=0,updates=[],learning_updates=0))
    print('ROUTED_TURN_HOLDOUT_FROZEN',spec['record'],sha(PLAN),flush=True)


def snapshot(env,obs,memory,spec,target):
    with patch.object(original,'switch_models',pair.prior.ORIGINAL_SWITCH):
        actors,proof,fields,inputs = pair.prior.ORIGINAL_SNAPSHOT(env,obs,memory,spec,target)
    check = pair.feedback_fixture(spec,env.device)
    assert check == read(OUT/'freeze.json')['turn_fixture']
    proof['sequence_boundaries']['turn_input_restore'] = check
    assert env.common_step_counter == env._sim_step_counter == 0
    return actors,proof,fields,inputs


def collect(env,actors,walking,memory,spec,record,state,publish,guard):
    ready = record/'windows-observer-ready.json'
    native = OUT/'native-zero.json'
    independent = OUT/'independent-zero/proof.json'
    checksum = sha(PLAN)
    assert state['status'] == 'running' and read(ready)['protocol_sha256'] == checksum
    assert read(native)['status'] == 'passed-native-zero' and read(independent)['physical_substeps'] == 0
    selected = {ARM:acquisition.controllers(spec,env.device)[ARM]}
    step = env.step
    consumption = record/'holdout-consumption.json'
    claimed = False

    def first_step(*args,**kwargs):
        nonlocal claimed
        if not claimed:
            guard()
            assert env.common_step_counter == env._sim_step_counter == 0
            assert state['actual']['controls'] == state['actual']['substeps'] == 0
            receipt = dict(status='claimed-before-first-physics',seed=spec['environment']['seed'],protocol_sha256=checksum,
                windows_observer_sha256=sha(ready),native_zero_sha256=sha(native),independent_zero_sha256=sha(independent),
                physical_substeps_before_claim=0,automatic_retry=False,heldout_used_for_tuning=False,
                claimed_at=datetime.now(timezone.utc).isoformat())
            with consumption.open('x',encoding='utf-8') as f:
                json.dump(receipt,f,indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            state.update(heldout_used=True,holdout_consumption_sha256=sha(consumption))
            claimed = True
        return step(*args,**kwargs)

    with patch.multiple(acquisition,ROLES=(ARM,),controllers=lambda spec,device:selected), patch.object(env,'step',first_step):
        acquisition.collect(env,actors,walking,memory,spec,record,state,publish,guard)
    assert claimed and len(state['branches']) == 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','zero','sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    phase(PLAN)
    spec = read(PLAN)
    assert not (ROOT/spec['record']/'holdout-consumption.json').exists()
    seed = spec['environment']['seed']
    with patch.multiple(pair.pair_runner,PLAN=PLAN,OUT=OUT,PAGE=PAGE,SEED=seed,phase=phase,collect=collect), \
         patch.multiple(route.prior,PLAN=PLAN,OUT=OUT,PAGE=PAGE,SEED=seed,phase=phase,snapshot=snapshot), \
         patch.object(base.shared,'prepare',original.prepare), \
         patch.object(base.shared,'actor_model',route.guarded_actor), \
         patch.object(original,'switch_models',pair.prior.route_models):
        (route.prior.zero if mode == 'zero' else pair.pair_runner.sample)()


if __name__ == '__main__':
    main()
