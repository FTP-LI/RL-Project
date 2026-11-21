"""Freeze and run the ramped continuous sequence on its reserved seed once."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_routed_getup_ramped_forward as development


original, base, route = development.original, development.base, development.route
ROOT, read, sha, write = development.ROOT, development.read, development.sha, development.write
DRAFT = ROOT / 'configs/experiments/continuous_routed_getup_ramped_forward_holdout_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_getup_ramped_forward_holdout')
OUT = ROOT / 'records/reports/continuous-routed-getup-ramped-forward/holdout'
PAGE = ROOT / 'records/continuous-routed-getup-ramped-forward-holdout-live.html'
RECEIPT = OUT / 'consumption.json'
STAGE, SEED = 'S5-routed-getup-ramped-forward-holdout', 2172


def phase(protocol):
    original.require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def dynamics(spec):
    """Only the seed and the pose permutation may differ at collection time."""
    value = deepcopy(spec)
    for key in ('status', 'stage', 'record', 'page', 'references', 'source_sha256',
                'reserved_holdout', 'data_scope', 'holdout'):
        value.pop(key, None)
    for key in ('seed', 'stage', 'run_id', 'split'):
        value['environment'].pop(key)
    value['dual_route'].pop('seed')
    for policy in (*value['policies'].values(), value['common_fallback'], *value['switch_policies'].values()):
        policy.pop('seed')
    return value


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['reserved_holdout_seed'] == SEED and SEED not in d['used_for_development'] + d['other_reserved_or_consumed_seeds']
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io',
                                      'automatic_retry', 'automatic_extension'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source, scored = (route.pinned(d[key]) for key in ('development_protocol', 'development_verdict'))
    assert source == development.PLAN
    spec, verdict = read(source), read(scored)
    assert verdict['status'] == 'passed-independent-verification' and verdict['protocol_sha256'] == sha(source)
    assert verdict['record'] == spec['record'] and verdict['score']['passed_development_gate'] and not verdict['heldout_used']
    p, s = spec['environment'], spec['sequence']
    n, total = d['worlds'], d['maximum_unique_controls']
    assert p['seed'] in d['used_for_development'] and n == p['num_envs'] == sum(d['requested_pose_counts'])
    assert d['requested_pose_counts'] == spec['dual_route']['requested_pose_counts']
    assert total == spec['maximum_unique_controls'] == 1600 and total*n == d['maximum_transitions'] == s['maximum_transitions']
    assert 5*total == d['maximum_batched_physics_substeps'] == spec['maximum_physics_substeps']
    assert d['boundaries'] == s['boundary_controls'] + [total] and d['control_hz'] == 50
    assert d['ramp'] == spec['ramped_forward']['ramp'] and d['maximum_gpu_memory_fraction'] == spec['draft']['gpu_memory_fraction_cap']
    assert d['minimum_complete_sequence_fraction'] == s['proposed_minimum_complete_sequence_fraction'] == .95
    for key in ('minimum_actual_right_coverage', 'minimum_other_actual_pose_coverage'):
        assert d[key] == s[key]
    assert disk_usage(ROOT).free >= 12 * 1024 ** 3
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    run_id = f'duck_switch_routed-getup-ramped-forward_holdout_s{SEED}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=groups))
    spec['dual_route']['seed'] = SEED
    for policy in (*spec['policies'].values(), spec['common_fallback'], *spec['switch_policies'].values()):
        policy['seed'] = SEED
    spec.update(status='frozen-routed-getup-ramped-forward-holdout', stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
        reserved_holdout=True, data_scope='reserved-holdout', holdout=d)
    spec['references'].update({base.rel(path): sha(path) for path in (DRAFT, source, scored)})
    for path in (Path(__file__).resolve(), Path(__file__).with_name('verify_routed_getup_ramped_forward_holdout.py').resolve()):
        spec['source_sha256'][base.rel(path)] = sha(path)
    assert dynamics(spec) == dynamics(read(source))
    base.shared.assert_sources(spec)
    cfg, _, _ = original.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and cfg.env.decimation == 5
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    torch.set_num_threads(4)
    checks = development.boundary_fixture(spec, original.switch_models(spec, 'cpu'), 'cpu')
    assert checks == read(development.OUT / 'freeze.json')['boundaries'] and not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN), physical_substeps=0,
        cpu_scene_exact=True, dynamics_match_development=True, boundaries=checks))
    (record / 'README.md').write_text('# 起身、站立与渐升前进留出\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-getup-ramped-forward-holdout-live.html)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身、站立与渐升前进连续留出', run_id=run_id,
        controls=0, maximum_controls=total, transitions=0, updates=[], learning_updates=0))
    print('ROUTED_GETUP_RAMPED_HOLDOUT_FROZEN', base.rel(record), sha(PLAN), flush=True)


def consume_and_collect(env, actors, walking_actors, meta, spec, record, state, publish, check_active):
    assert env.common_step_counter == env._sim_step_counter == 0
    assert state['actual'] == dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=1)
    assert sha(PLAN) == state['protocol_sha256'] and spec['reserved_holdout']
    check_active()
    ready = record / 'windows-observer-ready.json'
    native = OUT / 'native-zero.json'
    assert read(native)['status'] == 'passed-native-zero' and read(ready)['protocol_sha256'] == sha(PLAN)
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(dict(status='claimed-before-physics', seed=SEED, record=base.rel(record),
            protocol_sha256=sha(PLAN), native_zero_sha256=sha(native), observer_ready_sha256=sha(ready),
            actual_before_claim=state['actual'], retry_allowed=False, holdout_eligible_for_tuning=False,
            claimed_at=datetime.now(timezone.utc).isoformat()), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    state.update(heldout_used=True, data_scope=spec['data_scope'], holdout_consumption_sha256=sha(RECEIPT))
    publish()
    return development.acquisition.collect(env, actors, walking_actors, meta, spec, record, state, publish, check_active)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    phase(PLAN)
    assert not RECEIPT.exists()
    spec = read(PLAN)
    assert spec['reserved_holdout'] and spec['data_scope'] == 'reserved-holdout'
    factory = partial(development.RampedSequenceController, ramp=spec['ramped_forward']['ramp'])
    with patch.multiple(original, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, collect=consume_and_collect), \
         patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=original.snapshot), \
         patch.object(base.shared, 'prepare', original.prepare), \
         patch.object(base.shared, 'actor_model', route.guarded_actor), \
         patch.object(development.acquisition, 'SequenceController', factory), \
         patch.object(original, 'boundary_fixture', development.boundary_fixture):
        (route.prior.zero if mode == 'zero' else original.sample)()


if __name__ == '__main__':
    main()
