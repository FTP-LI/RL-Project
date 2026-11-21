"""Re-evaluate the full continuous sequence with the admitted startup ramp."""

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_routed_getup_sequence as original
import collect_routed_getup_sequence as acquisition
from routed_forward_entry_ramp import command_table
from routed_getup_sequence import FORWARD, STAND, ForwardFeedback, SequenceController


base, route = original.base, original.route
ROOT, read, sha, write = original.ROOT, original.read, original.sha, original.write
ORIGINAL_BOUNDARY_FIXTURE = original.boundary_fixture
DRAFT = ROOT / 'configs/experiments/continuous_routed_getup_ramped_forward_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_getup_ramped_forward')
OUT = ROOT / 'records/reports/continuous-routed-getup-ramped-forward'
PAGE = ROOT / 'records/continuous-routed-getup-ramped-forward-live.html'
STAGE, SEED = 'S5-routed-getup-ramped-forward', 1172


def phase(protocol):
    original.require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


class RampedSequenceController(SequenceController):
    def __init__(self, n, device, settings, homes, limits, forward, *, ramp, **boundaries):
        super().__init__(n, device, settings, homes, limits, forward, **boundaries)
        self.startup = torch.as_tensor(command_table(ramp, ramp['ramp_controls']), device=device)

    def advance(self, control, sensed):
        actor_input, bridge = super().advance(control, sensed)
        since_forward = control - self.forward_at
        if 0 <= since_forward < len(self.startup):
            actor_input[self.mode == FORWARD, 48:51] = self.startup[since_forward]
        return actor_input, bridge


def boundary_fixture(spec, actors, device, actual_term=None):
    p, s, d = (spec[key] for key in ('environment', 'sequence', 'ramped_forward'))
    factory = partial(RampedSequenceController, ramp=d['ramp'])
    with patch.object(original, 'SequenceController', factory):
        checks = ORIGINAL_BOUNDARY_FIXTURE(spec, actors, device, actual_term)
    homes = [p['model_home'], spec['switch_policies']['stand']['home'], spec['switch_policies']['forward']['home']]
    controller = factory(2, device, s['controller'], homes, p['joint_limits'], ForwardFeedback(spec['walking'], 2))
    controller.mode[:] = STAND
    controller.dwell[0] = controller.dwell_steps
    sensed = torch.zeros((2, 61), device=device)
    sensed[0, 5], sensed[1, 3] = -1., 1.
    logged = []
    with torch.inference_mode():
        for k in range(controller.forward_at, controller.forward_at + len(controller.startup) + 1):
            x, bridge = controller.advance(k, sensed)
            assert controller.mode.tolist() == [FORWARD, STAND] and not x[1, 48:].any()
            assert bool(bridge[0]) == (k < controller.forward_at + len(controller.startup))
            logged.append(base.shared.cpu(x[0, 48:51]))
    ramp = d['ramp']
    expected = command_table(ramp, ramp['ramp_controls'] + 1)
    np.testing.assert_array_equal(logged, expected)
    checks['ramp'] = dict(mapped_commands=expected.tolist(), existing_bridge=True, skipped_world_unchanged=True)
    return checks


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['development_seed'] == SEED and SEED not in d['excluded_for_tuning']
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io',
                                      'automatic_retry', 'automatic_extension'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source, trial, result = (route.pinned(d[key]) for key in ('source_sequence', 'admitted_pair', 'admission_result'))
    spec, pair, admission = map(read, (source, trial, result))
    assert admission['passed_diagnostic_gate'] and admission['protocol_sha256'] == sha(trial)
    assert admission['record'] == pair['record'] and not admission['heldout_used']
    assert pair['pair']['source_protocol'] == d['source_sequence']
    assert pair['pair']['branches']['ramped-command'] == d['ramp']
    p, s = spec['environment'], spec['sequence']
    n, total = d['worlds'], d['maximum_unique_controls']
    assert n == p['num_envs'] == sum(d['requested_pose_counts']) and total == spec['maximum_unique_controls'] == 1600
    assert d['requested_pose_counts'] == spec['dual_route']['requested_pose_counts']
    assert total * n == d['maximum_transitions'] == s['maximum_transitions']
    assert 5 * total == d['maximum_physics_substeps'] == spec['maximum_physics_substeps'] == 8000
    assert s['boundary_controls'] + [total] == d['boundaries'] and d['control_hz'] == 50
    assert d['ramp']['ramp_controls'] == s['controller']['cross_model_bridge_controls'] == 25
    assert d['ramp']['external_forward_m_s'] == s['controller']['forward_command_m_per_s'] == .15
    assert d['minimum_complete_sequence_fraction'] == s['proposed_minimum_complete_sequence_fraction'] == .95
    for key in ('minimum_actual_right_coverage', 'minimum_other_actual_pose_coverage'):
        assert d[key] == s[key]
    assert disk_usage(ROOT).free >= 12 * 1024 ** 3
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    run_id = f'duck_switch_routed-getup-ramped-forward_development_s{SEED}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=groups))
    spec['dual_route']['seed'] = SEED
    for policy in (*spec['policies'].values(), spec['common_fallback'], *spec['switch_policies'].values()):
        policy['seed'] = SEED
    spec.update(status='frozen-routed-getup-ramped-forward', stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
        ramped_forward=d)
    spec['draft']['gpu_memory_fraction_cap'] = d['maximum_gpu_memory_fraction']
    spec['references'].update({base.rel(path): sha(path) for path in (DRAFT, source, trial, result)})
    names = ('routed_forward_entry_ramp.py', 'verify_routed_forward_entry_ramp.py',
        'verify_routed_getup_ramped_forward.py', 'verify_walk_live_page.py')
    for path in (Path(__file__).resolve(), *(Path(__file__).with_name(name).resolve() for name in names)):
        spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    cfg, _, _ = original.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and cfg.env.decimation == 5
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    torch.set_num_threads(4)
    checks = boundary_fixture(spec, original.switch_models(spec, 'cpu'), 'cpu')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN), physical_substeps=0,
        cpu_scene_exact=True, boundaries=checks))
    (record / 'README.md').write_text('# 起身、站立与渐升前进复验\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-getup-ramped-forward-live.html)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身、站立与渐升前进连续复验', run_id=run_id,
        controls=0, maximum_controls=total, transitions=0, updates=[], learning_updates=0))
    print('ROUTED_GETUP_RAMPED_FORWARD_FROZEN', base.rel(record), sha(PLAN), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    phase(PLAN)
    spec = read(PLAN)
    factory = partial(RampedSequenceController, ramp=spec['ramped_forward']['ramp'])
    # Reuse the archived acquisition and its guards. Only the mapped startup command differs.
    with patch.multiple(original, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase), \
         patch.object(route.prior, 'snapshot', original.snapshot), \
         patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase), \
         patch.object(base.shared, 'prepare', original.prepare), \
         patch.object(base.shared, 'actor_model', route.guarded_actor), \
         patch.object(acquisition, 'SequenceController', factory), \
         patch.object(original, 'boundary_fixture', boundary_fixture):
        (route.prior.zero if mode == 'zero' else original.sample)()


if __name__ == '__main__':
    main()
